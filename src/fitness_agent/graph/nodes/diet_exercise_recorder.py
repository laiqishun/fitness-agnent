"""
饮食/运动记录节点
- 解析饮食内容，估算营养成分（QwenPlus）
- 解析运动内容，估算热量消耗
- 信息不足则追问
- 写入 diet_records / exercise_records 表
- 支持混合意图（同时记录饮食+设置提醒）
"""
from __future__ import annotations

import json
import uuid as uuid_module
from datetime import datetime

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from fitness_agent.config import get_settings
from fitness_agent.graph.state import AgentState

logger = structlog.get_logger(__name__)


def _build_plus_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.qwen.plus_model,
        api_key=settings.qwen.api_key,
        base_url=settings.qwen.base_url,
        temperature=0.2,
        max_tokens=2048,
        timeout=settings.qwen.request_timeout,
    )


async def _parse_diet(llm: ChatOpenAI, query: str, current_time: str) -> dict:
    """解析饮食信息并估算营养成分"""
    system = SystemMessage(content=f"""你是营养分析专家。当前时间：{current_time}

从用户输入中提取饮食记录，以 JSON 格式返回：
{{
  "has_diet_info": true/false,
  "meal_type": "breakfast|lunch|dinner|snack|other",
  "recorded_at_iso": "ISO时间",
  "food_items": [
    {{
      "name": "食物名称",
      "amount_g": 200,
      "amount_desc": "一碗",
      "calories_kcal": 232.0,
      "protein_g": 4.3,
      "carbs_g": 50.9,
      "fat_g": 0.5,
      "fiber_g": 0.5
    }}
  ],
  "total_calories_kcal": 232.0,
  "total_protein_g": 4.3,
  "total_carbs_g": 50.9,
  "total_fat_g": 0.5,
  "estimate_confidence": 0.85,
  "notes": "备注"
}}

营养估算规则：
- 米饭(100g): 热量116kcal, 蛋白2.6g, 碳水25.6g, 脂肪0.3g
- 鸡蛋(1个50g): 热量72kcal, 蛋白6.2g, 碳水0.6g, 脂肪4.8g
- 如果不确定重量，根据描述合理估算
- 请基于中国食物营养数据库估算

只返回 JSON，不要其他文字。""")

    human = HumanMessage(content=f"用户输入：{query}")

    try:
        resp = await llm.ainvoke([system, human])
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        logger.error("diet_parser: 解析失败", error=str(e))
        return {"has_diet_info": False}


async def _parse_exercise(llm: ChatOpenAI, query: str, user_profile: dict, current_time: str) -> dict:
    """解析运动信息并估算热量消耗"""
    weight_kg = user_profile.get("weight_kg", 65)

    system = SystemMessage(content=f"""你是运动科学专家。当前时间：{current_time}
用户体重：{weight_kg}kg

从用户输入中提取运动记录，以 JSON 格式返回：
{{
  "has_exercise_info": true/false,
  "exercise_type": "strength|cardio|flexibility|balance|hiit|other",
  "duration_min": 45,
  "recorded_at_iso": "ISO时间",
  "exercise_items": [
    {{
      "name": "哑铃卧推",
      "sets": 3,
      "reps": 12,
      "weight_kg": 20.0,
      "rest_sec": 60,
      "calories_kcal": 45.0
    }}
  ],
  "calories_burned_kcal": 300.0,
  "estimate_confidence": 0.8,
  "notes": "备注"
}}

热量估算规则（MET值法）：
- 力量训练: MET=3.5-6, 有氧: MET=4-10, HIIT: MET=8-12
- 热量 = MET × 体重(kg) × 时长(h) × 1.05

只返回 JSON，不要其他文字。""")

    human = HumanMessage(content=f"用户输入：{query}")

    try:
        resp = await llm.ainvoke([system, human])
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        logger.error("exercise_parser: 解析失败", error=str(e))
        return {"has_exercise_info": False}


async def diet_exercise_recorder_node(state: AgentState) -> dict:
    """
    饮食/运动记录节点

    支持：
    - 纯饮食记录
    - 纯运动记录
    - 混合记录（饮食+运动）
    - 混合意图（记录+设置提醒，提醒部分路由到 reminder_node）
    """
    query = state.get("rewritten_query") or state.get("original_query", "")
    user_profile = state.get("user_profile", {})
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    sub_intents = state.get("sub_intents", [])

    import pytz
    from datetime import datetime as dt_cls
    timezone_str = user_profile.get("timezone", "Asia/Shanghai")
    try:
        tz = pytz.timezone(timezone_str)
        current_time = dt_cls.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        current_time = dt_cls.now().strftime("%Y-%m-%d %H:%M:%S")

    llm = _build_plus_llm()

    # ── 并行解析饮食和运动 ────────────────────────────────────
    import asyncio
    diet_task = _parse_diet(llm, query, current_time)
    exercise_task = _parse_exercise(llm, query, user_profile, current_time)
    diet_result, exercise_result = await asyncio.gather(diet_task, exercise_task)

    has_diet = diet_result.get("has_diet_info", False)
    has_exercise = exercise_result.get("has_exercise_info", False)

    # ── 如果都没有解析到，追问 ────────────────────────────────
    if not has_diet and not has_exercise:
        clarification = "请告诉我您想记录的饮食或运动内容，例如：'刚吃了一碗米饭两个鸡蛋' 或 '跑步30分钟'。"
        return {
            "need_clarification": True,
            "clarification_question": clarification,
            "final_response": clarification,
            "messages": [AIMessage(content=clarification)],
        }

    reply_parts = []
    diet_info = {}
    exercise_info = {}

    # ── 写入饮食记录 ───────────────────────────────────────────
    if has_diet:
        diet_info = diet_result
        try:
            from fitness_agent.models.database import DietRecord, get_session, create_engine
            from dateutil.parser import parse as parse_dt

            engine = create_engine()
            async for session in get_session(engine):
                recorded_at = None
                if diet_result.get("recorded_at_iso"):
                    try:
                        recorded_at = parse_dt(diet_result["recorded_at_iso"])
                    except Exception:
                        recorded_at = dt_cls.now()

                record = DietRecord(
                    user_id=uuid_module.UUID(user_id) if user_id else None,
                    session_id=uuid_module.UUID(session_id) if session_id else None,
                    recorded_at=recorded_at or dt_cls.now(),
                    meal_type=diet_result.get("meal_type", "other"),
                    raw_input=query,
                    food_items=diet_result.get("food_items", []),
                    total_calories_kcal=float(diet_result.get("total_calories_kcal", 0)),
                    total_protein_g=float(diet_result.get("total_protein_g", 0)),
                    total_carbs_g=float(diet_result.get("total_carbs_g", 0)),
                    total_fat_g=float(diet_result.get("total_fat_g", 0)),
                    total_fiber_g=float(diet_result.get("total_fiber_g", 0)),
                    estimate_confidence=float(diet_result.get("estimate_confidence", 0.8)),
                    notes=diet_result.get("notes"),
                )
                session.add(record)
                logger.info("diet_recorder: 饮食记录写入成功")
        except Exception as e:
            logger.error("diet_recorder: 写入失败", error=str(e))

        total_cal = diet_result.get("total_calories_kcal", 0)
        meal_name = {"breakfast": "早餐", "lunch": "午餐", "dinner": "晚餐",
                     "snack": "加餐", "other": "饮食"}.get(
            diet_result.get("meal_type", "other"), "饮食"
        )
        reply_parts.append(f"✅ {meal_name}已记录！共摄入约 **{total_cal:.0f} 千卡**")

        # 添加营养摘要
        protein = diet_result.get("total_protein_g", 0)
        carbs = diet_result.get("total_carbs_g", 0)
        fat = diet_result.get("total_fat_g", 0)
        reply_parts.append(f"   蛋白质 {protein:.1f}g | 碳水 {carbs:.1f}g | 脂肪 {fat:.1f}g")

    # ── 写入运动记录 ───────────────────────────────────────────
    if has_exercise:
        exercise_info = exercise_result
        try:
            from fitness_agent.models.database import ExerciseRecord, get_session, create_engine
            from dateutil.parser import parse as parse_dt

            engine = create_engine()
            async for session in get_session(engine):
                recorded_at = None
                if exercise_result.get("recorded_at_iso"):
                    try:
                        recorded_at = parse_dt(exercise_result["recorded_at_iso"])
                    except Exception:
                        recorded_at = dt_cls.now()

                record = ExerciseRecord(
                    user_id=uuid_module.UUID(user_id) if user_id else None,
                    session_id=uuid_module.UUID(session_id) if session_id else None,
                    recorded_at=recorded_at or dt_cls.now(),
                    duration_min=exercise_result.get("duration_min"),
                    raw_input=query,
                    exercise_type=exercise_result.get("exercise_type", "other"),
                    exercise_items=exercise_result.get("exercise_items", []),
                    calories_burned_kcal=float(exercise_result.get("calories_burned_kcal", 0)),
                    estimate_confidence=float(exercise_result.get("estimate_confidence", 0.8)),
                    notes=exercise_result.get("notes"),
                )
                session.add(record)
                logger.info("exercise_recorder: 运动记录写入成功")
        except Exception as e:
            logger.error("exercise_recorder: 写入失败", error=str(e))

        cal_burned = exercise_result.get("calories_burned_kcal", 0)
        duration = exercise_result.get("duration_min", 0)
        reply_parts.append(
            f"💪 运动已记录！时长约 **{duration} 分钟**，消耗约 **{cal_burned:.0f} 千卡**"
        )

    # ── 混合意图：同时有 set_reminder ────────────────────────
    if "set_reminder" in sub_intents:
        reply_parts.append("\n📌 检测到您还想设置提醒，正在为您处理...")

    final_response = "\n".join(reply_parts)

    return {
        "need_clarification": False,
        "diet_info": diet_info,
        "exercise_info": exercise_info,
        "final_response": final_response,
        "structured_output": {
            "diet": diet_info if has_diet else None,
            "exercise": exercise_info if has_exercise else None,
        },
        "messages": [AIMessage(content=final_response)],
    }
