"""
提醒设置节点
- 解析用户输入的时间和提醒内容
- 信息不足时追问
- 调用 App 后端接口上报提醒
- 写入 reminders 表
"""
from __future__ import annotations

import json
from datetime import datetime

import pytz
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
        temperature=0.1,
        max_tokens=512,
        timeout=settings.qwen.request_timeout,
    )


def _get_current_datetime_str(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间字符串（供 LLM 做相对时间解析）"""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z（%A）")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def reminder_node(state: AgentState) -> dict:
    """
    提醒设置节点

    流程：
    1. 用 QwenPlus 解析时间和提醒内容
    2. 如果信息不足，生成追问
    3. 调用 App API 上报
    4. 写入数据库
    """
    query = state.get("rewritten_query") or state.get("original_query", "")
    user_profile = state.get("user_profile", {})
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    timezone_str = user_profile.get("timezone", "Asia/Shanghai")

    current_time = _get_current_datetime_str(timezone_str)
    llm = _build_plus_llm()

    # ── 步骤 1：解析提醒信息 ──────────────────────────────────
    parse_system = SystemMessage(content=f"""你是一个时间和提醒解析专家。
当前时间：{current_time}

请从用户输入中提取提醒信息，以 JSON 格式返回：
{{
  "has_enough_info": true/false,
  "missing_fields": ["time", "content"],  // 缺少的字段
  "clarification_question": "追问内容",   // has_enough_info=false 时填写
  "title": "提醒标题（简洁）",
  "description": "详细描述",
  "reminder_type": "workout|meal|medication|general",
  "remind_at_iso": "2024-01-15T15:00:00+08:00",  // ISO 8601 格式，has_enough_info=true 时填写
  "recurrence_rule": null  // 重复规则，例如 "FREQ=DAILY;INTERVAL=1"，不重复则为 null
}}

时间解析规则：
- "下午3点" → 今天 15:00
- "明天早上8点" → 明天 08:00
- "每天" → recurrence_rule: "FREQ=DAILY;INTERVAL=1"
- "每周一" → recurrence_rule: "FREQ=WEEKLY;BYDAY=MO"

只返回 JSON，不要其他文字。""")

    parse_human = HumanMessage(content=f"用户输入：{query}")

    try:
        parse_resp = await llm.ainvoke([parse_system, parse_human])
        raw = parse_resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
    except Exception as e:
        logger.error("reminder_node: 解析失败", error=str(e))
        parsed = {"has_enough_info": False, "clarification_question": "请告诉我提醒的具体时间和内容？"}

    # ── 步骤 2：信息不足则追问 ────────────────────────────────
    if not parsed.get("has_enough_info", False):
        clarification = parsed.get(
            "clarification_question",
            "请提供更多信息，例如提醒时间和提醒内容。"
        )
        logger.info("reminder_node: 信息不足，追问", question=clarification)
        return {
            "need_clarification": True,
            "clarification_question": clarification,
            "reminder_info": parsed,
            "final_response": clarification,
            "messages": [AIMessage(content=clarification)],
        }

    # ── 步骤 3：写入数据库 ────────────────────────────────────
    reminder_data = {
        "user_id": user_id,
        "session_id": session_id or None,
        "title": parsed.get("title", "提醒"),
        "description": parsed.get("description"),
        "reminder_type": parsed.get("reminder_type", "general"),
        "remind_at": parsed.get("remind_at_iso"),
        "timezone": timezone_str,
        "recurrence_rule": parsed.get("recurrence_rule"),
    }

    app_reminder_id = None
    try:
        from fitness_agent.services.app_api import AppAPIClient
        async with AppAPIClient() as client:
            app_result = await client.create_reminder(reminder_data)
            app_reminder_id = app_result.get("reminder_id")
            logger.info("reminder_node: App 提醒创建成功", app_reminder_id=app_reminder_id)
    except Exception as e:
        logger.warning("reminder_node: App 接口调用失败，仅写入数据库", error=str(e))

    # 数据库写入
    try:
        from fitness_agent.models.database import Reminder, get_session, create_engine
        import uuid as uuid_module

        engine = create_engine()
        async for session in get_session(engine):
            remind_at_dt = None
            if reminder_data.get("remind_at"):
                from dateutil.parser import parse as parse_dt
                remind_at_dt = parse_dt(reminder_data["remind_at"])

            reminder_obj = Reminder(
                user_id=uuid_module.UUID(user_id) if user_id else None,
                session_id=uuid_module.UUID(session_id) if session_id else None,
                title=reminder_data["title"],
                description=reminder_data.get("description"),
                reminder_type=reminder_data.get("reminder_type", "general"),
                remind_at=remind_at_dt,
                timezone=timezone_str,
                recurrence_rule=reminder_data.get("recurrence_rule"),
                app_reminder_id=app_reminder_id,
            )
            session.add(reminder_obj)
            logger.info("reminder_node: 数据库写入成功")
    except Exception as e:
        logger.error("reminder_node: 数据库写入失败", error=str(e))

    # ── 步骤 4：生成回复 ──────────────────────────────────────
    title = parsed.get("title", "提醒")
    remind_at_str = parsed.get("remind_at_iso", "")
    # 格式化时间显示
    try:
        from dateutil.parser import parse as parse_dt
        dt = parse_dt(remind_at_str)
        tz = pytz.timezone(timezone_str)
        dt_local = dt.astimezone(tz)
        time_display = dt_local.strftime("%Y年%m月%d日 %H:%M")
    except Exception:
        time_display = remind_at_str

    recurrence = parsed.get("recurrence_rule")
    if recurrence:
        final_response = f"✅ 已为您设置【{title}】的循环提醒！首次提醒时间：{time_display}"
    else:
        final_response = f"✅ 已为您设置提醒：【{title}】，提醒时间：{time_display}"

    return {
        "need_clarification": False,
        "reminder_info": reminder_data,
        "final_response": final_response,
        "structured_output": {"reminder": reminder_data, "app_reminder_id": app_reminder_id},
        "messages": [AIMessage(content=final_response)],
    }
