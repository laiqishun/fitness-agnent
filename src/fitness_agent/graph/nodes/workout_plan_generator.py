"""
运动计划生成节点
实现「构思（Think）→ 执行（Execute）→ 反思（Reflect）」循环
- 最多循环 3 次，超过则降级为通用建议
- 缺少关键信息时生成结构化提问
"""
from __future__ import annotations

import json
import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from fitness_agent.config import get_settings
from fitness_agent.graph.state import AgentState

logger = structlog.get_logger(__name__)

MAX_ITERATIONS = 3   # 最大循环次数


def _build_plus_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.qwen.plus_model,
        api_key=settings.qwen.api_key,
        base_url=settings.qwen.base_url,
        temperature=0.7,
        max_tokens=4096,
        timeout=settings.qwen.request_timeout,
    )


def _profile_to_context(profile: dict) -> str:
    """将用户画像转为详细文字描述"""
    if not profile:
        return "用户画像：未知"
    lines = [
        f"健身水平：{profile.get('fitness_level', '未知')}",
        f"主要目标：{profile.get('primary_goal', '未知')}",
        f"每周锻炼天数：{profile.get('weekly_workout_days', '未知')} 天",
        f"单次锻炼时长偏好：{profile.get('preferred_workout_duration_min', '未知')} 分钟",
        f"可用器械：{', '.join(profile.get('available_equipment', [])) or '未知'}",
        f"锻炼场地：{profile.get('workout_location', '未知')}",
        f"伤病史：{', '.join(profile.get('injury_history', [])) or '无'}",
        f"健康状况：{', '.join(profile.get('health_conditions', [])) or '无'}",
        f"身高/体重：{profile.get('height_cm', '?')}cm / {profile.get('weight_kg', '?')}kg",
        f"饮食限制：{', '.join(profile.get('dietary_restrictions', [])) or '无'}",
    ]
    return "\n".join(lines)


# ── 三阶段节点函数 ──────────────────────────────────────────────


async def workout_plan_think_node(state: AgentState) -> dict:
    """
    构思阶段：分析用户情况，判断是否信息充足，制定计划框架

    输出：plan_thoughts, need_clarification（可能）
    """
    query = state.get("rewritten_query") or state.get("original_query", "")
    user_profile = state.get("user_profile", {})
    iteration = state.get("plan_iteration", 0)

    logger.info("workout_plan_think: 开始构思", iteration=iteration)

    profile_context = _profile_to_context(user_profile)
    llm = _build_plus_llm()

    think_system = SystemMessage(content="""你是一位资深 NSCA/ACSM 认证的健身私教，正在为用户制定运动计划。

【构思阶段】
请分析用户情况，判断是否有足够信息制定个性化计划。

输出 JSON：
{
  "has_enough_info": true/false,
  "missing_info": ["需要补充的信息列表"],
  "clarification_question": "如果信息不足，提出结构化问题",
  "analysis": {
    "fitness_assessment": "对用户当前健身状态的评估",
    "goal_analysis": "目标分析与可行性评估",
    "constraint_analysis": "器械/时间/伤病限制分析",
    "training_principle": "拟采用的训练原则（渐进超负荷/分化训练/功能性训练等）",
    "plan_framework": "计划框架概述（几天/周，每天什么类型）"
  }
}

只返回 JSON，不要其他文字。""")

    think_human = HumanMessage(content=f"""用户画像：
{profile_context}

用户请求：{query}

请进行构思分析：""")

    try:
        resp = await llm.ainvoke([think_system, think_human])
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        think_result = json.loads(raw.strip())
    except Exception as e:
        logger.error("workout_plan_think: 解析失败", error=str(e))
        think_result = {"has_enough_info": True, "analysis": {}}

    # 信息不足，追问
    if not think_result.get("has_enough_info", True):
        clarification = think_result.get(
            "clarification_question",
            "为了制定个性化运动计划，我需要了解一些信息：\n"
            "1. 您的主要健身目标是什么？\n"
            "2. 您每周能安排几天锻炼？\n"
            "3. 您家里有哪些健身器材？\n"
            "4. 您是否有任何伤病或身体限制？"
        )
        logger.info("workout_plan_think: 信息不足，追问")
        return {
            "need_clarification": True,
            "clarification_question": clarification,
            "plan_thoughts": json.dumps(think_result, ensure_ascii=False),
            "final_response": clarification,
            "messages": [AIMessage(content=clarification)],
        }

    plan_thoughts = json.dumps(think_result.get("analysis", {}), ensure_ascii=False, indent=2)
    logger.info("workout_plan_think: 构思完成")
    return {
        "need_clarification": False,
        "plan_thoughts": plan_thoughts,
        "plan_iteration": iteration,
    }


async def workout_plan_execute_node(state: AgentState) -> dict:
    """
    执行阶段：检索权威资料，生成详细运动计划初稿

    输出：plan_draft
    """
    query = state.get("rewritten_query") or state.get("original_query", "")
    user_profile = state.get("user_profile", {})
    plan_thoughts = state.get("plan_thoughts", "")
    settings = get_settings()

    logger.info("workout_plan_execute: 开始生成计划")

    # ── 检索权威训练资料 ──────────────────────────────────────
    rag_docs = ""
    try:
        from fitness_agent.rag.retriever import HybridRetriever
        from fitness_agent.rag.reranker import GteReranker

        retriever = HybridRetriever()
        reranker = GteReranker()

        plan_query = f"运动计划 {query} {user_profile.get('primary_goal', '')}"
        docs_raw = await retriever.retrieve_documents(plan_query, top_k=8)
        if docs_raw:
            docs_reranked = await reranker.rerank(query=plan_query, docs=docs_raw, top_n=4)
            rag_docs = "\n\n".join([
                f"[参考{i+1}] {d.get('title', '')}: {d.get('content', '')[:300]}"
                for i, d in enumerate(docs_reranked)
            ])
    except Exception as e:
        logger.warning("workout_plan_execute: RAG 检索失败", error=str(e))

    profile_context = _profile_to_context(user_profile)
    llm = _build_plus_llm()

    execute_system = SystemMessage(content=f"""你是 NSCA/ACSM 认证的专业健身私教。
基于以下分析和参考资料，生成完整的个性化运动计划。

## 用户画像
{profile_context}

## 构思分析
{plan_thoughts}

## 参考资料
{rag_docs or '（无检索资料，使用专业知识）'}

## 计划要求
请生成结构化的运动计划，以 JSON 格式返回：
{{
  "plan_name": "计划名称",
  "goal": "计划目标",
  "duration_weeks": 4,
  "weekly_schedule": [
    {{
      "day": 1,
      "name": "胸肩训练日",
      "exercises": [
        {{
          "name": "哑铃卧推",
          "sets": 3,
          "reps": 12,
          "weight_kg": null,
          "rest_sec": 60,
          "notes": "保持核心收紧"
        }}
      ],
      "estimated_duration_min": 45,
      "notes": "训练注意事项"
    }}
  ],
  "equipment_required": ["所需器械"],
  "warm_up_tips": "热身建议",
  "cool_down_tips": "整理运动建议",
  "nutrition_tips": "营养配合建议",
  "safety_notes": "安全注意事项",
  "progression_guide": "进阶指南"
}}

只返回 JSON，不要其他文字。""")

    execute_human = HumanMessage(content=f"用户原始请求：{query}\n\n请生成详细运动计划：")

    try:
        resp = await llm.ainvoke([execute_system, execute_human])
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        plan_draft = raw.strip()
        # 验证 JSON 合法性
        json.loads(plan_draft)
        logger.info("workout_plan_execute: 计划生成完成")
    except Exception as e:
        logger.error("workout_plan_execute: 生成失败", error=str(e))
        plan_draft = json.dumps({
            "plan_name": "个性化健身计划",
            "goal": "综合健身",
            "duration_weeks": 4,
            "weekly_schedule": [],
            "error": "计划生成失败，请重试",
        }, ensure_ascii=False)

    return {"plan_draft": plan_draft}


async def workout_plan_reflect_node(state: AgentState) -> dict:
    """
    反思阶段：评估计划的完整性、安全性、个性化程度，决定是否需要再次迭代

    输出：plan_reflection, plan_is_complete, plan_iteration+1
    """
    plan_draft = state.get("plan_draft", "")
    user_profile = state.get("user_profile", {})
    plan_thoughts = state.get("plan_thoughts", "")
    iteration = state.get("plan_iteration", 0)

    logger.info("workout_plan_reflect: 开始反思", iteration=iteration)

    # 超过最大迭代次数，直接通过
    if iteration >= MAX_ITERATIONS - 1:
        logger.warning("workout_plan_reflect: 达到最大迭代次数，强制通过")
        return {
            "plan_reflection": "已达最大迭代次数，使用当前计划。",
            "plan_is_complete": True,
            "plan_iteration": iteration + 1,
        }

    profile_context = _profile_to_context(user_profile)
    llm = _build_plus_llm()

    reflect_system = SystemMessage(content="""你是专业运动计划质量审核专家。
请评估运动计划的质量，以 JSON 格式返回：
{
  "is_complete": true/false,
  "score": 8.5,
  "issues": ["问题1", "问题2"],
  "improvements": ["改进建议1", "改进建议2"],
  "reflection_summary": "总体评价",
  "need_revision": false
}

评估维度（各10分）：
1. 完整性：是否包含所有必要信息
2. 安全性：是否考虑伤病/禁忌，有无安全提示
3. 个性化：是否针对用户的目标/水平/器械定制
4. 科学性：是否符合 FITT 原则（频率/强度/时间/类型）
5. 可执行性：动作描述是否清晰，负荷是否合理

score >= 8.0 且 need_revision=false 时，计划通过。
只返回 JSON，不要其他文字。""")

    reflect_human = HumanMessage(content=f"""用户画像：
{profile_context}

构思分析：
{plan_thoughts[:500]}

运动计划（JSON）：
{plan_draft[:2000]}

请进行评估：""")

    try:
        resp = await llm.ainvoke([reflect_system, reflect_human])
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        reflect_result = json.loads(raw.strip())
    except Exception as e:
        logger.error("workout_plan_reflect: 解析失败", error=str(e))
        reflect_result = {"is_complete": True, "score": 8.0, "need_revision": False}

    is_complete = reflect_result.get("is_complete", True)
    score = float(reflect_result.get("score", 8.0))
    need_revision = reflect_result.get("need_revision", False)
    plan_is_complete = is_complete and score >= 8.0 and not need_revision

    logger.info(
        "workout_plan_reflect: 评估完成",
        score=score,
        is_complete=plan_is_complete,
        iteration=iteration + 1,
    )

    return {
        "plan_reflection": json.dumps(reflect_result, ensure_ascii=False),
        "plan_is_complete": plan_is_complete,
        "plan_iteration": iteration + 1,
    }


async def workout_plan_format_node(state: AgentState) -> dict:
    """
    格式化节点：将 JSON 计划转为用户友好的 Markdown 文本

    输出：final_response, structured_output
    """
    plan_draft = state.get("plan_draft", "")
    plan_reflection = state.get("plan_reflection", "")
    iterations = state.get("plan_iteration", 1)

    try:
        plan_data = json.loads(plan_draft)
    except Exception:
        plan_data = {}

    # 生成 Markdown 格式计划
    lines = []
    plan_name = plan_data.get("plan_name", "个性化健身计划")
    goal = plan_data.get("goal", "")
    weeks = plan_data.get("duration_weeks", 4)

    lines.append(f"# 🏋️ {plan_name}")
    lines.append(f"\n**目标**：{goal} | **周期**：{weeks} 周\n")

    if plan_data.get("warm_up_tips"):
        lines.append(f"**热身建议**：{plan_data['warm_up_tips']}\n")

    schedule = plan_data.get("weekly_schedule", [])
    for day_plan in schedule:
        day = day_plan.get("day", "?")
        name = day_plan.get("name", "训练日")
        duration = day_plan.get("estimated_duration_min", "?")
        lines.append(f"\n## 第 {day} 天 — {name}（约 {duration} 分钟）")

        exercises = day_plan.get("exercises", [])
        if exercises:
            lines.append("\n| 动作 | 组数 | 次数 | 重量 | 休息 |")
            lines.append("|------|------|------|------|------|")
            for ex in exercises:
                name_ex = ex.get("name", "未知")
                sets = ex.get("sets", "-")
                reps = ex.get("reps", "-")
                weight = f"{ex['weight_kg']}kg" if ex.get("weight_kg") else "自重"
                rest = f"{ex.get('rest_sec', '-')}s"
                lines.append(f"| {name_ex} | {sets} | {reps} | {weight} | {rest} |")
            notes = day_plan.get("notes", "")
            if notes:
                lines.append(f"\n> 💡 {notes}")

    if plan_data.get("nutrition_tips"):
        lines.append(f"\n## 🥗 营养建议\n{plan_data['nutrition_tips']}")

    if plan_data.get("safety_notes"):
        lines.append(f"\n## ⚠️ 安全提示\n{plan_data['safety_notes']}")

    if plan_data.get("progression_guide"):
        lines.append(f"\n## 📈 进阶指南\n{plan_data['progression_guide']}")

    lines.append(f"\n\n---\n*计划经过 {iterations} 轮优化生成*")

    final_response = "\n".join(lines)

    return {
        "final_response": final_response,
        "structured_output": {"workout_plan": plan_data, "iterations": iterations},
        "messages": [AIMessage(content=final_response)],
    }
