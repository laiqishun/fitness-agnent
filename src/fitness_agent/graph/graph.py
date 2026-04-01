"""
LangGraph 主图组装
整合所有节点，实现完整的 AI 私教对话流程

图结构：
START
  → load_user_profile       （加载用户画像）
  → query_rewriter          （查询改写+扩展）
  → intent_classifier       （意图识别）
  → [条件路由]
      → qa_node
      → course_recommendation
      → reminder_node
      → diet_exercise_recorder
      → workout_plan_think   →  workout_plan_execute
                              → workout_plan_reflect
                              → [loop or format]
                              → workout_plan_format
  → [need_clarification?]   → END（等待用户继续）
END
"""
from __future__ import annotations

import uuid
from typing import Literal

import structlog
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from fitness_agent.config import get_settings
from fitness_agent.graph.nodes.course_recommendation import course_recommendation_node
from fitness_agent.graph.nodes.diet_exercise_recorder import diet_exercise_recorder_node
from fitness_agent.graph.nodes.intent_classifier import intent_classifier_node
from fitness_agent.graph.nodes.qa_node import qa_node
from fitness_agent.graph.nodes.query_rewriter import query_rewriter_node
from fitness_agent.graph.nodes.reminder_node import reminder_node
from fitness_agent.graph.nodes.workout_plan_generator import (
    workout_plan_execute_node,
    workout_plan_format_node,
    workout_plan_reflect_node,
    workout_plan_think_node,
)
from fitness_agent.graph.state import AgentState

logger = structlog.get_logger(__name__)


# =============================================================
# 辅助节点
# =============================================================

async def load_user_profile_node(state: AgentState) -> dict:
    """
    加载用户画像节点
    从数据库读取最新画像，同时从 Redis 加载短期记忆
    """
    user_id = state.get("user_id", "")
    session_id = state.get("session_id", "")

    if not user_id:
        logger.warning("load_user_profile: user_id 为空，跳过画像加载")
        return {"user_profile": {}}

    user_profile: dict = {}

    # 从数据库加载画像
    try:
        from fitness_agent.models.database import UserProfile, get_session, create_engine
        from sqlalchemy import select
        import uuid as uuid_module

        engine = create_engine()
        async for session in get_session(engine):
            result = await session.execute(
                select(UserProfile)
                .where(
                    UserProfile.user_id == uuid_module.UUID(user_id),
                    UserProfile.is_current == True,
                )
                .limit(1)
            )
            profile_obj = result.scalar_one_or_none()
            if profile_obj:
                user_profile = {
                    "id": str(profile_obj.id),
                    "fitness_level": profile_obj.fitness_level,
                    "primary_goal": profile_obj.primary_goal,
                    "secondary_goals": profile_obj.secondary_goals or [],
                    "available_equipment": profile_obj.available_equipment or [],
                    "workout_location": profile_obj.workout_location,
                    "injury_history": profile_obj.injury_history or [],
                    "health_conditions": profile_obj.health_conditions or [],
                    "weekly_workout_days": profile_obj.weekly_workout_days,
                    "preferred_workout_duration_min": profile_obj.preferred_workout_duration_min,
                    "preferred_workout_time": profile_obj.preferred_workout_time,
                    "dietary_restrictions": profile_obj.dietary_restrictions or [],
                    "height_cm": float(profile_obj.height_cm) if profile_obj.height_cm else None,
                    "weight_kg": float(profile_obj.weight_kg) if profile_obj.weight_kg else None,
                    "profile_summary": profile_obj.profile_summary,
                }
                logger.info("load_user_profile: 画像加载成功", user_id=user_id)
    except Exception as e:
        logger.error("load_user_profile: 数据库加载失败", error=str(e))

    return {"user_profile": user_profile}


async def save_message_node(state: AgentState) -> dict:
    """
    保存消息到数据库，更新会话统计
    在图执行完毕后调用
    """
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    messages = state.get("messages", [])
    intent = state.get("intent", "unknown")

    if not session_id or not user_id or not messages:
        return {}

    try:
        from fitness_agent.models.database import ChatMessage, ChatSession, get_session, create_engine
        from sqlalchemy import select, func
        import uuid as uuid_module

        engine = create_engine()
        async for db_session in get_session(engine):
            # 获取当前最大序号
            result = await db_session.execute(
                select(func.max(ChatMessage.sequence_num))
                .where(ChatMessage.session_id == uuid_module.UUID(session_id))
            )
            max_seq = result.scalar() or 0

            # 保存最新的消息（最后两条：用户消息 + 助手回复）
            new_messages = []
            for msg in messages[-2:]:
                role = "user" if msg.type == "human" else "assistant"
                max_seq += 1
                new_messages.append(
                    ChatMessage(
                        session_id=uuid_module.UUID(session_id),
                        user_id=uuid_module.UUID(user_id),
                        role=role,
                        content=msg.content,
                        intent=intent if role == "user" else None,
                        sequence_num=max_seq,
                    )
                )
            db_session.add_all(new_messages)
        logger.info("save_message: 消息保存成功", count=len(new_messages))
    except Exception as e:
        logger.error("save_message: 保存失败", error=str(e))

    return {}


# =============================================================
# 条件路由函数
# =============================================================

def route_by_intent(state: AgentState) -> Literal[
    "qa_node",
    "course_recommendation_node",
    "reminder_node",
    "diet_exercise_recorder_node",
    "workout_plan_think_node",
]:
    """根据意图识别结果路由到对应节点"""
    intent = state.get("intent", "unknown")

    intent_route_map = {
        "qa": "qa_node",
        "course_recommendation": "course_recommendation_node",
        "set_reminder": "reminder_node",
        "record_diet_exercise": "diet_exercise_recorder_node",
        "generate_workout_plan": "workout_plan_think_node",
        "need_clarification": "qa_node",   # 降级到 QA 处理追问
        "unknown": "qa_node",
    }

    route = intent_route_map.get(intent, "qa_node")
    logger.info("route_by_intent", intent=intent, route=route)
    return route


def should_end_after_response(state: AgentState) -> Literal["save_message_node", END]:
    """判断是否需要结束（追问时直接结束，等待用户继续）"""
    if state.get("need_clarification", False):
        return "save_message_node"
    return "save_message_node"


def should_continue_workout_plan(
    state: AgentState,
) -> Literal["workout_plan_execute_node", "save_message_node"]:
    """构思阶段后，如果需要追问则结束，否则继续执行"""
    if state.get("need_clarification", False):
        return "save_message_node"
    return "workout_plan_execute_node"


def should_iterate_or_format(
    state: AgentState,
) -> Literal["workout_plan_think_node", "workout_plan_format_node"]:
    """
    反思阶段后判断：
    - 计划未达标 且 未超过最大迭代次数 → 重新构思
    - 否则 → 格式化输出
    """
    plan_is_complete = state.get("plan_is_complete", True)
    iteration = state.get("plan_iteration", 0)

    if not plan_is_complete and iteration < 3:
        logger.info("should_iterate_or_format: 继续迭代", iteration=iteration)
        return "workout_plan_think_node"

    logger.info("should_iterate_or_format: 进入格式化", iteration=iteration)
    return "workout_plan_format_node"


# =============================================================
# 构建图
# =============================================================

def build_graph(checkpointer=None) -> StateGraph:
    """
    构建完整的 LangGraph 图

    Args:
        checkpointer: LangGraph checkpointer（用于持久化，可选）

    Returns:
        编译后的 CompiledGraph
    """
    builder = StateGraph(AgentState)

    # ── 注册节点 ─────────────────────────────────────────────
    builder.add_node("load_user_profile_node", load_user_profile_node)
    builder.add_node("query_rewriter_node", query_rewriter_node)
    builder.add_node("intent_classifier_node", intent_classifier_node)

    # 业务节点
    builder.add_node("qa_node", qa_node)
    builder.add_node("course_recommendation_node", course_recommendation_node)
    builder.add_node("reminder_node", reminder_node)
    builder.add_node("diet_exercise_recorder_node", diet_exercise_recorder_node)

    # 运动计划节点（构思-执行-反思循环）
    builder.add_node("workout_plan_think_node", workout_plan_think_node)
    builder.add_node("workout_plan_execute_node", workout_plan_execute_node)
    builder.add_node("workout_plan_reflect_node", workout_plan_reflect_node)
    builder.add_node("workout_plan_format_node", workout_plan_format_node)

    # 收尾节点
    builder.add_node("save_message_node", save_message_node)

    # ── 主流程边 ─────────────────────────────────────────────
    builder.add_edge(START, "load_user_profile_node")
    builder.add_edge("load_user_profile_node", "query_rewriter_node")
    builder.add_edge("query_rewriter_node", "intent_classifier_node")

    # 意图路由
    builder.add_conditional_edges(
        "intent_classifier_node",
        route_by_intent,
        {
            "qa_node": "qa_node",
            "course_recommendation_node": "course_recommendation_node",
            "reminder_node": "reminder_node",
            "diet_exercise_recorder_node": "diet_exercise_recorder_node",
            "workout_plan_think_node": "workout_plan_think_node",
        },
    )

    # 普通节点 → 保存消息 → END
    for node in [
        "qa_node",
        "course_recommendation_node",
        "reminder_node",
        "diet_exercise_recorder_node",
        "workout_plan_format_node",
    ]:
        builder.add_edge(node, "save_message_node")

    # 运动计划子循环
    builder.add_conditional_edges(
        "workout_plan_think_node",
        should_continue_workout_plan,
        {
            "workout_plan_execute_node": "workout_plan_execute_node",
            "save_message_node": "save_message_node",
        },
    )
    builder.add_edge("workout_plan_execute_node", "workout_plan_reflect_node")
    builder.add_conditional_edges(
        "workout_plan_reflect_node",
        should_iterate_or_format,
        {
            "workout_plan_think_node": "workout_plan_think_node",
            "workout_plan_format_node": "workout_plan_format_node",
        },
    )

    # 最终 END
    builder.add_edge("save_message_node", END)

    # ── 编译图 ────────────────────────────────────────────────
    compile_kwargs: dict = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    return builder.compile(**compile_kwargs)


# =============================================================
# 图工厂（带 PostgreSQL checkpointer）
# =============================================================

async def create_graph_with_pg_checkpointer():
    """
    创建带 PostgreSQL checkpointer 的生产级图实例
    在 FastAPI lifespan 中初始化
    """
    settings = get_settings()
    try:
        async with await AsyncPostgresSaver.from_conn_string(
            settings.database.psycopg_url
        ) as checkpointer:
            await checkpointer.setup()  # 自动建表
            graph = build_graph(checkpointer=checkpointer)
            logger.info("create_graph: 图初始化成功（带 PG checkpointer）")
            return graph
    except Exception as e:
        logger.warning(
            "create_graph: PG checkpointer 初始化失败，降级为内存模式",
            error=str(e)
        )
        return build_graph()


# =============================================================
# 便捷调用接口
# =============================================================

async def run_agent(
    graph,
    user_message: str,
    user_id: str,
    app_user_id: str,
    session_id: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """
    运行 AI 私教智能体

    Args:
        graph: 编译后的 LangGraph 图
        user_message: 用户消息
        user_id: 内部用户 UUID
        app_user_id: App 侧用户 ID
        session_id: 会话 ID（None 则自动生成）
        metadata: 额外元数据

    Returns:
        包含 reply, intent, session_id 等信息的字典
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    # LangGraph thread_id = session_id，保证对话持久化
    config = {"configurable": {"thread_id": session_id}}

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_message)],
        "user_id": user_id,
        "app_user_id": app_user_id,
        "session_id": session_id,
        "original_query": user_message,
        "rewritten_query": "",
        "expanded_queries": [],
        "intent": "unknown",
        "intent_confidence": 0.0,
        "sub_intents": [],
        "need_clarification": False,
        "clarification_question": "",
        "clarification_context": {},
        "user_profile": {},
        "retrieved_docs": [],
        "course_results": [],
        "reminder_info": {},
        "diet_info": {},
        "exercise_info": {},
        "plan_iteration": 0,
        "plan_thoughts": "",
        "plan_draft": "",
        "plan_reflection": "",
        "plan_is_complete": False,
        "final_response": "",
        "structured_output": {},
        "metadata": metadata or {},
        "error": None,
    }

    try:
        final_state = await graph.ainvoke(initial_state, config=config)

        return {
            "session_id": session_id,
            "reply": final_state.get("final_response", ""),
            "intent": final_state.get("intent", "unknown"),
            "need_clarification": final_state.get("need_clarification", False),
            "structured_output": final_state.get("structured_output", {}),
            "metadata": {
                "rewritten_query": final_state.get("rewritten_query", ""),
                "intent_confidence": final_state.get("intent_confidence", 0.0),
            },
        }
    except Exception as e:
        logger.error("run_agent: 图执行失败", error=str(e), session_id=session_id)
        raise
