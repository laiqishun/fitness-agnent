"""
LangGraph AgentState 定义
所有节点共享的状态结构，通过 TypedDict 强类型声明
"""
from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    AI 私教智能体的全局状态

    注意：
    - messages 使用 add_messages reducer，自动追加而非覆盖
    - 其他字段默认 last-write-wins
    - total=False 允许字段可选（节点只更新自己关心的字段）
    """

    # ── 消息历史（LangChain Message 对象列表）────────────────
    messages: Annotated[list, add_messages]

    # ── 会话上下文 ────────────────────────────────────────────
    user_id: str           # 内部数据库 UUID（str 形式）
    app_user_id: str       # App 侧透传的用户 ID
    session_id: str        # 会话 UUID（str 形式）

    # ── 查询处理 ──────────────────────────────────────────────
    original_query: str    # 用户原始输入
    rewritten_query: str   # 改写后的查询（消歧义、上下文补全）
    expanded_queries: list[str]   # 扩展查询词列表（用于 RAG 多路召回）

    # ── 意图识别结果 ──────────────────────────────────────────
    intent: str            # "qa" | "course_recommendation" | "set_reminder"
                           # | "record_diet_exercise" | "generate_workout_plan"
                           # | "need_clarification" | "unknown"
    intent_confidence: float   # 意图置信度 [0, 1]
    sub_intents: list[str]     # 混合意图列表（如同时记录饮食+设置提醒）

    # ── 追问机制 ──────────────────────────────────────────────
    need_clarification: bool       # 是否需要追问用户
    clarification_question: str    # 追问内容
    clarification_context: dict[str, Any]  # 追问上下文（缺失字段信息）

    # ── 用户画像 ──────────────────────────────────────────────
    user_profile: dict[str, Any]   # 当前用户画像（从数据库加载）

    # ── RAG 检索结果 ──────────────────────────────────────────
    retrieved_docs: list[dict[str, Any]]   # 检索到的知识文档列表
    # 每条文档结构:
    # {
    #   "chunk_id": str,
    #   "document_id": str,
    #   "title": str,
    #   "content": str,
    #   "score": float,
    #   "source_url": str | None,
    # }

    # ── 课程推荐结果 ──────────────────────────────────────────
    course_results: list[dict[str, Any]]   # 推荐课程列表

    # ── 提醒信息（reminder_node 使用）────────────────────────
    reminder_info: dict[str, Any]   # 解析后的提醒信息

    # ── 饮食/运动记录（diet_exercise_recorder 使用）──────────
    diet_info: dict[str, Any]       # 解析后的饮食信息
    exercise_info: dict[str, Any]   # 解析后的运动信息

    # ── 运动计划生成（构思-执行-反思循环）────────────────────
    plan_iteration: int             # 当前迭代次数（0-based，最多 3 次）
    plan_thoughts: str              # 构思阶段：分析用户情况的思考过程
    plan_draft: str                 # 执行阶段：生成的初稿计划
    plan_reflection: str            # 反思阶段：对计划的评估与改进建议
    plan_is_complete: bool          # 反思后判断计划是否达标

    # ── 最终输出 ──────────────────────────────────────────────
    final_response: str             # 最终回复文本
    structured_output: dict[str, Any]  # 结构化输出（课程列表、计划 JSON 等）

    # ── 运行时元数据 ──────────────────────────────────────────
    metadata: dict[str, Any]       # 运行时临时数据（token 计数、耗时等）
    error: str | None              # 错误信息（节点执行失败时写入）
