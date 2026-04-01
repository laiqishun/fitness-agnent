"""
查询改写节点
使用 Qwen-Turbo（轻量快速）对用户输入进行：
1. 上下文消歧义改写（rewrite）：利用历史对话补全指代词、省略信息
2. 查询扩展（expand）：生成多个语义相关的查询词，提升 RAG 召回率
"""
from __future__ import annotations

import json
import structlog

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from fitness_agent.config import get_settings
from fitness_agent.graph.state import AgentState

logger = structlog.get_logger(__name__)


def _build_turbo_llm() -> ChatOpenAI:
    """创建 Qwen-Turbo 实例（用于轻量任务）"""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.qwen.turbo_model,
        api_key=settings.qwen.api_key,
        base_url=settings.qwen.base_url,
        temperature=0.3,   # 改写任务需要较低随机性
        max_tokens=1024,
        timeout=settings.qwen.request_timeout,
    )


def _format_history_for_context(messages: list) -> str:
    """将最近 N 条消息格式化为上下文字符串（避免过长）"""
    if not messages:
        return "（无历史对话）"
    # 取最近 6 条（用户+助手轮流）
    recent = messages[-6:]
    lines = []
    for msg in recent:
        role = "用户" if msg.type == "human" else "助手"
        content = msg.content[:200] if len(msg.content) > 200 else msg.content
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


async def query_rewriter_node(state: AgentState) -> dict:
    """
    查询改写节点

    输入：state.original_query, state.messages（历史）
    输出：state.rewritten_query, state.expanded_queries
    """
    original_query = state.get("original_query", "")
    if not original_query:
        # 从最后一条 human 消息取
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                original_query = msg.content
                break

    if not original_query:
        logger.warning("query_rewriter: 未找到用户输入，跳过改写")
        return {
            "original_query": "",
            "rewritten_query": "",
            "expanded_queries": [],
        }

    # 构建历史上下文
    history_context = _format_history_for_context(state.get("messages", [])[:-1])  # 去掉最后一条（当前消息）
    llm = _build_turbo_llm()

    # ── 步骤 1：上下文改写 ─────────────────────────────────────
    rewrite_system = SystemMessage(content="""你是一个健身助手查询改写专家。
你的任务是根据对话历史，将用户的模糊/简短查询改写为完整清晰的问题。

改写原则：
1. 补全指代词（"这个动作" → 上文提到的具体动作名称）
2. 补全省略的上下文（"怎么做" → "XXX 动作怎么做"）
3. 保持用户原始意图不变
4. 如果查询已经清晰完整，原样返回
5. 只输出改写后的查询文本，不要解释

只返回改写后的查询，不要多余的文字。""")

    rewrite_human = HumanMessage(content=f"""对话历史：
{history_context}

当前用户输入：{original_query}

请改写为完整清晰的查询：""")

    try:
        rewrite_response = await llm.ainvoke([rewrite_system, rewrite_human])
        rewritten_query = rewrite_response.content.strip()
        logger.info("query_rewriter: 改写完成", original=original_query, rewritten=rewritten_query)
    except Exception as e:
        logger.error("query_rewriter: 改写失败，使用原始查询", error=str(e))
        rewritten_query = original_query

    # ── 步骤 2：查询扩展 ──────────────────────────────────────
    expand_system = SystemMessage(content="""你是一个健身领域的查询扩展专家。
根据用户的查询，生成 3-5 个语义相关的扩展查询，用于从知识库检索更全面的信息。

扩展原则：
1. 覆盖查询的不同语义维度（同义词、上下位概念、相关概念）
2. 专注于健身、营养、运动、健康领域
3. 中文输出

以 JSON 数组格式返回，例如：
["扩展查询1", "扩展查询2", "扩展查询3"]

只返回 JSON 数组，不要其他文字。""")

    expand_human = HumanMessage(content=f"原始查询：{rewritten_query}\n\n请生成扩展查询 JSON 数组：")

    try:
        expand_response = await llm.ainvoke([expand_system, expand_human])
        raw_expand = expand_response.content.strip()
        # 去掉可能的 markdown 代码块
        if raw_expand.startswith("```"):
            raw_expand = raw_expand.split("```")[1]
            if raw_expand.startswith("json"):
                raw_expand = raw_expand[4:]
        expanded_queries: list[str] = json.loads(raw_expand)
        if not isinstance(expanded_queries, list):
            expanded_queries = [rewritten_query]
        logger.info("query_rewriter: 扩展完成", count=len(expanded_queries))
    except Exception as e:
        logger.error("query_rewriter: 扩展失败，使用改写后查询", error=str(e))
        expanded_queries = [rewritten_query]

    return {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "expanded_queries": expanded_queries,
    }
