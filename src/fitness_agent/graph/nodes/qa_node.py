"""
普通问答节点（QA Node）
- 检索向量知识库（document_chunks）
- 结合用户画像个性化回答
- 信息不足时生成追问
- 引用来源文档
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from fitness_agent.config import get_settings
from fitness_agent.graph.state import AgentState

if TYPE_CHECKING:
    from fitness_agent.rag.retriever import HybridRetriever
    from fitness_agent.rag.reranker import GteReranker

logger = structlog.get_logger(__name__)


def _build_plus_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.qwen.plus_model,
        api_key=settings.qwen.api_key,
        base_url=settings.qwen.base_url,
        temperature=settings.qwen.temperature,
        max_tokens=settings.qwen.max_tokens,
        timeout=settings.qwen.request_timeout,
    )


def _format_docs_for_prompt(docs: list[dict]) -> str:
    """将检索文档格式化为 Prompt 中的参考资料块"""
    if not docs:
        return "（未检索到相关资料）"
    parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.get("title", "未知来源")
        content = doc.get("content", "")
        score = doc.get("score", 0.0)
        parts.append(f"[参考{i}] 《{title}》（相关度：{score:.2f}）\n{content}")
    return "\n\n".join(parts)


def _format_profile_for_prompt(profile: dict) -> str:
    """将用户画像格式化为简洁的自然语言描述"""
    if not profile:
        return "暂无用户画像信息"
    lines = []
    if profile.get("fitness_level"):
        level_map = {
            "beginner": "初学者", "intermediate": "中级", "advanced": "高级", "elite": "专业"
        }
        lines.append(f"健身水平：{level_map.get(profile['fitness_level'], profile['fitness_level'])}")
    if profile.get("primary_goal"):
        goal_map = {
            "lose_weight": "减脂", "gain_muscle": "增肌", "improve_endurance": "提升耐力",
            "maintain_health": "维持健康", "rehabilitation": "康复训练", "other": "其他"
        }
        lines.append(f"主要目标：{goal_map.get(profile['primary_goal'], profile['primary_goal'])}")
    if profile.get("injury_history"):
        lines.append(f"伤病史：{', '.join(profile['injury_history'])}")
    if profile.get("health_conditions"):
        lines.append(f"健康状况：{', '.join(profile['health_conditions'])}")
    if profile.get("weight_kg") and profile.get("height_cm"):
        lines.append(f"体重/身高：{profile['weight_kg']}kg / {profile['height_cm']}cm")
    return "；".join(lines) if lines else "暂无详细画像信息"


async def qa_node(state: AgentState) -> dict:
    """
    问答节点

    流程：
    1. 并行检索知识库文档
    2. 判断是否有足够信息回答
    3. 如果信息不足，生成追问
    4. 否则，生成引用来源的回答
    """
    query = state.get("rewritten_query") or state.get("original_query", "")
    expanded_queries = state.get("expanded_queries", [query])
    user_profile = state.get("user_profile", {})
    retrieved_docs = state.get("retrieved_docs", [])  # 可能已由上游节点填充

    settings = get_settings()
    llm = _build_plus_llm()

    # ── 步骤 1：如果 retrieved_docs 为空，则执行检索 ──────────
    if not retrieved_docs:
        try:
            # 动态导入避免循环依赖
            from fitness_agent.rag.retriever import HybridRetriever
            from fitness_agent.rag.reranker import GteReranker

            retriever = HybridRetriever()
            reranker = GteReranker()

            # 多查询召回
            all_docs: list[dict] = []
            for q in expanded_queries[:3]:  # 最多 3 个扩展查询
                docs = await retriever.retrieve_documents(q, top_k=settings.rag.top_k)
                all_docs.extend(docs)

            # 去重（按 chunk_id）
            seen_ids: set[str] = set()
            unique_docs = []
            for doc in all_docs:
                doc_id = doc.get("chunk_id", "")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(doc)

            # 重排序
            if unique_docs:
                retrieved_docs = await reranker.rerank(
                    query=query,
                    docs=unique_docs,
                    top_n=settings.rag.rerank_top_n,
                )
            else:
                retrieved_docs = []

            logger.info("qa_node: 检索完成", doc_count=len(retrieved_docs))
        except Exception as e:
            logger.error("qa_node: 检索失败", error=str(e))
            retrieved_docs = []

    # ── 步骤 2：判断是否有足够信息 ────────────────────────────
    docs_str = _format_docs_for_prompt(retrieved_docs)
    profile_str = _format_profile_for_prompt(user_profile)

    # 判断是否需要追问（信息不足的情况）
    check_system = SystemMessage(content="""你是 AI 健身私教的信息充足度评估模块。
判断现有参考资料是否足够回答用户问题。

返回 JSON：
{
  "has_enough_info": true/false,
  "clarification_question": "追问内容（has_enough_info=false 时填写）",
  "reason": "判断理由"
}

只在以下情况返回 false：
1. 参考资料完全空白且问题非常专业/个性化
2. 问题涉及用户的具体伤病/健康情况但画像中没有
注意：即使参考资料不完整，也尽量回答，不要轻易追问。""")

    check_human = HumanMessage(content=f"""用户问题：{query}
用户画像：{profile_str}
参考资料数量：{len(retrieved_docs)}

请判断信息是否充足：""")

    try:
        import json
        check_resp = await llm.ainvoke([check_system, check_human])
        raw = check_resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        check_result = json.loads(raw.strip())
        has_enough_info = check_result.get("has_enough_info", True)
        clarification_question = check_result.get("clarification_question", "")
    except Exception:
        has_enough_info = True
        clarification_question = ""

    if not has_enough_info and clarification_question:
        logger.info("qa_node: 信息不足，生成追问")
        return {
            "retrieved_docs": retrieved_docs,
            "need_clarification": True,
            "clarification_question": clarification_question,
            "final_response": clarification_question,
            "messages": [AIMessage(content=clarification_question)],
        }

    # ── 步骤 3：生成回答 ───────────────────────────────────────
    answer_system = SystemMessage(content=f"""你是 Speediance AI 健身私教助手，专业、友善、实用。

## 用户画像
{profile_str}

## 参考资料
{docs_str}

## 回答原则
1. 基于参考资料回答，如果资料不足则根据健身专业知识回答
2. 回答要个性化，考虑用户的健身水平和目标
3. 注意安全提示，特别是有伤病史的用户
4. 如果引用了参考资料，在末尾注明来源（用 [参考N] 标注）
5. 语言：中文，语气专业但亲切
6. 长度：适中，不要过于冗长""")

    answer_human = HumanMessage(content=f"用户问题：{query}")

    try:
        answer_resp = await llm.ainvoke([answer_system, answer_human])
        final_response = answer_resp.content.strip()
        logger.info("qa_node: 回答生成完成", length=len(final_response))
    except Exception as e:
        logger.error("qa_node: 生成回答失败", error=str(e))
        final_response = "抱歉，我暂时无法回答这个问题，请稍后再试。"

    return {
        "retrieved_docs": retrieved_docs,
        "need_clarification": False,
        "clarification_question": "",
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)],
    }
