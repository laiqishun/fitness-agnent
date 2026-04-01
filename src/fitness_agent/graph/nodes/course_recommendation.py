"""
课程推荐节点
- 根据改写查询和用户画像向量检索 courses 表
- 使用 gte-rerank 重排序
- 生成个性化推荐理由
"""
from __future__ import annotations

import structlog
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
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
        temperature=0.7,
        max_tokens=1024,
        timeout=settings.qwen.request_timeout,
    )


def _format_courses_for_prompt(courses: list[dict]) -> str:
    """将课程列表格式化为 Prompt 内容"""
    if not courses:
        return "（未找到相关课程）"
    parts = []
    for i, course in enumerate(courses, 1):
        name = course.get("name", "未知课程")
        desc = course.get("description", "")[:100]
        difficulty = course.get("difficulty", "moderate")
        duration = course.get("duration_min", "未知")
        muscles = ", ".join(course.get("muscle_groups", []))
        equipment = ", ".join(course.get("equipment_needed", []))
        parts.append(
            f"{i}. 《{name}》\n"
            f"   难度：{difficulty} | 时长：{duration}分钟\n"
            f"   目标肌群：{muscles or '全身'}\n"
            f"   所需器械：{equipment or '无'}\n"
            f"   简介：{desc}"
        )
    return "\n\n".join(parts)


async def course_recommendation_node(state: AgentState) -> dict:
    """
    课程推荐节点

    流程：
    1. 用改写查询 + 用户画像向量检索课程
    2. gte-rerank 重排序
    3. QwenPlus 生成个性化推荐理由
    """
    query = state.get("rewritten_query") or state.get("original_query", "")
    user_profile = state.get("user_profile", {})
    settings = get_settings()

    # ── 步骤 1：检索课程 ──────────────────────────────────────
    try:
        from fitness_agent.rag.retriever import HybridRetriever
        from fitness_agent.rag.reranker import GteReranker

        retriever = HybridRetriever()
        reranker = GteReranker()

        # 检索课程（指定 table=courses）
        courses_raw = await retriever.retrieve_courses(
            query=query,
            user_profile=user_profile,
            top_k=settings.rag.top_k,
        )

        # 重排序
        if courses_raw:
            courses_reranked = await reranker.rerank(
                query=query,
                docs=[
                    {
                        "chunk_id": str(c.get("id", "")),
                        "content": f"{c.get('name', '')} {c.get('description', '')}",
                        **c,
                    }
                    for c in courses_raw
                ],
                top_n=settings.rag.rerank_top_n,
            )
            # 还原为完整 course 数据
            course_id_map = {str(c.get("id", "")): c for c in courses_raw}
            courses = [
                course_id_map.get(r.get("chunk_id", ""), r)
                for r in courses_reranked
            ]
        else:
            courses = []

        logger.info("course_recommendation: 检索完成", count=len(courses))
    except Exception as e:
        logger.error("course_recommendation: 检索失败", error=str(e))
        courses = []

    # ── 步骤 2：生成推荐理由 ───────────────────────────────────
    profile_goal = user_profile.get("primary_goal", "maintain_health")
    profile_level = user_profile.get("fitness_level", "beginner")
    profile_summary = user_profile.get("profile_summary", "")

    courses_text = _format_courses_for_prompt(courses)

    llm = _build_plus_llm()
    system_msg = SystemMessage(content=f"""你是 Speediance AI 健身私教，正在为用户推荐课程。

用户信息：
- 健身水平：{profile_level}
- 主要目标：{profile_goal}
- 画像摘要：{profile_summary or '无'}

已筛选课程列表：
{courses_text}

请根据用户需求和以上课程，给出：
1. 个性化推荐理由（说明为什么这些课程适合用户）
2. 重点推荐 1-2 门最适合的课程（标注序号）
3. 简洁温暖的语气，不超过 150 字""")

    human_msg = HumanMessage(content=f"用户询问：{query}")

    try:
        response = await llm.ainvoke([system_msg, human_msg])
        reason_text = response.content.strip()
    except Exception as e:
        logger.error("course_recommendation: 生成推荐理由失败", error=str(e))
        reason_text = f"为您找到了 {len(courses)} 门相关课程，请查看详情选择适合您的课程。"

    # 构建结构化输出
    structured_output = {
        "courses": courses,
        "reason": reason_text,
        "total": len(courses),
    }

    # 生成最终回复
    final_response = reason_text
    if courses:
        course_names = [c.get("name", "") for c in courses[:3]]
        final_response += f"\n\n为您推荐：{'、'.join(course_names)}"

    return {
        "course_results": courses,
        "final_response": final_response,
        "structured_output": structured_output,
        "messages": [AIMessage(content=final_response)],
    }
