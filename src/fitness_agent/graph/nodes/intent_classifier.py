"""
意图识别节点
使用 QwenPlus 对改写后的查询进行意图分类，输出结构化意图及置信度
支持混合意图（如同时记录饮食+设置提醒）
"""
from __future__ import annotations

import json
import structlog

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from fitness_agent.config import get_settings
from fitness_agent.graph.state import AgentState

logger = structlog.get_logger(__name__)

# 意图描述映射（用于 Prompt）
INTENT_DESCRIPTIONS = {
    "qa": "普通问答 - 询问健身知识、动作说明、营养知识等",
    "course_recommendation": "课程推荐 - 请求推荐课程、锻炼视频、训练方案",
    "set_reminder": "设置提醒 - 设定锻炼、饮食、用药等各类提醒",
    "record_diet_exercise": "记录饮食或运动 - 记录今日吃了什么或做了什么运动",
    "generate_workout_plan": "生成运动计划 - 制定个性化健身/减脂/增肌计划",
    "need_clarification": "信息不足 - 用户意图不明确，需要追问",
    "unknown": "无法识别的意图",
}


def _build_plus_llm() -> ChatOpenAI:
    """创建 QwenPlus 实例"""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.qwen.plus_model,
        api_key=settings.qwen.api_key,
        base_url=settings.qwen.base_url,
        temperature=0.1,   # 分类任务，尽量确定性输出
        max_tokens=512,
        timeout=settings.qwen.request_timeout,
    )


async def intent_classifier_node(state: AgentState) -> dict:
    """
    意图识别节点

    输入：state.rewritten_query, state.messages（历史）, state.user_profile
    输出：state.intent, state.intent_confidence, state.sub_intents
    """
    query = state.get("rewritten_query") or state.get("original_query", "")
    if not query:
        return {"intent": "unknown", "intent_confidence": 0.0, "sub_intents": []}

    # 构建意图描述供 LLM 参考
    intent_list = "\n".join([f"- {k}: {v}" for k, v in INTENT_DESCRIPTIONS.items()])

    # 用户画像摘要（帮助理解意图）
    profile = state.get("user_profile", {})
    profile_summary = profile.get("profile_summary", "暂无用户画像信息")

    system_prompt = f"""你是 Speediance AI 私教的意图识别器。
请分析用户的输入，准确识别其意图。

## 可用意图列表
{intent_list}

## 注意事项
1. 支持混合意图，例如"吃了早饭，记一下，顺便提醒我下午3点去健身"同时包含 record_diet_exercise 和 set_reminder
2. 如果用户意图明确，confidence 应 >= 0.85
3. 如果信息不足以判断，返回 need_clarification
4. 混合意图时，primary 填主要意图，sub_intents 填次要意图列表

## 输出格式（严格 JSON，不要其他文字）
{{
  "intent": "意图名称",
  "confidence": 0.95,
  "sub_intents": [],
  "reasoning": "简短推理说明（中文）"
}}"""

    human_prompt = f"""用户画像：{profile_summary}

用户输入：{query}

请识别意图："""

    llm = _build_plus_llm()
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        raw = response.content.strip()
        # 去掉 markdown 代码块
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw.strip())
        intent = result.get("intent", "unknown")
        confidence = float(result.get("confidence", 0.5))
        sub_intents = result.get("sub_intents", [])
        reasoning = result.get("reasoning", "")

        logger.info(
            "intent_classifier: 识别完成",
            intent=intent,
            confidence=confidence,
            sub_intents=sub_intents,
            reasoning=reasoning,
        )

        # 验证意图合法性
        if intent not in INTENT_DESCRIPTIONS:
            logger.warning("intent_classifier: 非法意图，降级为 unknown", intent=intent)
            intent = "unknown"
            confidence = 0.0

        return {
            "intent": intent,
            "intent_confidence": confidence,
            "sub_intents": sub_intents,
        }

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error("intent_classifier: JSON 解析失败", error=str(e))
        return {"intent": "qa", "intent_confidence": 0.5, "sub_intents": []}
    except Exception as e:
        logger.error("intent_classifier: 意图识别异常", error=str(e))
        return {"intent": "qa", "intent_confidence": 0.3, "sub_intents": []}
