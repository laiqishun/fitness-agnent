"""
会话摘要服务
模拟人类"模糊记忆"机制：
- 当会话消息数达到阈值时，自动生成摘要
- 更新 chat_sessions.summary 和 key_facts
- 从摘要中提取用户画像更新信息

设计理念：
- 对话细节随时间模糊，但关键事实（用户目标、伤病史等）长期保留
- 摘要比原始消息更紧凑，减少长对话时的 token 消耗
"""
from __future__ import annotations

import json
import uuid as uuid_module
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sqlalchemy import select

from fitness_agent.config import get_settings
from fitness_agent.models.database import (
    ChatMessage,
    ChatSession,
    UserProfile,
    UserProfileUpdateHistory,
    create_engine,
    get_session,
)

logger = structlog.get_logger(__name__)


class SessionSummarizer:
    """
    会话摘要生成服务

    功能：
    1. 生成会话摘要（condensed summary）
    2. 提取关键事实（key facts）
    3. 根据摘要更新用户画像
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._engine = None

    def _get_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self._settings.qwen.plus_model,
            api_key=self._settings.qwen.api_key,
            base_url=self._settings.qwen.base_url,
            temperature=0.3,
            max_tokens=2048,
            timeout=self._settings.qwen.request_timeout,
        )

    def _get_engine(self):
        if self._engine is None:
            self._engine = create_engine(self._settings)
        return self._engine

    async def should_summarize(self, session_id: str) -> bool:
        """判断会话是否需要生成摘要"""
        threshold = self._settings.summarize_after_messages
        engine = self._get_engine()

        async for session in get_session(engine):
            result = await session.execute(
                select(ChatSession).where(
                    ChatSession.id == uuid_module.UUID(session_id)
                )
            )
            chat_session = result.scalar_one_or_none()
            if chat_session and chat_session.message_count >= threshold:
                return True
        return False

    async def summarize_session(
        self,
        session_id: str,
        user_id: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        为指定会话生成摘要

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            force: 强制生成（不检查阈值）

        Returns:
            包含 summary 和 key_facts 的字典
        """
        if not force and not await self.should_summarize(session_id):
            logger.debug("session_summarizer: 未达摘要阈值，跳过", session_id=session_id)
            return {}

        engine = self._get_engine()
        messages_text = ""
        existing_summary = ""

        # 读取会话消息和现有摘要
        async for session in get_session(engine):
            # 获取现有摘要
            session_result = await session.execute(
                select(ChatSession).where(
                    ChatSession.id == uuid_module.UUID(session_id)
                )
            )
            chat_session = session_result.scalar_one_or_none()
            if chat_session:
                existing_summary = chat_session.summary or ""

            # 获取最近的消息（超过摘要阈值的那部分）
            threshold = self._settings.summarize_after_messages
            msg_result = await session.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == uuid_module.UUID(session_id))
                .order_by(ChatMessage.sequence_num)
                .limit(threshold)
            )
            messages = msg_result.scalars().all()

            # 格式化为对话文本
            msg_lines = []
            for msg in messages:
                role = "用户" if msg.role == "user" else "助手"
                msg_lines.append(f"{role}：{msg.content[:500]}")
            messages_text = "\n".join(msg_lines)

        if not messages_text:
            return {}

        # ── 生成摘要 ──────────────────────────────────────────
        llm = self._get_llm()
        prior_context = f"【历史摘要】\n{existing_summary}\n\n" if existing_summary else ""

        summary_system = SystemMessage(content="""你是 AI 健身私教的记忆管理系统。
请对以下对话记录生成结构化摘要，模拟人类的"关键记忆"机制：

输出 JSON：
{
  "summary": "对话摘要（150字以内，保留核心信息）",
  "key_facts": [
    {"category": "goal", "fact": "用户目标是减脂10斤"},
    {"category": "injury", "fact": "用户有膝盖半月板损伤"},
    {"category": "preference", "fact": "用户偏好早晨锻炼"},
    {"category": "equipment", "fact": "用户有 Speediance Gym Monster"},
    {"category": "progress", "fact": "用户本周完成了3次训练"}
  ],
  "profile_updates": {
    "weight_kg": 75.0,
    "fitness_level": "intermediate",
    "injury_history": ["膝盖半月板损伤"]
  }
}

category 枚举：goal, injury, preference, equipment, progress, diet, other
profile_updates 只包含有明确新信息的字段，没有则为空对象 {}
只返回 JSON，不要其他文字。""")

        summary_human = HumanMessage(content=f"""{prior_context}【当前对话记录】
{messages_text}

请生成摘要：""")

        try:
            resp = await llm.ainvoke([summary_system, summary_human])
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            summary_data = json.loads(raw.strip())
        except Exception as e:
            logger.error("session_summarizer: 摘要生成失败", error=str(e))
            return {}

        summary = summary_data.get("summary", "")
        key_facts = summary_data.get("key_facts", [])
        profile_updates = summary_data.get("profile_updates", {})

        # ── 更新数据库 ────────────────────────────────────────
        async for session in get_session(engine):
            # 更新会话摘要
            session_result = await session.execute(
                select(ChatSession).where(
                    ChatSession.id == uuid_module.UUID(session_id)
                )
            )
            chat_session = session_result.scalar_one_or_none()
            if chat_session:
                chat_session.summary = summary
                chat_session.key_facts = key_facts
                chat_session.status = "summarized"
                logger.info("session_summarizer: 会话摘要已更新", session_id=session_id)

            # 如果有画像更新，写入用户画像
            if profile_updates and user_id:
                await self._update_user_profile(
                    session=session,
                    user_id=user_id,
                    session_id=session_id,
                    updates=profile_updates,
                )

        return {
            "summary": summary,
            "key_facts": key_facts,
            "profile_updates": profile_updates,
        }

    async def _update_user_profile(
        self,
        session,
        user_id: str,
        session_id: str,
        updates: dict[str, Any],
    ) -> None:
        """根据摘要更新用户画像"""
        try:
            # 获取当前画像
            result = await session.execute(
                select(UserProfile).where(
                    UserProfile.user_id == uuid_module.UUID(user_id),
                    UserProfile.is_current == True,
                )
            )
            current_profile = result.scalar_one_or_none()

            if not current_profile:
                logger.warning("session_summarizer: 未找到用户画像，跳过更新", user_id=user_id)
                return

            # 记录变更
            changed_fields: dict[str, dict] = {}
            for field, new_value in updates.items():
                old_value = getattr(current_profile, field, None)
                if old_value != new_value:
                    changed_fields[field] = {"old": old_value, "new": new_value}
                    setattr(current_profile, field, new_value)

            if not changed_fields:
                return

            # 写入变更历史
            history = UserProfileUpdateHistory(
                user_id=uuid_module.UUID(user_id),
                profile_id=current_profile.id,
                changed_fields=changed_fields,
                change_reason="session_summary",
                session_id=uuid_module.UUID(session_id) if session_id else None,
            )
            session.add(history)
            logger.info(
                "session_summarizer: 用户画像已更新",
                user_id=user_id,
                changed_fields=list(changed_fields.keys()),
            )
        except Exception as e:
            logger.error("session_summarizer: 画像更新失败", error=str(e))
