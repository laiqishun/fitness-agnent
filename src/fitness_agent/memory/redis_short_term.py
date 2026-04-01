"""
Redis 短期记忆管理
- 存储最近 N 轮对话消息（FIFO 队列）
- 会话元数据缓存（意图历史、用户画像快照）
- 自动 TTL 管理
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import redis.asyncio as aioredis
import structlog

from fitness_agent.config import get_settings

logger = structlog.get_logger(__name__)

# Redis Key 前缀
KEY_PREFIX = "fitness_agent"
KEY_SHORT_TERM_MSGS = f"{KEY_PREFIX}:short_term:{{session_id}}"    # 短期消息队列
KEY_SESSION_META = f"{KEY_PREFIX}:session_meta:{{session_id}}"     # 会话元数据
KEY_USER_PROFILE_CACHE = f"{KEY_PREFIX}:profile_cache:{{user_id}}" # 用户画像缓存


class RedisShortTermMemory:
    """
    基于 Redis List 的短期记忆管理器

    数据结构：
    - 消息队列：Redis List（LPUSH + LTRIM 保持最近 N 条）
    - 会话元数据：Redis Hash
    - 用户画像缓存：Redis String（JSON 序列化）

    使用示例：
        memory = RedisShortTermMemory()
        await memory.add_message(session_id, {"role": "user", "content": "你好"})
        messages = await memory.get_recent_messages(session_id, limit=10)
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._redis_client: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis:
        """懒初始化 Redis 客户端"""
        if self._redis_client is None:
            self._redis_client = aioredis.from_url(
                self._settings.redis.url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis_client

    async def close(self) -> None:
        """关闭 Redis 连接"""
        if self._redis_client:
            await self._redis_client.aclose()
            self._redis_client = None

    # ── 短期消息管理 ──────────────────────────────────────────

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        添加一条消息到短期记忆队列

        Args:
            session_id: 会话 ID
            role: 角色（user / assistant / system）
            content: 消息内容
            metadata: 额外信息（意图、token 数等）
        """
        client = await self._get_client()
        key = KEY_SHORT_TERM_MSGS.format(session_id=session_id)
        max_msgs = self._settings.redis.max_short_term_messages
        ttl = self._settings.redis.ttl_seconds

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        async with client.pipeline(transaction=True) as pipe:
            # RPUSH 追加到队列尾部
            pipe.rpush(key, json.dumps(message, ensure_ascii=False))
            # 保留最近 max_msgs 条（从右边保留）
            pipe.ltrim(key, -max_msgs, -1)
            # 刷新 TTL
            pipe.expire(key, ttl)
            await pipe.execute()

        logger.debug("short_term_memory: 消息已添加", session_id=session_id, role=role)

    async def get_recent_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        获取最近 N 条消息

        Args:
            session_id: 会话 ID
            limit: 返回数量（默认全部）

        Returns:
            消息列表（时间正序）
        """
        client = await self._get_client()
        key = KEY_SHORT_TERM_MSGS.format(session_id=session_id)

        try:
            if limit:
                # 取最后 limit 条
                raw_messages = await client.lrange(key, -limit, -1)
            else:
                raw_messages = await client.lrange(key, 0, -1)

            messages = []
            for raw in raw_messages:
                try:
                    messages.append(json.loads(raw))
                except json.JSONDecodeError:
                    logger.warning("short_term_memory: 消息 JSON 解析失败", raw=raw[:100])

            return messages
        except Exception as e:
            logger.error("short_term_memory: 读取消息失败", error=str(e), session_id=session_id)
            return []

    async def get_message_count(self, session_id: str) -> int:
        """获取当前会话消息数量"""
        client = await self._get_client()
        key = KEY_SHORT_TERM_MSGS.format(session_id=session_id)
        return await client.llen(key)

    async def clear_session(self, session_id: str) -> None:
        """清除会话的短期记忆"""
        client = await self._get_client()
        key = KEY_SHORT_TERM_MSGS.format(session_id=session_id)
        await client.delete(key)
        logger.info("short_term_memory: 会话已清除", session_id=session_id)

    # ── 会话元数据 ────────────────────────────────────────────

    async def set_session_meta(
        self,
        session_id: str,
        meta: dict[str, Any],
    ) -> None:
        """设置会话元数据（如当前意图、上下文信息）"""
        client = await self._get_client()
        key = KEY_SESSION_META.format(session_id=session_id)
        ttl = self._settings.redis.ttl_seconds

        # 序列化每个字段
        serialized = {k: json.dumps(v, ensure_ascii=False) for k, v in meta.items()}

        async with client.pipeline() as pipe:
            if serialized:
                pipe.hset(key, mapping=serialized)
            pipe.expire(key, ttl)
            await pipe.execute()

    async def get_session_meta(self, session_id: str) -> dict[str, Any]:
        """获取会话元数据"""
        client = await self._get_client()
        key = KEY_SESSION_META.format(session_id=session_id)
        raw = await client.hgetall(key)

        result: dict[str, Any] = {}
        for k, v in raw.items():
            try:
                result[k] = json.loads(v)
            except json.JSONDecodeError:
                result[k] = v

        return result

    async def update_session_meta(self, session_id: str, updates: dict[str, Any]) -> None:
        """部分更新会话元数据"""
        await self.set_session_meta(session_id, updates)

    # ── 用户画像缓存 ──────────────────────────────────────────

    async def cache_user_profile(
        self,
        user_id: str,
        profile: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """缓存用户画像（避免频繁查数据库）"""
        client = await self._get_client()
        key = KEY_USER_PROFILE_CACHE.format(user_id=user_id)
        cache_ttl = ttl or self._settings.redis.ttl_seconds

        await client.setex(
            key,
            cache_ttl,
            json.dumps(profile, ensure_ascii=False, default=str),
        )
        logger.debug("short_term_memory: 用户画像已缓存", user_id=user_id)

    async def get_cached_user_profile(self, user_id: str) -> dict[str, Any] | None:
        """获取缓存的用户画像，不存在则返回 None"""
        client = await self._get_client()
        key = KEY_USER_PROFILE_CACHE.format(user_id=user_id)
        raw = await client.get(key)

        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("short_term_memory: 画像缓存 JSON 解析失败")
            return None

    async def invalidate_user_profile_cache(self, user_id: str) -> None:
        """使用户画像缓存失效"""
        client = await self._get_client()
        key = KEY_USER_PROFILE_CACHE.format(user_id=user_id)
        await client.delete(key)

    # ── 健康检查 ──────────────────────────────────────────────

    async def ping(self) -> bool:
        """Redis 连接健康检查"""
        try:
            client = await self._get_client()
            await client.ping()
            return True
        except Exception as e:
            logger.error("short_term_memory: Redis ping 失败", error=str(e))
            return False


# 全局单例
_memory_instance: RedisShortTermMemory | None = None


def get_short_term_memory() -> RedisShortTermMemory:
    """获取全局 RedisShortTermMemory 单例"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = RedisShortTermMemory()
    return _memory_instance
