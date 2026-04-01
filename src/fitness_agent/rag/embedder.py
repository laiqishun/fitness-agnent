"""
文本向量化模块
封装通义千问 text-embedding-v3（OpenAI 兼容接口）
支持单条/批量向量化
"""
from __future__ import annotations

import asyncio
from typing import Sequence

import httpx
import structlog

from fitness_agent.config import get_settings

logger = structlog.get_logger(__name__)


class TextEmbedder:
    """
    文本向量化器
    使用 DashScope text-embedding-v3，向量维度 1536

    使用示例：
        embedder = TextEmbedder()
        vec = await embedder.embed("俯卧撑的标准动作是什么")
        vecs = await embedder.embed_batch(["查询1", "查询2"])
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._model = self._settings.qwen.embedding_model
        self._dimensions = self._settings.qwen.embedding_dimensions
        self._api_key = self._settings.qwen.api_key
        self._base_url = self._settings.qwen.base_url
        # 批量请求最大 size（DashScope 限制单次最多 25 条）
        self._batch_size = 25

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        """
        对单条文本进行向量化

        Args:
            text: 输入文本

        Returns:
            float 列表，长度为 self.dimensions
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """
        批量向量化

        Args:
            texts: 文本列表

        Returns:
            向量列表，与输入顺序一致
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        # 分批处理
        text_list = list(texts)
        for i in range(0, len(text_list), self._batch_size):
            batch = text_list[i: i + self._batch_size]
            embeddings = await self._call_embedding_api(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def _call_embedding_api(self, texts: list[str]) -> list[list[float]]:
        """
        调用 DashScope Embedding API（OpenAI 兼容格式）

        接口文档：https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-api-details
        """
        url = f"{self._base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": texts,
            "encoding_format": "float",
            "dimensions": self._dimensions,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

                # 按 index 排序，确保顺序正确
                embeddings_data = sorted(data["data"], key=lambda x: x["index"])
                embeddings = [item["embedding"] for item in embeddings_data]
                logger.debug("embed_batch: 向量化成功", count=len(embeddings))
                return embeddings

        except httpx.HTTPStatusError as e:
            logger.error(
                "embed_batch: API 请求失败",
                status_code=e.response.status_code,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error("embed_batch: 向量化失败", error=str(e))
            raise


# 全局单例（避免重复初始化）
_embedder_instance: TextEmbedder | None = None


def get_embedder() -> TextEmbedder:
    """获取全局 TextEmbedder 单例"""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = TextEmbedder()
    return _embedder_instance
