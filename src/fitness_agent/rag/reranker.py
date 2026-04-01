"""
重排序模块
使用 gte-rerank 对检索结果进行重排序，提升精度
"""
from __future__ import annotations

from typing import Any

import httpx
import structlog

from fitness_agent.config import get_settings

logger = structlog.get_logger(__name__)


class GteReranker:
    """
    基于 gte-rerank 的重排序器
    使用 DashScope 的 rerank API

    API 文档：https://help.aliyun.com/zh/dashscope/developer-reference/rerank-api
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._model = self._settings.qwen.rerank_model
        self._api_key = self._settings.qwen.api_key
        # DashScope rerank API 端点（非 OpenAI 兼容，使用原生接口）
        self._rerank_url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    async def rerank(
        self,
        query: str,
        docs: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        对文档列表进行重排序

        Args:
            query: 检索查询文本
            docs: 文档列表，每条包含 content 字段
            top_n: 重排后返回数量（默认使用配置值）

        Returns:
            重排后的文档列表（按相关度降序）
        """
        if not docs:
            return []

        top_n = top_n or self._settings.rag.rerank_top_n
        top_n = min(top_n, len(docs))

        # 提取文档内容
        doc_contents = [doc.get("content", "") for doc in docs]

        try:
            rerank_scores = await self._call_rerank_api(query, doc_contents, top_n)
        except Exception as e:
            logger.warning(
                "reranker: 重排序失败，使用原始顺序",
                error=str(e),
                doc_count=len(docs),
            )
            # 降级：返回原始顺序的前 top_n 条
            return docs[:top_n]

        # 按重排分数重组结果
        results = []
        for item in rerank_scores:
            idx = item["index"]
            if 0 <= idx < len(docs):
                doc = docs[idx].copy()
                doc["rerank_score"] = item["relevance_score"]
                results.append(doc)

        logger.info(
            "reranker: 重排序完成",
            input_count=len(docs),
            output_count=len(results),
        )
        return results

    async def _call_rerank_api(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[dict[str, Any]]:
        """
        调用 DashScope Rerank API

        Returns:
            按相关度排序的 [{"index": int, "relevance_score": float}] 列表
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "top_n": top_n,
                "return_documents": False,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self._rerank_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        # 解析响应
        output = data.get("output", {})
        results = output.get("results", [])

        # 格式化
        formatted = []
        for item in results:
            formatted.append({
                "index": item.get("index", 0),
                "relevance_score": float(item.get("relevance_score", 0.0)),
            })

        # 按 relevance_score 降序排列
        formatted.sort(key=lambda x: x["relevance_score"], reverse=True)
        return formatted
