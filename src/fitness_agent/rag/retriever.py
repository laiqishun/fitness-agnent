"""
向量检索模块
支持：
- 文档分块向量检索（document_chunks）
- 课程向量检索（courses）
- 混合检索：向量相似度 + 关键词全文检索（RRF 融合）
"""
from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from fitness_agent.config import get_settings
from fitness_agent.models.database import create_engine, get_session
from fitness_agent.rag.embedder import get_embedder

logger = structlog.get_logger(__name__)


class HybridRetriever:
    """
    混合检索器：向量检索 + 关键词检索，使用 RRF（Reciprocal Rank Fusion）融合

    RRF 公式：score(d) = Σ 1/(k + rank_i(d))，k=60
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._embedder = get_embedder()
        self._engine = None  # 懒初始化

    def _get_engine(self):
        if self._engine is None:
            self._engine = create_engine(self._settings)
        return self._engine

    async def retrieve_documents(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        从 document_chunks 表检索相关文档分块

        Args:
            query: 检索查询文本
            top_k: 返回数量（默认使用配置值）
            category: 文档分类过滤（可选）
            min_score: 最低相似度阈值（可选）

        Returns:
            文档列表，每条包含 chunk_id, document_id, title, content, score
        """
        top_k = top_k or self._settings.rag.top_k
        min_score = min_score if min_score is not None else self._settings.rag.min_score

        # 并行获取向量和关键词结果
        import asyncio
        query_embedding = await self._embedder.embed(query)

        engine = self._get_engine()
        async for session in get_session(engine):
            # ── 向量检索 ───────────────────────────────────────
            vector_sql = text("""
                SELECT
                    dc.id::text            AS chunk_id,
                    dc.document_id::text   AS document_id,
                    sd.title               AS title,
                    dc.content             AS content,
                    sd.oss_url             AS source_url,
                    sd.category            AS category,
                    1 - (dc.embedding <=> cast(:embedding AS vector)) AS score,
                    ROW_NUMBER() OVER (ORDER BY dc.embedding <=> cast(:embedding AS vector)) AS vector_rank
                FROM document_chunks dc
                JOIN source_documents sd ON dc.document_id = sd.id
                WHERE sd.status = 'active'
                  AND (1 - (dc.embedding <=> cast(:embedding AS vector))) >= :min_score
                  {% if category %} AND sd.category = :category {% endif %}
                ORDER BY vector_rank
                LIMIT :top_k
            """.replace(
                "{% if category %} AND sd.category = :category {% endif %}",
                "AND sd.category = :category" if category else "",
            ))

            # ── 全文检索（BM25 近似）──────────────────────────
            fts_sql = text("""
                SELECT
                    dc.id::text            AS chunk_id,
                    dc.document_id::text   AS document_id,
                    sd.title               AS title,
                    dc.content             AS content,
                    sd.oss_url             AS source_url,
                    sd.category            AS category,
                    ts_rank(
                        to_tsvector('simple', dc.content),
                        plainto_tsquery('simple', :query)
                    ) AS score,
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank(
                            to_tsvector('simple', dc.content),
                            plainto_tsquery('simple', :query)
                        ) DESC
                    ) AS fts_rank
                FROM document_chunks dc
                JOIN source_documents sd ON dc.document_id = sd.id
                WHERE sd.status = 'active'
                  AND to_tsvector('simple', dc.content) @@ plainto_tsquery('simple', :query)
                ORDER BY fts_rank
                LIMIT :top_k
            """)

            params_vec = {
                "embedding": str(query_embedding),
                "min_score": min_score,
                "top_k": top_k * 2,  # 多取些，方便 RRF 融合
            }
            if category:
                params_vec["category"] = category

            params_fts = {"query": query, "top_k": top_k * 2}

            try:
                vec_result = await session.execute(vector_sql, params_vec)
                fts_result = await session.execute(fts_sql, params_fts)

                vec_rows = [dict(r._mapping) for r in vec_result.fetchall()]
                fts_rows = [dict(r._mapping) for r in fts_result.fetchall()]
            except Exception as e:
                logger.error("retrieve_documents: SQL 执行失败", error=str(e))
                return []

            # ── RRF 融合 ──────────────────────────────────────
            results = self._rrf_merge(vec_rows, fts_rows, top_k=top_k)
            logger.info(
                "retrieve_documents: 检索完成",
                query_preview=query[:50],
                count=len(results),
            )
            return results

        return []

    async def retrieve_courses(
        self,
        query: str,
        user_profile: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        从 courses 表检索相关课程
        结合用户画像做过滤（健身水平、器械）

        Args:
            query: 检索查询文本
            user_profile: 用户画像（可选，用于过滤）
            top_k: 返回数量

        Returns:
            课程列表
        """
        top_k = top_k or self._settings.rag.top_k

        query_embedding = await self._embedder.embed(query)
        engine = self._get_engine()

        # 构建过滤条件
        filters = ["c.is_active = true", "c.embedding IS NOT NULL"]
        params: dict[str, Any] = {
            "embedding": str(query_embedding),
            "top_k": top_k,
        }

        if user_profile:
            fitness_level = user_profile.get("fitness_level")
            if fitness_level:
                params["fitness_level"] = fitness_level
                filters.append("(:fitness_level = ANY(c.suitable_levels) OR c.suitable_levels = '{}')")

            equipment = user_profile.get("available_equipment", [])
            if equipment:
                # 有任意所需器械即可
                params["equipment"] = equipment
                # 要求课程所需器械是用户可用器械的子集
                # 简化为：课程器械列表与用户器械有交集，或课程不需要器械
                filters.append(
                    "(c.equipment_needed = '{}' OR c.equipment_needed && :equipment::text[])"
                )

        where_clause = " AND ".join(filters)

        course_sql = text(f"""
            SELECT
                c.id::text           AS id,
                c.name               AS name,
                c.description        AS description,
                c.instructor         AS instructor,
                c.category           AS category,
                c.difficulty         AS difficulty,
                c.duration_min       AS duration_min,
                c.tags               AS tags,
                c.muscle_groups      AS muscle_groups,
                c.equipment_needed   AS equipment_needed,
                c.thumbnail_url      AS thumbnail_url,
                c.rating             AS rating,
                c.is_premium         AS is_premium,
                1 - (c.embedding <=> cast(:embedding AS vector)) AS score
            FROM courses c
            WHERE {where_clause}
            ORDER BY c.embedding <=> cast(:embedding AS vector)
            LIMIT :top_k
        """)

        async for session in get_session(engine):
            try:
                result = await session.execute(course_sql, params)
                rows = [dict(r._mapping) for r in result.fetchall()]
                logger.info("retrieve_courses: 检索完成", count=len(rows))
                return rows
            except Exception as e:
                logger.error("retrieve_courses: SQL 执行失败", error=str(e))
                return []

        return []

    @staticmethod
    def _rrf_merge(
        vec_rows: list[dict],
        fts_rows: list[dict],
        top_k: int,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """
        RRF（Reciprocal Rank Fusion）融合向量检索和关键词检索结果

        Args:
            vec_rows: 向量检索结果（已排序）
            fts_rows: 关键词检索结果（已排序）
            top_k: 最终返回数量
            k: RRF 平滑参数（默认 60）

        Returns:
            融合后的排序结果
        """
        rrf_scores: dict[str, float] = {}
        doc_data: dict[str, dict] = {}

        # 处理向量检索排名
        for rank, row in enumerate(vec_rows, 1):
            cid = row["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank)
            doc_data[cid] = row

        # 处理关键词检索排名
        for rank, row in enumerate(fts_rows, 1):
            cid = row["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank)
            if cid not in doc_data:
                doc_data[cid] = row

        # 按 RRF score 排序
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for cid in sorted_ids[:top_k]:
            row = doc_data[cid].copy()
            row["score"] = round(rrf_scores[cid], 4)
            results.append(row)

        return results
