"""
PostgreSQL 持久化 Checkpointer
继承 LangGraph 的 BaseCheckpointSaver，实现对话状态的 PostgreSQL 持久化

注意：推荐直接使用 langgraph-checkpoint-postgres 提供的 AsyncPostgresSaver，
本模块作为封装层，统一管理连接池和初始化逻辑。
"""
from __future__ import annotations

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from fitness_agent.config import get_settings

logger = structlog.get_logger(__name__)


class PGCheckpointer:
    """
    PostgreSQL Checkpointer 工厂类

    封装 AsyncPostgresSaver 的创建和初始化逻辑，
    确保 checkpoint 所需表在首次使用前自动创建。

    使用示例：
        checkpointer = PGCheckpointer()
        saver = await checkpointer.create()
        graph = build_graph(checkpointer=saver)
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._saver: AsyncPostgresSaver | None = None

    async def create(self) -> AsyncPostgresSaver:
        """
        创建并初始化 AsyncPostgresSaver

        自动执行 setup() 创建必要的数据库表：
        - checkpoints
        - checkpoint_blobs
        - checkpoint_writes

        Returns:
            初始化完成的 AsyncPostgresSaver 实例
        """
        conn_string = self._settings.database.psycopg_url
        logger.info("pg_checkpointer: 初始化 PostgreSQL checkpointer")

        try:
            saver = await AsyncPostgresSaver.from_conn_string(conn_string)
            # 自动创建所需数据库表
            await saver.setup()
            self._saver = saver
            logger.info("pg_checkpointer: 初始化成功")
            return saver
        except Exception as e:
            logger.error("pg_checkpointer: 初始化失败", error=str(e))
            raise

    async def close(self) -> None:
        """关闭 checkpointer 连接"""
        if self._saver is not None:
            try:
                await self._saver.conn.close()
                logger.info("pg_checkpointer: 连接已关闭")
            except Exception as e:
                logger.warning("pg_checkpointer: 关闭连接失败", error=str(e))
            finally:
                self._saver = None

    @property
    def saver(self) -> AsyncPostgresSaver | None:
        """获取当前 saver 实例（未初始化时为 None）"""
        return self._saver


# 全局单例管理
_pg_checkpointer: PGCheckpointer | None = None


def get_pg_checkpointer() -> PGCheckpointer:
    """获取全局 PGCheckpointer 单例"""
    global _pg_checkpointer
    if _pg_checkpointer is None:
        _pg_checkpointer = PGCheckpointer()
    return _pg_checkpointer
