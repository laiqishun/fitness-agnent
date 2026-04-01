"""
FastAPI 应用入口
包含：
- 应用创建和配置
- 生命周期管理（startup/shutdown）
- 中间件注册
- 路由注册
- 结构化日志初始化
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fitness_agent.api.routes import router
from fitness_agent.config import get_settings

# =============================================================
# 日志配置
# =============================================================

def configure_logging(debug: bool = False) -> None:
    """配置 structlog 结构化日志"""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if debug else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            10 if debug else 20   # DEBUG=10, INFO=20
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


logger = structlog.get_logger(__name__)


# =============================================================
# 应用生命周期
# =============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理
    startup: 初始化数据库连接池、Redis、LangGraph 图
    shutdown: 优雅关闭连接
    """
    settings = get_settings()
    configure_logging(debug=settings.app_debug)

    logger.info("🚀 Fitness Agent 启动中...", env=settings.app_env)

    # ── Startup ────────────────────────────────────────────────
    # 1. 初始化数据库引擎
    try:
        from fitness_agent.models.database import create_engine
        engine = create_engine(settings)
        app.state.db_engine = engine
        logger.info("✅ 数据库引擎初始化完成")
    except Exception as e:
        logger.error("❌ 数据库引擎初始化失败", error=str(e))
        app.state.db_engine = None

    # 2. 初始化 Redis 连接
    try:
        from fitness_agent.memory.redis_short_term import get_short_term_memory
        memory = get_short_term_memory()
        if await memory.ping():
            app.state.redis_memory = memory
            logger.info("✅ Redis 连接初始化完成")
        else:
            logger.warning("⚠️ Redis 连接失败，短期记忆不可用")
            app.state.redis_memory = None
    except Exception as e:
        logger.warning("⚠️ Redis 初始化异常", error=str(e))
        app.state.redis_memory = None

    # 3. 初始化 LangGraph 图（带 PostgreSQL checkpointer）
    try:
        from fitness_agent.graph.graph import create_graph_with_pg_checkpointer, build_graph
        graph = await create_graph_with_pg_checkpointer()
        app.state.graph = graph
        logger.info("✅ LangGraph 图初始化完成")
    except Exception as e:
        logger.error("❌ LangGraph 图初始化失败，使用内存模式", error=str(e))
        from fitness_agent.graph.graph import build_graph
        app.state.graph = build_graph()

    logger.info("🎯 Fitness Agent 启动完成", host=settings.app_host, port=settings.app_port)

    yield  # 应用运行中

    # ── Shutdown ───────────────────────────────────────────────
    logger.info("🛑 Fitness Agent 正在关闭...")

    if hasattr(app.state, "db_engine") and app.state.db_engine:
        await app.state.db_engine.dispose()
        logger.info("✅ 数据库连接池已释放")

    if hasattr(app.state, "redis_memory") and app.state.redis_memory:
        await app.state.redis_memory.close()
        logger.info("✅ Redis 连接已关闭")

    logger.info("👋 Fitness Agent 已关闭")


# =============================================================
# FastAPI 应用创建
# =============================================================

def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    settings = get_settings()

    app = FastAPI(
        title="Speediance AI 私教智能体",
        description="基于 LangGraph 的多意图健身助手 API",
        version="0.1.0",
        docs_url="/docs" if settings.is_dev else None,    # 生产环境关闭 Swagger
        redoc_url="/redoc" if settings.is_dev else None,
        lifespan=lifespan,
    )

    # ── 中间件 ────────────────────────────────────────────────
    # CORS（开发环境允许所有来源）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_dev else ["https://app.speediance.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 请求日志中间件
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            path=request.url.path,
            method=request.method,
        )

        response = await call_next(request)

        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            "HTTP 请求",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response

    # ── 全局异常处理 ──────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("未捕获的异常", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={"code": "INTERNAL_ERROR", "message": "服务器内部错误，请稍后重试"},
        )

    # ── 注册路由 ──────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1", tags=["AI 私教"])

    return app


# 全局应用实例（供 uvicorn 和测试使用）
app = create_app()


# =============================================================
# 命令行入口
# =============================================================

def cli_main() -> None:
    """命令行启动入口（hatch run dev）"""
    settings = get_settings()
    uvicorn.run(
        "fitness_agent.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_dev,
        log_level="debug" if settings.app_debug else "info",
    )


if __name__ == "__main__":
    cli_main()
