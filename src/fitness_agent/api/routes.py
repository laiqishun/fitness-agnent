"""
FastAPI 路由定义
提供 AI 私教智能体的 HTTP API 接口
"""
from __future__ import annotations

import uuid as uuid_module
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from fitness_agent.config import Settings, get_settings
from fitness_agent.graph.graph import run_agent
from fitness_agent.models.database import (
    ChatMessage,
    ChatSession,
    User,
    get_session,
)
from fitness_agent.models.schemas import (
    ChatHistoryOut,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    MessageOut,
    SessionCreate,
    SessionOut,
    SuccessResponse,
    UserCreate,
    UserOut,
    UserProfileUpdate,
    UserProfileOut,
)
from fitness_agent.services.session_summarizer import SessionSummarizer

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================
# 依赖注入辅助
# =============================================================

async def get_db() -> AsyncSession:
    """获取数据库 session（FastAPI 依赖）"""
    from fitness_agent.models.database import create_engine
    engine = create_engine()
    async for session in get_session(engine):
        yield session


async def get_or_create_user(
    app_user_id: str,
    db: AsyncSession,
) -> User:
    """根据 app_user_id 获取或创建用户"""
    result = await db.execute(
        select(User).where(User.app_user_id == app_user_id, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()
    if user is None:
        user = User(app_user_id=app_user_id)
        db.add(user)
        await db.flush()   # 获取 user.id
        logger.info("routes: 新用户已创建", app_user_id=app_user_id, user_id=str(user.id))
    return user


# =============================================================
# 聊天接口（核心）
# =============================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="发送消息",
    description="向 AI 私教发送消息，自动处理意图识别、RAG 检索、记忆管理",
)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """
    核心聊天接口

    自动完成：
    1. 用户身份识别/创建
    2. 会话管理（新建/续接）
    3. 调用 LangGraph 智能体
    4. 触发会话摘要（如有需要）
    """
    # ── 获取/创建用户 ─────────────────────────────────────────
    user = await get_or_create_user(request.user_id, db)

    # ── 获取/创建会话 ─────────────────────────────────────────
    session_id = str(request.session_id) if request.session_id else None

    if session_id:
        # 验证会话归属
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == uuid_module.UUID(session_id),
                ChatSession.user_id == user.id,
            )
        )
        chat_session = result.scalar_one_or_none()
        if chat_session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="会话不存在或无权访问",
            )
    else:
        # 创建新会话
        chat_session = ChatSession(
            user_id=user.id,
            title=request.message[:50],  # 用第一条消息前 50 字作标题
        )
        db.add(chat_session)
        await db.flush()
        session_id = str(chat_session.id)
        logger.info("routes: 新会话已创建", session_id=session_id)

    await db.commit()

    # ── 从 app state 获取 graph ───────────────────────────────
    from fitness_agent.main import app as fastapi_app
    graph = fastapi_app.state.graph

    # ── 调用智能体 ────────────────────────────────────────────
    try:
        result = await run_agent(
            graph=graph,
            user_message=request.message,
            user_id=str(user.id),
            app_user_id=request.user_id,
            session_id=session_id,
            metadata=request.metadata,
        )
    except Exception as e:
        logger.error("routes: 智能体调用失败", error=str(e), session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"智能体处理失败：{str(e)}",
        )

    # ── 触发会话摘要（异步，不阻塞响应）────────────────────────
    try:
        summarizer = SessionSummarizer(settings)
        if await summarizer.should_summarize(session_id):
            import asyncio
            asyncio.create_task(
                summarizer.summarize_session(session_id, str(user.id))
            )
            logger.info("routes: 已触发后台摘要任务", session_id=session_id)
    except Exception as e:
        logger.warning("routes: 摘要触发失败（不影响响应）", error=str(e))

    # 生成消息 ID（LangGraph 消息的最后一条）
    message_id = uuid_module.uuid4()

    return ChatResponse(
        session_id=uuid_module.UUID(session_id),
        message_id=message_id,
        reply=result.get("reply", ""),
        intent=result.get("intent", "unknown"),
        need_clarification=result.get("need_clarification", False),
        structured_data=result.get("structured_output", {}),
        metadata=result.get("metadata", {}),
    )


# =============================================================
# 会话管理
# =============================================================

@router.post(
    "/sessions",
    response_model=SessionOut,
    status_code=status.HTTP_201_CREATED,
    summary="创建新会话",
)
async def create_session(
    body: SessionCreate,
    user_id: str,
    db: AsyncSession = Depends(get_db),
) -> SessionOut:
    """创建新对话会话"""
    user = await get_or_create_user(user_id, db)

    chat_session = ChatSession(
        user_id=user.id,
        title=body.title,
    )
    db.add(chat_session)
    await db.commit()
    await db.refresh(chat_session)

    return SessionOut.model_validate(chat_session)


@router.get(
    "/sessions/{session_id}/history",
    response_model=ChatHistoryOut,
    summary="获取会话历史",
)
async def get_session_history(
    session_id: uuid_module.UUID,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> ChatHistoryOut:
    """获取指定会话的消息历史"""
    user = await get_or_create_user(user_id, db)

    # 验证会话归属
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user.id,
        )
    )
    chat_session = result.scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 查询消息
    msg_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.sequence_num)
        .offset(offset)
        .limit(limit)
    )
    messages = msg_result.scalars().all()

    return ChatHistoryOut(
        session_id=session_id,
        messages=[MessageOut.model_validate(m) for m in messages],
        total=chat_session.message_count,
    )


@router.put(
    "/sessions/{session_id}/summarize",
    response_model=SuccessResponse,
    summary="手动触发会话摘要",
)
async def summarize_session(
    session_id: uuid_module.UUID,
    user_id: str,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> SuccessResponse:
    """手动触发指定会话的摘要生成"""
    user = await get_or_create_user(user_id, db)

    # 验证会话归属
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user.id,
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    summarizer = SessionSummarizer(settings)
    summary_result = await summarizer.summarize_session(
        session_id=str(session_id),
        user_id=str(user.id),
        force=True,
    )

    if not summary_result:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="摘要生成失败或会话无消息",
        )

    return SuccessResponse(message=f"摘要已生成，关键事实 {len(summary_result.get('key_facts', []))} 条")


# =============================================================
# 健康检查
# =============================================================

@router.get("/health", summary="服务健康检查")
async def health_check() -> dict[str, Any]:
    """检查各依赖服务状态"""
    from fitness_agent.memory.redis_short_term import get_short_term_memory

    health: dict[str, Any] = {"status": "ok", "services": {}}

    # Redis 检查
    try:
        memory = get_short_term_memory()
        redis_ok = await memory.ping()
        health["services"]["redis"] = "ok" if redis_ok else "error"
    except Exception as e:
        health["services"]["redis"] = f"error: {e}"

    # 数据库检查
    try:
        from fitness_agent.models.database import create_engine
        from sqlalchemy import text
        engine = create_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        health["services"]["postgres"] = "ok"
    except Exception as e:
        health["services"]["postgres"] = f"error: {e}"
        health["status"] = "degraded"

    return health
