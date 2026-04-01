"""
SQLAlchemy 异步 ORM 模型
对应 sql/schema.sql 中定义的所有表
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    ARRAY,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import NullPool

from fitness_agent.config import get_settings

# =============================================================
# Base 类 & 引擎工厂
# =============================================================


class Base(AsyncAttrs, DeclarativeBase):
    """所有 ORM 模型的基类"""

    type_annotation_map = {
        dict[str, Any]: JSON,
        list[str]: ARRAY(String),
        list[Any]: ARRAY(JSON),
    }


def create_engine(settings=None) -> AsyncEngine:
    """创建异步数据库引擎"""
    if settings is None:
        settings = get_settings()
    cfg = settings.database
    return create_async_engine(
        cfg.async_url,
        pool_size=cfg.pool_size,
        max_overflow=cfg.max_overflow,
        pool_pre_ping=True,         # 连接前 ping，自动处理断连
        echo=settings.app_debug,    # debug 模式输出 SQL
    )


async def get_session(engine: AsyncEngine | None = None):
    """
    FastAPI 依赖注入用的会话工厂
    用法: session = Depends(get_session)
    """
    if engine is None:
        engine = create_engine()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# =============================================================
# 枚举类型（对应 PostgreSQL ENUM）
# =============================================================

GenderEnum = Enum(
    "male", "female", "other", "unknown",
    name="gender_type", create_constraint=True
)
FitnessLevelEnum = Enum(
    "beginner", "intermediate", "advanced", "elite",
    name="fitness_level_type", create_constraint=True
)
GoalEnum = Enum(
    "lose_weight", "gain_muscle", "improve_endurance",
    "maintain_health", "rehabilitation", "other",
    name="goal_type", create_constraint=True
)
MessageRoleEnum = Enum(
    "user", "assistant", "system", "tool",
    name="message_role", create_constraint=True
)
IntentEnum = Enum(
    "qa", "course_recommendation", "set_reminder",
    "record_diet_exercise", "generate_workout_plan",
    "need_clarification", "unknown",
    name="intent_type", create_constraint=True
)
SessionStatusEnum = Enum(
    "active", "summarized", "archived",
    name="session_status", create_constraint=True
)
ReminderStatusEnum = Enum(
    "pending", "sent", "cancelled", "failed",
    name="reminder_status", create_constraint=True
)
DocumentStatusEnum = Enum(
    "active", "deprecated", "processing",
    name="document_status", create_constraint=True
)
CourseDifficultyEnum = Enum(
    "easy", "moderate", "hard", "extreme",
    name="course_difficulty", create_constraint=True
)
MealTypeEnum = Enum(
    "breakfast", "lunch", "dinner", "snack", "other",
    name="meal_type", create_constraint=True
)
ExerciseTypeEnum = Enum(
    "strength", "cardio", "flexibility", "balance", "hiit", "other",
    name="exercise_type", create_constraint=True
)

# =============================================================
# ORM 模型
# =============================================================


class User(Base):
    """基础用户表"""
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    app_user_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    username: Mapped[str | None] = mapped_column(String(64))
    email: Mapped[str | None] = mapped_column(String(256))
    phone: Mapped[str | None] = mapped_column(String(32))
    gender: Mapped[str] = mapped_column(GenderEnum, default="unknown")
    birth_date: Mapped[date | None] = mapped_column(Date)
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Shanghai")
    locale: Mapped[str] = mapped_column(String(16), default="zh-CN")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 关联
    profiles: Mapped[list["UserProfile"]] = relationship(back_populates="user")
    sessions: Mapped[list["ChatSession"]] = relationship(back_populates="user")


class UserProfile(Base):
    """用户健身画像表（含向量）"""
    __tablename__ = "user_profiles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    # 体征数据
    height_cm: Mapped[float | None] = mapped_column(Numeric(5, 1))
    weight_kg: Mapped[float | None] = mapped_column(Numeric(5, 1))
    body_fat_pct: Mapped[float | None] = mapped_column(Numeric(4, 1))
    # 健身目标与水平
    fitness_level: Mapped[str] = mapped_column(FitnessLevelEnum, default="beginner")
    primary_goal: Mapped[str] = mapped_column(GoalEnum, default="maintain_health")
    secondary_goals: Mapped[list[str]] = mapped_column(
        ARRAY(GoalEnum), server_default=text("'{}'")
    )
    # 器械与环境
    available_equipment: Mapped[list[str]] = mapped_column(
        ARRAY(String), server_default=text("'{}'")
    )
    workout_location: Mapped[str] = mapped_column(String(32), default="home")
    # 健康状况
    injury_history: Mapped[list[str]] = mapped_column(
        ARRAY(String), server_default=text("'{}'")
    )
    health_conditions: Mapped[list[str]] = mapped_column(
        ARRAY(String), server_default=text("'{}'")
    )
    # 运动习惯
    weekly_workout_days: Mapped[int] = mapped_column(Integer, default=3)
    preferred_workout_duration_min: Mapped[int] = mapped_column(Integer, default=30)
    preferred_workout_time: Mapped[str | None] = mapped_column(String(32))
    # 饮食偏好
    dietary_restrictions: Mapped[list[str]] = mapped_column(
        ARRAY(String), server_default=text("'{}'")
    )
    # 画像向量
    profile_embedding: Mapped[list[float] | None] = mapped_column(Vector(1536))
    # 文本摘要
    profile_summary: Mapped[str | None] = mapped_column(Text)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # 关联
    user: Mapped["User"] = relationship(back_populates="profiles")


class UserProfileUpdateHistory(Base):
    """用户画像变更历史"""
    __tablename__ = "user_profiles_update_history"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    profile_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("user_profiles.id", ondelete="CASCADE")
    )
    changed_fields: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    change_reason: Mapped[str | None] = mapped_column(String(256))
    session_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ChatSession(Base):
    """对话会话表"""
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(SessionStatusEnum, default="active")
    title: Mapped[str | None] = mapped_column(String(256))
    summary: Mapped[str | None] = mapped_column(Text)
    key_facts: Mapped[list[Any]] = mapped_column(JSON, default=list)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    last_message_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 关联
    user: Mapped["User"] = relationship(back_populates="sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", order_by="ChatMessage.sequence_num"
    )


class ChatMessage(Base):
    """消息表（含向量）"""
    __tablename__ = "chat_messages"
    __table_args__ = (
        UniqueConstraint("session_id", "sequence_num", name="uq_messages_session_seq"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(MessageRoleEnum, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(IntentEnum)
    intent_confidence: Mapped[float | None] = mapped_column(Numeric(3, 2))
    tool_calls: Mapped[list[Any]] = mapped_column(JSON, default=list)
    content_embedding: Mapped[list[float] | None] = mapped_column(Vector(1536))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    sequence_num: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # 关联
    session: Mapped["ChatSession"] = relationship(back_populates="messages")


class SourceDocument(Base):
    """知识库原始文档"""
    __tablename__ = "source_documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(64), default="article")
    author: Mapped[str | None] = mapped_column(String(256))
    published_date: Mapped[date | None] = mapped_column(Date)
    oss_bucket: Mapped[str | None] = mapped_column(String(128))
    oss_key: Mapped[str | None] = mapped_column(String(512))
    oss_url: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    raw_content: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(16), default="zh")
    status: Mapped[str] = mapped_column(DocumentStatusEnum, default="active")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    tags: Mapped[list[str]] = mapped_column(ARRAY(String), server_default=text("'{}'"))
    category: Mapped[str | None] = mapped_column(String(128))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # 关联
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document")


class DocumentChunk(Base):
    """文档分块（含向量）"""
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_doc_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_documents.id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    parent_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="SET NULL")
    )
    page_number: Mapped[int | None] = mapped_column(Integer)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    keywords: Mapped[list[str]] = mapped_column(ARRAY(String), server_default=text("'{}'"))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # 关联
    document: Mapped["SourceDocument"] = relationship(back_populates="chunks")


class Course(Base):
    """课程表（含向量）"""
    __tablename__ = "courses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    name_en: Mapped[str | None] = mapped_column(String(256))
    description: Mapped[str | None] = mapped_column(Text)
    instructor: Mapped[str | None] = mapped_column(String(128))
    category: Mapped[str | None] = mapped_column(String(128))
    tags: Mapped[list[str]] = mapped_column(ARRAY(String), server_default=text("'{}'"))
    muscle_groups: Mapped[list[str]] = mapped_column(ARRAY(String), server_default=text("'{}'"))
    equipment_needed: Mapped[list[str]] = mapped_column(ARRAY(String), server_default=text("'{}'"))
    difficulty: Mapped[str] = mapped_column(CourseDifficultyEnum, default="moderate")
    duration_min: Mapped[int | None] = mapped_column(Integer)
    suitable_levels: Mapped[list[str]] = mapped_column(
        ARRAY(FitnessLevelEnum), server_default=text("'{}'")
    )
    suitable_goals: Mapped[list[str]] = mapped_column(
        ARRAY(GoalEnum), server_default=text("'{}'")
    )
    thumbnail_url: Mapped[str | None] = mapped_column(Text)
    video_url: Mapped[str | None] = mapped_column(Text)
    oss_key: Mapped[str | None] = mapped_column(String(512))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536))
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    rating: Mapped[float] = mapped_column(Numeric(2, 1), default=0.0)
    rating_count: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_premium: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Reminder(Base):
    """提醒记录"""
    __tablename__ = "reminders"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="SET NULL")
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    reminder_type: Mapped[str] = mapped_column(String(64), default="general")
    remind_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Shanghai")
    recurrence_rule: Mapped[str | None] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(ReminderStatusEnum, default="pending")
    app_reminder_id: Mapped[str | None] = mapped_column(String(128))
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DietRecord(Base):
    """饮食记录"""
    __tablename__ = "diet_records"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="SET NULL")
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    meal_type: Mapped[str] = mapped_column(MealTypeEnum, default="other")
    raw_input: Mapped[str] = mapped_column(Text, nullable=False)
    food_items: Mapped[list[Any]] = mapped_column(JSON, default=list)
    total_calories_kcal: Mapped[float] = mapped_column(Numeric(7, 1), default=0)
    total_protein_g: Mapped[float] = mapped_column(Numeric(6, 1), default=0)
    total_carbs_g: Mapped[float] = mapped_column(Numeric(6, 1), default=0)
    total_fat_g: Mapped[float] = mapped_column(Numeric(6, 1), default=0)
    total_fiber_g: Mapped[float] = mapped_column(Numeric(6, 1), default=0)
    estimate_confidence: Mapped[float] = mapped_column(Numeric(3, 2), default=0.8)
    notes: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ExerciseRecord(Base):
    """运动记录"""
    __tablename__ = "exercise_records"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="SET NULL")
    )
    course_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("courses.id", ondelete="SET NULL")
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    duration_min: Mapped[int | None] = mapped_column(Integer)
    raw_input: Mapped[str] = mapped_column(Text, nullable=False)
    exercise_type: Mapped[str] = mapped_column(ExerciseTypeEnum, default="other")
    exercise_items: Mapped[list[Any]] = mapped_column(JSON, default=list)
    calories_burned_kcal: Mapped[float] = mapped_column(Numeric(6, 1), default=0)
    avg_heart_rate: Mapped[int | None] = mapped_column(Integer)
    max_heart_rate: Mapped[int | None] = mapped_column(Integer)
    estimate_confidence: Mapped[float] = mapped_column(Numeric(3, 2), default=0.8)
    notes: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
