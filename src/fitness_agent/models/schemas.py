"""
Pydantic Schemas — 用于 API 请求/响应的数据校验与序列化
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# =============================================================
# 通用基类
# =============================================================

class BaseSchema(BaseModel):
    """所有 Schema 的基类，开启 from_attributes 支持 ORM 对象直接转换"""
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# =============================================================
# 用户相关
# =============================================================

class UserCreate(BaseSchema):
    """创建用户请求"""
    app_user_id: str = Field(..., description="App 侧用户唯一 ID")
    username: str | None = None
    email: str | None = None
    phone: str | None = None
    gender: Literal["male", "female", "other", "unknown"] = "unknown"
    timezone: str = "Asia/Shanghai"
    locale: str = "zh-CN"


class UserOut(BaseSchema):
    """用户信息响应"""
    id: uuid.UUID
    app_user_id: str
    username: str | None
    email: str | None
    gender: str
    timezone: str
    is_active: bool
    created_at: datetime


# =============================================================
# 用户画像
# =============================================================

class UserProfileUpdate(BaseSchema):
    """更新用户画像请求（所有字段可选）"""
    height_cm: float | None = None
    weight_kg: float | None = None
    body_fat_pct: float | None = None
    fitness_level: Literal["beginner", "intermediate", "advanced", "elite"] | None = None
    primary_goal: str | None = None
    secondary_goals: list[str] | None = None
    available_equipment: list[str] | None = None
    workout_location: str | None = None
    injury_history: list[str] | None = None
    health_conditions: list[str] | None = None
    weekly_workout_days: int | None = Field(None, ge=0, le=7)
    preferred_workout_duration_min: int | None = None
    preferred_workout_time: str | None = None
    dietary_restrictions: list[str] | None = None


class UserProfileOut(BaseSchema):
    """用户画像响应"""
    id: uuid.UUID
    user_id: uuid.UUID
    height_cm: float | None
    weight_kg: float | None
    fitness_level: str
    primary_goal: str
    available_equipment: list[str]
    weekly_workout_days: int
    profile_summary: str | None
    is_current: bool
    updated_at: datetime


# =============================================================
# 会话相关
# =============================================================

class SessionCreate(BaseSchema):
    """创建会话请求"""
    title: str | None = Field(None, max_length=256, description="会话标题（可选，自动生成）")


class SessionOut(BaseSchema):
    """会话信息响应"""
    id: uuid.UUID
    user_id: uuid.UUID
    status: str
    title: str | None
    summary: str | None
    message_count: int
    last_message_at: datetime
    created_at: datetime


# =============================================================
# 消息相关
# =============================================================

class MessageOut(BaseSchema):
    """单条消息响应"""
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    intent: str | None
    sequence_num: int
    created_at: datetime


class ChatHistoryOut(BaseSchema):
    """会话历史响应"""
    session_id: uuid.UUID
    messages: list[MessageOut]
    total: int


# =============================================================
# 聊天请求/响应
# =============================================================

class ChatRequest(BaseSchema):
    """发送消息请求"""
    user_id: str = Field(..., description="App 侧用户 ID")
    session_id: uuid.UUID | None = Field(None, description="会话 ID，不填则创建新会话")
    message: str = Field(..., min_length=1, max_length=4096, description="用户消息内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class ChatResponse(BaseSchema):
    """聊天响应"""
    session_id: uuid.UUID
    message_id: uuid.UUID
    reply: str = Field(..., description="助手回复内容")
    intent: str = Field(..., description="识别到的意图")
    need_clarification: bool = Field(default=False, description="是否需要追问")
    # 结构化数据（根据意图不同返回不同内容）
    structured_data: dict[str, Any] = Field(
        default_factory=dict,
        description="结构化输出，如课程列表、运动计划等",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================
# 课程相关
# =============================================================

class CourseOut(BaseSchema):
    """课程信息响应"""
    id: uuid.UUID
    name: str
    description: str | None
    instructor: str | None
    category: str | None
    difficulty: str
    duration_min: int | None
    tags: list[str]
    muscle_groups: list[str]
    equipment_needed: list[str]
    thumbnail_url: str | None
    rating: float
    is_premium: bool


class CourseRecommendationOut(BaseSchema):
    """课程推荐响应"""
    courses: list[CourseOut]
    reason: str = Field(..., description="推荐原因说明")
    total: int


# =============================================================
# 提醒相关
# =============================================================

class ReminderCreate(BaseSchema):
    """创建提醒（内部使用）"""
    user_id: uuid.UUID
    session_id: uuid.UUID | None = None
    title: str
    description: str | None = None
    reminder_type: str = "general"
    remind_at: datetime
    timezone: str = "Asia/Shanghai"
    recurrence_rule: str | None = None


class ReminderOut(BaseSchema):
    """提醒信息响应"""
    id: uuid.UUID
    title: str
    description: str | None
    reminder_type: str
    remind_at: datetime
    status: str
    created_at: datetime


# =============================================================
# 饮食/运动记录
# =============================================================

class FoodItem(BaseSchema):
    """单种食物营养信息"""
    name: str
    amount_g: float | None = None
    amount_desc: str | None = None   # 例如 "一碗"、"两个"
    calories_kcal: float = 0
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0
    fiber_g: float = 0


class DietRecordOut(BaseSchema):
    """饮食记录响应"""
    id: uuid.UUID
    meal_type: str
    raw_input: str
    food_items: list[FoodItem]
    total_calories_kcal: float
    total_protein_g: float
    total_carbs_g: float
    total_fat_g: float
    estimate_confidence: float
    recorded_at: datetime


class ExerciseItem(BaseSchema):
    """单个运动动作信息"""
    name: str
    sets: int | None = None
    reps: int | None = None
    weight_kg: float | None = None
    duration_min: float | None = None
    distance_km: float | None = None
    rest_sec: int | None = None
    calories_kcal: float = 0


class ExerciseRecordOut(BaseSchema):
    """运动记录响应"""
    id: uuid.UUID
    exercise_type: str
    raw_input: str
    exercise_items: list[ExerciseItem]
    duration_min: int | None
    calories_burned_kcal: float
    estimate_confidence: float
    recorded_at: datetime


# =============================================================
# 运动计划
# =============================================================

class WorkoutPlanDay(BaseSchema):
    """单日训练计划"""
    day: int = Field(..., description="第几天（1-7）")
    name: str = Field(..., description="例如 '胸肩训练日'")
    exercises: list[ExerciseItem]
    estimated_duration_min: int
    notes: str | None = None


class WorkoutPlanOut(BaseSchema):
    """完整运动计划响应"""
    plan_name: str
    goal: str
    duration_weeks: int
    weekly_schedule: list[WorkoutPlanDay]
    equipment_required: list[str]
    warm_up_tips: str | None = None
    cool_down_tips: str | None = None
    nutrition_tips: str | None = None
    safety_notes: str | None = None
    reflection_notes: str | None = Field(None, description="AI 反思阶段的评估说明")


# =============================================================
# 通用响应
# =============================================================

class ErrorResponse(BaseSchema):
    """错误响应"""
    code: str
    message: str
    detail: dict[str, Any] | None = None


class SuccessResponse(BaseSchema):
    """成功响应（无数据）"""
    success: bool = True
    message: str = "操作成功"
