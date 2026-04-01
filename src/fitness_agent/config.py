"""
配置管理模块
使用 pydantic-settings 从环境变量加载所有配置，支持 .env 文件
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL 数据库配置"""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", extra="ignore")

    host: str = Field(default="localhost", description="数据库主机")
    port: int = Field(default=5432, description="数据库端口")
    db: str = Field(default="fitness_agent", description="数据库名")
    user: str = Field(default="postgres", description="数据库用户")
    password: str = Field(default="", description="数据库密码")
    pool_size: int = Field(default=10, description="连接池大小")
    max_overflow: int = Field(default=20, description="连接池溢出上限")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def async_url(self) -> str:
        """asyncpg 异步连接 URL"""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sync_url(self) -> str:
        """同步连接 URL（用于 alembic / checkpointer）"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def psycopg_url(self) -> str:
        """psycopg3 连接 URL（用于 langgraph checkpoint）"""
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class RedisSettings(BaseSettings):
    """Redis 配置"""

    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="ignore")

    host: str = Field(default="localhost", description="Redis 主机")
    port: int = Field(default=6379, description="Redis 端口")
    password: str = Field(default="", description="Redis 密码（空则无需认证）")
    db: int = Field(default=0, description="Redis DB 编号")
    ttl_seconds: int = Field(default=3600, description="短期记忆 TTL（秒）")
    max_short_term_messages: int = Field(default=20, description="短期记忆保留最近 N 条消息")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def url(self) -> str:
        """Redis 连接 URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class QwenSettings(BaseSettings):
    """通义千问 / DashScope 模型配置（OpenAI 兼容接口）"""

    model_config = SettingsConfigDict(env_prefix="DASHSCOPE_", extra="ignore")

    api_key: str = Field(default="", description="DashScope API Key")
    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="OpenAI 兼容接口 base URL",
    )

    # 模型名称（单独前缀，通过自定义 validator 处理）
    plus_model: str = Field(default="qwen-plus", alias="QWEN_PLUS_MODEL")
    turbo_model: str = Field(default="qwen-turbo", alias="QWEN_TURBO_MODEL")
    embedding_model: str = Field(default="text-embedding-v3", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")
    rerank_model: str = Field(default="gte-rerank", alias="RERANK_MODEL")

    # 调用参数
    temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")
    request_timeout: int = Field(default=60, alias="LLM_REQUEST_TIMEOUT")

    model_config = SettingsConfigDict(
        env_prefix="",          # 使用 alias，不要统一前缀
        populate_by_name=True,
        extra="ignore",
    )


class AppBackendSettings(BaseSettings):
    """App 后端接口配置"""

    model_config = SettingsConfigDict(env_prefix="APP_BACKEND_", extra="ignore")

    base_url: str = Field(default="https://api.speediance.com", description="App 后端 base URL")
    api_key: str = Field(default="", description="内部接口鉴权 Key")
    timeout: int = Field(default=10, description="请求超时（秒）")
    max_retries: int = Field(default=3, description="最大重试次数")


class RAGSettings(BaseSettings):
    """RAG 检索参数配置"""

    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    top_k: int = Field(default=10, description="初检召回数量")
    rerank_top_n: int = Field(default=5, description="重排后保留数量")
    min_score: float = Field(default=0.6, description="最低相似度阈值")


class Settings(BaseSettings):
    """全局配置聚合，自动从 .env 文件加载"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # 应用基础
    app_env: Literal["development", "production", "testing"] = Field(
        default="development", alias="APP_ENV"
    )
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_secret_key: str = Field(default="change-me", alias="APP_SECRET_KEY")

    # 会话摘要触发阈值
    summarize_after_messages: int = Field(default=20, alias="SUMMARIZE_AFTER_MESSAGES")

    # 嵌套配置（通过子 Settings 对象管理）
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    qwen: QwenSettings = Field(default_factory=QwenSettings)
    app_backend: AppBackendSettings = Field(default_factory=AppBackendSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)

    @model_validator(mode="after")
    def _validate_production(self) -> "Settings":
        """生产环境强制校验"""
        if self.app_env == "production":
            if not self.qwen.api_key:
                raise ValueError("生产环境必须设置 DASHSCOPE_API_KEY")
            if self.app_secret_key == "change-me":
                raise ValueError("生产环境必须修改 APP_SECRET_KEY")
        return self

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    获取全局配置单例（使用 lru_cache 保证只初始化一次）
    FastAPI 依赖注入：Depends(get_settings)
    """
    return Settings()
