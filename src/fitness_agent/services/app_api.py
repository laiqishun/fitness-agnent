"""
App 后端接口调用模块
负责与 Speediance App 服务端通信，目前支持：
- POST /reminders（创建提醒）

特性：
- 使用 httpx 异步客户端
- tenacity 自动重试（指数退避）
- 完整的错误处理和日志
"""
from __future__ import annotations

from typing import Any

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fitness_agent.config import get_settings

logger = structlog.get_logger(__name__)


class AppAPIError(Exception):
    """App API 调用异常"""

    def __init__(self, message: str, status_code: int | None = None, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AppAPIClient:
    """
    App 后端 HTTP 客户端（异步上下文管理器）

    使用示例：
        async with AppAPIClient() as client:
            result = await client.create_reminder({...})
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "AppAPIClient":
        cfg = self._settings.app_backend
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url,
            timeout=cfg.timeout,
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
                "X-Service": "fitness-agent",
            },
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        带重试的 HTTP 请求

        使用指数退避策略，最多重试 max_retries 次
        仅对网络错误和 5xx 服务端错误重试，4xx 直接抛出
        """
        if self._client is None:
            raise RuntimeError("AppAPIClient 未初始化，请使用 async with 上下文管理器")

        max_retries = self._settings.app_backend.max_retries

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(
                (httpx.NetworkError, httpx.TimeoutException)
            ),
            reraise=True,
        ):
            with attempt:
                logger.debug(
                    "app_api: 发送请求",
                    method=method,
                    path=path,
                    attempt=attempt.retry_state.attempt_number,
                )
                response = await self._client.request(method, path, **kwargs)

                # 4xx 错误直接抛出，不重试
                if 400 <= response.status_code < 500:
                    raise AppAPIError(
                        f"App API 返回 {response.status_code}",
                        status_code=response.status_code,
                        response_body=response.text,
                    )

                # 5xx 错误触发重试
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"服务端错误 {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                return response.json()

        raise AppAPIError("请求重试次数耗尽")

    # ── 提醒接口 ──────────────────────────────────────────────

    async def create_reminder(self, reminder_data: dict[str, Any]) -> dict[str, Any]:
        """
        创建提醒

        Args:
            reminder_data: 提醒信息，包含：
                - user_id: 用户 ID
                - title: 提醒标题
                - description: 详细描述（可选）
                - reminder_type: 提醒类型
                - remind_at: 提醒时间（ISO 格式）
                - timezone: 时区
                - recurrence_rule: 重复规则（可选）

        Returns:
            App 返回的提醒信息，包含 reminder_id

        Raises:
            AppAPIError: 接口调用失败
        """
        logger.info(
            "app_api: 创建提醒",
            title=reminder_data.get("title"),
            remind_at=reminder_data.get("remind_at"),
        )

        try:
            result = await self._request_with_retry(
                "POST",
                "/reminders",
                json={
                    "user_id": reminder_data.get("user_id"),
                    "title": reminder_data.get("title"),
                    "description": reminder_data.get("description"),
                    "type": reminder_data.get("reminder_type", "general"),
                    "remind_at": reminder_data.get("remind_at"),
                    "timezone": reminder_data.get("timezone", "Asia/Shanghai"),
                    "recurrence": reminder_data.get("recurrence_rule"),
                },
            )
            logger.info("app_api: 提醒创建成功", reminder_id=result.get("reminder_id"))
            return result
        except AppAPIError:
            raise
        except Exception as e:
            logger.error("app_api: 创建提醒失败", error=str(e))
            raise AppAPIError(f"创建提醒失败: {e}") from e

    async def cancel_reminder(self, app_reminder_id: str) -> dict[str, Any]:
        """
        取消提醒

        Args:
            app_reminder_id: App 侧提醒 ID

        Returns:
            操作结果
        """
        logger.info("app_api: 取消提醒", app_reminder_id=app_reminder_id)
        try:
            return await self._request_with_retry(
                "DELETE",
                f"/reminders/{app_reminder_id}",
            )
        except Exception as e:
            logger.error("app_api: 取消提醒失败", error=str(e))
            raise AppAPIError(f"取消提醒失败: {e}") from e
