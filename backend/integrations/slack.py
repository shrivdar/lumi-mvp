"""Slack HITL integration — send questions to researchers, collect responses."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from core.audit import AuditLogger
from core.config import settings
from core.constants import CACHE_TTL_DEFAULT
from core.exceptions import ToolError
from integrations.base_tool import BaseTool

logger = structlog.get_logger(__name__)
audit = AuditLogger("slack")


class SlackTool(BaseTool):
    tool_id = "slack"
    name = "slack_hitl"
    description = "Send human-in-the-loop questions to Slack and collect expert responses for uncertainty resolution."
    category = "hitl"
    rate_limit = 1.0  # conservative for Slack
    cache_ttl = CACHE_TTL_DEFAULT

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init Slack WebClient to avoid import overhead when unused."""
        if self._client is None:
            if not settings.slack_bot_token:
                raise ToolError(
                    "SLACK_BOT_TOKEN not configured",
                    error_code="SLACK_NOT_CONFIGURED",
                )
            from slack_sdk.web.async_client import AsyncWebClient
            self._client = AsyncWebClient(token=settings.slack_bot_token)
        return self._client

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "ask")
        if action == "ask":
            return await self._ask(
                question=kwargs["question"],
                channel=kwargs.get("channel", settings.slack_default_channel),
                context=kwargs.get("context", ""),
                timeout=kwargs.get("timeout", settings.hitl_timeout_seconds),
            )
        elif action == "notify":
            return await self._notify(
                message=kwargs["message"],
                channel=kwargs.get("channel", settings.slack_default_channel),
            )
        raise ValueError(f"Unknown Slack action: {action}")

    async def _ask(
        self,
        question: str,
        channel: str,
        context: str = "",
        timeout: int = 600,
    ) -> dict[str, Any]:
        client = self._get_client()

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "YOHAS — Expert Input Needed"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Question:*\n{question}"},
            },
        ]
        if context:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Context:*\n{context}"},
            })
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"_Please reply in a thread within {timeout // 60} minutes._"},
        })

        try:
            result = await client.chat_postMessage(
                channel=channel,
                text=f"YOHAS Expert Input: {question}",
                blocks=blocks,
            )
        except Exception as exc:
            raise ToolError(
                f"Failed to post Slack message: {exc}",
                error_code="SLACK_POST_FAILED",
            ) from exc

        ts = result.get("ts", "")
        channel_id = result.get("channel", channel)

        audit.log("slack_question_sent", channel=channel_id, ts=ts)

        # Poll for thread replies
        response = await self._poll_for_reply(
            client=client,
            channel=channel_id,
            thread_ts=ts,
            timeout=timeout,
        )

        return {
            "question": question,
            "channel": channel_id,
            "thread_ts": ts,
            "response": response,
            "responded": response is not None,
        }

    async def _notify(self, message: str, channel: str) -> dict[str, Any]:
        client = self._get_client()
        try:
            result = await client.chat_postMessage(
                channel=channel,
                text=message,
            )
            return {
                "channel": result.get("channel", channel),
                "ts": result.get("ts", ""),
                "sent": True,
            }
        except Exception as exc:
            raise ToolError(
                f"Failed to send Slack notification: {exc}",
                error_code="SLACK_NOTIFY_FAILED",
            ) from exc

    async def _poll_for_reply(
        self,
        client: Any,
        channel: str,
        thread_ts: str,
        timeout: int,
        poll_interval: int = 15,
    ) -> str | None:
        """Poll Slack thread for a reply from a human."""
        elapsed = 0
        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            try:
                result = await client.conversations_replies(
                    channel=channel,
                    ts=thread_ts,
                    limit=10,
                )
                messages = result.get("messages", [])
                # Skip the original message (first in thread)
                replies = [m for m in messages if m.get("ts") != thread_ts]
                if replies:
                    # Return the first human reply
                    for reply in replies:
                        if not reply.get("bot_id"):
                            audit.log("slack_response_received", channel=channel, thread_ts=thread_ts)
                            return reply.get("text", "")
            except Exception:
                logger.warning("slack_poll_error", channel=channel, thread_ts=thread_ts)

        audit.log("slack_timeout", channel=channel, thread_ts=thread_ts, timeout_s=timeout)
        return None
