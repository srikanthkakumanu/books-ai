"""
agents/tools/mcp_client.py

Typed HTTP client for the MCP server.

This is the ONLY place in the agent layer that communicates with the MCP server.
All tool calls go through here. Handles:
- Retry with exponential backoff
- OpenTelemetry span creation
- Timeout enforcement
- Structured logging
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog
from opentelemetry import trace

from agents.config import settings

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

_RETRY_DELAYS = [0.5, 1.0, 2.0]  # seconds between retries
_DEFAULT_TIMEOUT = 5.0
_SLOW_TOOL_TIMEOUT = 10.0  # recommendations take longer

_SLOW_TOOLS = {"get_recommendations"}


class MCPClient:
    """
    Async HTTP client for the MCP tool server.

    Usage:
        mcp = MCPClient()
        result = await mcp.call("search_books", {"query": "Dune"}, user_id="uuid")
    """

    def __init__(self) -> None:
        self._base_url = settings.MCP_SERVER_URL
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Content-Type": "application/json"},
        )

    async def call(
        self,
        tool_name: str,
        args: dict[str, Any],
        user_id: str,
    ) -> dict[str, Any]:
        """
        Calls an MCP tool by name with the given arguments.

        Args:
            tool_name: Registered MCP tool name (e.g. "search_books")
            args: Tool input arguments (validated against tool schema by MCP server)
            user_id: Authenticated user ID (passed as header, not in body)

        Returns:
            Tool response as a dict

        Raises:
            MCPToolError: If the tool returns an error after all retries exhausted
        """
        timeout = _SLOW_TOOL_TIMEOUT if tool_name in _SLOW_TOOLS else _DEFAULT_TIMEOUT
        payload = {"name": tool_name, "arguments": args}
        headers = {"X-User-ID": user_id}

        with tracer.start_as_current_span(f"mcp.{tool_name}") as span:
            span.set_attribute("mcp.tool", tool_name)
            span.set_attribute("mcp.user_id", user_id)

            for attempt, delay in enumerate([0.0, *_RETRY_DELAYS], start=1):
                if delay:
                    await asyncio.sleep(delay)

                try:
                    response = await self._client.post(
                        "/tools/call",
                        json=payload,
                        headers=headers,
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    data = response.json()

                    # MCP tools return structured errors in the body, not HTTP 4xx
                    if error := data.get("error"):
                        raise MCPToolError(tool_name, error)

                    logger.info(
                        "mcp_tool_success",
                        tool=tool_name,
                        user_id=user_id,
                        attempt=attempt,
                    )
                    return data

                except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
                    if attempt > len(_RETRY_DELAYS):
                        logger.error(
                            "mcp_tool_exhausted",
                            tool=tool_name,
                            user_id=user_id,
                            error=str(exc),
                        )
                        raise MCPToolError(tool_name, str(exc)) from exc

                    logger.warning(
                        "mcp_tool_retry",
                        tool=tool_name,
                        attempt=attempt,
                        error=str(exc),
                    )

        raise MCPToolError(tool_name, "Unexpected exit from retry loop")

    async def aclose(self) -> None:
        """Closes the underlying HTTP client. Call on app shutdown."""
        await self._client.aclose()


class MCPToolError(Exception):
    """Raised when an MCP tool call fails after all retries."""

    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"MCP tool '{tool_name}' failed: {reason}")
