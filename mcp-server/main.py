"""
mcp-server/main.py

FastAPI application and MCP tool registry.

Exposes a single POST /tools/call endpoint that the agent uses.
All registered tools are in mcp_server/tools/__init__.py.
"""

from __future__ import annotations

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mcp_server.middleware.auth import verify_caller_token
from mcp_server.middleware.observability import setup_telemetry
from mcp_server.middleware.rate_limit import check_rate_limit
from mcp_server.tools import TOOL_REGISTRY

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="Books AI MCP Server",
    description="Model Context Protocol tool server for the Books AI agent",
    version="1.0.0",
)

setup_telemetry(app)


# ── Request / Response schemas ────────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    """MCP tool call request body."""

    name: str
    arguments: dict


class ToolCallResponse(BaseModel):
    """MCP tool call response. Always returns either result or error."""

    result: dict | None = None
    error: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Liveness probe — returns immediately."""
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness() -> dict:
    """Readiness probe — checks downstream service connectivity."""
    # TODO: add book-service reachability check
    return {"status": "ok"}


@app.get("/tools")
async def list_tools() -> dict:
    """
    Returns the list of available tool definitions.

    The agent reads this on startup to know which tools it can call.
    Each entry is a full MCP tool schema (name, description, input_schema).
    """
    return {
        "tools": [entry["definition"] for entry in TOOL_REGISTRY.values()]
    }


@app.post("/tools/call", response_model=ToolCallResponse)
async def call_tool(
    body: ToolCallRequest,
    request: Request,
    user_id: str = Depends(verify_caller_token),
    _rate_check: None = Depends(check_rate_limit),
) -> ToolCallResponse:
    """
    Dispatches a tool call to the registered handler.

    The user_id is extracted from the caller JWT (not trusted from the body).
    Rate limiting is applied per user_id via Redis.
    """
    tool_name = body.name

    if tool_name not in TOOL_REGISTRY:
        logger.warning("unknown_tool_called", tool=tool_name, user_id=user_id)
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    entry = TOOL_REGISTRY[tool_name]
    handler = entry["handler"]
    schema_class = entry["schema"]

    logger.info("tool_call_received", tool=tool_name, user_id=user_id)

    try:
        # Validate args against the tool's Pydantic schema
        validated_args = schema_class(**body.arguments)
        result = await handler(validated_args, user_id=user_id)
        return ToolCallResponse(result=result.model_dump())

    except Exception as exc:  # noqa: BLE001
        logger.error("tool_call_failed", tool=tool_name, user_id=user_id, error=str(exc))
        return ToolCallResponse(error=str(exc))
