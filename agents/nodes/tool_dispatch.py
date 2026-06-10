"""
agents/nodes/tool_dispatch.py

MCP tool dispatch node.

Routes to the correct MCP tool based on the classified intent,
executes the tool call, and stores results in state.
On failure, sets tool_error so the conditional edge routes to error_handler.
"""

from __future__ import annotations

import structlog

from agents.state import AgentState
from agents.tools.mcp_client import MCPClient

logger = structlog.get_logger(__name__)

_mcp = MCPClient()

# Maps intent → MCP tool name
_INTENT_TO_TOOL: dict[str, str] = {
    "recommend": "get_recommendations",
    "search": "search_books",
    "review": "write_review",
    "shelf": "add_to_shelf",
}


async def dispatch_tool(state: AgentState) -> dict:
    """
    Calls the MCP tool matching the current intent.

    Returns:
        Partial state update: {"tool_results": dict} or {"tool_error": str}
    """
    intent = state["intent"]
    tool_name = _INTENT_TO_TOOL.get(intent)

    if not tool_name:
        # general intent — no tool call needed
        return {"tool_results": {}, "tool_error": None}

    logger.info(
        "dispatching_tool",
        user_id=state["user_id"],
        intent=intent,
        tool=tool_name,
    )

    try:
        tool_args = _build_tool_args(state, tool_name)
        result = await _mcp.call(
            tool_name=tool_name,
            args=tool_args,
            user_id=state["user_id"],
        )
        return {"tool_results": result, "tool_error": None}

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "tool_dispatch_failed",
            user_id=state["user_id"],
            tool=tool_name,
            error=str(exc),
        )
        return {"tool_results": {}, "tool_error": str(exc)}


def _build_tool_args(state: AgentState, tool_name: str) -> dict:
    """
    Constructs tool arguments from agent state.

    Each tool has different required args — extracted from the message
    content or from RAG sources.
    """
    user_message = state["messages"][-1].content

    if tool_name == "get_recommendations":
        return {
            "user_id": state["user_id"],
            "seed_book_ids": state.get("rag_sources", [])[:3],
            "query": user_message,
            "limit": 5,
        }

    if tool_name == "search_books":
        return {
            "query": user_message,
            "limit": 10,
        }

    if tool_name == "write_review":
        return {
            "user_id": state["user_id"],
            "query": user_message,
        }

    if tool_name == "add_to_shelf":
        return {
            "user_id": state["user_id"],
            "query": user_message,
        }

    return {"query": user_message}
