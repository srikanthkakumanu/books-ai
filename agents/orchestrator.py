"""
agents/orchestrator.py

LangGraph StateGraph definition for the Books AI agent.

Graph topology:
    Entry → classify → retrieve → tool_dispatch → respond → END
                                        ↓
                                  error_handler (on tool_error)
                                        ↓
                                     respond

Usage:
    from agents.orchestrator import graph

    result = await graph.ainvoke({
        "user_id": "uuid",
        "session_id": "uuid",
        "messages": [HumanMessage(content="Recommend a book like Dune")],
        "intent": "general",
        "rag_context": "",
        "rag_sources": [],
        "tool_results": {},
        "tool_error": None,
        "response_tokens": 0,
    })
"""

from __future__ import annotations

import structlog
from langgraph.graph import END, StateGraph

from agents.nodes.classify import classify_intent
from agents.nodes.error_handler import handle_tool_error
from agents.nodes.respond import generate_response
from agents.nodes.retrieve import retrieve_rag_context
from agents.nodes.tool_dispatch import dispatch_tool
from agents.state import AgentState

logger = structlog.get_logger(__name__)


def _route_after_tool_dispatch(state: AgentState) -> str:
    """
    Conditional edge: if a tool error occurred, route to error handler.
    Otherwise proceed to the response node.
    """
    if state.get("tool_error"):
        logger.warning(
            "tool_dispatch_error",
            user_id=state["user_id"],
            error=state["tool_error"],
        )
        return "error_handler"
    return "respond"


def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph StateGraph.

    Separated from module-level instantiation to make testing easier
    (tests can call build_graph() with patched nodes).
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("classify", classify_intent)
    builder.add_node("retrieve", retrieve_rag_context)
    builder.add_node("tool_dispatch", dispatch_tool)
    builder.add_node("respond", generate_response)
    builder.add_node("error_handler", handle_tool_error)

    # ── Wire edges ────────────────────────────────────────────────────────
    builder.set_entry_point("classify")
    builder.add_edge("classify", "retrieve")
    builder.add_edge("retrieve", "tool_dispatch")
    builder.add_conditional_edges(
        "tool_dispatch",
        _route_after_tool_dispatch,
        {"respond": "respond", "error_handler": "error_handler"},
    )
    builder.add_edge("error_handler", "respond")
    builder.add_edge("respond", END)

    return builder.compile()


# Module-level compiled graph — import this in main.py and tests
graph = build_graph()
