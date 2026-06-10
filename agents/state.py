"""
agents/state.py

Single source of truth for the LangGraph agent state.

Rules:
- Never add ad-hoc keys to a node return dict — extend AgentState here first.
- messages uses operator.add — nodes append rather than overwrite.
- All fields have explicit types — no Any.
- Add new fields at the END of their section to keep diffs clean.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage


IntentType = Literal["recommend", "search", "review", "shelf", "general"]


class AgentState(TypedDict):
    """State passed between all LangGraph nodes."""

    # ── Immutable inputs (set at entry, never mutated) ────────────────────
    user_id: str
    session_id: str

    # ── Message history (append-only via operator.add) ────────────────────
    messages: Annotated[list[BaseMessage], operator.add]

    # ── Populated by classify node ────────────────────────────────────────
    intent: IntentType
    # Sanitised version of the latest user query (after guardrail processing)
    sanitised_query: str

    # ── Populated by retrieve node ────────────────────────────────────────
    rag_context: str        # Formatted XML string injected into the final prompt
    rag_sources: list[str]  # Book IDs used as RAG context (for attribution)
    rag_token_count: int    # Estimated tokens of rag_context (for budget monitoring)

    # ── Populated by tool_dispatch node ──────────────────────────────────
    tool_results: dict      # Cleaned JSON response from MCP tool
    tool_error: str | None  # Set if tool call failed; triggers error_handler node

    # ── Populated by respond node ─────────────────────────────────────────
    response_tokens: int        # Output tokens (for cost tracking)
    cache_read_tokens: int      # Cache-hit input tokens (for cost tracking)
    cache_write_tokens: int     # Cache-miss input tokens written to cache
