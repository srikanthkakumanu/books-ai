"""
agents/prompts/prompt_config.py

Centralised prompt configuration and token budget constants.

Single source of truth for ALL token limits, model choices, and
temperature settings across every LLM call in the system.

Design principles applied:
  - Separate configuration from prompt text (change limits without touching prompts)
  - Explicit token budgets prevent runaway costs
  - Per-intent max_tokens reflects actual response length needs
  - Claude claude.ai/docs best practices: haiku for classification, sonnet for generation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from agents.state import IntentType


# ── Model selection ───────────────────────────────────────────────────────────
# Use the cheapest model that meets the task's quality bar.
# Classification: Haiku (fast, cheap, deterministic one-word output)
# Generation:     Sonnet (best quality/cost for multi-paragraph responses)

CLASSIFY_MODEL: Final[str] = "claude-haiku-4-5-20251001"   # ~40x cheaper than Sonnet
RESPOND_MODEL:  Final[str] = "claude-sonnet-4-20250514"


# ── Token budgets ─────────────────────────────────────────────────────────────
# Rule: set max_tokens to the MAXIMUM you'd ever accept, not a comfortable default.
# Over-generous budgets waste money. Under-generous budgets truncate responses.

CLASSIFY_MAX_TOKENS: Final[int] = 10    # "recommend" is 9 chars; 10 is safe
CLASSIFY_TEMPERATURE: Final[float] = 0.0  # Fully deterministic

@dataclass(frozen=True)
class ResponseBudget:
    max_tokens: int
    temperature: float
    # Prefill hint: steer the response format before the model starts generating
    # Saves tokens by skipping preamble ("Great question! I'd be happy to...")
    prefill: str | None = None

# Per-intent response budgets — tuned to actual content length needs
RESPONSE_BUDGETS: dict[IntentType, ResponseBudget] = {
    "recommend": ResponseBudget(
        max_tokens=600,
        temperature=0.7,
        prefill="Here are some books you'll love based on",
    ),
    "search":    ResponseBudget(
        max_tokens=400,
        temperature=0.3,  # Lower: search results need accuracy
        prefill="I found these books matching your search:",
    ),
    "review":    ResponseBudget(
        max_tokens=500,
        temperature=0.6,
        prefill=None,  # Reviews need natural opening
    ),
    "shelf":     ResponseBudget(
        max_tokens=200,
        temperature=0.3,  # Confirmations are short and factual
        prefill="Done! I've",
    ),
    "general":   ResponseBudget(
        max_tokens=350,
        temperature=0.5,
        prefill=None,
    ),
}


# ── Context window budgets ────────────────────────────────────────────────────
# Claude Sonnet 4 has a 200k context window.
# We budget explicitly to keep costs predictable.

SYSTEM_PROMPT_TARGET_TOKENS: Final[int] = 400   # Tight — system prompts are reused every call
RAG_CONTEXT_MAX_TOKENS:      Final[int] = 800   # ~5 book summaries at 160 tokens each
TOOL_RESULTS_MAX_TOKENS:     Final[int] = 300   # Trim tool JSON to this budget
MESSAGE_HISTORY_MAX_TURNS:   Final[int] = 10    # Sliding window: keep last N turns
MESSAGE_HISTORY_MAX_TOKENS:  Final[int] = 2000  # Hard cap on history tokens

# RAG context: per-book summary token budget (affects formatter truncation)
RAG_BOOK_SUMMARY_MAX_CHARS: Final[int] = 280  # ~70 tokens per book entry


# ── Prompt caching ────────────────────────────────────────────────────────────
# Anthropic prompt caching: prepend cache_control to static content blocks.
# System prompts and RAG context that repeats across turns should be cached.
# Min cacheable block: 1024 tokens (Haiku) / 2048 tokens (Sonnet).
# Cache TTL: 5 minutes. Savings: ~90% on cached input tokens.

ENABLE_PROMPT_CACHING: Final[bool] = True
