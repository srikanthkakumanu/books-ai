"""
agents/prompts/__init__.py

Public API for the prompts package.
Import from here — not from individual modules — to keep imports stable.
"""

from agents.prompts.intent import build_intent_prompt
from agents.prompts.prompt_config import (
    CLASSIFY_MAX_TOKENS,
    CLASSIFY_MODEL,
    CLASSIFY_TEMPERATURE,
    ENABLE_PROMPT_CACHING,
    MESSAGE_HISTORY_MAX_TOKENS,
    MESSAGE_HISTORY_MAX_TURNS,
    RAG_CONTEXT_MAX_TOKENS,
    RESPOND_MODEL,
    RESPONSE_BUDGETS,
    ResponseBudget,
    SYSTEM_PROMPT_TARGET_TOKENS,
    TOOL_RESULTS_MAX_TOKENS,
)
from agents.prompts.response import (
    build_response_prompt,
    get_response_budget,
)

__all__ = [
    "build_intent_prompt",
    "build_response_prompt",
    "get_response_budget",
    "CLASSIFY_MODEL",
    "CLASSIFY_MAX_TOKENS",
    "CLASSIFY_TEMPERATURE",
    "RESPOND_MODEL",
    "RESPONSE_BUDGETS",
    "ResponseBudget",
    "SYSTEM_PROMPT_TARGET_TOKENS",
    "RAG_CONTEXT_MAX_TOKENS",
    "TOOL_RESULTS_MAX_TOKENS",
    "MESSAGE_HISTORY_MAX_TURNS",
    "MESSAGE_HISTORY_MAX_TOKENS",
    "ENABLE_PROMPT_CACHING",
]
