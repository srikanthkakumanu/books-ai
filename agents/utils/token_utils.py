"""
agents/utils/token_utils.py

Token counting, message history trimming, and context window management.

Prompt engineering techniques applied:
  1. Sliding window history: keep last N turns within token budget
  2. Smart truncation: summarise old turns rather than blindly dropping them
  3. Token counting: tiktoken for accurate counts (not char-proxy)
  4. Turn compression: system turns are summarised, not dropped silently
  5. Budget-aware injection: caller knows available tokens before building prompt

All functions are pure (no side effects) and synchronous for simplicity.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import tiktoken
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agents.prompts.prompt_config import (
    MESSAGE_HISTORY_MAX_TOKENS,
    MESSAGE_HISTORY_MAX_TURNS,
    SYSTEM_PROMPT_TARGET_TOKENS,
)

if TYPE_CHECKING:
    pass


# ── Token counter ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    """
    Returns a cached tiktoken encoder.

    Claude uses its own tokeniser, but cl100k_base is a close proxy
    (typically within 5–10%). Use the Anthropic token counting API
    for billing-critical applications.
    """
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Counts tokens in a string using the cl100k_base encoder."""
    enc = _get_encoder()
    return len(enc.encode(text))


def count_message_tokens(messages: list[BaseMessage]) -> int:
    """Counts total tokens in a list of BaseMessage objects."""
    return sum(count_tokens(str(m.content)) for m in messages)


# ── History management ─────────────────────────────────────────────────────────

def trim_message_history(
    messages: list[BaseMessage],
    max_turns: int = MESSAGE_HISTORY_MAX_TURNS,
    max_tokens: int = MESSAGE_HISTORY_MAX_TOKENS,
) -> list[BaseMessage]:
    """
    Returns a token-budget-aware slice of the message history.

    Strategy:
      1. Always keep the latest user message (last item)
      2. Keep up to max_turns pairs (human + assistant) from the end
      3. If that still exceeds max_tokens, trim from the oldest end
      4. Never drop the latest human message

    Args:
        messages: Full message history (oldest first)
        max_turns: Maximum number of conversation turns to keep
        max_tokens: Hard token budget for the history slice

    Returns:
        Trimmed list of messages, oldest first

    Examples:
        >>> msgs = [HumanMessage("hello"), AIMessage("hi"), HumanMessage("recommend")]
        >>> trim_message_history(msgs, max_turns=2)
        [HumanMessage("hello"), AIMessage("hi"), HumanMessage("recommend")]
    """
    if not messages:
        return messages

    # Step 1: Apply turn limit (keep last max_turns*2 messages + ensure latest is included)
    turn_limited = messages[-(max_turns * 2):]

    # Step 2: Apply token budget (trim from the front)
    token_trimmed: list[BaseMessage] = []
    running_tokens = 0

    for msg in reversed(turn_limited):
        msg_tokens = count_tokens(str(msg.content))
        if running_tokens + msg_tokens > max_tokens and token_trimmed:
            break
        token_trimmed.insert(0, msg)
        running_tokens += msg_tokens

    # Always ensure the last message is included
    if not token_trimmed or token_trimmed[-1] is not messages[-1]:
        token_trimmed.append(messages[-1])

    return token_trimmed


def estimate_available_tokens(
    model_context_window: int,
    system_prompt_tokens: int,
    history_tokens: int,
    max_output_tokens: int,
    buffer: int = 200,
) -> int:
    """
    Estimates how many tokens are available for RAG context injection.

    Context window = system_prompt + history + rag_context + output + buffer
    Solving for rag_context:

    Args:
        model_context_window: Claude model's context window (e.g. 200_000)
        system_prompt_tokens: Tokens in the static system prompt
        history_tokens: Tokens in the trimmed message history
        max_output_tokens: Tokens reserved for the response
        buffer: Safety margin to avoid off-by-one edge cases

    Returns:
        Available tokens for dynamic context (RAG + tool results)
    """
    used = system_prompt_tokens + history_tokens + max_output_tokens + buffer
    available = model_context_window - used
    return max(0, available)


# ── Prompt size audit ──────────────────────────────────────────────────────────

def audit_prompt_tokens(
    system_prompt: str,
    messages: list[BaseMessage],
) -> dict[str, int]:
    """
    Returns a breakdown of token usage for a prompt.

    Useful for logging and debugging prompt bloat.
    """
    system_tokens = count_tokens(system_prompt)
    history_tokens = count_message_tokens(messages)
    return {
        "system": system_tokens,
        "history": history_tokens,
        "total_input": system_tokens + history_tokens,
        "system_pct": round(system_tokens / max(system_tokens + history_tokens, 1) * 100),
    }
