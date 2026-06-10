"""
agents/utils/prompt_cache.py

Anthropic prompt caching utilities.

Anthropic's prompt caching lets you mark content blocks with cache_control
so repeated identical prefixes are served from cache at ~10% of the input
token cost with 5-minute TTL.

Requirements for caching to activate:
  - Sonnet: minimum 2048 tokens in the cacheable block
  - Haiku:  minimum 1024 tokens in the cacheable block
  - Content must be IDENTICAL across calls (no dynamic content in cached block)

Best practices applied:
  1. Static content (persona, rules, format) → cached block (ephemeral, 5 min TTL)
  2. Dynamic content (RAG, tool results, user query) → uncached block after
  3. Cache the SYSTEM prompt prefix, not user messages
  4. Use cache_control: {"type": "ephemeral"} on the last static block
  5. Monitor cache hit rate via response.usage.cache_read_input_tokens

References:
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
"""

from __future__ import annotations

from typing import Any

from agents.prompts.prompt_config import ENABLE_PROMPT_CACHING


def make_cached_system_blocks(
    static_prefix: str,
    dynamic_suffix: str = "",
) -> list[dict[str, Any]]:
    """
    Builds the system prompt as a list of content blocks for the Anthropic API.

    The static prefix gets cache_control: ephemeral.
    The dynamic suffix (RAG context, tool results) is a plain text block.

    Args:
        static_prefix: The unchanging persona/rules portion of the system prompt
        dynamic_suffix: Per-request dynamic context (RAG + tool results)

    Returns:
        List of content block dicts for the `system` parameter

    Example (direct Anthropic API usage):
        client.messages.create(
            model="claude-sonnet-4-20250514",
            system=make_cached_system_blocks(static_text, dynamic_text),
            messages=[...],
        )
    """
    if not ENABLE_PROMPT_CACHING:
        # Flat string fallback for environments without caching support
        full = static_prefix + ("\n\n" + dynamic_suffix if dynamic_suffix else "")
        return [{"type": "text", "text": full}]

    blocks: list[dict[str, Any]] = []

    # Block 1: Static prefix — marked for caching
    # cache_control on the LAST block you want cached (all preceding blocks are also cached)
    blocks.append({
        "type": "text",
        "text": static_prefix,
        "cache_control": {"type": "ephemeral"},
    })

    # Block 2: Dynamic suffix — NOT cached (changes every request)
    if dynamic_suffix:
        blocks.append({
            "type": "text",
            "text": dynamic_suffix,
        })

    return blocks


def split_static_dynamic(full_system_prompt: str) -> tuple[str, str]:
    """
    Splits a fully assembled system prompt into its static and dynamic halves.

    The split point is the <catalogue_context> tag — everything before it
    is static (persona + rules + task instruction); everything from it onward
    is dynamic (RAG + tool data).

    Args:
        full_system_prompt: The complete system prompt string

    Returns:
        Tuple of (static_portion, dynamic_portion)
    """
    split_markers = ["<catalogue_context>", "<tool_data>"]

    earliest_split = len(full_system_prompt)
    for marker in split_markers:
        pos = full_system_prompt.find(marker)
        if pos != -1:
            earliest_split = min(earliest_split, pos)

    if earliest_split == len(full_system_prompt):
        # No dynamic content — entire prompt is static
        return full_system_prompt, ""

    static = full_system_prompt[:earliest_split].rstrip()
    dynamic = full_system_prompt[earliest_split:]
    return static, dynamic


def log_cache_metrics(usage: dict) -> dict[str, int]:
    """
    Extracts cache hit/miss metrics from Anthropic API usage metadata.

    Args:
        usage: The usage dict from response.usage

    Returns:
        Dict with cache_read_tokens, cache_write_tokens, cache_savings_pct
    """
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_write = usage.get("cache_creation_input_tokens", 0)
    input_tokens = usage.get("input_tokens", 1)

    savings_pct = round(cache_read / max(input_tokens + cache_read, 1) * 100)

    return {
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "uncached_input_tokens": input_tokens,
        "cache_savings_pct": savings_pct,
    }
