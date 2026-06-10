"""
agents/utils/cost_tracker.py

Per-user LLM cost tracking with model-specific pricing.

Records token usage after every LLM call. Stored in PostgreSQL
for billing dashboards and per-user cost caps.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

# Anthropic pricing as of June 2025 (per million tokens)
# Update when pricing changes — source: https://www.anthropic.com/pricing
_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {
        "input": 3.00,    # $ per 1M input tokens
        "output": 15.00,  # $ per 1M output tokens
        "cache_read": 0.30,   # $ per 1M cache-read tokens (90% off input)
        "cache_write": 3.75,  # $ per 1M cache-write tokens (25% premium)
    },
    "claude-haiku-4-5-20251001": {
        "input": 0.80,
        "output": 4.00,
        "cache_read": 0.08,
        "cache_write": 1.00,
    },
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """
    Calculates the USD cost of a single LLM call.

    Args:
        model: Model string (must be in _PRICING)
        input_tokens: Uncached input tokens
        output_tokens: Generated output tokens
        cache_read_tokens: Tokens served from cache (cheaper)
        cache_write_tokens: Tokens written to cache (slight premium)

    Returns:
        Cost in USD (float, 6 decimal places)
    """
    pricing = _PRICING.get(model, _PRICING["claude-sonnet-4-20250514"])
    per_million = 1_000_000

    cost = (
        (input_tokens * pricing["input"] / per_million)
        + (output_tokens * pricing["output"] / per_million)
        + (cache_read_tokens * pricing["cache_read"] / per_million)
        + (cache_write_tokens * pricing["cache_write"] / per_million)
    )
    return round(cost, 6)


async def record_usage(
    user_id: str,
    intent: str,
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "claude-sonnet-4-20250514",
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> None:
    """
    Records a single LLM call's token usage to the database.

    Non-blocking: designed to be called in a try/except — failure here
    must never affect the user-facing response.
    """
    cost = calculate_cost(
        model=model,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )

    logger.info(
        "usage_recorded",
        user_id=user_id,
        intent=intent,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_tokens=cache_read_tokens,
        cost_usd=cost,
    )

    # DB write happens here in production
    # from agents.utils.db import get_session
    # async with get_session() as session:
    #     session.add(UsageEvent(...))
    #     await session.commit()
