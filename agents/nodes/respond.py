"""
agents/nodes/respond.py

Final response generation node — with prompt caching and prefill.

Prompt engineering improvements over v1:
  1. Prompt caching: static persona block marked cache_control: ephemeral
  2. Assistant prefill: steer response format without consuming output tokens
  3. Per-intent max_tokens from ResponseBudget (not one-size-fits-all 1024)
  4. Message history trimming before LLM call (sliding window, token-budgeted)
  5. Cache metrics logged for monitoring cache hit rate
  6. Output guardrail validates response before returning
  7. Direct Anthropic SDK (not LangChain wrapper) for full API feature access

Token savings vs v1:
  - Prompt caching: ~90% off cached input tokens (persona = ~400 tokens)
  - Prefill: saves 5–15 tokens per response (skips preamble)
  - History trimming: prevents 8k+ token history accumulation
  - Per-intent budgets: shelf responses use 200 tokens, not 1024
"""

from __future__ import annotations

import structlog
from anthropic import AsyncAnthropic
from langchain_core.messages import AIMessage

from agents.prompts.response import (
    build_response_prompt,
    get_response_budget,
)
from agents.prompts.prompt_cache import (
    log_cache_metrics,
    make_cached_system_blocks,
    split_static_dynamic,
)
from agents.state import AgentState
from agents.utils.cost_tracker import record_usage
from agents.utils.guardrails import validate_response
from agents.utils.token_utils import audit_prompt_tokens, trim_message_history

logger = structlog.get_logger(__name__)

_client = AsyncAnthropic()


def _messages_to_api_format(messages) -> list[dict]:
    """Converts LangChain messages to Anthropic API message dicts."""
    role_map = {"human": "user", "ai": "assistant"}
    result = []
    for m in messages:
        role = role_map.get(m.type, "user")
        result.append({"role": role, "content": str(m.content)})
    return result


async def generate_response(state: AgentState) -> dict:
    """
    Generates the final user-facing response with all optimisations applied.

    Pipeline:
      1. Trim message history to token budget
      2. Build system prompt (static + dynamic sections)
      3. Split for prompt caching (static → cached block, dynamic → uncached)
      4. Get per-intent token budget and prefill
      5. Call Claude via Anthropic SDK with streaming
      6. Validate output via guardrails
      7. Log cost and cache metrics

    Returns:
        Partial state update: {"messages": [AIMessage], "response_tokens": int}
    """
    intent = state["intent"]
    budget = get_response_budget(intent)

    # ── 1. Trim history to token budget ────────────────────────────────────
    trimmed_messages = trim_message_history(state["messages"])

    # ── 2. Build full system prompt ─────────────────────────────────────────
    full_system = build_response_prompt(
        rag_context=state.get("rag_context", ""),
        tool_results=state.get("tool_results", {}),
        intent=intent,
    )

    # ── 3. Split for prompt caching ─────────────────────────────────────────
    static_prefix, dynamic_suffix = split_static_dynamic(full_system)
    system_blocks = make_cached_system_blocks(static_prefix, dynamic_suffix)

    # ── 4. Format messages ─────────────────────────────────────────────────
    api_messages = _messages_to_api_format(trimmed_messages)

    # Prefill: inject a partial assistant response to steer format
    # Claude will continue from this exact text — it saves ~10 tokens per call
    # and eliminates hollow openers ("Great question! I'd be happy to...")
    if budget.prefill:
        api_messages.append({"role": "assistant", "content": budget.prefill})

    # ── 5. Audit token usage (dev/debug logging) ───────────────────────────
    token_audit = audit_prompt_tokens(full_system, trimmed_messages)
    logger.debug(
        "prompt_token_audit",
        user_id=state["user_id"],
        intent=intent,
        **token_audit,
    )

    logger.info(
        "generating_response",
        user_id=state["user_id"],
        intent=intent,
        has_rag=bool(state.get("rag_context")),
        has_tools=bool(state.get("tool_results")),
        max_tokens=budget.max_tokens,
    )

    # ── 6. LLM call ────────────────────────────────────────────────────────
    from agents.prompts.prompt_config import RESPOND_MODEL
    response = await _client.messages.create(
        model=RESPOND_MODEL,
        max_tokens=budget.max_tokens,
        temperature=budget.temperature,
        system=system_blocks,
        messages=api_messages,
    )

    response_text = response.content[0].text

    # Prepend prefill text to the response (it was injected but not in response body)
    if budget.prefill:
        response_text = budget.prefill + response_text

    # ── 7. Output guardrail ─────────────────────────────────────────────────
    guard = validate_response(response_text, intent)
    if not guard.allowed:
        response_text = guard.reason or "I wasn't able to generate a response. Please try again."

    # ── 8. Log metrics ─────────────────────────────────────────────────────
    usage = response.usage
    cache_metrics = log_cache_metrics({
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "input_tokens": usage.input_tokens,
    })

    logger.info(
        "response_generated",
        user_id=state["user_id"],
        intent=intent,
        output_tokens=usage.output_tokens,
        **cache_metrics,
    )

    # Record usage for cost tracking (non-blocking)
    try:
        await record_usage(
            user_id=state["user_id"],
            intent=intent,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("cost_tracking_failed", error=str(exc))

    return {
        "messages": [AIMessage(content=response_text)],
        "response_tokens": usage.output_tokens,
    }
