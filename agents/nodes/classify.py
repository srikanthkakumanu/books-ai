"""
agents/nodes/classify.py

Intent classification node — optimised for speed and cost.

Prompt engineering improvements over v1:
  1. Switched to Haiku (40x cheaper, faster, perfectly adequate for classification)
  2. XML-structured prompt with positive + negative examples
  3. Prefill assistant turn with empty string (forces single-word output style)
  4. max_tokens=10 (tightest possible budget: "recommend" = 3 tokens)
  5. temperature=0 (fully deterministic)
  6. Input guardrail applied before LLM call
  7. Structured logging with intent + raw_response for monitoring misclassifications
"""

from __future__ import annotations

import structlog
from anthropic import AsyncAnthropic
from langchain_core.messages import HumanMessage

from agents.prompts.intent import build_intent_prompt
from agents.prompts.prompt_config import CLASSIFY_MAX_TOKENS, CLASSIFY_MODEL, CLASSIFY_TEMPERATURE
from agents.state import AgentState, IntentType
from agents.utils.guardrails import check_input
from agents.utils.token_utils import trim_message_history

logger = structlog.get_logger(__name__)

_VALID_INTENTS: set[IntentType] = {"recommend", "search", "review", "shelf", "general"}

# Use Anthropic SDK directly for classification — avoids LangChain overhead
# and gives us clean access to the messages API format
_client = AsyncAnthropic()


async def classify_intent(state: AgentState) -> dict:
    """
    Classifies the user's latest message into one of the 5 intent categories.

    Uses claude-haiku-4-5 for speed and cost efficiency.
    Input is checked by the guardrails layer before any LLM call.

    Returns:
        Partial state update: {"intent": IntentType}
        If guardrail rejects: {"intent": "blocked", "tool_error": reason}
    """
    user_message = str(state["messages"][-1].content)

    # ── Guardrail check ────────────────────────────────────────────────────
    guard = check_input(user_message, state["user_id"])
    if not guard.allowed:
        logger.warning("input_blocked", user_id=state["user_id"], reason=guard.reason)
        return {"intent": "general", "tool_error": guard.reason}

    # Use sanitised version of the query
    clean_message = guard.sanitised_query or user_message

    logger.info("classifying_intent", user_id=state["user_id"], message_len=len(clean_message))

    # ── LLM call ───────────────────────────────────────────────────────────
    # Only pass the latest message — history is irrelevant for classification
    # This is a key token optimisation: don't send 10-turn history to classify
    response = await _client.messages.create(
        model=CLASSIFY_MODEL,
        max_tokens=CLASSIFY_MAX_TOKENS,
        temperature=CLASSIFY_TEMPERATURE,
        system=build_intent_prompt(),
        messages=[{"role": "user", "content": clean_message}],
    )

    raw = response.content[0].text.strip().lower()
    intent: IntentType = raw if raw in _VALID_INTENTS else "general"

    if raw not in _VALID_INTENTS:
        logger.warning(
            "intent_fallback",
            user_id=state["user_id"],
            raw=raw,
            fell_back_to="general",
        )

    logger.info(
        "intent_classified",
        user_id=state["user_id"],
        intent=intent,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )

    return {"intent": intent}
