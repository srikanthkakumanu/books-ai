"""
tests/unit/agents/test_classify_node.py

Unit tests for the intent classification node.

The LLM is mocked — we test the node's logic, not Claude's responses.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from unittest.mock import AsyncMock, patch

from agents.nodes.classify import classify_intent
from agents.state import AgentState


def make_state(message: str) -> AgentState:
    """Helper: builds a minimal AgentState for testing."""
    return {
        "user_id": "test-user-123",
        "session_id": "test-session-456",
        "messages": [HumanMessage(content=message)],
        "intent": "general",
        "rag_context": "",
        "rag_sources": [],
        "tool_results": {},
        "tool_error": None,
        "response_tokens": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llm_raw_output, expected_intent",
    [
        ("recommend", "recommend"),
        ("RECOMMEND", "recommend"),      # case-insensitive
        ("search", "search"),
        ("review", "review"),
        ("shelf", "shelf"),
        ("general", "general"),
        ("unknown_garbage", "general"),  # fallback for invalid output
        ("", "general"),                 # empty fallback
    ],
)
async def test_classify_intent_maps_llm_output_to_valid_intent(
    llm_raw_output: str,
    expected_intent: str,
) -> None:
    """
    classify_intent should map LLM output to a valid IntentType.
    Invalid or unexpected values fall back to 'general'.
    """
    mock_llm_response = AIMessage(content=llm_raw_output)

    with patch("agents.nodes.classify._llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        state = make_state("Recommend me something like Dune")
        result = await classify_intent(state)

    assert result["intent"] == expected_intent


@pytest.mark.asyncio
async def test_classify_intent_passes_full_message_history_to_llm() -> None:
    """
    The LLM should receive the full message history, not just the latest message.
    This ensures context is preserved across multi-turn conversations.
    """
    mock_response = AIMessage(content="recommend")

    with patch("agents.nodes.classify._llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        state = make_state("Something like that")
        state["messages"] = [
            HumanMessage(content="I loved Dune"),
            AIMessage(content="Great choice! What kind of book are you in the mood for?"),
            HumanMessage(content="Something like that"),
        ]

        await classify_intent(state)

    call_args = mock_llm.ainvoke.call_args[0][0]
    # First arg is system message, followed by all user messages
    message_contents = [m.content for m in call_args[1:]]
    assert "I loved Dune" in message_contents


@pytest.mark.asyncio
async def test_classify_intent_returns_only_intent_key() -> None:
    """
    Node should return a partial state update with only the 'intent' key.
    It should not modify other state fields.
    """
    with patch("agents.nodes.classify._llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="search"))

        state = make_state("Find me books about machine learning")
        result = await classify_intent(state)

    assert set(result.keys()) == {"intent"}
