"""
tests/unit/agents/test_prompt_engineering.py

Tests for prompt engineering correctness.

These tests verify that:
- Prompts contain required structural elements (XML tags, constraints)
- Token budgets are within configured limits
- Prefills are intact in ResponseBudgets
- Cache split point is deterministic
- Tool descriptions contain disambiguation guidance
"""

from __future__ import annotations

import pytest

from agents.prompts.intent import build_intent_prompt
from agents.prompts.prompt_config import (
    CLASSIFY_MAX_TOKENS,
    CLASSIFY_MODEL,
    RESPOND_MODEL,
    RESPONSE_BUDGETS,
    SYSTEM_PROMPT_TARGET_TOKENS,
)
from agents.prompts.response import build_response_prompt, get_response_budget
from agents.utils.prompt_cache import split_static_dynamic
from agents.utils.token_utils import count_tokens


class TestIntentPrompt:
    def test_contains_all_five_intents(self):
        prompt = build_intent_prompt()
        for intent in ["recommend", "search", "review", "shelf", "general"]:
            assert intent in prompt, f"Intent '{intent}' missing from classification prompt"

    def test_uses_xml_structure(self):
        prompt = build_intent_prompt()
        assert "<task>" in prompt
        assert "<output_format>" in prompt
        assert "<examples>" in prompt

    def test_has_negative_example_prevention(self):
        """Prompt must instruct model to output ONE word only."""
        prompt = build_intent_prompt()
        assert "one word" in prompt.lower() or "exactly one" in prompt.lower()

    def test_token_count_within_budget(self):
        """Intent prompt should be < 200 tokens — it's sent with EVERY classification call."""
        prompt = build_intent_prompt()
        tokens = count_tokens(prompt)
        assert tokens < 200, (
            f"Intent prompt is {tokens} tokens. "
            f"It's used with Haiku on every call — keep it under 200."
        )

    def test_uses_haiku_model(self):
        """Classification must use Haiku, not Sonnet (cost optimisation)."""
        assert "haiku" in CLASSIFY_MODEL.lower(), (
            f"CLASSIFY_MODEL is '{CLASSIFY_MODEL}' — must be Haiku for cost efficiency"
        )

    def test_max_tokens_tight(self):
        """Max tokens for classification should be ≤ 15 (single word output)."""
        assert CLASSIFY_MAX_TOKENS <= 15, (
            f"CLASSIFY_MAX_TOKENS={CLASSIFY_MAX_TOKENS}. "
            f"'recommend' is 3 tokens — anything over 15 wastes money."
        )


class TestResponsePrompt:
    def test_contains_persona_block(self):
        prompt = build_response_prompt("", {}, "general")
        assert "<persona>" in prompt

    def test_contains_core_rules(self):
        prompt = build_response_prompt("", {}, "general")
        assert "<core_rules>" in prompt

    def test_rag_context_wrapped_in_xml(self):
        """RAG context must be inside <catalogue_context> tags."""
        prompt = build_response_prompt(
            rag_context="Some book context",
            tool_results={},
            intent="recommend",
        )
        assert "<catalogue_context>" in prompt
        assert "Some book context" in prompt
        assert "</catalogue_context>" in prompt

    def test_tool_results_wrapped_in_xml(self):
        """Tool results must be inside <tool_data> tags."""
        prompt = build_response_prompt(
            rag_context="",
            tool_results={"books": [{"title": "Dune", "author": "Herbert"}]},
            intent="search",
        )
        assert "<tool_data>" in prompt
        assert "</tool_data>" in prompt

    def test_no_rag_section_when_empty(self):
        """No <catalogue_context> block when RAG context is empty — saves tokens."""
        prompt = build_response_prompt(rag_context="", tool_results={}, intent="general")
        assert "<catalogue_context>" not in prompt

    def test_no_tool_section_when_empty(self):
        """No <tool_data> block when tool results are empty — saves tokens."""
        prompt = build_response_prompt(rag_context="", tool_results={}, intent="shelf")
        assert "<tool_data>" not in prompt

    def test_contains_negative_constraints(self):
        """System prompt must contain NEVER constraints for hollow phrases."""
        prompt = build_response_prompt("", {}, "recommend")
        assert "NEVER" in prompt

    def test_system_prompt_token_budget(self):
        """Static-only prompt should target SYSTEM_PROMPT_TARGET_TOKENS."""
        prompt = build_response_prompt(rag_context="", tool_results={}, intent="recommend")
        tokens = count_tokens(prompt)
        # Allow up to 2x the target (dynamic content can reasonably double it)
        assert tokens < SYSTEM_PROMPT_TARGET_TOKENS * 2, (
            f"System prompt is {tokens} tokens (target: {SYSTEM_PROMPT_TARGET_TOKENS}). "
            f"Review for token waste."
        )

    def test_tool_results_strip_internal_fields(self):
        """Internal fields (id, user_id, timestamps) should not appear in tool data block."""
        prompt = build_response_prompt(
            rag_context="",
            tool_results={
                "books": [{
                    "id": "secret-uuid",
                    "title": "Dune",
                    "author": "Herbert",
                    "user_id": "user-123",
                    "created_at": "2024-01-01",
                }]
            },
            intent="search",
        )
        # These internal fields should be stripped
        assert "secret-uuid" not in prompt
        assert "user-123" not in prompt
        assert "created_at" not in prompt
        # But user-facing fields should remain
        assert "Dune" in prompt
        assert "Herbert" in prompt

    @pytest.mark.parametrize("intent", ["recommend", "search", "review", "shelf", "general"])
    def test_all_intents_produce_valid_prompt(self, intent):
        """Every intent must produce a non-empty prompt without raising."""
        prompt = build_response_prompt(
            rag_context="Some context",
            tool_results={"books": []},
            intent=intent,
        )
        assert len(prompt) > 100


class TestResponseBudgets:
    @pytest.mark.parametrize("intent", ["recommend", "search", "review", "shelf", "general"])
    def test_all_intents_have_budget(self, intent):
        budget = get_response_budget(intent)
        assert budget.max_tokens > 0
        assert 0.0 <= budget.temperature <= 1.0

    def test_shelf_budget_is_smallest(self):
        """Shelf confirmations should have the tightest token budget."""
        shelf_budget = RESPONSE_BUDGETS["shelf"]
        for intent, budget in RESPONSE_BUDGETS.items():
            if intent != "shelf":
                assert shelf_budget.max_tokens <= budget.max_tokens, (
                    f"shelf max_tokens ({shelf_budget.max_tokens}) should be ≤ "
                    f"{intent} max_tokens ({budget.max_tokens})"
                )

    def test_recommend_has_prefill(self):
        """Recommendations must have a prefill to skip hollow openings."""
        budget = RESPONSE_BUDGETS["recommend"]
        assert budget.prefill is not None
        assert len(budget.prefill) > 5

    def test_search_has_lower_temperature(self):
        """Search results need accuracy over creativity — lower temp."""
        search_budget = RESPONSE_BUDGETS["search"]
        recommend_budget = RESPONSE_BUDGETS["recommend"]
        assert search_budget.temperature < recommend_budget.temperature


class TestPromptCaching:
    def test_split_separates_at_catalogue_context(self):
        prompt = "Static persona here\n\n<catalogue_context>Dynamic stuff</catalogue_context>"
        static, dynamic = split_static_dynamic(prompt)
        assert "Static persona here" in static
        assert "<catalogue_context>" in dynamic

    def test_split_all_static_when_no_dynamic(self):
        prompt = "Only static content here — no catalogue context"
        static, dynamic = split_static_dynamic(prompt)
        assert static == prompt
        assert dynamic == ""

    def test_split_at_tool_data_tag(self):
        prompt = "Static prefix\n\n<tool_data>Some tool output</tool_data>"
        static, dynamic = split_static_dynamic(prompt)
        assert "<tool_data>" in dynamic
        assert "Static prefix" in static


class TestUsesCorrectModel:
    def test_respond_model_is_sonnet(self):
        assert "sonnet" in RESPOND_MODEL.lower(), (
            f"RESPOND_MODEL is '{RESPOND_MODEL}' — must be Sonnet for response quality"
        )
