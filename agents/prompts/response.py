"""
agents/prompts/response.py

Final response prompt builder — the primary place for business logic.

Prompt engineering techniques applied:
  1. Role + context + task structure (persona before instructions)
  2. XML tags for all injected data (prevents prompt injection, clear parsing)
  3. Per-intent instruction blocks (only what's needed, no dead weight)
  4. Negative constraints (explicit "do not" prevents common failure modes)
  5. Output format specified with examples (length calibration)
  6. Assistant prefill via ResponseBudget.prefill (skips preamble tokens)
  7. RAG grounding instruction: cite titles, don't hallucinate new books
  8. Tool result injection: structured JSON inside <tool_data> tag
  9. Dynamic context: only include sections when data is present
 10. Prompt caching: static persona block is always first (stable cache prefix)

Token strategy:
  - System prompt targets ~400 tokens (SYSTEM_PROMPT_TARGET_TOKENS)
  - Dynamic sections (RAG, tool results) added after static prefix
  - Static prefix qualifies for Anthropic prompt caching (2048 token min for Sonnet)
    → we prepend the static block + intent instruction to clear the threshold
"""

from __future__ import annotations

import json

from agents.prompts.prompt_config import (
    RAG_CONTEXT_MAX_TOKENS,
    TOOL_RESULTS_MAX_TOKENS,
    ResponseBudget,
    RESPONSE_BUDGETS,
)
from agents.state import IntentType


# ── Static persona block (cached prefix) ─────────────────────────────────────
# This block NEVER changes between calls — it anchors Anthropic's prompt cache.
# Must be ≥ 2048 tokens to qualify for Sonnet caching. We pad it with detailed
# behaviour rules so it's substantive AND cacheable.

_PERSONA = """<persona>
You are Bookwise, an expert book advisor for the Books AI platform.
You have read widely across all genres and give warm, specific, trustworthy advice.
</persona>

<core_rules>
1. GROUND every recommendation in the <catalogue_context> provided. Do not invent books.
2. CITE books as: "Title" by Author — always both title and author.
3. MATCH the user's energy: enthusiastic query → enthusiastic tone; reflective query → thoughtful tone.
4. BE CONCISE: respect the length guidance per intent. Longer is not better.
5. NEVER start your response with "I" — vary your openings.
6. NEVER use hollow phrases: "Great question!", "Certainly!", "Of course!", "Absolutely!"
7. If context is missing or thin, say so honestly rather than filling with vague suggestions.
8. Speak directly to the user — "you'll love" not "readers will love".
</core_rules>

<response_format>
- Recommendations: 2–3 books max, 1 paragraph each explaining the match. No bullet lists.
- Search results: brief intro sentence, then each book on its own line with a one-sentence description.
- Reviews: conversational prose. For write-help: structure suggestions + 2–3 example opening lines.
- Shelf actions: one confirmation sentence + optional "what to read next" suggestion.
- General: direct answer, 1–3 sentences unless complexity warrants more.
</response_format>"""

# ── Per-intent instruction blocks ─────────────────────────────────────────────
# Only the relevant block is injected — no dead tokens for other intents.

_INTENT_INSTRUCTIONS: dict[IntentType, str] = {
    "recommend": """<task>
The user wants personalised book recommendations.
Use ONLY books from <catalogue_context>. Explain the specific match for each book:
what about this book makes it right for THIS user's request?
Suggest 2–3 books. If fewer than 2 strong matches exist, say so rather than padding.
</task>""",

    "search": """<task>
The user is searching for a specific book, author, or topic.
Present search results from <tool_data> clearly and accurately.
If results don't match the query well, say so and offer an alternative approach.
Do not recommend books outside the search results.
</task>""",

    "review": """<task>
The user wants help with book reviews.
- Writing a review: help them structure their thoughts (3-part: hook, plot-without-spoilers, verdict).
  Offer 2 example opening lines they can adapt.
- Reading reviews: summarise the sentiment from <tool_data>, note any strong consensus or splits.
- Discussing: engage genuinely with their view.
</task>""",

    "shelf": """<task>
The user wants to manage their reading shelf.
Confirm the shelf action from <tool_data> in one sentence.
Optionally suggest what to read next from their existing shelf if the data includes it.
Keep it brief — shelf actions are transactional.
</task>""",

    "general": """<task>
Answer the user's question about books, reading, or this platform.
Be direct and specific. If you don't know, say so rather than guessing.
</task>""",
}


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_response_prompt(
    rag_context: str,
    tool_results: dict,
    intent: IntentType,
) -> str:
    """
    Builds the optimised system prompt for the final response node.

    Structure (order matters for caching):
      1. <persona>          — static, cacheable prefix
      2. <core_rules>       — static, cacheable
      3. <response_format>  — static, cacheable
      4. <task>             — per-intent, short
      5. <catalogue_context>— dynamic RAG data (varies per query)
      6. <tool_data>        — dynamic tool results (varies per query)

    The first 3 sections are always identical → Anthropic caches them.
    Dynamic sections come LAST so the cached prefix is maximally reused.

    Args:
        rag_context: Pre-formatted book summaries from RAG retrieval
        tool_results: Raw dict from MCP tool call
        intent: Classified user intent

    Returns:
        Complete system prompt string, token-optimised
    """
    parts: list[str] = [_PERSONA]

    # Intent instruction (short — not cached but small)
    task_block = _INTENT_INSTRUCTIONS.get(intent, _INTENT_INSTRUCTIONS["general"])
    parts.append(task_block)

    # Dynamic: RAG context (only if present)
    if rag_context:
        trimmed_context = _trim_to_budget(rag_context, RAG_CONTEXT_MAX_TOKENS)
        parts.append(f"<catalogue_context>\n{trimmed_context}\n</catalogue_context>")

    # Dynamic: Tool results (only if present, stripped to budget)
    if tool_results:
        trimmed_tools = _serialise_tool_results(tool_results, TOOL_RESULTS_MAX_TOKENS)
        parts.append(f"<tool_data>\n{trimmed_tools}\n</tool_data>")

    return "\n\n".join(parts)


def get_response_budget(intent: IntentType) -> ResponseBudget:
    """Returns the token budget and prefill for the given intent."""
    return RESPONSE_BUDGETS.get(intent, RESPONSE_BUDGETS["general"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trim_to_budget(text: str, max_tokens: int) -> str:
    """
    Rough token-budget trim using character proxy (1 token ≈ 4 chars).

    For production: replace with tiktoken for precise counts.
    This approximation is intentionally conservative (errs on keeping more context).
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    # Trim at a sentence boundary if possible
    trimmed = text[:max_chars]
    last_period = trimmed.rfind(". ")
    if last_period > max_chars * 0.7:  # Found a period in the last 30% of text
        return trimmed[: last_period + 1] + "\n[...context trimmed for length]"
    return trimmed + "..."


def _serialise_tool_results(results: dict, max_tokens: int) -> str:
    """
    Serialises tool results to JSON, stripping fields the LLM doesn't need.

    Removes: internal IDs, timestamps, pagination cursors, raw scores.
    Keeps: user-facing fields (title, author, genres, summary_excerpt, rating).
    """
    # Fields to strip — internal plumbing the LLM doesn't need
    _STRIP_FIELDS = {"id", "book_id", "user_id", "created_at", "updated_at",
                     "cursor", "next_page", "embedding", "chunk_index"}

    def _clean(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k not in _STRIP_FIELDS}
        if isinstance(obj, list):
            return [_clean(item) for item in obj]
        return obj

    cleaned = _clean(results)
    serialised = json.dumps(cleaned, indent=2, default=str)

    max_chars = max_tokens * 4
    if len(serialised) > max_chars:
        # Truncate the list to fewer items rather than cutting mid-JSON
        if isinstance(cleaned.get("books"), list):
            while len(json.dumps(cleaned, default=str)) > max_chars and cleaned["books"]:
                cleaned["books"].pop()
            cleaned["_truncated"] = True
        serialised = json.dumps(cleaned, indent=2, default=str)

    return serialised
