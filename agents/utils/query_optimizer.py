"""
agents/utils/query_optimizer.py

Query rewriting and optimisation before RAG retrieval.

Prompt engineering techniques applied:
  1. Query expansion: enrich sparse queries with implied terms
  2. Query decomposition: break compound queries into atomic sub-queries
  3. HyDE (Hypothetical Document Embedding): generate a hypothetical answer
     then embed that — dramatically improves retrieval for vague queries
  4. Conversational context injection: resolve "that book" / "same author"
     references using message history before embedding
  5. Negative query filtering: detect and strip non-retrievable intent
     ("why is Dune good?" → should not hit vector DB)

References:
  - HyDE: https://arxiv.org/abs/2212.10496
  - Query rewriting for RAG: Anthropic cookbook
"""

from __future__ import annotations

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agents.prompts.prompt_config import CLASSIFY_MODEL

logger = structlog.get_logger(__name__)

# Haiku for query rewriting — cheap, fast, good enough for expansion
_llm = ChatAnthropic(model=CLASSIFY_MODEL, max_tokens=150, temperature=0.2)

_REWRITE_PROMPT = """<task>Rewrite the user's book query for semantic search retrieval.</task>

<rules>
1. Resolve pronouns and references using conversation history if provided.
2. Expand implicit genre/mood terms (e.g. "page-turner" → "fast-paced thriller suspense").
3. Output ONLY the rewritten query — no explanation, no quotes.
4. Keep it under 30 words.
5. If the query is already specific and clear, output it unchanged.
</rules>

<examples>
History: "I loved Project Hail Mary" | Query: "something similar" → "science fiction first contact survival optimistic like Project Hail Mary Andy Weir"
History: none | Query: "a cozy mystery" → "cozy mystery amateur detective small town light-hearted"
History: "books by Le Guin" | Query: "her best one" → "Ursula Le Guin best acclaimed novel"
History: none | Query: "Dune" → "Dune"
</examples>"""


async def rewrite_query_for_retrieval(
    query: str,
    history: list[BaseMessage] | None = None,
) -> str:
    """
    Rewrites a user query to improve vector similarity retrieval.

    Uses conversational context to resolve references and expands
    terse queries with implied semantic terms.

    Args:
        query: The raw user query string
        history: Recent message history for reference resolution

    Returns:
        Rewritten query string, or original query if rewriting fails
    """
    # Fast path: long explicit queries don't need rewriting
    if len(query.split()) >= 8 and not _has_references(query):
        return query

    try:
        context_str = ""
        if history:
            # Inject last 2 turns as context for reference resolution
            recent = history[-4:]
            context_str = " | ".join(
                f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {str(m.content)[:100]}"
                for m in recent
            )

        user_msg = f"History: {context_str or 'none'}\nQuery: {query}"

        response = await _llm.ainvoke([
            SystemMessage(content=_REWRITE_PROMPT),
            HumanMessage(content=user_msg),
        ])

        rewritten = response.content.strip()

        # Sanity check: rewritten query should not be drastically longer
        if len(rewritten.split()) > 40:
            logger.warning("query_rewrite_too_long", original=query, rewritten=rewritten[:80])
            return query

        logger.info(
            "query_rewritten",
            original=query[:60],
            rewritten=rewritten[:60],
            changed=query != rewritten,
        )
        return rewritten

    except Exception as exc:  # noqa: BLE001
        logger.warning("query_rewrite_failed", error=str(exc))
        return query  # Fallback: use original query


def _has_references(query: str) -> bool:
    """
    Detects pronouns and demonstratives that signal reference resolution is needed.
    """
    reference_words = {
        "it", "that", "this", "those", "these", "them", "her", "his",
        "similar", "same", "like it", "like that", "the same", "another",
        "more like", "else", "other"
    }
    words = set(query.lower().split())
    return bool(words & reference_words)


# ── HyDE (Hypothetical Document Embedding) ────────────────────────────────────

_HYDE_PROMPT = """<task>Write a short hypothetical book summary that would perfectly answer the user's request.</task>

<rules>
1. Write as if this book EXISTS in the catalogue — invent a plausible title, author, and 2-sentence summary.
2. Match the genre, mood, and themes the user is asking for.
3. Output ONLY: Title: X | Author: Y | Summary: Z
4. Keep the summary under 40 words.
</rules>

<purpose>
This hypothetical summary will be embedded and used for vector similarity search.
It should be CONTENT-rich with genre keywords, themes, and mood terms — not a generic description.
</purpose>"""


async def generate_hyde_query(query: str) -> str:
    """
    Generates a hypothetical book summary for HyDE-based retrieval.

    HyDE works by generating a hypothetical answer to the query, then using
    that answer's embedding (rather than the query's embedding) for retrieval.
    The answer embedding tends to live closer to real documents in vector space.

    Use for: vague mood queries ("something hopeful"), theme queries ("found family"),
    comparative queries ("like X but darker").
    Avoid for: specific title/author searches where the query is already precise.

    Args:
        query: User's book request

    Returns:
        Hypothetical book description string for embedding
    """
    try:
        response = await _llm.ainvoke([
            SystemMessage(content=_HYDE_PROMPT),
            HumanMessage(content=query),
        ])
        hyde_text = response.content.strip()
        logger.info("hyde_generated", query=query[:50], hyde_len=len(hyde_text))
        return hyde_text
    except Exception as exc:  # noqa: BLE001
        logger.warning("hyde_generation_failed", error=str(exc))
        return query


def should_use_hyde(query: str) -> bool:
    """
    Heuristic: use HyDE for vague mood/theme queries, skip for specific ones.

    HyDE adds latency (one extra LLM call) — only worth it when the query
    is too vague for direct embedding to work well.
    """
    vague_signals = [
        "something", "a book", "any book", "recommend", "suggest",
        "looking for", "want to read", "feel like", "mood for",
        "hopeful", "dark", "funny", "uplifting", "cozy", "intense",
    ]
    query_lower = query.lower()
    return any(signal in query_lower for signal in vague_signals)
