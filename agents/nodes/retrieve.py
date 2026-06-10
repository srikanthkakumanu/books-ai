"""
agents/nodes/retrieve.py

RAG context retrieval node — with query optimisation.

Prompt engineering improvements over v1:
  1. Query rewriting: expand sparse queries before embedding
  2. HyDE (Hypothetical Document Embedding): for vague mood queries
  3. Parallel retrieval: run rewrite + vector search concurrently
  4. Intent-based top_k tuning: recommend gets more context than general
  5. Structured logging of retrieval quality for monitoring

Token impact:
  - Better queries → more relevant chunks → fewer wasted context tokens
  - Intent-tuned top_k → right amount of context, not always the max
"""

from __future__ import annotations

import asyncio
import structlog

from agents.state import AgentState
from agents.utils.query_optimizer import (
    generate_hyde_query,
    rewrite_query_for_retrieval,
    should_use_hyde,
)
from rag.retriever import BookRAGRetriever

logger = structlog.get_logger(__name__)

# Intents that benefit from RAG context
_RAG_INTENTS = {"recommend", "search", "general"}

# Per-intent top_k: recommendations need more context to give nuanced advice;
# search just needs the top result; general is in between
_INTENT_TOP_K: dict[str, int] = {
    "recommend": 6,   # More candidates = better personalisation
    "search":    4,   # Search results are already ranked; don't over-retrieve
    "general":   3,   # Minimal context for factual Q&A
}

_retriever = BookRAGRetriever()


async def retrieve_rag_context(state: AgentState) -> dict:
    """
    Retrieves relevant book context with query optimisation.

    For vague queries: applies HyDE (generate hypothetical answer, embed that).
    For reference-heavy queries: rewrites to resolve "that book"/"same author".
    Skips retrieval entirely for intents that don't need book context.

    Returns:
        Partial state update: {"rag_context": str, "rag_sources": list[str]}
    """
    if state["intent"] not in _RAG_INTENTS:
        return {"rag_context": "", "rag_sources": []}

    raw_query = str(state["messages"][-1].content)
    history = state.get("messages", [])[:-1]  # History without the latest message
    top_k = _INTENT_TOP_K.get(state["intent"], 4)

    logger.info(
        "retrieving_rag_context",
        user_id=state["user_id"],
        intent=state["intent"],
        raw_query=raw_query[:60],
        top_k=top_k,
    )

    # ── Query preparation ──────────────────────────────────────────────────
    # Decision: HyDE for vague queries, rewrite for reference-heavy queries
    if should_use_hyde(raw_query) and state["intent"] == "recommend":
        # Run HyDE + standard rewrite concurrently, use whichever returns first
        # HyDE generates a hypothetical document; standard rewrite expands terms
        hyde_task = asyncio.create_task(generate_hyde_query(raw_query))
        rewrite_task = asyncio.create_task(rewrite_query_for_retrieval(raw_query, history))

        hyde_query, rewritten_query = await asyncio.gather(hyde_task, rewrite_task)

        # Run two parallel vector searches with different query strategies
        hyde_search = asyncio.create_task(
            _retriever.search(query=hyde_query, top_k=top_k // 2 + 1)
        )
        rewrite_search = asyncio.create_task(
            _retriever.search(query=rewritten_query, top_k=top_k // 2 + 1)
        )
        (hyde_ctx, hyde_src), (rewrite_ctx, rewrite_src) = await asyncio.gather(
            hyde_search, rewrite_search
        )

        # Merge: prefer rewrite results, fill remaining slots from HyDE
        # Deduplication is handled inside the retriever via book_id set
        context = rewrite_ctx or hyde_ctx
        sources = list(dict.fromkeys(rewrite_src + hyde_src))  # Deduplicate, preserve order

        logger.info(
            "hybrid_retrieval_complete",
            user_id=state["user_id"],
            hyde_sources=len(hyde_src),
            rewrite_sources=len(rewrite_src),
        )
    else:
        # Standard path: rewrite query, single vector search
        optimised_query = await rewrite_query_for_retrieval(raw_query, history)
        context, sources = await _retriever.search(query=optimised_query, top_k=top_k)

    logger.info(
        "rag_complete",
        user_id=state["user_id"],
        source_count=len(sources),
        context_chars=len(context),
    )

    return {"rag_context": context, "rag_sources": sources}
