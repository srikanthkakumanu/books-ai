"""
rag/retriever.py

Main entry point for the RAG pipeline — with re-ranking and caching.

Improvements over v1:
  1. Embedding cache: Redis TTL cache for identical queries (saves embed API call)
  2. Re-ranking: LLM-as-judge scoring for recommend intent
  3. Diversity enforcement: no author-dominated results
  4. Structured return: FormattedContext with token count metadata
  5. Similarity threshold configurable per call site

Usage (from agent layer):
    retriever = BookRAGRetriever()
    context, sources = await retriever.search("book like Dune", top_k=5)
"""

from __future__ import annotations

import hashlib
import json

import structlog

from rag.embedders.openai_embedder import OpenAIEmbedder
from rag.retrievers.pgvector_retriever import BookChunk, PgVectorRetriever
from rag.utils.context_ranker import enforce_diversity, rerank_chunks
from rag.utils.formatter import format_context_structured

logger = structlog.get_logger(__name__)

# Intents that merit the extra re-ranking LLM call
_RERANK_INTENTS = {"recommend"}


class BookRAGRetriever:
    """
    Orchestrates the full RAG retrieval pipeline:
      1. Check embedding cache (Redis)
      2. Embed the (rewritten) query
      3. pgvector cosine similarity search
      4. Optional LLM re-ranking (recommend intent only)
      5. Diversity enforcement
      6. Format as token-budgeted prompt context

    Returns both the formatted context string AND source book IDs
    so the agent can attribute recommendations.
    """

    def __init__(
        self,
        embedder: OpenAIEmbedder | None = None,
        retriever: PgVectorRetriever | None = None,
        redis_client=None,
    ) -> None:
        self._embedder = embedder or OpenAIEmbedder()
        self._retriever = retriever or PgVectorRetriever()
        self._redis = redis_client  # Optional — gracefully skipped if None

    async def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.70,
        intent: str = "general",
        rerank: bool | None = None,
    ) -> tuple[str, list[str]]:
        """
        Retrieves relevant book context for the given query.

        Args:
            query: Query string (may be rewritten by the agent layer)
            top_k: Maximum number of books to return after re-ranking
            similarity_threshold: Minimum cosine similarity for initial retrieval
            intent: Agent intent — affects re-ranking and top_k behaviour
            rerank: Override re-ranking decision (None = auto based on intent)

        Returns:
            Tuple of (formatted_context_string, list_of_book_ids)
        """
        logger.info("rag_search_start", query=query[:60], top_k=top_k, intent=intent)

        # ── 1. Embed (with cache) ──────────────────────────────────────────
        embedding = await self._get_embedding_cached(query)

        # ── 2. Vector search (retrieve more than needed for re-ranking) ────
        # Retrieve 2x top_k to give re-ranker more to work with
        retrieval_k = top_k * 2 if intent in _RERANK_INTENTS else top_k
        chunks: list[BookChunk] = await self._retriever.search(
            embedding=embedding,
            top_k=retrieval_k,
            similarity_threshold=similarity_threshold,
        )

        if not chunks:
            logger.info("rag_no_results", query=query[:60])
            return "", []

        # ── 3. Optional re-ranking ─────────────────────────────────────────
        should_rerank = rerank if rerank is not None else (intent in _RERANK_INTENTS)
        if should_rerank and len(chunks) > 1:
            chunks = await rerank_chunks(chunks, query=query)

        # ── 4. Diversity enforcement ───────────────────────────────────────
        chunks = enforce_diversity(chunks)

        # ── 5. Trim to final top_k ─────────────────────────────────────────
        chunks = chunks[:top_k]

        # ── 6. Format ─────────────────────────────────────────────────────
        formatted = format_context_structured(chunks)
        source_ids = [chunk.book_id for chunk in chunks]

        logger.info(
            "rag_search_complete",
            book_count=formatted.book_count,
            estimated_tokens=formatted.estimated_tokens,
            source_ids=source_ids,
        )

        return formatted.text, source_ids

    async def _get_embedding_cached(self, query: str) -> list[float]:
        """
        Returns the embedding for a query, using Redis cache if available.

        Cache key: SHA256 of the query string (safe, collision-resistant).
        TTL: 1 hour — embeddings are deterministic so longer TTL is fine.
        """
        if self._redis is None:
            return await self._embedder.embed(query)

        cache_key = f"emb:{hashlib.sha256(query.encode()).hexdigest()[:16]}"
        try:
            cached = await self._redis.get(cache_key)
            if cached:
                logger.debug("embedding_cache_hit", key=cache_key[:20])
                return json.loads(cached)
        except Exception:  # noqa: BLE001
            pass  # Cache failure is non-fatal — fall through to live embed

        embedding = await self._embedder.embed(query)

        try:
            await self._redis.setex(cache_key, 3600, json.dumps(embedding))
        except Exception:  # noqa: BLE001
            pass  # Cache write failure is non-fatal

        return embedding
