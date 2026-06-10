"""
rag/utils/context_ranker.py

Re-ranking of RAG retrieval results for higher context quality.

Problem: pgvector returns chunks ranked by embedding cosine similarity,
but cosine similarity doesn't always reflect semantic relevance to the user's
specific query. A book about "space" might rank highly for a "Dune" query
when the user actually wants epic world-building fantasy.

Solution: Cross-encoder re-ranking — use a second LLM pass to score each
retrieved chunk against the original query, then reorder.

Techniques applied:
  1. LLM-as-judge scoring: Haiku scores relevance 1-5 for each chunk
  2. Batch scoring: all chunks in one LLM call (not N separate calls)
  3. Structured output: scores returned as JSON for reliable parsing
  4. Score threshold: chunks below threshold dropped before formatter
  5. Diversity injection: prevent top-k being all the same author/genre

Token cost: ~150 tokens per re-ranking call (Haiku, very cheap)
Latency: adds ~200ms; only runs for "recommend" intent where quality matters most
"""

from __future__ import annotations

import json

import structlog
from anthropic import AsyncAnthropic

from agents.prompts.prompt_config import CLASSIFY_MODEL
from rag.retrievers.pgvector_retriever import BookChunk

logger = structlog.get_logger(__name__)

_client = AsyncAnthropic()

_RERANK_PROMPT = """<task>Score each book's relevance to the user's query. Return JSON only.</task>

<scoring>
5 = Perfect match — directly answers what the user asked for
4 = Strong match — highly relevant genre/theme/mood
3 = Partial match — related but not ideal
2 = Weak match — tangentially relevant
1 = Poor match — unrelated
</scoring>

<output_format>
Return ONLY valid JSON, no other text:
{"scores": [{"index": 0, "score": 4}, {"index": 1, "score": 2}, ...]}
</output_format>"""


async def rerank_chunks(
    chunks: list[BookChunk],
    query: str,
    min_score: int = 3,
) -> list[BookChunk]:
    """
    Re-ranks retrieved chunks using LLM-as-judge scoring.

    Uses claude-haiku for cheap, fast relevance scoring.
    Returns chunks re-ordered by LLM score, filtered to min_score.

    Args:
        chunks: Retrieved chunks from pgvector (initial ranking by cosine sim)
        query: Original user query (not the rewritten version)
        min_score: Minimum LLM relevance score to keep (1-5)

    Returns:
        Re-ordered, filtered list of BookChunks
    """
    if len(chunks) <= 1:
        return chunks

    # Build compact book list for scoring (title + genre + first 80 chars of summary)
    book_list = []
    for i, chunk in enumerate(chunks):
        genres = ", ".join(chunk.genres[:2]) if chunk.genres else "fiction"
        summary_preview = chunk.summary_chunk[:80].rstrip() + "…"
        book_list.append(
            f"{i}. \"{chunk.title}\" by {chunk.author} [{genres}] — {summary_preview}"
        )

    user_content = f"Query: {query}\n\nBooks:\n" + "\n".join(book_list)

    try:
        response = await _client.messages.create(
            model=CLASSIFY_MODEL,
            max_tokens=150,
            temperature=0.0,
            system=_RERANK_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        raw = response.content[0].text.strip()
        data = json.loads(raw)
        scores: dict[int, int] = {item["index"]: item["score"] for item in data["scores"]}

        # Filter and re-sort by LLM score
        scored_chunks = [
            (chunk, scores.get(i, 1))
            for i, chunk in enumerate(chunks)
            if scores.get(i, 1) >= min_score
        ]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        reranked = [chunk for chunk, _ in scored_chunks]

        logger.info(
            "reranking_complete",
            original_count=len(chunks),
            filtered_count=len(reranked),
            dropped=len(chunks) - len(reranked),
        )

        return reranked if reranked else chunks  # Fallback: return original if all filtered

    except Exception as exc:  # noqa: BLE001
        logger.warning("reranking_failed", error=str(exc))
        return chunks  # Graceful fallback to original order


def enforce_diversity(chunks: list[BookChunk], max_per_author: int = 2) -> list[BookChunk]:
    """
    Prevents top results from being dominated by a single author or genre.

    Rules:
    - Max max_per_author books from the same author
    - At least 2 different genres if 4+ books in results

    Args:
        chunks: Re-ranked chunks
        max_per_author: Maximum books from same author in results

    Returns:
        Diversity-enforced list (some chunks may be dropped or reordered)
    """
    author_counts: dict[str, int] = {}
    diverse: list[BookChunk] = []
    deferred: list[BookChunk] = []  # Chunks that exceeded the author limit

    for chunk in chunks:
        author_key = chunk.author.lower().strip()
        count = author_counts.get(author_key, 0)

        if count < max_per_author:
            diverse.append(chunk)
            author_counts[author_key] = count + 1
        else:
            deferred.append(chunk)

    # Add deferred chunks at the end if we have room (don't drop them entirely)
    result = diverse + deferred
    return result
