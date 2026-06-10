"""
rag/retrievers/pgvector_retriever.py

Cosine similarity search against PostgreSQL with pgvector extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rag.utils.db_factory import get_db_session

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class BookChunk:
    """A single retrieved book chunk from the vector database."""

    book_id: str
    title: str
    author: str
    genres: list[str]
    summary_chunk: str
    similarity: float


class PgVectorRetriever:
    """
    Retrieves book chunks from PostgreSQL using pgvector cosine similarity.

    The query uses the <=> operator (cosine distance) with an ivfflat index.
    Results are filtered by a minimum similarity threshold to avoid returning
    irrelevant books.
    """

    _QUERY = text("""
        SELECT
            be.book_id::text,
            b.title,
            b.author,
            b.genres,
            be.summary_chunk,
            1 - (be.embedding <=> :embedding ::vector) AS similarity
        FROM books_schema.book_embeddings be
        JOIN books_schema.books b ON b.id = be.book_id
        WHERE 1 - (be.embedding <=> :embedding ::vector) > :threshold
        ORDER BY be.embedding <=> :embedding ::vector
        LIMIT :limit
    """)

    async def search(
        self,
        embedding: list[float],
        top_k: int,
        similarity_threshold: float,
    ) -> list[BookChunk]:
        """
        Finds the top_k most similar book chunks to the given embedding vector.

        Args:
            embedding: 1536-dim float list from the embedder
            top_k: Maximum number of chunks to return
            similarity_threshold: Minimum cosine similarity score to include

        Returns:
            List of BookChunk ordered by similarity descending
        """
        async with get_db_session() as session:
            result = await session.execute(
                self._QUERY,
                {
                    "embedding": str(embedding),
                    "threshold": similarity_threshold,
                    "limit": top_k,
                },
            )
            rows = result.mappings().all()

        return [
            BookChunk(
                book_id=row["book_id"],
                title=row["title"],
                author=row["author"],
                genres=row["genres"] or [],
                summary_chunk=row["summary_chunk"],
                similarity=float(row["similarity"]),
            )
            for row in rows
        ]
