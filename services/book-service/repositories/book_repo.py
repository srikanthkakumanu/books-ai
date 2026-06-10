"""
services/book-service/repositories/book_repo.py

Data access layer for the books table.

ALL SQL lives here. Routers never touch the database directly.
Uses SQLAlchemy async ORM throughout — no raw SQL strings except where
the ORM cannot express the query (noted with a comment when used).
"""

from __future__ import annotations

from uuid import UUID

import structlog
from fastapi import Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from book_service.models.domain import BookCreateRequest, BookModel
from book_service.utils.db import get_session

logger = structlog.get_logger(__name__)


class BookRepository:
    """
    Repository for the books_schema.books table.

    All methods are async. Session is injected via FastAPI Depends.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, book_id: UUID) -> BookModel | None:
        """Fetches a single book by primary key. Returns None if not found."""
        result = await self._session.execute(
            select(BookModel).where(BookModel.id == book_id)
        )
        return result.scalar_one_or_none()

    async def get_by_isbn(self, isbn: str) -> BookModel | None:
        """Fetches a book by ISBN. Returns None if not found."""
        result = await self._session.execute(
            select(BookModel).where(BookModel.isbn == isbn)
        )
        return result.scalar_one_or_none()

    async def search(
        self,
        query: str,
        genre: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> tuple[list[BookModel], int]:
        """
        Full-text search using PostgreSQL tsvector.

        Returns a tuple of (results, total_count).
        total_count is the count without pagination for the UI.
        """
        # Base query using the generated search_vector column
        # plainto_tsquery is safer than to_tsquery — handles user input gracefully
        base = select(BookModel).where(
            BookModel.search_vector.op("@@")(func.plainto_tsquery("english", query))
        )

        if genre:
            base = base.where(BookModel.genres.contains([genre]))

        # Get total count (execute separately to avoid subquery complexity)
        count_result = await self._session.execute(
            select(func.count()).select_from(base.subquery())
        )
        total = count_result.scalar_one()

        # Get paginated results ordered by relevance
        paginated = (
            base
            .order_by(
                func.ts_rank(
                    BookModel.search_vector,
                    func.plainto_tsquery("english", query),
                ).desc()
            )
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(paginated)
        books = list(result.scalars().all())

        return books, total

    async def create(self, data: BookCreateRequest) -> BookModel:
        """Creates and persists a new book. Commits the transaction."""
        book = BookModel(
            title=data.title,
            author=data.author,
            isbn=data.isbn,
            genres=data.genres,
            summary=data.summary,
            cover_url=str(data.cover_url) if data.cover_url else None,
            page_count=data.page_count,
            language=data.language,
        )
        self._session.add(book)
        await self._session.commit()
        await self._session.refresh(book)
        return book


async def get_book_repo(session: AsyncSession = Depends(get_session)) -> BookRepository:
    """FastAPI dependency that provides a BookRepository with an injected session."""
    return BookRepository(session)
