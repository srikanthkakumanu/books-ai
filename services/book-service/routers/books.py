"""
services/book-service/routers/books.py

HTTP router for the /books resource.

Routers are thin:
- Validate inputs (FastAPI + Pydantic handle this)
- Call the repository
- Map domain models to response schemas
- Raise appropriate HTTP exceptions

No SQL, no business logic — that lives in the repository.
"""

from __future__ import annotations

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from book_service.models.domain import BookCreateRequest, BookResponse, BookSearchResponse
from book_service.repositories.book_repo import BookRepository, get_book_repo

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/books", tags=["books"])


@router.get("/{book_id}", response_model=BookResponse)
async def get_book(
    book_id: UUID,
    repo: BookRepository = Depends(get_book_repo),
) -> BookResponse:
    """
    Fetches a single book by ID.

    Returns 404 if the book does not exist.
    """
    book = await repo.get_by_id(book_id)
    if not book:
        raise HTTPException(status_code=404, detail=f"Book {book_id} not found")

    return BookResponse.model_validate(book)


@router.get("/search", response_model=BookSearchResponse)
async def search_books(
    q: str = Query(min_length=1, max_length=500, description="Search query"),
    genre: str | None = Query(default=None, description="Genre filter"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=50),
    repo: BookRepository = Depends(get_book_repo),
) -> BookSearchResponse:
    """
    Full-text search across book titles, authors, and summaries.

    Uses PostgreSQL tsvector for efficient full-text matching.
    Supports optional genre filtering.
    """
    offset = (page - 1) * page_size

    books, total = await repo.search(
        query=q,
        genre=genre,
        limit=page_size,
        offset=offset,
    )

    return BookSearchResponse(
        books=[BookResponse.model_validate(b) for b in books],
        total=total,
        query=q,
        page=page,
        page_size=page_size,
    )


@router.post("", response_model=BookResponse, status_code=201)
async def create_book(
    body: BookCreateRequest,
    repo: BookRepository = Depends(get_book_repo),
    # TODO: add admin auth dependency
) -> BookResponse:
    """
    Creates a new book in the catalogue.

    Admin-only endpoint. ISBN must be unique if provided.
    """
    if body.isbn:
        existing = await repo.get_by_isbn(body.isbn)
        if existing:
            raise HTTPException(status_code=409, detail=f"ISBN {body.isbn} already exists")

    book = await repo.create(body)
    logger.info("book_created", book_id=str(book.id), title=book.title)

    return BookResponse.model_validate(book)
