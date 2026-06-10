"""
services/book-service/models/domain.py

SQLAlchemy ORM models for the books_schema.

Rules:
- Models map 1:1 to database tables
- No business logic in models — that lives in repositories
- Use UUID primary keys throughout
- updated_at is maintained by a DB trigger (see migration 002)
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import ARRAY, Column, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class BookModel(Base):
    """
    Represents a book in the catalogue.

    The search_vector column is a PostgreSQL GENERATED column — it is
    maintained automatically by the DB engine. Never set it in Python code.
    """

    __tablename__ = "books"
    __table_args__ = {"schema": "books_schema"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False)
    author = Column(Text, nullable=False)
    isbn = Column(String(13), unique=True, nullable=True)
    genres = Column(ARRAY(Text), nullable=False, default=list)
    summary = Column(Text, nullable=True)
    cover_url = Column(Text, nullable=True)
    page_count = Column(Integer, nullable=True)
    language = Column(String(5), nullable=False, default="en")
    # search_vector is a GENERATED ALWAYS column — do not include in INSERT/UPDATE
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


"""
services/book-service/models/schemas.py

Pydantic v2 request/response schemas for the book-service API.

Separate from ORM models intentionally — the API contract should not
be coupled to the database schema.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class BookResponse(BaseModel):
    """Full book representation returned by the API."""

    id: UUID
    title: str
    author: str
    isbn: str | None = None
    genres: list[str]
    summary: str | None = None
    cover_url: str | None = None
    page_count: int | None = None
    language: str
    created_at: datetime

    model_config = {"from_attributes": True}


class BookCreateRequest(BaseModel):
    """Request body for creating a new book (admin only)."""

    title: str = Field(min_length=1, max_length=500)
    author: str = Field(min_length=1, max_length=200)
    isbn: str | None = Field(default=None, min_length=10, max_length=13)
    genres: list[str] = Field(default_factory=list)
    summary: str | None = None
    cover_url: str | None = None
    page_count: int | None = Field(default=None, ge=1)
    language: str = Field(default="en", min_length=2, max_length=5)


class BookSearchResponse(BaseModel):
    """Paginated search results."""

    books: list[BookResponse]
    total: int
    query: str
    page: int
    page_size: int
