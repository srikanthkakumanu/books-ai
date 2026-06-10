"""
services/book-service/main.py

Book Service FastAPI application.

Owns: books catalogue, full-text search, book embeddings table.
Port: 8010

Exposes:
    GET  /books/{book_id}        — fetch a single book
    GET  /books/search           — full-text search
    POST /books                  — create a book (admin only)
    GET  /health
    GET  /health/ready
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from book_service.routers.books import router as books_router
from book_service.utils.db import engine, Base

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: startup and shutdown tasks."""
    logger.info("book_service_starting")
    # Tables are managed by Alembic — never use create_all in production
    # Base.metadata.create_all(engine)  # DEV ONLY
    yield
    logger.info("book_service_stopping")
    await engine.dispose()


app = FastAPI(
    title="Book Service",
    description="Books catalogue microservice",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(books_router)


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness() -> dict:
    """Readiness probe — verifies DB connectivity."""
    from sqlalchemy import text
    from book_service.utils.db import get_session
    async with get_session() as session:
        await session.execute(text("SELECT 1"))
    return {"status": "ok"}
