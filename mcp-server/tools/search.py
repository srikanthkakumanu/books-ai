"""
mcp-server/tools/search.py

search_books MCP tool — with optimised tool description for LLM comprehension.

Prompt engineering techniques for MCP tool descriptions:
  1. Verb-first descriptions: "Search the catalogue..." not "This tool searches..."
  2. When-to-use guidance: explicit "Use this when..." prevents agent calling wrong tool
  3. When-NOT-to-use: prevents the agent calling search when recommend is correct
  4. Output description: tells the agent what it will receive (calibrates expectations)
  5. Field descriptions include examples (not just types)
  6. Enum constraints where possible (reduces free-form hallucination)
  7. Required fields minimal — fewer required fields → fewer tool call failures
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from mcp_server.utils.http import service_client

logger = structlog.get_logger(__name__)


# ── MCP Tool Definition ────────────────────────────────────────────────────────
# The description is read by Claude on every tool selection decision.
# Every word counts — this is a prompt, not documentation.

TOOL_DEFINITION = {
    "name": "search_books",
    "description": (
        "Search the book catalogue by title, author name, or keyword. "
        "Returns matching books with title, author, genres, and a brief summary. "
        "Use this when: the user names a specific book or author, asks to browse by topic, "
        "or wants to know if a specific book is in the catalogue. "
        "Do NOT use this for open-ended recommendations — use get_recommendations instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The search term. Can be a book title ('Dune'), author name "
                    "('Ursula Le Guin'), or topic keyword ('space opera'). "
                    "Extract the specific search term from the user's message — "
                    "do not pass the full user message verbatim."
                ),
            },
            "genre": {
                "type": "string",
                "description": (
                    "Optional genre filter to narrow results. "
                    "Use only when the user explicitly specifies a genre. "
                    "Examples: 'science fiction', 'fantasy', 'mystery', 'romance', "
                    "'thriller', 'historical fiction', 'biography', 'self-help'."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Number of results to return. Default 10. Max 50. Use 5 for quick lookups.",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": ["query"],
    },
}


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class SearchBooksRequest(BaseModel):
    query: str = Field(
        description="Search term — book title, author, or keyword",
        min_length=1,
        max_length=200,
    )
    genre: str | None = Field(default=None, description="Optional genre filter")
    limit: int = Field(default=10, ge=1, le=50)


class BookResult(BaseModel):
    id: str
    title: str
    author: str
    genres: list[str]
    summary_excerpt: str
    cover_url: str | None = None
    average_rating: float | None = None


class SearchBooksResponse(BaseModel):
    books: list[BookResult]
    total: int
    query: str  # Echo back the query for the LLM to reference
    error: str | None = None


# ── Handler ───────────────────────────────────────────────────────────────────

async def handle_search_books(
    args: SearchBooksRequest,
    user_id: str,
) -> SearchBooksResponse:
    """Handles search_books tool call — delegates to book-service."""
    logger.info("search_books", user_id=user_id, query=args.query, genre=args.genre)

    params: dict = {"q": args.query, "limit": args.limit}
    if args.genre:
        params["genre"] = args.genre

    data = await service_client.get(
        service="book-service",
        path="/books/search",
        params=params,
    )

    books = [BookResult(**b) for b in data.get("books", [])]

    return SearchBooksResponse(
        books=books,
        total=data.get("total", len(books)),
        query=args.query,
    )
