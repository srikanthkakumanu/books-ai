"""
mcp-server/tools/recommend.py

get_recommendations MCP tool.

Prompt engineering for tool description:
  - Explicit input guidance: what makes a good seed_book_ids list
  - Output description: what the agent should do with confidence scores
  - Clear disambiguation from search_books tool
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from mcp_server.utils.http import service_client

logger = structlog.get_logger(__name__)

TOOL_DEFINITION = {
    "name": "get_recommendations",
    "description": (
        "Get personalised book recommendations for a user based on their reading history "
        "and/or seed books similar to what they want. "
        "Use this when: the user asks for recommendations, says 'something like X', "
        "or asks what to read next. "
        "Do NOT use this to look up a specific book — use search_books for that. "
        "Returns a ranked list of books with confidence scores and match reasons."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "The authenticated user's ID. Always use the user_id from context — never make one up.",
            },
            "seed_book_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Book IDs to use as recommendation seeds. "
                    "Use book IDs from the RAG catalogue context if available. "
                    "Pass an empty list to use only the user's reading history."
                ),
            },
            "query": {
                "type": "string",
                "description": "The user's request in their own words. Used for theme/mood matching.",
            },
            "limit": {
                "type": "integer",
                "description": "Number of recommendations to return. Default 5. Max 10.",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["user_id", "seed_book_ids", "query"],
    },
}


class RecommendBooksRequest(BaseModel):
    user_id: str
    seed_book_ids: list[str] = Field(default_factory=list)
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=10)


class RecommendedBook(BaseModel):
    id: str
    title: str
    author: str
    genres: list[str]
    summary_excerpt: str
    confidence: float  # 0.0–1.0 — how well this matches the request
    match_reason: str  # Brief explanation for the LLM to use in its response


class RecommendBooksResponse(BaseModel):
    books: list[RecommendedBook]
    error: str | None = None


async def handle_get_recommendations(
    args: RecommendBooksRequest,
    user_id: str,
) -> RecommendBooksResponse:
    """Handles get_recommendations tool call — delegates to recommend-service."""
    logger.info(
        "get_recommendations",
        user_id=user_id,
        seed_count=len(args.seed_book_ids),
        query=args.query[:50],
    )

    data = await service_client.post(
        service="recommend-service",
        path="/recommendations",
        json={
            "user_id": user_id,
            "seed_book_ids": args.seed_book_ids,
            "query": args.query,
            "limit": args.limit,
        },
    )

    books = [RecommendedBook(**b) for b in data.get("books", [])]
    return RecommendBooksResponse(books=books)
