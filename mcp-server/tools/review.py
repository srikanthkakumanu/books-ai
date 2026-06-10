"""
mcp-server/tools/review.py

write_review and get_reviews MCP tools.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from mcp_server.utils.http import service_client

logger = structlog.get_logger(__name__)

TOOL_DEFINITION = {
    "name": "write_review",
    "description": (
        "Save a book review written by the user. "
        "Use this ONLY when the user explicitly provides review text or a rating to save. "
        "Do NOT use this just to help them write a review — that requires no tool call. "
        "Returns confirmation and the saved review."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "user_id":  {"type": "string", "description": "Authenticated user ID from context."},
            "book_id":  {"type": "string", "description": "Book ID being reviewed. Extract from catalogue context or search results."},
            "rating":   {"type": "integer", "description": "Star rating 1–5.", "minimum": 1, "maximum": 5},
            "body":     {"type": "string", "description": "The review text. May be null if user only provides a rating."},
        },
        "required": ["user_id", "book_id", "rating"],
    },
}


class WriteReviewRequest(BaseModel):
    user_id: str
    book_id: str
    rating: int = Field(ge=1, le=5)
    body: str | None = None


class WriteReviewResponse(BaseModel):
    review_id: str
    book_title: str
    rating: int
    error: str | None = None


async def handle_write_review(args: WriteReviewRequest, user_id: str) -> WriteReviewResponse:
    """Saves a user review via review-service."""
    data = await service_client.post(
        service="review-service",
        path="/reviews",
        json={"user_id": user_id, "book_id": args.book_id,
              "rating": args.rating, "body": args.body},
    )
    return WriteReviewResponse(**data)
