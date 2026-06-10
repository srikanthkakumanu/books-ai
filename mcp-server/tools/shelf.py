"""
mcp-server/tools/shelf.py

add_to_shelf MCP tool.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from mcp_server.utils.http import service_client

logger = structlog.get_logger(__name__)

TOOL_DEFINITION = {
    "name": "add_to_shelf",
    "description": (
        "Add a book to one of the user's reading shelves (e.g. 'Want to read', 'Read', 'Favourites'). "
        "Use this when the user says 'add this to my list', 'save this book', or names a shelf. "
        "Returns the updated shelf and a confirmation message. "
        "If the shelf name is ambiguous, default to 'Want to read'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "user_id":    {"type": "string", "description": "Authenticated user ID from context."},
            "book_id":    {"type": "string", "description": "Book ID to add. Extract from catalogue context or search results."},
            "shelf_name": {
                "type": "string",
                "description": "Name of the shelf. Common values: 'Want to read', 'Currently reading', 'Read', 'Favourites'.",
                "default": "Want to read",
            },
        },
        "required": ["user_id", "book_id"],
    },
}


class AddToShelfRequest(BaseModel):
    user_id: str
    book_id: str
    shelf_name: str = Field(default="Want to read", max_length=100)


class AddToShelfResponse(BaseModel):
    shelf_name: str
    book_title: str
    shelf_book_count: int
    error: str | None = None


async def handle_add_to_shelf(args: AddToShelfRequest, user_id: str) -> AddToShelfResponse:
    """Adds a book to the user's shelf via shelf-service."""
    data = await service_client.post(
        service="shelf-service",
        path="/shelves/add",
        json={"user_id": user_id, "book_id": args.book_id, "shelf_name": args.shelf_name},
    )
    return AddToShelfResponse(**data)
