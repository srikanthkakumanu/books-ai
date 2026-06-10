"""
mcp-server/tools/__init__.py

Central registry of all MCP tools.

When adding a new tool:
1. Create tools/{name}.py with TOOL_DEFINITION, a Request schema, and handle_{name}
2. Import and register it here
3. Add unit test in tests/unit/mcp/test_{name}.py
4. Update agents/tools/mcp_client.py
"""

from mcp_server.tools.recommend import (
    TOOL_DEFINITION as RECOMMEND_DEF,
    RecommendBooksRequest,
    handle_get_recommendations,
)
from mcp_server.tools.review import (
    TOOL_DEFINITION as REVIEW_DEF,
    WriteReviewRequest,
    handle_write_review,
)
from mcp_server.tools.search import (
    TOOL_DEFINITION as SEARCH_DEF,
    SearchBooksRequest,
    handle_search_books,
)
from mcp_server.tools.shelf import (
    TOOL_DEFINITION as SHELF_DEF,
    AddToShelfRequest,
    handle_add_to_shelf,
)

# Registry maps tool_name → {definition, handler, schema}
# main.py uses this for routing and OpenAPI-style tool listing
TOOL_REGISTRY: dict[str, dict] = {
    "search_books": {
        "definition": SEARCH_DEF,
        "handler": handle_search_books,
        "schema": SearchBooksRequest,
    },
    "get_recommendations": {
        "definition": RECOMMEND_DEF,
        "handler": handle_get_recommendations,
        "schema": RecommendBooksRequest,
    },
    "write_review": {
        "definition": REVIEW_DEF,
        "handler": handle_write_review,
        "schema": WriteReviewRequest,
    },
    "add_to_shelf": {
        "definition": SHELF_DEF,
        "handler": handle_add_to_shelf,
        "schema": AddToShelfRequest,
    },
}
