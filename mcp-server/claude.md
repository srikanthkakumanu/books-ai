# MCP Server — Claude Code Context

## Purpose

Exposes typed, discoverable tools to the AI agent via the Model Context Protocol (MCP). Acts as the anti-corruption layer between the non-deterministic agent world and the deterministic microservice world.

Think of each MCP tool as: a strongly-typed REST endpoint + an OpenAPI description that the LLM reads at runtime.

## Directory Layout

```
mcp-server/
├── claude.md                   # THIS FILE
├── main.py                     # FastAPI app + MCP endpoint registration
├── config.py                   # Settings (pydantic-settings)
├── tools/
│   ├── __init__.py             # Registers all tools with the MCP registry
│   ├── search.py               # search_books tool
│   ├── recommend.py            # get_recommendations tool
│   ├── review.py               # write_review, get_reviews tools
│   └── shelf.py                # add_to_shelf, get_shelf, remove_from_shelf tools
├── schemas/
│   ├── __init__.py
│   ├── search.py               # SearchRequest / SearchResponse
│   ├── recommend.py            # RecommendRequest / RecommendResponse
│   ├── review.py               # ReviewRequest / ReviewResponse
│   └── shelf.py                # ShelfRequest / ShelfResponse
└── middleware/
    ├── __init__.py
    ├── auth.py                 # Validates caller JWT (agent must authenticate)
    ├── rate_limit.py           # Per-user rate limiting via Redis
    └── observability.py        # OpenTelemetry + structured logging
```

## Tool Definition Pattern

Every tool follows the same structure. Do not deviate from this pattern:

```python
# tools/search.py

TOOL_DEFINITION = {
    "name": "search_books",
    "description": (
        "Search the book catalogue by title, author, or keyword. "
        "Returns a list of matching books with title, author, genres, and a summary excerpt. "
        "Use this when the user wants to find a specific book or browse by topic."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — title, author name, or keyword"
            },
            "genre": {
                "type": "string",
                "description": "Optional genre filter (e.g. 'science fiction', 'mystery')"
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 10, max 50)",
                "default": 10
            }
        },
        "required": ["query"]
    }
}

async def handle_search_books(args: SearchRequest, user_id: str) -> SearchResponse:
    """
    Handles search_books MCP tool call.

    Validates args via Pydantic (done by caller), delegates to book-service,
    and returns structured response.
    """
    ...
```

## Naming Conventions

| Item | Convention | Example |
|------|-----------|---------|
| Tool name | `snake_case` verb_noun | `search_books`, `get_recommendations` |
| Tool file | `snake_case.py` | `search.py`, `recommend.py` |
| Request schema | `{Verb}{Noun}Request` | `SearchBooksRequest` |
| Response schema | `{Verb}{Noun}Response` | `SearchBooksResponse` |
| Handler function | `handle_{tool_name}` | `handle_search_books` |

## Schema Rules

- All schemas are Pydantic v2 `BaseModel` classes in `schemas/`
- Every field must have a `description` — the LLM reads this
- Use `Field(description=..., examples=[...])` for all fields
- Enums over freeform strings wherever possible
- Response schemas must include an `error: str | None = None` field — never raise exceptions to the agent; return structured errors

## Downstream Service Calls

Each tool handler calls exactly one microservice via the `ServiceClient` helper in `utils/http.py`:

```python
result = await service_client.get(
    service="book-service",
    path=f"/books/search",
    params={"q": args.query, "genre": args.genre, "limit": args.limit},
)
```

Service URLs are resolved from environment variables — never hardcoded.

## Authentication

The MCP server authenticates callers using a short-lived JWT issued by the agent layer. Middleware in `middleware/auth.py` validates this on every request. The `user_id` claim from the JWT is passed to every handler — never trust `user_id` from the tool args body.

## Adding a New Tool

1. Create `tools/{name}.py` with `TOOL_DEFINITION` dict and `handle_{name}` async function
2. Create `schemas/{name}.py` with Request and Response Pydantic models
3. Register in `tools/__init__.py`: `from .{name} import TOOL_DEFINITION, handle_{name}`
4. Add unit test in `tests/unit/mcp/test_{name}.py`
5. Update `agents/tools/mcp_client.py` to expose the new tool as a typed method
