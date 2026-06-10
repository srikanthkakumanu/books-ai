# Domain Microservices — Claude Code Context

## Purpose

Five bounded-context services, each owning its domain model, database schema, and business logic. These are conventional FastAPI services — no AI, no agents. The MCP server calls these via REST.

## Services and Their Ports

| Service | Port | Owns | DB Schema |
|---------|------|------|-----------|
| book-service | 8010 | Books catalogue | `books`, `book_embeddings` |
| user-service | 8011 | Users, auth, profiles | `users`, `user_sessions` |
| review-service | 8012 | Reviews, ratings | `reviews` |
| recommend-service | 8013 | Recommendation logic | `user_books` (read), `recommendations_log` |
| shelf-service | 8014 | Reading shelves | `shelves`, `shelf_books` |

## Standard Service Layout

Every service follows the same structure:

```
{service-name}/
├── claude.md               # Service-specific context (what this service owns)
├── main.py                 # FastAPI app, router registration, lifespan
├── config.py               # Pydantic Settings — reads from env vars
├── models/
│   ├── __init__.py
│   ├── domain.py           # SQLAlchemy ORM models (the DB tables)
│   └── schemas.py          # Pydantic request/response schemas
├── routers/
│   ├── __init__.py
│   └── {resource}.py       # One router file per resource (e.g. books.py)
├── repositories/
│   ├── __init__.py
│   └── {resource}_repo.py  # DB access layer — ALL SQL lives here
├── utils/
│   ├── __init__.py
│   └── db.py               # SQLAlchemy engine + session factory
├── Dockerfile
├── requirements.txt
└── tests/
    └── test_{resource}.py
```

## Repository Pattern (MANDATORY)

All database access goes through the repository layer. Router handlers never touch the DB directly.

```python
# repositories/book_repo.py
class BookRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, book_id: UUID) -> Book | None:
        result = await self.session.execute(
            select(BookModel).where(BookModel.id == book_id)
        )
        return result.scalar_one_or_none()

    async def search(self, query: str, genre: str | None, limit: int) -> list[Book]:
        stmt = (
            select(BookModel)
            .where(BookModel.search_vector.match(query))
            .limit(limit)
        )
        if genre:
            stmt = stmt.where(BookModel.genres.contains([genre]))
        result = await self.session.execute(stmt)
        return result.scalars().all()
```

## Router Pattern

Routers are thin — they validate input, call the repository, and return the response schema:

```python
# routers/books.py
router = APIRouter(prefix="/books", tags=["books"])

@router.get("/{book_id}", response_model=BookSchema)
async def get_book(
    book_id: UUID,
    repo: BookRepository = Depends(get_book_repo),
) -> BookSchema:
    book = await repo.get_by_id(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return BookSchema.model_validate(book)
```

## HTTP Status Code Conventions

| Scenario | Code |
|----------|------|
| Successful GET | 200 |
| Successful POST (created) | 201 |
| Successful DELETE (no body) | 204 |
| Not found | 404 |
| Validation error | 422 (FastAPI default) |
| Conflict (duplicate) | 409 |
| Unauthorized | 401 |
| Forbidden | 403 |

## Event Publishing

Services publish domain events to Kafka after write operations. Use the helper in `utils/events.py`:

```python
# After successful review creation:
await publish_event("review.created", {
    "review_id": str(review.id),
    "book_id": str(review.book_id),
    "user_id": str(review.user_id),
    "rating": review.rating,
})
```

Event topic naming: `{entity}.{verb}` in past tense (`book.created`, `review.deleted`, `shelf.updated`).

## Health Checks

Every service exposes:
- `GET /health` → `{"status": "ok"}` — liveness probe
- `GET /health/ready` → checks DB connectivity — readiness probe

## Inter-Service Communication

Services do NOT call each other directly. All inter-service orchestration goes through:
1. The agent layer (for user-facing flows)
2. Kafka events (for async side effects)

The only exception is `recommend-service`, which reads `user_books` from `book-service`'s DB via a dedicated read replica — documented in `recommend-service/claude.md`.

## Migrations

Alembic manages all schema changes. Never modify tables manually.

```bash
# In the service directory:
alembic revision --autogenerate -m "add search_vector to books"
alembic upgrade head
```

Migration files live in `db/migrations/` at the repo root, namespaced by service.
