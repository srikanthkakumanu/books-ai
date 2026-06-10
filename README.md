# Books AI

An AI-powered book discovery and recommendation platform. Users chat naturally — _"recommend something like Dune"_, _"add this to my shelf"_, _"write a review for Project Hail Mary"_ — and an AI agent handles intent routing, semantic retrieval, and tool dispatch.

## Architecture Overview

```text
Browser
  │  SSE stream
  ▼
Next.js UI (port 3000)
  │  REST / SSE  (BFF route — JWT validated here)
  ▼
Agent Orchestrator — LangGraph StateGraph (port 8000)
  │  classify intent → retrieve RAG context → dispatch MCP tool → stream response
  ├──► RAG Pipeline ──► pgvector (PostgreSQL 16)
  └──► MCP Server (port 8001)
         │  REST
         ├──► book-service     (8010)
         ├──► user-service     (8011)
         ├──► review-service   (8012)
         ├──► recommend-service(8013)
         └──► shelf-service    (8014)
```

**Key constraints:**

- Agent layer never calls microservices directly — always through MCP.
- Browser never calls the agent or MCP server directly — always through the Next.js BFF route.
- Services do not call each other; async side effects go through Kafka events.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | Next.js 14 (App Router), TypeScript, Tailwind CSS, SWR |
| Agent | LangGraph `StateGraph`, Anthropic Claude (Haiku + Sonnet), Python 3.12 |
| MCP Server | FastAPI, Model Context Protocol |
| RAG | pgvector, OpenAI `text-embedding-3-small` (1536-dim) |
| Microservices | FastAPI, SQLAlchemy (async), Pydantic v2, Alembic |
| Database | PostgreSQL 16 + pgvector extension |
| Cache / Sessions | Redis |
| Event bus | Kafka |
| Infrastructure | Docker Compose (dev), Kubernetes + Terraform (prod) |

## Repository Layout

```text
books-ai/
├── ui/                    ← Next.js 14 frontend
├── agents/                ← LangGraph orchestrator (FastAPI, port 8000)
├── mcp-server/            ← MCP tool server (FastAPI, port 8001)
├── rag/                   ← RAG pipeline (embedder + retriever + ingestor)
├── services/
│   ├── book-service/      ← Books catalogue (port 8010)
│   ├── user-service/      ← Auth + profiles (port 8011)
│   ├── review-service/    ← Reviews + ratings (port 8012)
│   ├── recommend-service/ ← Recommendation logic (port 8013)
│   └── shelf-service/     ← Reading shelves (port 8014)
├── db/                    ← Alembic migrations, seed data, DDL reference
├── infra/                 ← Docker Compose, Kubernetes manifests, Terraform
├── tests/                 ← Integration and end-to-end test suites
└── docs/                  ← Architecture docs, prompt engineering guide
```

## Prerequisites

- Docker + Docker Compose
- Node.js 20+
- Python 3.12+
- `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, and any other required values
```

### 2. Start backing services (Postgres, Redis, Kafka)

```bash
docker compose -f infra/docker/docker-compose.dev.yml up -d
```

### 3. Run database migrations and seed data

```bash
# From repo root
alembic upgrade head
python db/seeds/seed_dev.py
```

### 4. Start application services

```bash
# Agent orchestrator (hot-reload)
cd agents && uvicorn main:app --reload --port 8000

# MCP server (hot-reload)
cd mcp-server && uvicorn main:app --reload --port 8001

# Domain microservices (repeat for each)
cd services/book-service && uvicorn main:app --reload --port 8010
cd services/user-service  && uvicorn main:app --reload --port 8011
# ... etc.

# Frontend
cd ui && npm install && npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Environment Variables

All secrets are provided via environment variables. Never hardcode them. See `.env.example` for the full list.

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | Claude API (agent calls) |
| `OPENAI_API_KEY` | Embeddings (`text-embedding-3-small`) |
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Cache + session store |
| `KAFKA_BOOTSTRAP_SERVERS` | Event bus |
| `MCP_SERVER_URL` | MCP server base URL (used by agent) |
| `JWT_SECRET` | Token signing key |

## Running Tests

```bash
# Python (all services, agent, RAG)
pytest

# Watch mode
pytest-watch

# Frontend
cd ui && npm test

# RAG evaluation (run manually before releases — not in CI)
pytest tests/integration/test_rag.py --eval
```

## Domain Concepts

| Concept | Description |
| --- | --- |
| **Book** | Core entity: title, author, ISBN, genres (array), summary, cover_url |
| **Shelf** | A named reading list owned by a user (e.g. "Want to read", "Favourites") |
| **Review** | 1–5 star rating + optional text. One per user per book. |
| **Recommendation** | AI-ranked list of books for a query. Not stored permanently. |
| **Embedding** | 1536-dim vector of a book summary chunk, stored in `book_embeddings` |

## Agent Request Flow

1. **Classify** — Haiku classifies intent and runs input guardrails.
2. **Retrieve** — Query is rewritten, embedded, and used for pgvector similarity search (top-5 chunks, threshold 0.70).
3. **Tool dispatch** — If the intent requires an action (search, add to shelf, write review), the agent calls the MCP server.
4. **Respond** — Sonnet generates a streaming response using RAG context + tool results. The persona system prompt block is cached.

Approximate cost per request: ~$0.014 (dominated by the Sonnet response call).

## Development Guidelines

### Migrations

Never edit migration files directly. Always use Alembic:

```bash
cd services/<service-name>
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
```

Migration files live in `db/migrations/`, namespaced by service.

### Adding an MCP Tool

1. Create `mcp-server/tools/{name}.py` with `TOOL_DEFINITION` and `handle_{name}`.
2. Create `mcp-server/schemas/{name}.py` with Request/Response Pydantic models.
3. Register in `mcp-server/tools/__init__.py`.
4. Add a unit test in `tests/unit/mcp/test_{name}.py`.
5. Expose the new tool as a typed method in `agents/tools/mcp_client.py`.

### Coding Standards

**Python** — type hints everywhere, Pydantic v2 schemas, `ruff` + `black` (line length 100), `structlog` (no `print`), `pytest` + `pytest-asyncio`.

**TypeScript** — strict mode, no `any`, ESLint + Prettier, React Server Components by default, `zod` for all runtime validation.

**Both** — no circular imports, every new feature needs unit tests before merge, every new env var must be added to `.env.example`.

### Things to Never Do

- Modify `db/migrations/` files after they have been applied.
- Add environment variables without updating `.env.example`.
- Call microservices directly from the agent layer (use MCP).
- Use `time.sleep()` — use `asyncio.sleep()`.
- Use `SELECT *` in production queries.
- Commit secrets, API keys, or connection strings.
