# Database Layer — Claude Code Context

## Purpose

PostgreSQL 16 with the `pgvector` extension. This directory contains migrations, seed data, and schema documentation. Each service owns its own schema (PostgreSQL schema, not just tables) to enforce bounded-context isolation at the DB level.

## Directory Layout

```
db/
├── claude.md                           # THIS FILE
├── migrations/
│   ├── env.py                          # Alembic env — multi-schema aware
│   ├── script.py.mako                  # Migration template
│   ├── books/
│   │   ├── 001_create_books.py
│   │   ├── 002_add_search_vector.py
│   │   └── 003_add_book_embeddings.py
│   ├── users/
│   │   └── 001_create_users.py
│   ├── reviews/
│   │   └── 001_create_reviews.py
│   ├── shelves/
│   │   └── 001_create_shelves.py
│   └── recommendations/
│       └── 001_create_recommendations_log.py
├── schemas/
│   └── schema.sql                      # Full DDL for reference (generated, do not edit)
└── seeds/
    ├── seed_dev.py                     # Development seed (1000 books, test users)
    └── books_fixture.json              # 1000 book records for seeding
```

## Schema Ownership

Each service connects to a dedicated PostgreSQL schema. Isolation enforced by the connection user.

```
books_schema      → managed by book-service
users_schema      → managed by user-service
reviews_schema    → managed by review-service
shelves_schema    → managed by shelf-service
recommendations_schema → managed by recommend-service
```

## Master DDL

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;           -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";      -- UUID generation
CREATE EXTENSION IF NOT EXISTS pg_trgm;          -- Trigram similarity for fuzzy search

-- Create per-service schemas
CREATE SCHEMA IF NOT EXISTS books_schema;
CREATE SCHEMA IF NOT EXISTS users_schema;
CREATE SCHEMA IF NOT EXISTS reviews_schema;
CREATE SCHEMA IF NOT EXISTS shelves_schema;
CREATE SCHEMA IF NOT EXISTS recommendations_schema;

-- ============================================================
-- books_schema
-- ============================================================
CREATE TABLE books_schema.books (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title           TEXT NOT NULL,
    author          TEXT NOT NULL,
    isbn            VARCHAR(13) UNIQUE,
    genres          TEXT[]          NOT NULL DEFAULT '{}',
    summary         TEXT,
    cover_url       TEXT,
    published_date  DATE,
    page_count      INTEGER,
    language        VARCHAR(5)      DEFAULT 'en',
    -- Full-text search vector (auto-maintained by trigger)
    search_vector   TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english',
            coalesce(title, '') || ' ' ||
            coalesce(author, '') || ' ' ||
            coalesce(summary, '')
        )
    ) STORED,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_books_search ON books_schema.books USING gin(search_vector);
CREATE INDEX idx_books_genres ON books_schema.books USING gin(genres);
CREATE INDEX idx_books_author ON books_schema.books (author);

-- Embedding chunks for RAG
CREATE TABLE books_schema.book_embeddings (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    book_id         UUID            NOT NULL REFERENCES books_schema.books(id) ON DELETE CASCADE,
    chunk_index     INTEGER         NOT NULL,
    summary_chunk   TEXT            NOT NULL,
    genre_tags      TEXT,
    embedding       VECTOR(1536),   -- text-embedding-3-small dimensions
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE (book_id, chunk_index)
);

-- ivfflat index for approximate nearest neighbour (fast, good recall)
-- Tune lists = sqrt(row_count). Switch to hnsw for > 1M rows.
CREATE INDEX idx_book_embeddings_vec
    ON books_schema.book_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============================================================
-- users_schema
-- ============================================================
CREATE TABLE users_schema.users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email           TEXT            UNIQUE NOT NULL,
    name            TEXT,
    avatar_url      TEXT,
    preferences     JSONB           DEFAULT '{}',  -- genre prefs, UI settings
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE TABLE users_schema.user_sessions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID            NOT NULL REFERENCES users_schema.users(id) ON DELETE CASCADE,
    token_hash      TEXT            NOT NULL,  -- store hash, never plaintext
    expires_at      TIMESTAMPTZ     NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sessions_user_id ON users_schema.user_sessions (user_id);
CREATE INDEX idx_sessions_expires ON users_schema.user_sessions (expires_at);

-- ============================================================
-- reviews_schema
-- ============================================================
CREATE TABLE reviews_schema.reviews (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    book_id         UUID            NOT NULL,    -- FK to books_schema.books (cross-schema ref)
    user_id         UUID            NOT NULL,    -- FK to users_schema.users (cross-schema ref)
    rating          SMALLINT        NOT NULL CHECK (rating BETWEEN 1 AND 5),
    body            TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE (book_id, user_id)                   -- one review per user per book
);

CREATE INDEX idx_reviews_book_id ON reviews_schema.reviews (book_id);
CREATE INDEX idx_reviews_user_id ON reviews_schema.reviews (user_id);

-- ============================================================
-- shelves_schema
-- ============================================================
CREATE TABLE shelves_schema.shelves (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID            NOT NULL,
    name            TEXT            NOT NULL,
    description     TEXT,
    is_default      BOOLEAN         DEFAULT FALSE,  -- "Want to read", "Read" are defaults
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE TABLE shelves_schema.shelf_books (
    shelf_id        UUID            NOT NULL REFERENCES shelves_schema.shelves(id) ON DELETE CASCADE,
    book_id         UUID            NOT NULL,
    added_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (shelf_id, book_id)
);

-- ============================================================
-- recommendations_schema
-- ============================================================
CREATE TABLE recommendations_schema.recommendations_log (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID            NOT NULL,
    query           TEXT            NOT NULL,
    book_ids        UUID[]          NOT NULL,    -- ordered by relevance
    rag_sources     UUID[]          NOT NULL,    -- book IDs used as RAG context
    model           TEXT            NOT NULL,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- LLM cost tracking (all services write here)
CREATE TABLE recommendations_schema.usage_events (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID            NOT NULL,
    service         TEXT            NOT NULL,   -- 'agent', 'rag', etc.
    intent          TEXT,
    model           TEXT            NOT NULL,
    prompt_tokens   INTEGER         NOT NULL,
    completion_tokens INTEGER       NOT NULL,
    cost_usd        NUMERIC(10, 6),             -- computed and stored for billing
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_usage_user_id ON recommendations_schema.usage_events (user_id, created_at DESC);
```

## Migration Rules

1. Never modify a migration file after it has been applied to any environment
2. Always provide a `downgrade()` function — even if it's a no-op with a comment explaining why
3. Destructive changes (DROP COLUMN, DROP TABLE) require a 3-step migration: (1) stop writing, (2) remove from code, (3) then drop
4. Adding columns: always `nullable=True` or with a `server_default` — never non-nullable without a default
5. Index creation in production: use `CREATE INDEX CONCURRENTLY` (pass `postgresql_concurrently=True` in Alembic)

## Performance Rules

- Paginate all list endpoints: use `LIMIT`/`OFFSET` with a max of 100 rows
- Never use `SELECT *` — always name columns
- Always use connection pooling (PgBouncer in transaction mode for writes, session mode for long connections)
- `EXPLAIN ANALYZE` any query that touches > 10k rows before shipping
