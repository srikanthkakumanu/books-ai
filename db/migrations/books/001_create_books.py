"""
db/migrations/books/001_create_books.py

Initial migration: create books_schema with books and book_embeddings tables.

Alembic revision: 001_books_initial
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001_books_initial"
down_revision = None
branch_labels = ("books",)
depends_on = None


def upgrade() -> None:
    # Enable extensions (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.execute("CREATE SCHEMA IF NOT EXISTS books_schema")

    # books table
    op.create_table(
        "books",
        sa.Column("id", postgresql.UUID(as_uuid=True),
                  server_default=sa.text("uuid_generate_v4()"), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("author", sa.Text(), nullable=False),
        sa.Column("isbn", sa.String(13), nullable=True),
        sa.Column("genres", postgresql.ARRAY(sa.Text()), nullable=False,
                  server_default="{}"),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("cover_url", sa.Text(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("language", sa.String(5), nullable=False, server_default="en"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("isbn", name="uq_books_isbn"),
        schema="books_schema",
    )

    # Generated search_vector column (PostgreSQL 12+)
    op.execute("""
        ALTER TABLE books_schema.books
        ADD COLUMN search_vector TSVECTOR
        GENERATED ALWAYS AS (
            to_tsvector('english',
                coalesce(title, '') || ' ' ||
                coalesce(author, '') || ' ' ||
                coalesce(summary, '')
            )
        ) STORED
    """)

    # GIN index on search_vector for fast full-text search
    op.create_index(
        "idx_books_search",
        "books",
        ["search_vector"],
        schema="books_schema",
        postgresql_using="gin",
    )

    # GIN index on genres array for fast genre filtering
    op.create_index(
        "idx_books_genres",
        "books",
        ["genres"],
        schema="books_schema",
        postgresql_using="gin",
    )

    # book_embeddings table (RAG)
    op.create_table(
        "book_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True),
                  server_default=sa.text("uuid_generate_v4()"), primary_key=True),
        sa.Column("book_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("summary_chunk", sa.Text(), nullable=False),
        sa.Column("genre_tags", sa.Text(), nullable=True),
        # pgvector VECTOR type — 1536 dims for text-embedding-3-small
        # Alembic doesn't have native vector support; use raw SQL type
        sa.Column("embedding", sa.Text(), nullable=True),  # overridden below
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["book_id"], ["books_schema.books.id"],
            ondelete="CASCADE",
            name="fk_book_embeddings_book_id",
        ),
        sa.UniqueConstraint("book_id", "chunk_index", name="uq_book_embeddings_chunk"),
        schema="books_schema",
    )

    # Override embedding column type to VECTOR(1536)
    op.execute(
        "ALTER TABLE books_schema.book_embeddings "
        "ALTER COLUMN embedding TYPE vector(1536) "
        "USING embedding::vector(1536)"
    )

    # ivfflat approximate nearest-neighbour index
    # CONCURRENTLY not supported in transactions — run via Alembic post-migration
    op.execute("""
        CREATE INDEX idx_book_embeddings_vec
        ON books_schema.book_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS books_schema.book_embeddings CASCADE")
    op.execute("DROP TABLE IF EXISTS books_schema.books CASCADE")
    op.execute("DROP SCHEMA IF EXISTS books_schema CASCADE")
