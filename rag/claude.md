# RAG Pipeline — Claude Code Context

## Purpose

Retrieval-Augmented Generation pipeline. Embeds user queries and finds semantically relevant book summaries from the vector database, then formats them as LLM prompt context. Also handles offline ingestion of new book data.

## Directory Layout

```
rag/
├── claude.md                   # THIS FILE
├── retriever.py                # Main entry point: query → context string
├── embedders/
│   ├── __init__.py
│   └── openai_embedder.py      # text-embedding-3-small wrapper
├── retrievers/
│   ├── __init__.py
│   └── pgvector_retriever.py   # cosine similarity search against pgvector
├── ingestors/
│   ├── __init__.py
│   ├── book_ingestor.py        # Chunk + embed + upsert books
│   └── pipeline.py             # Orchestrates ingestion jobs
└── utils/
    ├── __init__.py
    ├── chunker.py              # Splits book summaries into overlapping chunks
    ├── formatter.py            # Formats retrieval results as prompt context
    └── db_factory.py           # Returns pgvector or Qdrant client based on config
```

## Retrieval Flow

```
query string
    ↓
openai_embedder.embed(query)  → 1536-dim float[]
    ↓
pgvector_retriever.search(embedding, top_k=5)  → list[BookChunk]
    ↓
formatter.to_context_string(chunks)  → str
    ↓
returned to agent as rag_context
```

## Embedding Model

- Model: `text-embedding-3-small` (OpenAI) — 1536 dimensions
- Batch size: 100 texts per API call during ingestion
- Cache embeddings in Redis with TTL=24h for repeated identical queries
- Never embed on every agent call if the query was recently seen

## Chunking Strategy

Book summaries are chunked before embedding. Settings in `utils/chunker.py`:

- Chunk size: 512 tokens
- Overlap: 64 tokens (prevents context loss at boundaries)
- Each chunk stores: `book_id`, `chunk_index`, `text`, `genre_tags`
- Use `tiktoken` (cl100k_base encoding) for token counting — never character counting

## pgvector Query Pattern

```sql
SELECT
    be.book_id,
    b.title,
    b.author,
    b.genres,
    be.summary_chunk,
    1 - (be.embedding <=> $1::vector) AS similarity
FROM book_embeddings be
JOIN books b ON b.id = be.book_id
WHERE 1 - (be.embedding <=> $1::vector) > 0.7  -- similarity threshold
ORDER BY be.embedding <=> $1::vector
LIMIT $2;
```

- Always filter by similarity threshold (0.7 default) — don't return irrelevant results
- Index type: `ivfflat` with `lists = 100` (tune to `sqrt(row_count)`)
- For production > 1M rows, switch to `hnsw` index

## Context Formatter

The formatter produces a string that is injected verbatim into the agent's system prompt. Format:

```
Relevant books from the catalogue:
1. "The Name of the Wind" by Patrick Rothfuss [genres: fantasy, epic]
   Summary: A young man grows up to be the most notorious wizard his world has ever seen...

2. "A Wizard of Earthsea" by Ursula K. Le Guin [genres: fantasy, classic]
   Summary: A young boy with unusual magical ability attends a school for wizards...
```

Never include raw book IDs in the context string — the LLM doesn't need them, and they waste tokens.

## Ingestion Pipeline

The ingestion pipeline (`ingestors/pipeline.py`) is an **offline job**, not called at query time. It runs:
- On new book data import
- Nightly to re-embed books with updated summaries
- Triggered by `book.updated` Kafka event

Idempotent: uses `ON CONFLICT (book_id, chunk_index) DO UPDATE` in the upsert.

## Similarity Threshold Tuning

If retrieval results feel irrelevant, adjust the threshold in `RAG_SIMILARITY_THRESHOLD` env var. Default: `0.70`. Range: `0.60` (broader) to `0.85` (stricter).

## Testing

- Unit tests mock the embedder and DB client
- Integration test in `tests/integration/test_rag.py` uses a test PostgreSQL instance with pgvector
- Eval metrics tracked with RAGAS: `context_precision`, `context_recall`, `faithfulness`
- Never run RAGAS evals in CI — they're expensive; run manually before releases
