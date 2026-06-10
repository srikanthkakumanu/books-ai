# Tests — Claude Code Context

## Purpose

Three test layers following the standard testing pyramid: unit (fast, isolated), integration (real DB/cache), and e2e (full stack via browser or HTTP).

## Directory Layout

```
tests/
├── claude.md
├── conftest.py                         # Shared fixtures (DB session, HTTP client, mocks)
├── pytest.ini                          # asyncio_mode = "auto", markers
├── unit/
│   ├── agents/
│   │   ├── test_classify_node.py       # Intent classification logic
│   │   ├── test_retrieve_node.py       # RAG context retrieval
│   │   ├── test_tool_dispatch.py       # MCP tool routing
│   │   └── test_orchestrator.py        # Full graph with mocked tools
│   ├── rag/
│   │   ├── test_chunker.py             # Text chunking
│   │   ├── test_formatter.py           # Context string formatting
│   │   └── test_retriever.py           # Retrieval with mocked DB
│   └── mcp/
│       ├── test_search_tool.py
│       ├── test_recommend_tool.py
│       └── test_review_tool.py
├── integration/
│   ├── test_book_service.py            # book-service against real PostgreSQL
│   ├── test_review_service.py
│   ├── test_rag_pipeline.py            # Full embed → search against pgvector
│   └── test_orchestrator.py            # Agent + MCP + real services (no LLM)
└── e2e/
    ├── test_chat_flow.py               # Full chat: UI → agent → services
    └── test_book_search.py             # Search, detail page, add to shelf
```

## Unit Test Rules

- No real HTTP calls, no real DB connections
- Mock LLM responses using `unittest.mock.AsyncMock` or `pytest-mock`
- Mock MCP client and RAG retriever at the boundary
- Each test file tests one module — same name as the module being tested
- Test method naming: `test_{scenario}_{expected_outcome}`
  - Good: `test_classify_recommend_intent_returns_recommend`
  - Bad: `test_classify_1`, `testClassify`

## Integration Test Rules

- Require `INTEGRATION=true` env var to run (excluded from fast CI)
- Use a dedicated test database: `DATABASE_URL=.../{service}_test`
- Truncate tables in a `setup`/`teardown` fixture — never assume clean state
- Never mock the DB in integration tests — that defeats the purpose
- Do mock: the LLM (Anthropic API), external APIs (OpenAI embeddings)

## Fixture Patterns

```python
# conftest.py
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine(TEST_DATABASE_URL)
    async with AsyncSession(engine) as session:
        yield session
        await session.rollback()    # always rollback after test

@pytest.fixture
def mock_mcp_client(mocker):
    return mocker.AsyncMock(spec=MCPClient)

@pytest.fixture
def mock_embedder(mocker):
    mock = mocker.AsyncMock(spec=OpenAIEmbedder)
    mock.embed.return_value = [0.1] * 1536  # 1536-dim zero vector
    return mock
```

## Coverage Requirements

- Unit tests: 80% line coverage minimum (enforced in CI)
- Integration tests: critical paths only (recommend flow, review creation, RAG retrieval)
- E2E tests: smoke test only in CI, full suite before release

## Running Tests

```bash
# Fast unit tests (no external deps)
pytest tests/unit/ -v

# Integration tests (requires Docker Compose running)
INTEGRATION=true pytest tests/integration/ -v

# With coverage report
pytest tests/unit/ --cov=agents --cov=rag --cov=mcp-server --cov-report=html

# E2E (requires full stack)
pytest tests/e2e/ -v --base-url=http://localhost:3000
```

## LLM Eval Tests (not pytest)

RAG quality is evaluated separately using RAGAS. Run manually before releases:

```bash
cd tests && python eval/run_ragas.py --dataset eval/rag_eval_set.json
```

Tracks: `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`. Alert if any metric drops > 5% from baseline.
