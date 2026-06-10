# Agent Orchestrator — Claude Code Context

## Purpose

The brain of the system. A LangGraph `StateGraph` that receives user messages, classifies intent, retrieves RAG context, dispatches MCP tool calls, and streams the final response. Runs as a FastAPI application on port 8000.

## Directory Layout

```
agents/
├── claude.md                   # THIS FILE
├── main.py                     # FastAPI app entry point
├── orchestrator.py             # LangGraph StateGraph (entry point for graph)
├── state.py                    # AgentState TypedDict — single source of truth
├── nodes/
│   ├── classify.py             # Haiku intent classification + guardrail
│   ├── retrieve.py             # Query optimisation + RAG retrieval
│   ├── tool_dispatch.py        # MCP tool routing and calling
│   ├── respond.py              # Sonnet response generation + caching
│   └── error_handler.py        # Graceful error recovery
├── prompts/
│   ├── __init__.py             # Public API — import from here
│   ├── prompt_config.py        # ALL token budgets, model names, temperatures
│   ├── intent.py               # Intent classification system prompt (XML-structured)
│   └── response.py             # Response system prompt builder + tool result serialiser
├── tools/
│   └── mcp_client.py           # Typed MCP client (retry + tracing + timeout)
└── utils/
    ├── __init__.py
    ├── cost_tracker.py         # Token usage → USD cost → DB
    ├── guardrails.py           # Input sanitation + output validation
    ├── prompt_cache.py         # Anthropic cache_control block builder
    ├── query_optimizer.py      # Query rewriting + HyDE generation
    └── token_utils.py          # tiktoken counting + history trimming
```

## LLM Call Inventory

Every LLM call in the system is documented here. When adding a new call, add it to this table.

| Location | Model | max_tokens | Purpose | Caching? |
|----------|-------|-----------|---------|---------|
| `nodes/classify.py` | Haiku | 10 | Intent classification | No (< min threshold) |
| `utils/query_optimizer.py:rewrite_query` | Haiku | 150 | Query expansion | No |
| `utils/query_optimizer.py:generate_hyde` | Haiku | 150 | HyDE document generation | No |
| `rag/utils/context_ranker.py` | Haiku | 150 | Re-ranking scoring | No |
| `nodes/respond.py` | Sonnet | 200–600 | User-facing response | YES — persona block |

**Total worst-case cost per request (Haiku + Sonnet):**
- Haiku calls: ~500 input + ~20 output = ~$0.0005
- Sonnet call: ~2100 input + ~500 output = ~$0.014
- Cached Sonnet input: ~400 × 0.10 = $0.00012 savings
- **Approximate total: ~$0.014 per user request**

## State Schema (CRITICAL — read before modifying)

All state lives in `AgentState` in `state.py`. Never add ad-hoc keys. Every new field needs:
1. A type annotation in `AgentState`
2. A comment indicating which node populates it
3. An update to this claude.md

## Prompt Engineering Rules

**READ `docs/prompt-engineering-guide.md` before modifying any prompt.**

Short rules:
- System prompts use XML tags, not markdown headers
- `CLASSIFY_MODEL = Haiku`, `RESPOND_MODEL = Sonnet` — never swap these
- All budgets in `prompt_config.py` — never inline `max_tokens=X` in node code
- Static prompt content goes BEFORE dynamic content (cache prefix stability)
- Prefill in `ResponseBudget.prefill` for every intent that has a predictable opener
- Input guardrails always run in `classify_intent` before any LLM call

## Error Handling

- Every node returns a partial state dict — never raises
- `tool_error` in state routes to `error_handler` node
- `error_handler` produces a user-friendly message, never exposes internals
- Cost tracking is always in a try/except — billing failure ≠ response failure

## Testing

- Unit tests mock `_client.messages.create` (the Anthropic SDK call)
- `test_prompt_engineering.py` tests structural correctness of all prompts
- Integration tests run the full graph with a mock MCP server
