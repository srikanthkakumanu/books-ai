# Books AI — Prompt Engineering & Token Optimisation Guide

## Overview

This document is the authoritative reference for all prompt engineering decisions in Books AI. Every LLM call in the system has been designed with specific techniques — this guide explains what, why, and how to extend each pattern.

---

## 1. Model Selection Strategy

### The Rule: Use the Cheapest Model That Meets Quality Bar

| Task | Model | Why | Cost Ratio |
|------|-------|-----|------------|
| Intent classification | `claude-haiku-4-5` | One-word output; quality = accuracy, not creativity | 1× |
| Query rewriting | `claude-haiku-4-5` | Short output; structured transformation | 1× |
| RAG re-ranking | `claude-haiku-4-5` | JSON scoring; no prose generation | 1× |
| Final response | `claude-sonnet-4` | Multi-paragraph prose; quality matters to users | ~40× |

**Where this lives:** `agents/prompts/prompt_config.py` — `CLASSIFY_MODEL`, `RESPOND_MODEL`

### Why Not GPT or Gemini?

- Anthropic's prompt caching is the most mature; 90% savings on cached tokens
- XML tags are Claude's native structured output format (trained on XML)
- Tool calling quality is highest in the Claude family for the book advisory use case

---

## 2. Prompt Structure Principles

### 2.1 XML Tags for Structure

Claude is trained on XML-heavy data. Use XML tags for:
- **Sections** within a system prompt: `<task>`, `<rules>`, `<examples>`, `<output_format>`
- **Injected data**: `<catalogue_context>`, `<tool_data>`, `<books>`, `<book>`
- **Output guidance**: `<format>`, `<length>`

**Never use markdown headers** (`##`) in system prompts — Claude parses XML more reliably.

```python
# ✅ Good — XML structure
system = """
<task>Classify the user's message into exactly one intent.</task>
<output_format>Output exactly one word. No punctuation.</output_format>
<examples>
"recommend me a book" → recommend
</examples>
"""

# ❌ Bad — markdown headers (less reliable parsing)
system = """
## Task
Classify the user's message.

## Output Format
One word only.
"""
```

### 2.2 Role Before Instructions

Always establish the persona FIRST. Claude performs better when it has a stable identity before receiving instructions.

```python
# ✅ Good — role first
"""
<persona>
You are Bookwise, an expert book advisor with encyclopaedic reading knowledge.
</persona>

<task>Recommend books based on the user's request...</task>
"""

# ❌ Bad — instructions before identity
"""
Recommend 2-3 books. Be specific. Cite titles and authors.
You are a book advisor.
"""
```

### 2.3 Static Before Dynamic

For prompt caching to work, **static content must come first**. The cache key is based on the prefix of the prompt — any dynamic content after the static prefix doesn't invalidate the cache.

```
[CACHED]  Persona + Core Rules + Format + Task Instruction
[DYNAMIC] <catalogue_context>...</catalogue_context>
[DYNAMIC] <tool_data>...</tool_data>
```

**Where this lives:** `agents/utils/prompt_cache.py` — `split_static_dynamic()`

### 2.4 Positive + Negative Constraints

Always pair every "do this" with the corresponding "don't do this". Claude follows both, and the negative form prevents the most common failure modes.

```python
# ✅ Good — paired constraints
"""
<core_rules>
5. NEVER start your response with "I" — vary your openings.
6. NEVER use hollow phrases: "Great question!", "Certainly!", "Of course!"
7. If context is missing, say so honestly rather than filling with vague suggestions.
</core_rules>
"""
```

### 2.5 Output Format with Examples

Specify format, then give examples. Don't just describe — show.

```python
# ✅ Good — format + example
"""
<output_format>
Output exactly one word. No punctuation, no explanation.
Correct: recommend
Incorrect: "I'd classify this as recommend" or "Recommend."
</output_format>
"""
```

---

## 3. Token Optimisation Techniques

### 3.1 Per-Intent Token Budgets

Don't use a single `max_tokens` for all intents. Response length varies dramatically:

| Intent | max_tokens | Rationale |
|--------|-----------|-----------|
| recommend | 600 | 2–3 paragraphs per book |
| search | 400 | Results list + brief intro |
| review | 500 | Prose analysis, moderate length |
| shelf | 200 | Single confirmation sentence |
| general | 350 | Direct answer, variable length |

**Where this lives:** `agents/prompts/prompt_config.py` — `RESPONSE_BUDGETS`

### 3.2 Assistant Prefill

Inject a partial assistant turn before generation to:
- Skip hollow preamble ("Great question! I'd be happy to help...")
- Steer the response format from the very first token
- Save 5–15 tokens per response

```python
# In the messages array sent to the API:
messages = [
    {"role": "user", "content": "recommend me a sci-fi book"},
    # Prefill — Claude continues from here:
    {"role": "assistant", "content": "Here are some books you'll love based on"},
]
```

**Where this lives:** `agents/prompts/prompt_config.py` — `ResponseBudget.prefill`

⚠️ **Important:** Prepend the prefill text to the response text before returning it to the user — the prefill is in the request, not the response body.

### 3.3 Message History Trimming

Message history grows unboundedly across a session. Without trimming:
- Session 1 (5 turns): ~500 tokens of history
- Session 20 (50 turns): ~5,000 tokens of history per call

The sliding window keeps the last `MESSAGE_HISTORY_MAX_TURNS` turn pairs, hard-capped at `MESSAGE_HISTORY_MAX_TOKENS`.

```python
# Key insight: classification node doesn't need history at all
# Only the latest message is passed to Haiku for intent classification
# This alone saves 500–2000 tokens per call on Haiku

# Response node gets trimmed history (last 10 turns max, 2000 tokens hard cap)
trimmed = trim_message_history(state["messages"])
```

**Where this lives:** `agents/utils/token_utils.py` — `trim_message_history()`

### 3.4 RAG Context Token Budget

RAG context is the most variable part of the prompt. Left unconstrained, it can balloon to thousands of tokens.

Budget enforcement:
- Max 800 tokens for the full `<catalogue_context>` block
- Max 280 chars (~70 tokens) per book summary
- Summary truncated at sentence boundary (not mid-word)
- Max 3 genre tags per book (additional tags add noise)

```python
# In formatter.py
def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Prefers '. ' as truncation point over hard character cut."""
```

**Where this lives:** `rag/utils/formatter.py`

### 3.5 Tool Result Stripping

MCP tools return full database objects. The LLM only needs user-facing fields.

Fields always stripped from tool results before prompt injection:
- `id`, `book_id`, `user_id` — internal IDs
- `created_at`, `updated_at` — timestamps (not useful to LLM)
- `embedding`, `chunk_index` — RAG internals
- `cursor`, `next_page` — pagination state

This typically saves 30–50% of the tool result tokens.

**Where this lives:** `agents/prompts/response.py` — `_serialise_tool_results()`

---

## 4. Prompt Caching

### How It Works

Anthropic caches the prefix of your prompt. On repeated calls with the same prefix, cached tokens cost ~10% of normal input token price.

**Requirements:**
- Sonnet: minimum 2048 tokens in the cached block
- Haiku: minimum 1024 tokens in the cached block
- Content must be byte-identical (even whitespace changes invalidate cache)
- Cache TTL: 5 minutes (refreshed on each cache hit)

### Our Caching Strategy

The `_PERSONA` block in `response.py` is the cached prefix. It contains the persona, core rules, and response format — always identical, ~400 tokens.

The dynamic sections (RAG context, tool data) come AFTER and don't affect the cache key.

```
Request 1:  [PERSONA 400t] [CATALOGUE 350t] [TOOLS 200t]  → cache MISS, cache written
Request 2:  [PERSONA 400t] [CATALOGUE 350t] [TOOLS 200t]  → cache HIT (same user, same turn)
Request 3:  [PERSONA 400t] [DIFFERENT CATALOGUE] [TOOLS]  → cache HIT on PERSONA only
```

### Monitoring Cache Hit Rate

```python
# In respond.py, after every LLM call:
cache_metrics = log_cache_metrics(response.usage)
# Logs: cache_read_tokens, cache_write_tokens, cache_savings_pct

# Target: > 60% savings_pct for active sessions
# If < 30%, investigate: dynamic content may have crept into the static prefix
```

**Where this lives:** `agents/utils/prompt_cache.py`

---

## 5. RAG Query Optimisation

### 5.1 Query Rewriting

Raw user queries are often too terse for good vector similarity. "something cozy" doesn't embed near "cozy mystery amateur detective small town light-hearted".

Rewriting adds implied terms, resolves pronouns, and normalises genre vocabulary.

```
"something cozy"  →  "cozy mystery amateur detective small town light-hearted"
"her best work"   →  "Ursula Le Guin best acclaimed novel"  (resolved from history)
"Dune but darker" →  "dark epic science fiction desert world political intrigue like Dune"
```

**When to skip:** Queries ≥ 8 specific words with no pronouns — already good enough.

**Where this lives:** `agents/utils/query_optimizer.py` — `rewrite_query_for_retrieval()`

### 5.2 HyDE (Hypothetical Document Embedding)

For vague mood/theme queries, generate a hypothetical book description and embed that instead of the query.

Why it works: "a hopeful book about resilience" is a query. "Title: The Lighthouse Year | Author: [fictional] | A young woman rebuilds her life after loss, finding unexpected community and purpose in a small coastal town. Themes: resilience, found family, hope." is a document. The document embedding lives much closer to real book embeddings in vector space.

```python
# Trigger condition (query_optimizer.py):
def should_use_hyde(query: str) -> bool:
    vague_signals = ["something", "a book", "recommend", "mood for", "cozy", ...]
    return any(signal in query.lower() for signal in vague_signals)
```

**When NOT to use:** Specific title/author queries. HyDE adds ~200ms latency from the extra LLM call.

**Where this lives:** `agents/utils/query_optimizer.py` — `generate_hyde_query()`

### 5.3 LLM Re-ranking (Cross-Encoder)

pgvector's cosine similarity ranks by embedding distance, not semantic relevance to the specific query. Re-ranking uses Haiku as a cross-encoder to score each retrieved chunk against the original query.

```
pgvector returns: [Space Opera A, Space Opera B, Science Fantasy C, Hard Sci-Fi D]
User query:       "something like Dune — epic world-building"

Haiku scores:     [Space Opera A: 4, Science Fantasy C: 5, Space Opera B: 3, Hard Sci-Fi D: 2]
Re-ranked:        [Science Fantasy C, Space Opera A, Space Opera B]
                   (Hard Sci-Fi D dropped — below min_score=3)
```

**Cost:** ~150 Haiku tokens per re-ranking call — negligible vs Sonnet response cost.
**When to use:** `recommend` intent only — quality matters most there.

**Where this lives:** `rag/utils/context_ranker.py` — `rerank_chunks()`

---

## 6. MCP Tool Description Engineering

Tool descriptions are prompts read by Claude on every tool selection decision. Write them accordingly.

### Rules for Tool Descriptions

1. **Verb-first:** "Search the catalogue..." not "This tool searches..."
2. **When-to-use guidance:** Explicit "Use this when..." prevents wrong tool selection
3. **When-NOT-to-use:** Disambiguation between similar tools prevents confusion
4. **Output preview:** "Returns a ranked list with confidence scores..." calibrates agent expectations
5. **Field descriptions with examples:** Don't just say "string" — say "e.g. 'science fiction', 'mystery'"
6. **Enum constraints where possible:** Reduces hallucinated field values

```python
# ✅ Good tool description
"description": (
    "Search the book catalogue by title, author name, or keyword. "
    "Use this when: the user names a specific book or author. "
    "Do NOT use this for open-ended recommendations — use get_recommendations instead."
)

# ❌ Bad tool description  
"description": "Search for books."
```

**Where this lives:** `mcp-server/tools/*.py` — `TOOL_DEFINITION["description"]`

---

## 7. Guardrails

### Input Guardrails (before LLM call)

Applied in `classify_intent` before the first LLM call — catches injection attempts cheaply.

- Prompt injection patterns (`ignore previous instructions`, injected `<system>` tags)
- Off-scope content (violence, weapons — not relevant to a book platform)
- PII detection (logged but not blocked — "credit card number" might be a book title)

### Output Guardrails (after LLM response)

Applied in `generate_response` after the Sonnet call.

- Empty/too-short response detection
- Unexpected refusal pattern detection (signals misconfigured system prompt)

**Where this lives:** `agents/utils/guardrails.py`

---

## 8. Token Budget Reference

| Component | Target | Hard Limit | Notes |
|-----------|--------|-----------|-------|
| System prompt (static) | 400 tokens | 600 tokens | Includes persona + rules + format + task |
| RAG context | 500 tokens | 800 tokens | ~5 books at 70 tokens each |
| Tool results | 200 tokens | 300 tokens | Stripped of internal fields |
| Message history | 1000 tokens | 2000 tokens | Sliding window, last 10 turns |
| Total input | ~2100 tokens | 3700 tokens | Well within Sonnet's 200k window |
| Output (recommend) | 600 tokens | 600 tokens | Per ResponseBudget |
| Output (shelf) | 200 tokens | 200 tokens | Per ResponseBudget |

---

## 9. Adding a New Prompt

Checklist when adding or modifying any LLM call:

- [ ] Is this call using the cheapest model that meets quality bar?
- [ ] Is the system prompt using XML tags for structure (not markdown)?
- [ ] Does it have positive AND negative constraints?
- [ ] Does it have concrete examples for expected output?
- [ ] Is `max_tokens` set to the tightest reasonable budget?
- [ ] Is `temperature` set appropriately (0.0 for deterministic, 0.7 for creative)?
- [ ] For the response node: does it use the cached system prefix pattern?
- [ ] Is there an assistant prefill to skip hollow preamble?
- [ ] Are there unit tests covering the classification/format of the output?
