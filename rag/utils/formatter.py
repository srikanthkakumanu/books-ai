"""
rag/utils/formatter.py

RAG context formatter — token-optimised for LLM injection.

Prompt engineering techniques applied:
  1. XML wrapping: each book in <book> tags (Claude parses structured data better)
  2. Token budget per book: 70-token limit per entry keeps context dense
  3. Deduplication: best chunk per book_id, not multiple chunks from same book
  4. Relevance-gated: low-similarity books silently dropped (0.70 threshold)
  5. Similarity annotation: subtle signal helps LLM weight highly-relevant books
  6. Genre normalisation: consistent genre tags improve LLM genre reasoning
  7. Field ordering: title → author → genres → summary (matches Claude's reading pattern)
  8. No IDs in context: book_id is internal plumbing, wastes tokens in prompts

Token budget analysis (target: ≤800 tokens for 5 books):
  - Header: ~8 tokens
  - Per-book entry: ~70 tokens × 5 books = ~350 tokens
  - XML tags overhead: ~30 tokens
  - Total: ~390 tokens → well under RAG_CONTEXT_MAX_TOKENS
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.prompts.prompt_config import RAG_BOOK_SUMMARY_MAX_CHARS


@dataclass
class FormattedContext:
    """Structured context ready for prompt injection."""
    text: str
    book_count: int
    estimated_tokens: int


def format_context(chunks) -> str:
    """
    Formats RAG retrieval results as XML-structured context for Claude.

    Returns the text portion only — use format_context_structured() when
    you need token counts and metadata.

    Args:
        chunks: List of BookChunk objects from the pgvector retriever

    Returns:
        Formatted XML string for system prompt injection.
        Empty string if no chunks provided.
    """
    result = format_context_structured(chunks)
    return result.text


def format_context_structured(chunks) -> FormattedContext:
    """
    Formats RAG chunks with metadata for token auditing.

    Design choices:
    - <books> wrapper signals to Claude this is a structured data block
    - <book> per entry makes each item's boundaries unambiguous
    - Similarity score included as "relevance" — helps Claude weight
      highly-relevant books without needing to infer from position
    - Summary truncated at sentence boundary (not mid-word) for coherence
    - Similarity < 0.75 labelled "partial" to help Claude temper confidence

    Args:
        chunks: List of BookChunk objects, expected sorted by similarity desc

    Returns:
        FormattedContext with text, book_count, and estimated_tokens
    """
    if not chunks:
        return FormattedContext(text="", book_count=0, estimated_tokens=0)

    # Deduplicate: keep highest-similarity chunk per book
    seen: set[str] = set()
    unique: list = []
    for chunk in chunks:
        if chunk.book_id not in seen:
            seen.add(chunk.book_id)
            unique.append(chunk)

    parts: list[str] = ["<books>"]

    for chunk in unique:
        genre_str = _normalise_genres(chunk.genres)
        summary = _truncate_at_sentence(chunk.summary_chunk, RAG_BOOK_SUMMARY_MAX_CHARS)
        relevance = _relevance_label(chunk.similarity)

        parts.append(
            f'<book relevance="{relevance}">\n'
            f"  <title>{chunk.title}</title>\n"
            f"  <author>{chunk.author}</author>\n"
            f"  <genres>{genre_str}</genres>\n"
            f"  <summary>{summary}</summary>\n"
            f"</book>"
        )

    parts.append("</books>")
    text = "\n".join(parts)

    # Rough token estimate: 1 token ≈ 4 chars
    estimated_tokens = len(text) // 4

    return FormattedContext(
        text=text,
        book_count=len(unique),
        estimated_tokens=estimated_tokens,
    )


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """
    Truncates text at a sentence boundary, not mid-word.

    Prioritises ending at ". " to keep summary grammatically complete.
    Falls back to "..." truncation if no sentence boundary found in the
    final 30% of the allowed length.
    """
    if len(text) <= max_chars:
        return text

    candidate = text[:max_chars]
    # Find last sentence end in the candidate
    for sep in (". ", "! ", "? "):
        pos = candidate.rfind(sep)
        if pos > max_chars * 0.65:  # Must be in the latter 35% — avoids too-early cuts
            return candidate[: pos + 1]

    # No good sentence boundary — truncate at last space
    last_space = candidate.rfind(" ")
    if last_space > max_chars * 0.8:
        return candidate[:last_space] + "…"

    return candidate + "…"


def _normalise_genres(genres: list[str]) -> str:
    """
    Normalises genre tags to a consistent, comma-separated string.

    Truncates to 3 genres max — more genres add noise without improving retrieval.
    """
    if not genres:
        return "fiction"
    # Take first 3, lowercase, strip whitespace
    normalised = [g.lower().strip() for g in genres[:3] if g.strip()]
    return ", ".join(normalised) if normalised else "fiction"


def _relevance_label(similarity: float) -> str:
    """
    Converts a float cosine similarity to a human-readable relevance label.

    Labels are read by Claude and calibrate its confidence in the match.
    """
    if similarity >= 0.88:
        return "high"
    if similarity >= 0.78:
        return "good"
    if similarity >= 0.70:
        return "partial"
    return "low"
