"""
agents/prompts/intent.py

Intent classification prompt — highly optimised for Haiku.

Prompt engineering techniques applied:
  1. XML tags for structure (Claude's native format; improves parsing accuracy)
  2. Minimal tokens: instruction + examples only, no filler
  3. Positive + negative examples (shows edge cases explicitly)
  4. Output format stated before examples (primes model output style)
  5. Exhaustive edge cases in examples to reduce misclassification
  6. Imperative phrasing: "Output exactly one word" not "Please output..."

Token budget: ~120 tokens for this system prompt (< SYSTEM_PROMPT_TARGET_TOKENS)
Model: claude-haiku-4-5 — deterministic classification at 40x lower cost than Sonnet
"""

from __future__ import annotations


# Static prompt — built once at import time, not on every call
# This enables prompt caching (Haiku: 1024 token min block, this is under that
# but we prepend it to a long enough system block with the examples)
_INTENT_PROMPT: str = """<task>Classify the user's book-related message into exactly one intent.</task>

<output_format>Output exactly one word. No punctuation, no explanation.</output_format>

<intents>
recommend — wants book suggestions or "something like X"
search    — wants to find a specific book, author, or topic
review    — wants to write, read, or discuss a book review
shelf     — wants to add/remove/view their reading lists
general   — anything else (platform questions, greetings, etc.)
</intents>

<examples>
"something like The Martian" → recommend
"books by Ursula Le Guin" → search
"find me a fantasy novel" → search
"add Dune to my want-to-read list" → shelf
"remove that book from my shelf" → shelf
"write a review for Project Hail Mary" → review
"what do other readers think of this?" → review
"I loved Dune, what should I read next?" → recommend
"is this app free?" → general
"hello" → general
"what genres do you have?" → general
"show me my reading list" → shelf
</examples>"""


def build_intent_prompt() -> str:
    """
    Returns the intent classification system prompt.

    Pre-built at module load — safe to call on every request
    without allocation overhead.
    """
    return _INTENT_PROMPT
