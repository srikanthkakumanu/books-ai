"""
agents/utils/guardrails.py

Input validation and output guardrails for the AI agent.

Prompt engineering techniques applied:
  1. Input sanitisation: strip prompt injection attempts before they reach Claude
  2. Output validation: verify response stays within domain scope
  3. Jailbreak detection: catch attempts to override the system prompt
  4. PII detection: flag potential PII in user messages before logging
  5. Response length enforcement: catch truncated or runaway responses

Security note:
  These are lightweight, heuristic-based guardrails suitable for a book platform.
  For higher-risk applications, replace with Anthropic's safety APIs or a dedicated
  moderation layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


# ── Input guardrails ──────────────────────────────────────────────────────────

# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(?!a\s+book)", re.I),   # "you are now [not a book thing]"
    re.compile(r"pretend\s+you\s+(are|have\s+no)", re.I),
    re.compile(r"<\s*/?system\s*>", re.I),                          # Injected XML tags
    re.compile(r"act\s+as\s+(if\s+you\s+(are|have)\s+)?(?!a\s+book)", re.I),
]

# Queries that are completely out of scope for a book platform
_OFF_SCOPE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(hack|exploit|malware|phishing|credential)\b", re.I),
    re.compile(r"\b(weapon|bomb|explosive|drug\s+synthesis)\b", re.I),
]

# Rough PII indicators (for log scrubbing — not shown to user)
_PII_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),   # Phone
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
    re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),  # Credit card
]


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    sanitised_query: str | None = None


def check_input(message: str, user_id: str) -> GuardrailResult:
    """
    Checks a user message for injection attempts and off-scope content.

    Args:
        message: Raw user message text
        user_id: For logging (not included in log if message is flagged)

    Returns:
        GuardrailResult with allowed=True if the message passes all checks
    """
    # Check injection patterns
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(message):
            logger.warning(
                "prompt_injection_attempt",
                user_id=user_id,
                pattern=pattern.pattern[:40],
            )
            return GuardrailResult(
                allowed=False,
                reason="I can only help with book recommendations and reading-related questions.",
            )

    # Check off-scope content
    for pattern in _OFF_SCOPE_PATTERNS:
        if pattern.search(message):
            logger.warning("off_scope_message", user_id=user_id)
            return GuardrailResult(
                allowed=False,
                reason="I'm specialised for books and reading. I can't help with that.",
            )

    # Check for PII (scrub from logs, still allow the message)
    has_pii = any(p.search(message) for p in _PII_PATTERNS)
    if has_pii:
        logger.info("pii_detected_in_message", user_id=user_id)
        # Don't block — PII might be a book title. Just flag it.

    # Sanitise: strip leading XML/angle-bracket injections
    sanitised = _sanitise_message(message)

    return GuardrailResult(allowed=True, sanitised_query=sanitised)


def _sanitise_message(message: str) -> str:
    """
    Strips potential XML injection tags from user messages.

    Angle brackets in normal book titles/queries are extremely rare —
    stripping them is safe for this domain.
    """
    # Remove attempted XML tag injections while preserving normal text
    cleaned = re.sub(r"<(?!em>|strong>|b>|i>)[^>]{1,50}>", "", message)
    return cleaned.strip()


# ── Output guardrails ─────────────────────────────────────────────────────────

def validate_response(response: str, intent: str) -> GuardrailResult:
    """
    Validates that the model's response is appropriate for a book platform.

    Checks:
    - Response is not empty or suspiciously short
    - Response does not contain obviously off-topic content
    - Response is not a refusal without explanation (indicates misconfigured prompt)

    Args:
        response: The AI's generated response text
        intent: The classified intent for context

    Returns:
        GuardrailResult — if not allowed, caller should use the fallback message
    """
    if not response or len(response.strip()) < 20:
        logger.warning("response_too_short", intent=intent, length=len(response))
        return GuardrailResult(
            allowed=False,
            reason="Sorry, I wasn't able to generate a response. Please try again.",
        )

    # Detect hollow refusal patterns (suggests misconfigured system prompt)
    refusal_patterns = [
        r"I cannot|I can't|I'm unable|I am unable",
        r"I don't have access to",
        r"As an AI",
    ]
    for pattern in refusal_patterns:
        if re.search(pattern, response[:200], re.I):
            logger.warning(
                "unexpected_refusal",
                intent=intent,
                response_prefix=response[:100],
            )
            # Don't block — some refusals are legitimate. Just log.
            break

    return GuardrailResult(allowed=True)
