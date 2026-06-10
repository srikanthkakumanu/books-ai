"""
agents/utils/__init__.py

Utility package for the agent layer.
"""
from agents.utils.cost_tracker import calculate_cost, record_usage
from agents.utils.guardrails import GuardrailResult, check_input, validate_response
from agents.utils.prompt_cache import (
    log_cache_metrics,
    make_cached_system_blocks,
    split_static_dynamic,
)
from agents.utils.query_optimizer import (
    generate_hyde_query,
    rewrite_query_for_retrieval,
    should_use_hyde,
)
from agents.utils.token_utils import (
    audit_prompt_tokens,
    count_tokens,
    estimate_available_tokens,
    trim_message_history,
)

__all__ = [
    "calculate_cost", "record_usage",
    "GuardrailResult", "check_input", "validate_response",
    "log_cache_metrics", "make_cached_system_blocks", "split_static_dynamic",
    "generate_hyde_query", "rewrite_query_for_retrieval", "should_use_hyde",
    "audit_prompt_tokens", "count_tokens", "estimate_available_tokens", "trim_message_history",
]
