"""Protocol contract helpers shared by proxy translation paths."""

import logging

log = logging.getLogger("litellm-proxy.v2.contracts")

# Sentinel stop reason for mid-stream upstream errors in translated SSE.
# Distinct from "end_turn" so clients can distinguish stream errors from
# normal completion without parsing message content.
UPSTREAM_ERROR_STOP = "upstream_error"
ANTHROPIC_TERMINAL_STOP_REASONS = frozenset({
    "end_turn",
    "tool_use",
    "max_tokens",
    UPSTREAM_ERROR_STOP,
})


def map_openai_finish_reason(finish_reason):
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    if finish_reason == "stop":
        return "end_turn"
    if finish_reason in ("tool_calls", "function_call"):
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "content_filter":
        log.warning("Upstream content_filter triggered — response may be truncated")
        return "end_turn"
    if finish_reason:
        log.debug("Unknown finish_reason %s mapped to end_turn", finish_reason)
    return "end_turn"


def resolve_terminal_stop_reason(reason):
    """Resolve a terminal reason for translated Anthropic output.

    V2 translation must not silently normalize transport abort reasons into
    valid Anthropic completion semantics. Only contract stop reasons are
    accepted here.
    """
    if reason in ANTHROPIC_TERMINAL_STOP_REASONS:
        return reason
    raise ValueError("non-contract terminal reason: %s" % reason)


def resolve_legacy_terminal_stop_reason(reason):
    """Legacy compatibility adapter for V1 proxy semantics.

    V1 still coerces non-contract terminal reasons like server shutdown into
    regular Anthropic stop reasons. Keep that behavior explicit and logged until
    the V2 event model owns shutdown/abort semantics.
    """
    if reason in ANTHROPIC_TERMINAL_STOP_REASONS:
        return reason
    log.debug(
        "Coercing non-contract terminal reason %s through legacy finish-reason mapping",
        reason,
    )
    return map_openai_finish_reason(reason)
