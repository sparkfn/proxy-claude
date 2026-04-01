"""Normalized V2 event types for translated upstream streams."""

from dataclasses import dataclass
import json

try:
    from gateway.proxy_v2.errors import ProxyError
except ImportError:
    from proxy_v2.errors import ProxyError

class IncompleteMessageError(Exception):
    """Raised when transport ends before a validated semantic stop."""


class TransportAbortError(Exception):
    """Raised when the stream aborts for a transport/runtime cause."""


@dataclass(eq=True)
class OpenAIChunk:
    chunk_id: str = ""
    model: str = ""
    usage: dict | None = None
    delta: dict | None = None
    finish_reason: str | None = None
    error: dict | str | None = None


@dataclass(eq=True)
class MessageStart:
    message_id: str
    model: str
    input_tokens: int = 0


@dataclass(eq=True)
class TextDelta:
    text: str


@dataclass(eq=True)
class ToolUseStart:
    index: int
    tool_id: str
    name: str


@dataclass(eq=True)
class ToolUseArgsDelta:
    index: int
    partial_json: str


@dataclass(eq=True)
class ToolCallStart:
    index: int
    tool_call_id: str
    name: str


@dataclass(eq=True)
class ToolCallArgsDelta:
    index: int
    partial_json: str


@dataclass(eq=True)
class ToolCallComplete:
    index: int
    input: dict


@dataclass(eq=True)
class UsageDelta:
    input_tokens: int
    output_tokens: int


@dataclass(eq=True)
class MessageStop:
    stop_reason: str
    output_tokens: int = 0


@dataclass(eq=True)
class Abort:
    reason: str
    message: str | None = None


def decode_openai_chunk(data):
    """Decode one OpenAI-style SSE JSON payload into a normalized chunk."""
    try:
        payload = json.loads(data)
    except (TypeError, ValueError) as exc:
        raise ProxyError(
            502,
            "Malformed upstream stream payload",
            "upstream_error",
            code="malformed_upstream_json",
        ) from exc

    if not isinstance(payload, dict):
        raise ProxyError(
            502,
            "Malformed upstream stream payload",
            "upstream_error",
            code="malformed_upstream_json",
        )

    if "error" in payload:
        return OpenAIChunk(
            chunk_id=payload.get("id", ""),
            model=payload.get("model", ""),
            usage=payload.get("usage") or {},
            delta={},
            finish_reason=None,
            error=payload["error"],
        )

    choices = payload.get("choices")
    if choices == []:
        if "usage" not in payload:
            raise ProxyError(
                502,
                "Malformed upstream stream payload",
                "upstream_error",
                code="missing_upstream_usage",
            )
        usage = payload.get("usage")
        if usage is not None and not isinstance(usage, dict):
            raise ProxyError(
                502,
                "Malformed upstream usage payload",
                "upstream_error",
                code="malformed_upstream_usage",
            )
        return OpenAIChunk(
            chunk_id=payload.get("id", ""),
            model=payload.get("model", ""),
            usage=usage or {},
            delta={},
            finish_reason=None,
            error=None,
        )

    if "choices" not in payload:
        raise ProxyError(
            502,
            "Malformed upstream stream payload",
            "upstream_error",
            code="missing_upstream_choices",
        )
    if not isinstance(choices, list) or not choices:
        raise ProxyError(
            502,
            "Malformed upstream stream payload",
            "upstream_error",
            code="invalid_upstream_choices",
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ProxyError(
            502,
            "Malformed upstream stream payload",
            "upstream_error",
            code="invalid_upstream_choice",
        )
    if "delta" not in choice:
        raise ProxyError(
            502,
            "Malformed upstream stream payload",
            "upstream_error",
            code="missing_upstream_delta",
        )
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        raise ProxyError(
            502,
            "Malformed upstream delta payload",
            "upstream_error",
            code="malformed_upstream_delta",
        )

    return OpenAIChunk(
        chunk_id=payload.get("id", ""),
        model=payload.get("model", ""),
        usage=payload.get("usage") or {},
        delta=delta,
        finish_reason=choice.get("finish_reason"),
        error=None,
    )
