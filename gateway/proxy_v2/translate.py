"""Compatibility wrappers over the split V2 request/response translators."""

import json
import logging

try:
    from gateway.proxy_v2.errors import ProxyError
    from gateway.proxy_v2.request_translate import (
        translate_anthropic_request,
        validate_anthropic_messages_request as _validate_anthropic_messages_request,
    )
    from gateway.proxy_v2.response_translate import (
        serialize_anthropic_message,
        strip_think_tags,
        translate_openai_response,
    )
except ImportError:
    from proxy_v2.errors import ProxyError
    from proxy_v2.request_translate import (
        translate_anthropic_request,
        validate_anthropic_messages_request as _validate_anthropic_messages_request,
    )
    from proxy_v2.response_translate import (
        serialize_anthropic_message,
        strip_think_tags,
        translate_openai_response,
    )

log = logging.getLogger("litellm-proxy.v2.translate")


def validate_anthropic_messages_request(body_json):
    try:
        _validate_anthropic_messages_request(body_json)
    except ProxyError as exc:
        raise ValueError(exc.message) from exc


def anthropic_to_openai_request(body_json, thinking_effort=None, thinking_contract=None):
    try:
        translated = translate_anthropic_request(
            body_json,
            thinking_effort=thinking_effort,
            thinking_contract=thinking_contract,
        )
    except ProxyError as exc:
        raise ValueError(exc.message) from exc
    return json.dumps(translated).encode("utf-8")


def openai_to_anthropic_response(response_bytes):
    try:
        body_json = json.loads(response_bytes)
    except (TypeError, ValueError) as exc:
        raise ValueError("OpenAI response body must be valid JSON") from exc

    try:
        return serialize_anthropic_message(translate_openai_response(body_json))
    except ProxyError as exc:
        raise ValueError(exc.message) from exc
