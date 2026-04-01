"""Test-only V2 runtime harness for translated responses."""

import json
import logging

try:
    from gateway.proxy_v2.anthropic_sse import AnthropicSSEWriter
    from gateway.proxy_v2.errors import ProxyError
    from gateway.proxy_v2.events import decode_openai_chunk
    from gateway.proxy_v2.response_translate import serialize_anthropic_message, translate_openai_response
    from gateway.proxy_v2.sse import SSEParser
    from gateway.proxy_v2.state import TranslationState
except ImportError:
    from proxy_v2.anthropic_sse import AnthropicSSEWriter
    from proxy_v2.errors import ProxyError
    from proxy_v2.events import decode_openai_chunk
    from proxy_v2.response_translate import serialize_anthropic_message, translate_openai_response
    from proxy_v2.sse import SSEParser
    from proxy_v2.state import TranslationState

log = logging.getLogger("litellm-proxy.v2.runtime")


def translate_buffered_response(response_bytes):
    try:
        body_json = json.loads(response_bytes)
    except (TypeError, ValueError) as exc:
        raise ProxyError(502, "Buffered upstream response must be valid JSON", "upstream_error") from exc
    return serialize_anthropic_message(translate_openai_response(body_json))


def translate_stream(upstream_byte_iterable, *, abort_signal, logger):
    stream_log = logger or log
    parser = SSEParser()
    state = TranslationState()
    writer = AnthropicSSEWriter()

    try:
        for raw_chunk in upstream_byte_iterable:
            if abort_signal and abort_signal():
                for payload in writer.write(state.abort("server_shutdown")):
                    yield payload
                return

            for frame in parser.feed(raw_chunk):
                if frame.data == "[DONE]":
                    for payload in writer.write(state.finish_eof()):
                        yield payload
                    return
                for payload in writer.write(state.apply_chunk(decode_openai_chunk(frame.data))):
                    yield payload

        parser.finish()
        for payload in writer.write(state.finish_eof()):
            yield payload
    except ProxyError as exc:
        stream_log.error("V2 translated stream aborted: %s", exc.message)
        for payload in writer.write(state.abort(exc.code or "upstream_error", message=exc.message)):
            yield payload
