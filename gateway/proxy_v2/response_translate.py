"""Buffered response translation for V2."""

import json
import logging
import re

try:
    from gateway.proxy_v2.contracts import map_openai_finish_reason
    from gateway.proxy_v2.errors import ProxyError
except ImportError:
    from proxy_v2.contracts import map_openai_finish_reason
    from proxy_v2.errors import ProxyError

log = logging.getLogger("litellm-proxy.v2.response_translate")
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_tags(text):
    if not text or "<think>" not in text:
        return text
    return _THINK_TAG_RE.sub("", text).strip()


def translate_openai_response(body_json):
    if not isinstance(body_json, dict):
        raise ProxyError(502, "OpenAI response body must be a JSON object", "upstream_error")

    choices = body_json.get("choices", [])
    content = []
    stop_reason = "end_turn"

    if choices:
        choice = choices[0] or {}
        msg = choice.get("message") or {}
        finish_reason = choice.get("finish_reason", "stop")
        text = _extract_visible_text(msg)
        if text:
            content.append({"type": "text", "text": text})

        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            tool_blocks = []
            all_tool_calls_valid = True
            for tool_call in tool_calls:
                content_block = _tool_call_to_anthropic_content_block(tool_call)
                tool_blocks.append(content_block)
                if content_block.get("type") != "tool_use":
                    all_tool_calls_valid = False
            if all_tool_calls_valid:
                content.extend(tool_blocks)
                stop_reason = "tool_use"
            else:
                content.extend(
                    block for block in tool_blocks
                    if block.get("type") != "tool_use"
                )
                stop_reason = "end_turn"
        else:
            stop_reason = map_openai_finish_reason(finish_reason)

    usage = body_json.get("usage") or {}
    return {
        "id": body_json.get("id", ""),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": body_json.get("model", ""),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def serialize_anthropic_message(message):
    return json.dumps(message).encode("utf-8")


def _extract_visible_text(message):
    if "content" in message:
        return strip_think_tags(message.get("content") or "")
    reasoning_text = message.get("reasoning_content")
    if reasoning_text:
        return strip_think_tags(reasoning_text)
    return ""


def _tool_call_to_anthropic_content_block(tool_call):
    function = tool_call.get("function") or {}
    raw_arguments = function.get("arguments", "{}")
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, ValueError):
        tool_name = function.get("name", "unknown")
        log.warning(
            "Malformed tool arguments from upstream (tool=%s): %s",
            tool_name,
            str(raw_arguments)[:200],
        )
        return {
            "type": "text",
            "text": "[Tool call failed: malformed arguments for %s]" % tool_name,
        }
    if not isinstance(arguments, dict):
        tool_name = function.get("name", "unknown")
        log.warning(
            "Non-object tool arguments from upstream (tool=%s): %s",
            tool_name,
            str(raw_arguments)[:200],
        )
        return {
            "type": "text",
            "text": "[Tool call failed: malformed arguments for %s]" % tool_name,
        }
    return {
        "type": "tool_use",
        "id": tool_call.get("id", ""),
        "name": function.get("name", ""),
        "input": arguments,
    }
