"""Anthropic request translation for V2."""

import json
import logging

try:
    from gateway.proxy_v2.errors import ProxyError
except ImportError:
    from proxy_v2.errors import ProxyError

log = logging.getLogger("litellm-proxy.v2.request_translate")


def validate_anthropic_messages_request(body_json):
    if not isinstance(body_json, dict):
        raise ProxyError(400, "Request body must be a JSON object", "invalid_request")
    model = body_json.get("model")
    if not model or not isinstance(model, str):
        raise ProxyError(400, "model field is required and must be a string", "invalid_request")
    messages = body_json.get("messages")
    if messages is None:
        raise ProxyError(400, "messages field is required", "invalid_request")
    if not isinstance(messages, list) or not messages:
        raise ProxyError(400, "messages field must be a non-empty list", "invalid_request")
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg:
            raise ProxyError(400, "each message must be an object with a role field", "invalid_request")


def translate_anthropic_request(body_json, *, thinking_effort, thinking_contract):
    validate_anthropic_messages_request(body_json)
    openai_body = {
        "model": body_json.get("model"),
        "messages": _translate_messages(body_json),
    }

    metadata = body_json.get("metadata")
    if isinstance(metadata, dict) and metadata.get("user_id"):
        openai_body["user"] = metadata["user_id"]
    if body_json.get("max_tokens") is not None:
        openai_body["max_tokens"] = body_json["max_tokens"]
        openai_body["max_completion_tokens"] = body_json["max_tokens"]
    if body_json.get("temperature") is not None:
        openai_body["temperature"] = body_json["temperature"]
    if body_json.get("top_p") is not None:
        openai_body["top_p"] = body_json["top_p"]
    if body_json.get("stop_sequences"):
        openai_body["stop"] = body_json["stop_sequences"]
    if body_json.get("top_k") is not None:
        openai_body["top_k"] = body_json["top_k"]
    if body_json.get("stream"):
        openai_body["stream"] = True
        openai_body["stream_options"] = {"include_usage": True}

    _apply_verified_thinking_contract(openai_body, thinking_contract, thinking_effort)
    apply_openai_tools_and_choice(
        openai_body,
        tools=body_json.get("tools"),
        tool_choice=body_json.get("tool_choice"),
    )

    response_format = body_json.get("response_format")
    if response_format:
        openai_body["response_format"] = response_format
    return openai_body


def _translate_messages(body_json):
    messages = []
    system_message = _translate_system_message(body_json.get("system"))
    if system_message is not None:
        messages.append(system_message)

    for msg in body_json.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            messages.append({"role": role, "content": str(content) if content else ""})
            continue
        messages.extend(_translate_content_blocks(role, content))
    return messages


def _translate_system_message(system):
    if not system:
        return None
    if isinstance(system, str):
        return {"role": "system", "content": system}
    if not isinstance(system, list):
        return {"role": "system", "content": str(system)}

    if any(isinstance(item, dict) and "cache_control" in item for item in system):
        log.debug("System cache_control present but not forwarded")

    parts = []
    for item in system:
        if isinstance(item, dict):
            parts.append(item.get("text", ""))
        else:
            parts.append(str(item))
    text = "\n".join(parts)
    if not text:
        return None
    return {"role": "system", "content": text}


def _translate_content_blocks(role, content_blocks):
    text_parts = []
    tool_calls = []
    tool_results = []

    for block in content_blocks:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "image":
            image_part = _translate_image_block(block)
            if image_part is not None:
                text_parts.append(image_part)
        elif block_type == "thinking":
            thinking_text = block.get("thinking", "")
            if thinking_text:
                text_parts.append({"type": "text", "text": thinking_text})
        elif block_type == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })
        elif block_type == "tool_result":
            tool_results.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": _flatten_tool_result_content(block),
            })

    flat_content = _flatten_openai_content(text_parts)
    if role == "assistant":
        assistant_message = {"role": "assistant", "content": flat_content if text_parts else None}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        return [assistant_message]
    if tool_results:
        translated = list(tool_results)
        if text_parts:
            translated.append({"role": role, "content": flat_content})
        return translated
    return [{"role": role, "content": flat_content if text_parts else ""}]


def _translate_image_block(block):
    source = block.get("source") or {}
    media_type = source.get("media_type", "image/png")
    b64_data = source.get("data", "")
    if not b64_data:
        return None
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64_data}"}}


def _flatten_tool_result_content(block):
    result_content = block.get("content", "")
    if isinstance(result_content, list):
        parts = []
        for item in result_content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if item.get("type") == "image":
                source = item.get("source") or {}
                parts.append("[image: %s]" % source.get("media_type", "image/png"))
            else:
                parts.append(item.get("text", ""))
        result_content = "\n".join(parts)
    if block.get("is_error"):
        result_content = "[ERROR] %s" % result_content
    return str(result_content)


def _flatten_openai_content(text_parts):
    has_images = any(isinstance(part, dict) and part.get("type") == "image_url" for part in text_parts)
    if has_images:
        return text_parts
    return "\n".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in text_parts
    )


def _apply_verified_thinking_contract(openai_body, thinking_contract, thinking_effort):
    if not thinking_effort:
        return
    if not thinking_contract:
        raise ProxyError(400, "thinking_effort requires a verified thinking contract", "invalid_request")
    strategy = thinking_contract.get("strategy")
    if strategy == "openai_chat_reasoning_effort":
        openai_body["reasoning_effort"] = thinking_effort
        return
    raise ProxyError(
        400,
        "Unsupported thinking strategy '%s' for provider '%s'"
        % (strategy, thinking_contract.get("provider", "unknown")),
        "invalid_request",
    )


def apply_openai_tools_and_choice(openai_body, *, tools, tool_choice):
    translated_tools = _translate_tools(tools)
    if translated_tools:
        openai_body["tools"] = translated_tools
    _attach_openai_tool_choice(openai_body, tool_choice)


def _translate_tools(tools):
    if not tools:
        return []
    translated_tools = []
    for tool in tools:
        if tool.get("cache_control"):
            log.debug("Tool cache_control present but not forwarded")
        translated_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return translated_tools


def _attach_openai_tool_choice(openai_body, tool_choice):
    if not tool_choice:
        return
    translated_tools = openai_body.get("tools") or []
    if not translated_tools:
        raise ProxyError(400, "tool_choice requires tools to be defined", "invalid_request")
    choice_type = tool_choice.get("type", "auto") if isinstance(tool_choice, dict) else tool_choice
    if choice_type == "auto":
        openai_body["tool_choice"] = "auto"
    elif choice_type == "any":
        openai_body["tool_choice"] = "required"
    elif choice_type == "none":
        openai_body["tool_choice"] = "none"
    elif choice_type == "tool":
        tool_name = tool_choice.get("name", "") if isinstance(tool_choice, dict) else ""
        if not tool_name:
            raise ProxyError(400, "tool_choice.type=tool requires a tool name", "invalid_request")
        openai_body["tools"] = _filter_translated_tools(translated_tools, tool_name)
        # Some OpenAI-compatible backends reject named tool_choice objects but do
        # accept required tool calling. Filtering tools to the requested tool keeps
        # the caller's intent while staying within the accepted contract.
        openai_body["tool_choice"] = "required"
    else:
        raise ProxyError(400, "Unsupported tool_choice type: %s" % choice_type, "invalid_request")


def _filter_translated_tools(translated_tools, tool_name):
    filtered_tools = [
        tool for tool in translated_tools
        if (tool.get("function") or {}).get("name") == tool_name
    ]
    if not filtered_tools:
        raise ProxyError(
            400,
            "tool_choice references unknown tool: %s" % tool_name,
            "invalid_request",
        )
    if len(filtered_tools) != len(translated_tools):
        log.debug("Restricted translated tools to forced tool_choice=%s", tool_name)
    return filtered_tools
