"""
Reverse proxy: strips system messages from Anthropic /v1/messages requests
before forwarding to LiteLLM. Supports SSE streaming pass-through so
Claude Code sees tokens incrementally. Runs on the host at port 2555.

Required because chatgpt/ provider rejects system messages and LiteLLM
doesn't strip them in the Anthropic-to-Responses translation path.
"""

import atexit
import json
import os
import re
import sys
import time
import http.client
import socket
import ssl
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging

log = logging.getLogger("litellm-proxy")


def _parse_size(value, default):
    """Parse a human-readable size string (e.g. '10MB', '512KB', '1GB') to bytes.

    Accepts B, KB, MB, GB suffixes (case-insensitive). Raw integers treated as bytes.
    Returns default with a warning on invalid input.
    """
    if not value:
        return default
    value = str(value).strip()
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    match = re.match(r'^(\d+(?:\.\d+)?)\s*(GB|MB|KB|B)?$', value, re.IGNORECASE)
    if match:
        num = float(match.group(1))
        unit = (match.group(2) or "B").upper()
        return int(num * multipliers[unit])
    log.warning("Invalid size '%s', using default %d", value, default)
    return default


def _env_int(name, default):
    """Read an integer from environment, falling back to default."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        log.warning("Invalid integer for %s='%s', using default %d", name, val, default)
        return default


# --- Configuration (all from environment, no config.py dependency) ---
LITELLM_HOST = os.environ.get("PROXY_LITELLM_HOST", "localhost")
LITELLM_PORT = _env_int("PROXY_LITELLM_PORT", 4000)
LISTEN_PORT = _env_int("PROXY_LISTEN_PORT", int(sys.argv[1]) if len(sys.argv) > 1 else 2555)
MAX_WORKERS = _env_int("PROXY_MAX_WORKERS", 20)
MAX_REQUEST_BODY = _parse_size(os.environ.get("PROXY_MAX_REQUEST_BODY"), 100 * 1024**2)  # 100MB
MAX_RESPONSE_BODY = _parse_size(os.environ.get("PROXY_MAX_RESPONSE_BODY"), 50 * 1024**2)  # 50MB
CONNECT_TIMEOUT = _env_int("PROXY_CONNECT_TIMEOUT", 10)
READ_TIMEOUT = _env_int("PROXY_READ_TIMEOUT", 300)
NON_STREAM_READ_TIMEOUT = _env_int("PROXY_NON_STREAM_READ_TIMEOUT", 30)
STREAM_IDLE_TIMEOUT = _env_int("PROXY_STREAM_IDLE_TIMEOUT", 900)  # 15 min
MAX_STREAM_LIFETIME = _env_int("PROXY_MAX_STREAM_LIFETIME", 21600)  # 6 hours
MAX_STREAM_BYTES = _parse_size(os.environ.get("PROXY_MAX_STREAM_BYTES"), 250 * 1024**2)  # 250MB
SOCKET_TIMEOUT = _env_int("PROXY_SOCKET_TIMEOUT", 30)


_COUNTERS = {
    "stream_budget_killed": 0,
    "idle_timeout_inactive": 0,
    "truncated_stream": 0,
    "invalid_request": 0,
    "upstream_timeout": 0,
    "upstream_refused": 0,
    "upstream_http_error": 0,
    "upstream_io_error": 0,
    "handler_errors": 0,
    "xlate_stream_errors": 0,
    "xlate_stream_eof_no_finish": 0,
    "circuit_breaker_rejected": 0,
}
_COUNTERS_LOCK = threading.Lock()


def _inc_counter(key):
    with _COUNTERS_LOCK:
        _COUNTERS[key] += 1


def _print_counters():
    # Uses print() instead of log.warning() because atexit handlers run during
    # interpreter shutdown after logging.shutdown() has flushed and closed all
    # handlers. log calls here would be silently dropped.
    with _COUNTERS_LOCK:
        if any(_COUNTERS.values()):
            print("Proxy counters: %s" % json.dumps(_COUNTERS, sort_keys=True), file=sys.stderr, flush=True)


atexit.register(_print_counters)


class _CircuitBreaker:
    """Per-provider circuit breaker. Opens after consecutive failures, half-opens after cooldown."""

    def __init__(self, failure_threshold=5, cooldown_seconds=30):
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        # provider -> {"failures": int, "opened_at": float|None, "probe_in_flight": bool}
        self._state = {}

    def is_open(self, provider):
        with self._lock:
            s = self._state.get(provider)
            if not s or s["opened_at"] is None:
                return False
            if time.monotonic() - s["opened_at"] >= self._cooldown_seconds:
                if not s.get("probe_in_flight"):
                    s["probe_in_flight"] = True
                    log.info("Circuit half-open for %s, allowing single probe", provider)
                    return False  # Allow this one through
                return True  # Another probe already in flight
            return True

    def record_success(self, provider):
        with self._lock:
            self._state[provider] = {"failures": 0, "opened_at": None, "probe_in_flight": False}

    def record_failure(self, provider):
        with self._lock:
            s = self._state.setdefault(provider, {"failures": 0, "opened_at": None, "probe_in_flight": False})
            s["failures"] += 1
            s["probe_in_flight"] = False
            if s["failures"] >= self._failure_threshold:
                if s["opened_at"] is None:
                    log.warning("Circuit opened for %s after %d failures", provider, s["failures"])
                s["opened_at"] = time.monotonic()


_circuit = _CircuitBreaker()


def _error_response(status_code, message, error_type="proxy_error"):
    """Build a JSON error body and return (status_code, body_bytes)."""
    return status_code, json.dumps(
        {"error": {"message": message, "type": error_type}}
    ).encode()


def _validate_messages(body_bytes):
    """Validate /v1/messages request schema. Returns error string or None."""
    try:
        data = json.loads(body_bytes)
    except (ValueError, TypeError):
        return "Request body must be valid JSON"
    if not isinstance(data, dict):
        return "Request body must be a JSON object"
    model = data.get("model")
    if not model or not isinstance(model, str):
        return "model field is required and must be a string"
    messages = data.get("messages")
    if messages is None:
        return "messages field is required"
    if not isinstance(messages, list) or len(messages) == 0:
        return "messages field must be a non-empty list"
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg:
            return "each message must be an object with a role field"
    return None


def strip_system(body_bytes):
    """Remove 'system' field, merge into first user message.

    Caller MUST run _validate_messages() first.
    Returns modified body bytes, or original bytes if no system field.
    """
    try:
        data = json.loads(body_bytes)
    except (ValueError, TypeError):
        return body_bytes

    if not isinstance(data, dict):
        return body_bytes

    system = data.pop("system", None)
    if not system:
        return body_bytes

    messages = data.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        return body_bytes  # validation already caught this

    if isinstance(system, str):
        text = system
    elif isinstance(system, list):
        text = "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in system
        )
    else:
        text = str(system)

    if text:
        msg = messages[0]
        if msg.get("role") == "user":
            c = msg.get("content", "")
            if isinstance(c, str):
                msg["content"] = text + "\n\n" + c
            elif isinstance(c, list):
                msg["content"] = [{"type": "text", "text": text + "\n\n"}] + c
            else:
                messages.insert(0, {"role": "user", "content": text})
        else:
            messages.insert(0, {"role": "user", "content": text})

    return json.dumps(data).encode()


# Cache: models that need OpenAI translation (loaded once at startup)
_OPENAI_TRANSLATED_MODELS = None
# Cache: ordered list of configured model names (for fallback routing)
_ALL_CONFIGURED_MODELS = None
# Cache: model_name -> (host, base_path, api_key_env_var) for native Anthropic forwarding
_NATIVE_ANTHROPIC_MODELS = None

def _load_translated_models():
    """Load model routing tables from config + provider registry.

    Populates three caches:
    - _ALL_CONFIGURED_MODELS: ordered list of model names (first = default fallback)
    - _OPENAI_TRANSLATED_MODELS: set of models needing Anthropic→OpenAI translation
    - _NATIVE_ANTHROPIC_MODELS: dict of models with native Anthropic endpoints
    Called once at startup, cached for all subsequent requests."""
    global _OPENAI_TRANSLATED_MODELS, _ALL_CONFIGURED_MODELS, _NATIVE_ANTHROPIC_MODELS
    translated = set()
    all_models = []  # ordered list, first model is the default fallback
    native = {}  # model_name -> {"url": parsed, "api_key_env": str}

    # Build provider lookup: model_name -> provider instance
    provider_for_model = {}
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import providers
        for p in providers.all_providers():
            if p.anthropic_base_url:
                for model_name in p.models:
                    provider_for_model[model_name] = p
    except Exception as e:
        log.warning("Could not load providers for native routing: %s", e)

    try:
        import yaml
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "litellm_config.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        for entry in cfg.get("model_list", []):
            name = entry.get("model_name", "")
            all_models.append(name)
            litellm_model = entry.get("litellm_params", {}).get("model", "")

            # Check if this model's provider has a native Anthropic endpoint
            if name in provider_for_model:
                p = provider_for_model[name]
                if not p.native_auth:
                    log.warning("Provider %s has anthropic_base_url but no native_auth — skipping native route", p.name)
                    continue
                api_key_env = p.native_auth.get("env")
                auth_header = p.native_auth.get("header", "x-api-key")
                parsed = urllib.parse.urlparse(p.anthropic_base_url)
                native[name] = {"host": parsed.hostname, "port": parsed.port or 443,
                                "path": parsed.path.rstrip("/"), "api_key_env": api_key_env,
                                "auth_header": auth_header}
                log.info("Native Anthropic route: %s → %s", name, p.anthropic_base_url)
            elif litellm_model.startswith("openai/"):
                translated.add(name)
    except Exception as e:
        log.warning("Failed to load litellm_config.yaml: %s — model routing disabled", e)
    _OPENAI_TRANSLATED_MODELS = translated
    _ALL_CONFIGURED_MODELS = all_models
    _NATIVE_ANTHROPIC_MODELS = native
    log.debug("Models needing OpenAI translation: %s", translated)
    log.debug("Native Anthropic models: %s", list(native.keys()))
    log.debug("All configured models (ordered): %s", all_models)


def _remap_model_if_needed(body_bytes):
    """If the request targets a model that isn't configured, remap to a configured one.

    Claude Code sends background requests (title gen, summarization) to models like
    claude-haiku-4-5-20251001 which may not be configured. We remap these to the
    first configured model so they don't 400.
    """
    if not _ALL_CONFIGURED_MODELS:
        return body_bytes
    try:
        data = json.loads(body_bytes)
    except (ValueError, TypeError):
        return body_bytes
    model = data.get("model", "")
    if model and model not in _ALL_CONFIGURED_MODELS:
        # Pick first configured model as fallback (deterministic, config file order)
        fallback = _ALL_CONFIGURED_MODELS[0]
        log.info("Remapping unconfigured model %s → %s", model, fallback)
        data["model"] = fallback
        return json.dumps(data).encode()
    return body_bytes


def _get_native_route(body_bytes):
    """If the request targets a model with a native Anthropic endpoint, return its route dict.
    Returns None if the model should go through LiteLLM."""
    if not _NATIVE_ANTHROPIC_MODELS:
        return None
    try:
        data = json.loads(body_bytes)
    except (ValueError, TypeError) as e:
        log.debug("Native route lookup: invalid JSON: %s", e)
        return None
    return _NATIVE_ANTHROPIC_MODELS.get(data.get("model", ""))


def _needs_openai_translation(body_bytes):
    """Check if this Anthropic /v1/messages request targets a model that
    needs OpenAI-compatible translation (e.g. MiniMax via openai/ prefix)."""
    if not _OPENAI_TRANSLATED_MODELS:
        return False
    try:
        data = json.loads(body_bytes)
    except (ValueError, TypeError):
        return False
    return data.get("model", "") in _OPENAI_TRANSLATED_MODELS


def _anthropic_to_openai(body_bytes, thinking_effort=None):
    """Convert Anthropic /v1/messages request to OpenAI /v1/chat/completions format.

    Handles: system messages, text content, tool definitions, tool_use (assistant),
    and tool_result (user) blocks for full agentic coding tool support.
    """
    data = json.loads(body_bytes)
    messages = []

    # Convert system to a system message
    system = data.pop("system", None)
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in system
            )
            if text:
                messages.append({"role": "system", "content": text})

    # Convert messages (handles text, tool_use, tool_result content blocks)
    for msg in data.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            messages.append({"role": role, "content": str(content) if content else ""})
            continue

        # Process content blocks
        text_parts = []
        tool_calls = []
        tool_results = []

        for block in content:
            if not isinstance(block, dict):
                text_parts.append(str(block))
                continue
            btype = block.get("type", "")

            if btype == "text":
                text_parts.append({"type": "text", "text": block.get("text", "")})

            elif btype == "image":
                # Anthropic image → OpenAI image_url (base64 data URI)
                source = block.get("source", {})
                media_type = source.get("media_type", "image/png")
                b64_data = source.get("data", "")
                if b64_data:
                    text_parts.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64_data}"}})

            elif btype == "tool_use":
                # Assistant's tool call → OpenAI tool_calls
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

            elif btype == "tool_result":
                # User's tool result → OpenAI tool role message
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_content = "\n".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in result_content
                    )
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": str(result_content),
                })

        # Flatten text_parts: if all items are text-only, use a plain string;
        # if any images are present, use the OpenAI multimodal content array.
        has_images = any(isinstance(p, dict) and p.get("type") == "image_url" for p in text_parts)
        if has_images:
            flat_content = text_parts  # already in OpenAI multimodal format
        else:
            flat_content = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text_parts)

        # Emit messages in the right order
        # OpenAI requires: assistant (with tool_calls) → tool results → user text
        if role == "assistant":
            msg_obj = {"role": "assistant"}
            if isinstance(flat_content, str) and flat_content:
                msg_obj["content"] = flat_content
            else:
                msg_obj["content"] = None
            if tool_calls:
                msg_obj["tool_calls"] = tool_calls
            messages.append(msg_obj)
        elif tool_results:
            # Tool results MUST come immediately after assistant's tool_calls
            for tr in tool_results:
                messages.append(tr)
            # Any user text/images go AFTER tool results
            if text_parts:
                messages.append({"role": "user", "content": flat_content})
        else:
            messages.append({"role": role, "content": flat_content if text_parts else ""})

    openai_body = {
        "model": data.get("model"),
        "messages": messages,
    }
    if data.get("max_tokens"):
        openai_body["max_tokens"] = data["max_tokens"]
    if data.get("temperature") is not None:
        openai_body["temperature"] = data["temperature"]
    if data.get("stream"):
        openai_body["stream"] = True

    # Inject thinking/reasoning effort if set
    if thinking_effort:
        openai_body["reasoning_effort"] = thinking_effort

    # Convert Anthropic tools → OpenAI tools
    tools = data.get("tools")
    if tools:
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        openai_body["tools"] = openai_tools

    return json.dumps(openai_body).encode()


def _openai_to_anthropic(response_bytes):
    """Convert OpenAI /v1/chat/completions response to Anthropic /v1/messages format.

    Handles text content and tool_calls (function calling).
    """
    try:
        data = json.loads(response_bytes)
    except (ValueError, TypeError):
        return response_bytes

    choices = data.get("choices", [])
    content = []
    stop_reason = "end_turn"

    if choices:
        msg = choices[0].get("message", {})
        finish_reason = choices[0].get("finish_reason", "stop")

        # Text content
        text = _strip_think_tags(msg.get("content", "") or "")
        if text:
            content.append({"type": "text", "text": text})

        # Tool calls
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            stop_reason = "tool_use"
            for tc in tool_calls:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (ValueError, TypeError):
                    args = {}
                content.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args,
                })
        elif finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"

    usage = data.get("usage", {})
    anthropic_resp = {
        "id": data.get("id", ""),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": data.get("model", ""),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }
    return json.dumps(anthropic_resp).encode()


def _strip_think_tags(text):
    """Strip MiniMax <think>...</think> reasoning blocks from response text."""
    if not text or "<think>" not in text:
        return text
    # Remove <think>...</think> blocks (including multiline)
    cleaned = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    return cleaned.strip()


def _is_streaming(resp):
    """Return True if the upstream response should be streamed to the client."""
    ct = resp.getheader("Content-Type", "").lower()
    te = resp.getheader("Transfer-Encoding", "").lower()
    return "text/event-stream" in ct or "chunked" in te


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_error(self, status_code, message, error_type="proxy_error"):
        code, body = _error_response(status_code, message, error_type)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _try_send_error(self, status_code, message, error_type="proxy_error"):
        """Send error response, logging if client already disconnected."""
        try:
            self._send_error(status_code, message, error_type)
        except (BrokenPipeError, ConnectionResetError, OSError):
            log.debug("Client disconnected before %d response could be sent", status_code)

    def _handle_upstream_error(self, exc, method, path, circuit_provider=None):
        """Handle upstream connection errors with logging, counters, and optional circuit breaker."""
        if isinstance(exc, socket.timeout):
            log.warning("Upstream timeout for %s %s: %s", method, path, exc)
            _inc_counter("upstream_timeout")
            self._try_send_error(504, "Upstream timeout", "upstream_error")
        elif isinstance(exc, ConnectionRefusedError):
            log.error("Upstream refused for %s %s: %s", method, path, exc)
            _inc_counter("upstream_refused")
            self._try_send_error(502, "Upstream connection refused", "upstream_error")
        elif isinstance(exc, http.client.HTTPException):
            log.error("Upstream HTTP error for %s %s: %s", method, path, exc)
            _inc_counter("upstream_http_error")
            self._try_send_error(502, "Upstream HTTP error", "upstream_error")
        elif isinstance(exc, OSError):
            log.error("Upstream I/O error for %s %s: %s", method, path, exc)
            _inc_counter("upstream_io_error")
            self._try_send_error(502, "Upstream I/O error", "upstream_error")
        if circuit_provider:
            _circuit.record_failure(circuit_provider)

    def _proxy(self, method):
        raw_cl = self.headers.get("Content-Length")
        if raw_cl is None:
            if method in ("POST", "PUT", "PATCH"):
                self._send_error(411, "Content-Length required", "validation_error")
                return
            length = 0
        else:
            try:
                length = int(raw_cl)
            except ValueError:
                self._send_error(400, "Invalid Content-Length header", "validation_error")
                return
            if length < 0:
                self._send_error(400, "Invalid Content-Length header", "validation_error")
                return

        if length > MAX_REQUEST_BODY:
            self._send_error(413, "Request body too large", "validation_error")
            return

        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            self._send_error(400, "Chunked transfer encoding is not supported", "validation_error")
            return

        body = self.rfile.read(length) if length else b""

        translate_response = False
        native_route = None
        if method == "POST" and "/v1/messages" in self.path:
            err = _validate_messages(body)
            if err:
                _inc_counter("invalid_request")
                self._send_error(400, err, "validation_error")
                return
            # Remap unconfigured models to a configured fallback
            # (Claude Code sends background requests to claude-haiku which isn't configured)
            body = _remap_model_if_needed(body)
            # Check routing: native Anthropic endpoint > OpenAI translation > LiteLLM passthrough
            native_route = _get_native_route(body)
            if native_route:
                log.debug("Native Anthropic route for %s", self.path)
            elif _needs_openai_translation(body):
                # Read thinking effort from custom header
                thinking = self.headers.get("x-thinking-effort")
                # Rewrite Anthropic request to OpenAI format for openai/ models
                body = _anthropic_to_openai(body, thinking_effort=thinking)
                self.path = "/v1/chat/completions"
                translate_response = True
                log.debug("Translated Anthropic→OpenAI for %s (thinking=%s)", self.path, thinking or "default")
            else:
                body = strip_system(body)

        # Extract model name for logging
        _model = ""
        if method == "POST" and body:
            try:
                _model = json.loads(body).get("model", "")
            except (ValueError, TypeError):
                pass
        _t0 = time.time()

        # --- Native Anthropic forwarding (bypass LiteLLM) ---
        if native_route:
            self._forward_native(method, body, native_route, _model, _t0)
            return

        conn = http.client.HTTPConnection(LITELLM_HOST, LITELLM_PORT, timeout=CONNECT_TIMEOUT)
        try:
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "content-length", "transfer-encoding")}
            headers["Content-Length"] = str(len(body))
            headers["Host"] = f"{LITELLM_HOST}:{LITELLM_PORT}"

            # When translating Anthropic→OpenAI, convert auth header (case-insensitive)
            if translate_response:
                api_key_value = None
                for k, v in self.headers.items():
                    if k.lower() == "x-api-key":
                        api_key_value = v
                        break
                # Remove all case variants of x-api-key from forwarded headers
                headers = {k: v for k, v in headers.items() if k.lower() != "x-api-key"}
                if api_key_value:
                    headers["Authorization"] = "Bearer %s" % api_key_value

            conn.request(method, self.path, body=body if method in ("POST", "PUT", "PATCH") else None, headers=headers)
            # Use shorter timeout for non-streaming requests
            try:
                is_stream_request = json.loads(body).get("stream", False) if body else False
            except (ValueError, TypeError):
                is_stream_request = False
            conn.sock.settimeout(STREAM_IDLE_TIMEOUT if is_stream_request else NON_STREAM_READ_TIMEOUT)
            resp = conn.getresponse()
            _t1 = time.time()
            _xlate = " [translated]" if translate_response else ""
            log.info("%s %s model=%s status=%d %.1fs%s",
                     method, self.path, _model or "-", resp.status, _t1 - _t0, _xlate)

            if _is_streaming(resp):
                self._stream_response(resp, conn, translate_response)
            elif resp.getheader("Content-Length") is None:
                self._stream_response(resp, conn, translate_response)
            else:
                try:
                    upstream_cl = int(resp.getheader("Content-Length"))
                except (TypeError, ValueError):
                    upstream_cl = 0
                if upstream_cl > MAX_RESPONSE_BODY:
                    self._stream_response(resp, conn, translate_response)
                else:
                    self._buffer_response(resp, conn, translate_response)
        except (socket.timeout, ConnectionRefusedError, http.client.HTTPException, OSError) as e:
            self._handle_upstream_error(e, method, self.path)
        finally:
            conn.close()

    def _buffer_response(self, resp, conn, translate=False):
        """Forward a non-streaming response after fully buffering it (with size cap)."""
        buf = bytearray()
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
            if len(buf) > MAX_RESPONSE_BODY:
                log.warning("Upstream response exceeded %d bytes for %s", MAX_RESPONSE_BODY, self.path)
                self._send_error(502, "Upstream response too large", "upstream_error")
                return
        if translate and resp.status == 200:
            buf = _openai_to_anthropic(bytes(buf))
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "keep-alive", "content-length"):
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(buf)))
        self.end_headers()
        self.wfile.write(buf)

    def _forward_native(self, method, body, route, model_name, t0):
        """Forward request directly to a provider's native Anthropic endpoint via HTTPS."""
        host = route["host"]
        port = route["port"]
        base_path = route["path"]
        api_key_env = route["api_key_env"]
        auth_header = route.get("auth_header", "x-api-key")

        # Circuit breaker: reject immediately if provider is in open state
        if _circuit.is_open(model_name):
            log.warning("Circuit breaker open for %s, rejecting request", model_name)
            _inc_counter("circuit_breaker_rejected")
            self._send_error(503, "Provider temporarily unavailable (circuit open)", "upstream_error")
            return

        # Get API key from environment (loaded by litellm.sh from .env)
        api_key = os.environ.get(api_key_env, "") if api_key_env else None
        if not api_key:
            log.error("Native forward: no API key for model %s", model_name)
            self._send_error(502, "Provider authentication not configured", "auth_error")
            return

        # Build path: base_path + original request path
        # e.g. /anthropic + /v1/messages = /anthropic/v1/messages
        forward_path = base_path + self.path
        log.info("Native forward: %s %s model=%s → %s:%d%s", method, self.path, model_name, host, port, forward_path)

        ctx = ssl.create_default_context()
        conn = http.client.HTTPSConnection(host, port, timeout=CONNECT_TIMEOUT, context=ctx)
        try:
            # Case-insensitive exclusion of hop-by-hop and auth headers
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "content-length", "transfer-encoding",
                                            "x-api-key", "authorization")}
            headers["Content-Length"] = str(len(body))
            headers["Host"] = host
            # Add provider's API key
            headers[auth_header] = api_key

            conn.request(method, forward_path, body=body if method in ("POST", "PUT", "PATCH") else None, headers=headers)
            # Determine if request is streaming
            try:
                is_stream_request = json.loads(body).get("stream", False) if body else False
            except (ValueError, TypeError):
                is_stream_request = False
            conn.sock.settimeout(STREAM_IDLE_TIMEOUT if is_stream_request else NON_STREAM_READ_TIMEOUT)
            resp = conn.getresponse()
            _t1 = time.time()
            log.info("Native response: %s %s model=%s status=%d %.1fs",
                     method, forward_path, model_name, resp.status, _t1 - t0)

            if resp.status >= 400:
                _circuit.record_failure(model_name)
                try:
                    raw_err = resp.read(4096)
                    log.debug("Native upstream %d raw body: %s", resp.status, raw_err.decode("utf-8", errors="replace")[:500])
                except Exception as e:
                    log.error("Failed to read native error body: %s", e)
                    raw_err = b""
                log.warning("Native upstream error %d for %s %s", resp.status, method, forward_path)

                # Map upstream status to appropriate proxy status
                if resp.status == 401 or resp.status == 403:
                    err_code, err_body = _error_response(502, "Provider authentication failed", "auth_error")
                elif resp.status == 429:
                    err_code, err_body = _error_response(429, "Provider rate limited — retry later", "upstream_error")
                elif resp.status >= 500:
                    err_code, err_body = _error_response(502, "Provider temporarily unavailable", "upstream_error")
                else:
                    err_code, err_body = _error_response(502, "Provider request failed", "upstream_error")

                try:
                    self.send_response(err_code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_body)))
                    if resp.status == 429:
                        retry_after = resp.getheader("Retry-After")
                        if retry_after:
                            self.send_header("Retry-After", retry_after)
                    self.end_headers()
                    self.wfile.write(err_body)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    log.debug("Client disconnected before native error response could be sent")
                return

            # Record success for circuit breaker
            _circuit.record_success(model_name)

            # Validate Content-Type on non-streaming success responses
            ct = resp.getheader("Content-Type", "")
            if resp.status == 200 and not _is_streaming(resp):
                if "application/json" not in ct.lower() and "text/event-stream" not in ct.lower():
                    log.warning("Native upstream returned unexpected Content-Type: %s for %s", ct, forward_path)
                    self._send_error(502, "Provider returned unexpected content type", "upstream_error")
                    return

            if _is_streaming(resp):
                self._stream_response(resp, conn, translate=False)
            elif resp.getheader("Content-Length") is None:
                self._stream_response(resp, conn, translate=False)
            else:
                try:
                    upstream_cl = int(resp.getheader("Content-Length"))
                except (TypeError, ValueError):
                    upstream_cl = 0
                if upstream_cl > MAX_RESPONSE_BODY:
                    self._stream_response(resp, conn, translate=False)
                else:
                    self._buffer_response(resp, conn, translate=False)
        except (socket.timeout, ConnectionRefusedError, http.client.HTTPException, OSError) as e:
            self._handle_upstream_error(e, method, forward_path, circuit_provider=model_name)
        finally:
            conn.close()

    def _stream_translated(self, resp, conn):
        """Read OpenAI SSE stream, translate each chunk to Anthropic SSE events inline.

        Strips MiniMax <think>...</think> reasoning blocks incrementally:
        suppresses text while inside a think block, streams normally after.
        """
        try:
            resp.fp.raw._sock.settimeout(STREAM_IDLE_TIMEOUT)
            log.debug("SSE xlate: idle timeout set to %ds", STREAM_IDLE_TIMEOUT)
        except (AttributeError, TypeError) as e:
            log.warning("SSE xlate: could not set idle timeout: %s", e)

        started = False
        content_block_index = 0  # tracks current content block index
        text_block_open = False
        msg_id = ""
        model_name = ""
        input_tokens = 0
        output_tokens = 0
        total_bytes_read = 0
        chunks_received = 0
        sse_events_sent = 0
        text_chars_sent = 0
        text_chars_suppressed = 0  # eaten by think filter
        # Think-tag state machine
        in_think = False
        think_buf = ""
        past_think = False
        # Tool call accumulation: {index: {"id": str, "name": str, "arguments": str}}
        tool_calls = {}
        tool_blocks_started = set()  # tool call indices we've sent content_block_start for
        buf = b""
        done = False
        start_time = time.monotonic()
        last_chunk_time = start_time

        def _send_event(event_str):
            nonlocal sse_events_sent
            event_bytes = event_str.encode()
            self.wfile.write(f"{len(event_bytes):x}\r\n".encode())
            self.wfile.write(event_bytes)
            self.wfile.write(b"\r\n")
            self.wfile.flush()
            sse_events_sent += 1

        def _send_text_delta(text):
            nonlocal text_block_open, content_block_index, text_chars_sent
            if not text:
                return
            if not text_block_open:
                text_block_open = True
                _send_event(f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n")
                _send_event("event: ping\ndata: {\"type\": \"ping\"}\n\n")
            evt = {"type": "content_block_delta", "index": content_block_index, "delta": {"type": "text_delta", "text": text}}
            _send_event(f"event: content_block_delta\ndata: {json.dumps(evt)}\n\n")
            text_chars_sent += len(text)

        def _process_text(text):
            """Filter text through think-tag state machine. Sends clean text immediately."""
            nonlocal in_think, think_buf, past_think, text_chars_suppressed
            if past_think:
                _send_text_delta(text)
                return
            think_buf += text
            while think_buf:
                if in_think:
                    end_idx = think_buf.find("</think>")
                    if end_idx >= 0:
                        suppressed = think_buf[:end_idx]
                        text_chars_suppressed += len(suppressed)
                        think_buf = think_buf[end_idx + 8:]
                        in_think = False
                        past_think = True
                        log.debug("SSE xlate: </think> found, suppressed %d chars, resuming stream", text_chars_suppressed)
                        remaining = think_buf.lstrip()
                        think_buf = ""
                        if remaining:
                            _send_text_delta(remaining)
                        return
                    else:
                        text_chars_suppressed += len(think_buf)
                        think_buf = ""
                        return
                else:
                    start_idx = think_buf.find("<think>")
                    if start_idx >= 0:
                        before = think_buf[:start_idx]
                        if before.strip():
                            _send_text_delta(before)
                        think_buf = think_buf[start_idx + 7:]
                        in_think = True
                        log.debug("SSE xlate: <think> entered, suppressing content")
                    elif "<" in think_buf and len(think_buf) < 7:
                        return
                    else:
                        past_think = True
                        _send_text_delta(think_buf)
                        think_buf = ""
                        return

        def _log_summary(exit_reason):
            elapsed = time.monotonic() - start_time
            log.info("SSE xlate done: reason=%s elapsed=%.1fs model=%s chunks=%d bytes_read=%d "
                     "sse_sent=%d text_sent=%d text_suppressed=%d tokens_in=%d tokens_out=%d "
                     "started=%s done=%s in_think=%s finish=%s",
                     exit_reason, elapsed, model_name, chunks_received, total_bytes_read,
                     sse_events_sent, text_chars_sent, text_chars_suppressed,
                     input_tokens, output_tokens,
                     started, done, in_think, finish_reason_seen or "none")

        finish_reason_seen = None

        try:
            while not done:
                if time.monotonic() - start_time > MAX_STREAM_LIFETIME:
                    log.warning("SSE xlate: lifetime exceeded %ds for %s", MAX_STREAM_LIFETIME, self.path)
                    _log_summary("lifetime_exceeded")
                    break

                data = resp.read(4096)
                now = time.monotonic()

                if not data:
                    if not done and not finish_reason_seen:
                        log.warning("SSE xlate: upstream EOF without [DONE] or finish_reason "
                                    "(chunks=%d, bytes=%d, in_think=%s, tokens=%d, idle=%.1fs)",
                                    chunks_received, total_bytes_read, in_think, output_tokens,
                                    now - last_chunk_time)
                        _inc_counter("xlate_stream_eof_no_finish")
                    _log_summary("eof")
                    break

                idle_gap = now - last_chunk_time
                last_chunk_time = now
                total_bytes_read += len(data)
                chunks_received += 1

                if idle_gap > 5.0:
                    log.debug("SSE xlate: chunk %d after %.1fs idle (%d bytes)", chunks_received, idle_gap, len(data))

                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str or not line_str.startswith("data: "):
                        continue
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        log.debug("SSE xlate: [DONE] received (tokens=%d, in_think=%s)", output_tokens, in_think)
                        done = True
                        break
                    try:
                        chunk = json.loads(data_str)
                    except (ValueError, TypeError) as e:
                        log.debug("SSE xlate: unparseable chunk: %s (data=%s)", e, data_str[:200])
                        continue

                    if not msg_id:
                        msg_id = chunk.get("id", "msg_translated")
                        model_name = chunk.get("model", "")
                        log.debug("SSE xlate: stream started id=%s model=%s", msg_id, model_name)
                    usage = chunk.get("usage", {})
                    if usage:
                        input_tokens = usage.get("prompt_tokens", input_tokens)
                        output_tokens = usage.get("completion_tokens", output_tokens)

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    finish_reason = choices[0].get("finish_reason")

                    if not started:
                        started = True
                        evt = {
                            "type": "message_start",
                            "message": {
                                "id": msg_id, "type": "message", "role": "assistant",
                                "content": [], "model": model_name,
                                "stop_reason": None, "stop_sequence": None,
                                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                            },
                        }
                        _send_event(f"event: message_start\ndata: {json.dumps(evt)}\n\n")

                    # Handle text content
                    text = delta.get("content", "")
                    if text:
                        output_tokens += 1
                        _process_text(text)

                    # Handle tool calls
                    delta_tool_calls = delta.get("tool_calls", [])
                    for tc in delta_tool_calls:
                        tc_idx = tc.get("index", 0)
                        fn = tc.get("function", {})

                        if tc_idx not in tool_calls:
                            tool_calls[tc_idx] = {"id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": ""}
                            log.debug("SSE xlate: tool_call[%d] started name=%s", tc_idx, fn.get("name", "?"))

                        if fn.get("name"):
                            tool_calls[tc_idx]["name"] = fn["name"]
                        if tc.get("id"):
                            tool_calls[tc_idx]["id"] = tc["id"]
                        if fn.get("arguments"):
                            tool_calls[tc_idx]["arguments"] += fn["arguments"]

                            # Stream the tool call arguments as input_json_delta
                            if tc_idx not in tool_blocks_started:
                                # Close text block if open
                                if text_block_open:
                                    _send_event(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n")
                                    content_block_index += 1
                                    text_block_open = False
                                tool_blocks_started.add(tc_idx)
                                block_idx = content_block_index + tc_idx
                                _send_event(f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_idx, 'content_block': {'type': 'tool_use', 'id': tool_calls[tc_idx]['id'], 'name': tool_calls[tc_idx]['name'], 'input': {}}})}\n\n")

                            block_idx = content_block_index + tc_idx
                            evt = {"type": "content_block_delta", "index": block_idx, "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}}
                            _send_event(f"event: content_block_delta\ndata: {json.dumps(evt)}\n\n")

                    if finish_reason:
                        finish_reason_seen = finish_reason
                        log.debug("SSE xlate: finish_reason=%s (tokens=%d, text_sent=%d)",
                                  finish_reason, output_tokens, text_chars_sent)
                        # Close any open text block
                        if text_block_open:
                            _send_event(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n")
                            content_block_index += 1
                            text_block_open = False
                        # Close any open tool blocks (indices are relative to content_block_index)
                        for tc_idx in sorted(tool_blocks_started):
                            block_idx = content_block_index + tc_idx
                            _send_event(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n")

                        stop = "end_turn" if finish_reason == "stop" else "tool_use" if finish_reason == "tool_calls" else finish_reason
                        evt = {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": output_tokens}}
                        _send_event(f"event: message_delta\ndata: {json.dumps(evt)}\n\n")
                        _send_event("event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n")

            if done or finish_reason_seen:
                _log_summary("complete")

        except (socket.timeout, BrokenPipeError, ConnectionResetError, OSError) as e:
            log.warning("SSE xlate: stream error: %s [%s]", type(e).__name__, e)
            _inc_counter("xlate_stream_errors")
            _log_summary("error_%s" % type(e).__name__)
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            try:
                conn.close()
            except OSError:
                pass

    def _stream_response(self, resp, conn, translate=False):
        """Forward an SSE / chunked response incrementally with idle timeout."""
        self.send_response(resp.status)

        if translate and resp.status == 200:
            # Send Anthropic-style SSE headers instead of forwarding upstream headers
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self._stream_translated(resp, conn)
            return

        for k, v in resp.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "keep-alive", "content-length"):
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        # Set idle timeout on the upstream socket so resp.read() raises
        # socket.timeout if upstream goes silent. This works correctly with
        # buffered reads (unlike select.select which can false-negative when
        # the BufferedReader has data in its internal buffer).
        try:
            resp.fp.raw._sock.settimeout(STREAM_IDLE_TIMEOUT)
        except (AttributeError, TypeError) as e:
            log.warning("Could not set stream idle timeout: %s (protection inactive)", e)
            _inc_counter("idle_timeout_inactive")
            fallback_timeout = min(STREAM_IDLE_TIMEOUT, 60)
            try:
                conn.sock.settimeout(fallback_timeout)
            except (AttributeError, OSError) as fallback_err:
                log.warning("Fallback stream timeout also failed: %s", fallback_err)

        start_time = time.monotonic()
        total_streamed = 0
        budget_killed = False
        try:
            while True:
                if time.monotonic() - start_time > MAX_STREAM_LIFETIME:
                    log.warning("Stream lifetime exceeded %ds for %s", MAX_STREAM_LIFETIME, self.path)
                    budget_killed = True
                    _inc_counter("stream_budget_killed")
                    break
                chunk = resp.read(4096)
                if not chunk:
                    break
                total_streamed += len(chunk)
                if total_streamed > MAX_STREAM_BYTES:
                    log.warning("Stream byte budget exceeded %d for %s", MAX_STREAM_BYTES, self.path)
                    budget_killed = True
                    _inc_counter("stream_budget_killed")
                    break
                self.wfile.write(f"{len(chunk):x}\r\n".encode())
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            if not budget_killed:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    log.debug("Could not send stream terminator: %s", e)
        except socket.timeout:
            log.warning("Stream idle timeout %ds for %s", STREAM_IDLE_TIMEOUT, self.path)
            _inc_counter("truncated_stream")
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            log.debug("Client disconnected during stream: %s", e)

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")

    def log_message(self, fmt, *args):
        log.debug(fmt, *args)


class BoundedThreadServer(HTTPServer):
    """HTTPServer with bounded thread pool and client socket timeout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self._semaphore = threading.Semaphore(MAX_WORKERS)

    def process_request(self, req, addr):
        if not self._semaphore.acquire(blocking=False):
            # Pool is full — reject immediately
            try:
                req.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: 62\r\n"
                    b"Connection: close\r\n\r\n"
                    b'{"error":{"message":"Server overloaded","type":"overload_error"}}'
                )
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                log.debug("503 overload response failed (client gone): %s", e)
            self.shutdown_request(req)
            return
        self._pool.submit(self._handle, req, addr)

    def _handle(self, req, addr):
        try:
            req.settimeout(SOCKET_TIMEOUT)
            self.finish_request(req, addr)
        except Exception as e:
            log.error("Unhandled request error for %s: %s", addr, e, exc_info=True)
            _inc_counter("handler_errors")
        finally:
            self.shutdown_request(req)
            self._semaphore.release()

    def server_close(self):
        super().server_close()
        self._pool.shutdown(wait=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info("Proxy :%d -> LiteLLM :%d (workers=%d)", LISTEN_PORT, LITELLM_PORT, MAX_WORKERS)
    _load_translated_models()
    BoundedThreadServer(("127.0.0.1", LISTEN_PORT), Handler).serve_forever()
