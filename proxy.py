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
import threading
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
MAX_REQUEST_BODY = _parse_size(os.environ.get("PROXY_MAX_REQUEST_BODY"), 10 * 1024**2)   # 10MB
MAX_RESPONSE_BODY = _parse_size(os.environ.get("PROXY_MAX_RESPONSE_BODY"), 2 * 1024**2)   # 2MB
CONNECT_TIMEOUT = _env_int("PROXY_CONNECT_TIMEOUT", 10)
READ_TIMEOUT = _env_int("PROXY_READ_TIMEOUT", 300)
STREAM_IDLE_TIMEOUT = _env_int("PROXY_STREAM_IDLE_TIMEOUT", 60)
MAX_STREAM_LIFETIME = _env_int("PROXY_MAX_STREAM_LIFETIME", 600)  # 10 min
MAX_STREAM_BYTES = _parse_size(os.environ.get("PROXY_MAX_STREAM_BYTES"), 100 * 1024**2)  # 100MB
SOCKET_TIMEOUT = _env_int("PROXY_SOCKET_TIMEOUT", 30)


_COUNTERS = {
    "stream_budget_killed": 0,
    "idle_timeout_inactive": 0,
    "truncated_stream": 0,
    "invalid_request": 0,
}


def _print_counters():
    # Uses print() instead of log.warning() because atexit handlers run during
    # interpreter shutdown after logging.shutdown() has flushed and closed all
    # handlers. log calls here would be silently dropped.
    if any(_COUNTERS.values()):
        print(f"Proxy counters: {_COUNTERS}", file=sys.stderr, flush=True)


atexit.register(_print_counters)


def _error_response(status_code, message):
    """Build a JSON error body and return (status_code, body_bytes)."""
    return status_code, json.dumps(
        {"error": {"message": message, "type": "proxy_error"}}
    ).encode()


def _validate_messages(body_bytes):
    """Validate /v1/messages request schema. Returns error string or None."""
    try:
        data = json.loads(body_bytes)
    except (ValueError, TypeError):
        return "Request body must be valid JSON"
    if not isinstance(data, dict):
        return "Request body must be a JSON object"
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

def _load_translated_models():
    """Load the set of model names that use openai/ prefix (need Anthropic→OpenAI translation).
    Called once at startup, cached for all subsequent requests."""
    global _OPENAI_TRANSLATED_MODELS
    models = set()
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "litellm_config.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        for entry in cfg.get("model_list", []):
            litellm_model = entry.get("litellm_params", {}).get("model", "")
            if litellm_model.startswith("openai/"):
                models.add(entry.get("model_name", ""))
    except Exception:
        pass
    _OPENAI_TRANSLATED_MODELS = models
    log.debug("Models needing OpenAI translation: %s", models)


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


def _anthropic_to_openai(body_bytes):
    """Convert Anthropic /v1/messages request body to OpenAI /v1/chat/completions format."""
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

    # Convert messages
    for msg in data.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Flatten content blocks to text
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)
        messages.append({"role": role, "content": content})

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

    return json.dumps(openai_body).encode()


def _openai_to_anthropic(response_bytes):
    """Convert OpenAI /v1/chat/completions response to Anthropic /v1/messages format."""
    try:
        data = json.loads(response_bytes)
    except (ValueError, TypeError):
        return response_bytes

    choices = data.get("choices", [])
    content = []
    if choices:
        msg = choices[0].get("message", {})
        text = msg.get("content", "")
        if text:
            content.append({"type": "text", "text": text})

    usage = data.get("usage", {})
    anthropic_resp = {
        "id": data.get("id", ""),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": data.get("model", ""),
        "stop_reason": choices[0].get("finish_reason", "end_turn") if choices else "end_turn",
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }
    return json.dumps(anthropic_resp).encode()


def _is_streaming(resp):
    """Return True if the upstream response should be streamed to the client."""
    ct = resp.getheader("Content-Type", "").lower()
    te = resp.getheader("Transfer-Encoding", "").lower()
    return "text/event-stream" in ct or "chunked" in te


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_error(self, status_code, message):
        code, body = _error_response(status_code, message)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _try_send_error(self, status_code, message):
        """Send error response, logging if client already disconnected."""
        try:
            self._send_error(status_code, message)
        except (BrokenPipeError, ConnectionResetError, OSError):
            log.debug("Client disconnected before %d response could be sent", status_code)

    def _proxy(self, method):
        raw_cl = self.headers.get("Content-Length")
        if raw_cl is None:
            if method in ("POST", "PUT", "PATCH"):
                self._send_error(411, "Content-Length required")
                return
            length = 0
        else:
            try:
                length = int(raw_cl)
            except ValueError:
                self._send_error(400, "Invalid Content-Length header")
                return
            if length < 0:
                self._send_error(400, "Invalid Content-Length header")
                return

        if length > MAX_REQUEST_BODY:
            self._send_error(413, "Request body too large")
            return

        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            self._send_error(400, "Chunked transfer encoding is not supported")
            return

        body = self.rfile.read(length) if length else b""

        translate_response = False
        if method == "POST" and "/v1/messages" in self.path:
            err = _validate_messages(body)
            if err:
                _COUNTERS["invalid_request"] += 1
                self._send_error(400, err)
                return
            if _needs_openai_translation(body):
                # Rewrite Anthropic request to OpenAI format for openai/ models
                body = _anthropic_to_openai(body)
                self.path = "/v1/chat/completions"
                translate_response = True
                log.debug("Translated Anthropic→OpenAI for %s", self.path)
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

        conn = http.client.HTTPConnection(LITELLM_HOST, LITELLM_PORT, timeout=CONNECT_TIMEOUT)
        try:
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "content-length", "transfer-encoding")}
            headers["Content-Length"] = str(len(body))
            headers["Host"] = f"{LITELLM_HOST}:{LITELLM_PORT}"

            # When translating Anthropic→OpenAI, convert auth header
            if translate_response:
                api_key = headers.pop("x-api-key", None) or headers.pop("X-Api-Key", None)
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

            conn.request(method, self.path, body=body if method in ("POST", "PUT", "PATCH") else None, headers=headers)
            conn.sock.settimeout(READ_TIMEOUT)
            resp = conn.getresponse()
            _t1 = time.time()
            _xlate = " [translated]" if translate_response else ""
            log.info("%s %s model=%s status=%d %.1fs%s",
                     method, self.path, _model or "-", resp.status, _t1 - _t0, _xlate)

            if _is_streaming(resp):
                self._stream_response(resp, conn)
            elif resp.getheader("Content-Length") is None:
                self._stream_response(resp, conn)
            else:
                try:
                    upstream_cl = int(resp.getheader("Content-Length"))
                except (TypeError, ValueError):
                    upstream_cl = 0
                if upstream_cl > MAX_RESPONSE_BODY:
                    self._stream_response(resp, conn)
                else:
                    self._buffer_response(resp, conn, translate_response)
        except socket.timeout as e:
            log.warning("Upstream timeout for %s %s: %s", method, self.path, e)
            self._try_send_error(504, "Upstream timeout")
        except ConnectionRefusedError as e:
            log.error("Upstream refused for %s %s: %s", method, self.path, e)
            self._try_send_error(502, "Upstream connection refused")
        except http.client.HTTPException as e:
            log.error("Upstream HTTP error for %s %s: %s", method, self.path, e)
            self._try_send_error(502, "Upstream HTTP error")
        except OSError as e:
            log.error("Upstream I/O error for %s %s: %s", method, self.path, e)
            self._try_send_error(502, "Upstream I/O error")
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
                self._send_error(502, "Upstream response too large")
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

    def _stream_response(self, resp, conn):
        """Forward an SSE / chunked response incrementally with idle timeout."""
        self.send_response(resp.status)
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
            _COUNTERS["idle_timeout_inactive"] += 1
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
                    _COUNTERS["stream_budget_killed"] += 1
                    break
                chunk = resp.read(4096)
                if not chunk:
                    break
                total_streamed += len(chunk)
                if total_streamed > MAX_STREAM_BYTES:
                    log.warning("Stream byte budget exceeded %d for %s", MAX_STREAM_BYTES, self.path)
                    budget_killed = True
                    _COUNTERS["stream_budget_killed"] += 1
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
            _COUNTERS["truncated_stream"] += 1
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
                    b'{"error":{"message":"Server overloaded","type":"proxy_error"}}'
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
        except Exception:
            log.error("Unhandled request error for %s", addr, exc_info=True)
        finally:
            self.shutdown_request(req)
            self._semaphore.release()

    def server_close(self):
        super().server_close()
        self._pool.shutdown(wait=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info("Proxy :%d -> LiteLLM :%d (workers=%d)", LISTEN_PORT, LITELLM_PORT, MAX_WORKERS)
    _load_translated_models()
    BoundedThreadServer(("127.0.0.1", LISTEN_PORT), Handler).serve_forever()
