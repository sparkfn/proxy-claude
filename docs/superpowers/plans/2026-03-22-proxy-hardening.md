# Proxy Hardening Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the LiteLLM proxy against upstream misbehavior, resource exhaustion, state corruption, and protocol violations.

**Architecture:** Five files are modified in-place. No new files except updating `.env.example`. proxy.py gets the most changes (config system, bounded concurrency, timeouts, size limits). container.py gets FD leak fix and subprocess timeouts. config.py gets a corruption guard. Both provider files get response validation.

**Tech Stack:** Python stdlib (http.server, http.client, concurrent.futures, threading, select, socket, json), PyYAML, requests

**Spec:** `docs/superpowers/specs/2026-03-22-proxy-hardening-design.md`

---

## Task 1: proxy.py — Configuration system and size parser

**Files:**
- Modify: `proxy.py:1-20` (imports and module-level constants)

- [ ] **Step 1: Add imports and `_parse_size` helper**

Replace lines 1-20 of `proxy.py` with:

```python
"""
Reverse proxy: strips system messages from Anthropic /v1/messages requests
before forwarding to LiteLLM. Supports SSE streaming pass-through so
Claude Code sees tokens incrementally. Runs on the host at port 2555.

Required because chatgpt/ provider rejects system messages and LiteLLM
doesn't strip them in the Anthropic-to-Responses translation path.
"""

import json
import os
import re
import sys
import http.client
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse


def _parse_size(value, default):
    """Parse a human-readable size string (e.g. '10MB', '512KB', '1GB') to bytes.

    Accepts B, KB, MB, GB suffixes (case-insensitive). Raw integers treated as bytes.
    Returns default with a stderr warning on invalid input.
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
    print(f"Warning: invalid size '{value}', using default {default}", file=sys.stderr, flush=True)
    return default


def _env_int(name, default):
    """Read an integer from environment, falling back to default."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        print(f"Warning: invalid integer for {name}='{val}', using default {default}", file=sys.stderr, flush=True)
        return default


# --- Configuration (all from environment, no config.py dependency) ---
LITELLM_HOST = os.environ.get("PROXY_LITELLM_HOST", "localhost")
LITELLM_PORT = _env_int("PROXY_LITELLM_PORT", 4000)
LISTEN_PORT = _env_int("PROXY_LISTEN_PORT", int(sys.argv[1]) if len(sys.argv) > 1 else 2555)
MAX_WORKERS = _env_int("PROXY_MAX_WORKERS", 20)
MAX_REQUEST_BODY = _parse_size(os.environ.get("PROXY_MAX_REQUEST_BODY"), 10 * 1024**2)   # 10MB
MAX_RESPONSE_BODY = _parse_size(os.environ.get("PROXY_MAX_RESPONSE_BODY"), 50 * 1024**2)  # 50MB
CONNECT_TIMEOUT = _env_int("PROXY_CONNECT_TIMEOUT", 10)
READ_TIMEOUT = _env_int("PROXY_READ_TIMEOUT", 300)
STREAM_IDLE_TIMEOUT = _env_int("PROXY_STREAM_IDLE_TIMEOUT", 60)
SOCKET_TIMEOUT = _env_int("PROXY_SOCKET_TIMEOUT", 30)
```

- [ ] **Step 2: Verify proxy.py still parses**

Run: `cd /Users/noonoon/Dev/proxy-claude && python3 -c "import py_compile; py_compile.compile('proxy.py', doraise=True)"`
Expected: No output (success)

- [ ] **Step 3: Commit**

```bash
git add proxy.py
git commit -m "feat(proxy): add env-based configuration system with human-readable size parser"
```

---

## Task 2: proxy.py — strip_system crash fix and protocol version

**Files:**
- Modify: `proxy.py:strip_system` function (around line 23-57 after Task 1)
- Modify: `proxy.py:Handler` class (protocol_version attribute)

- [ ] **Step 1: Harden strip_system**

Replace the `strip_system` function with:

```python
def strip_system(body_bytes):
    """Remove 'system' field, merge into first user message."""
    try:
        data = json.loads(body_bytes)
    except Exception:
        return body_bytes

    system = data.pop("system", None)
    if not system:
        return body_bytes

    if isinstance(system, str):
        text = system
    elif isinstance(system, list):
        text = "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in system
        )
    else:
        text = str(system)

    messages = data.get("messages")
    if text and isinstance(messages, list) and len(messages) > 0:
        msg = messages[0]
        if isinstance(msg, dict) and msg.get("role") == "user":
            c = msg.get("content", "")
            if isinstance(c, str):
                msg["content"] = text + "\n\n" + c
            elif isinstance(c, list):
                msg["content"] = [{"type": "text", "text": text + "\n\n"}] + c
            else:
                data["messages"].insert(0, {"role": "user", "content": text})
        else:
            data["messages"].insert(0, {"role": "user", "content": text})

    return json.dumps(data).encode()
```

Key changes vs original:
- `isinstance(messages, list)` guard before indexing
- `isinstance(msg, dict)` guard before calling `.get()` on first element
- If `content` is neither str nor list, fall back to prepending a user message

- [ ] **Step 2: Add protocol_version to Handler**

Add `protocol_version = "HTTP/1.1"` as the first line inside the `Handler` class body:

```python
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
```

- [ ] **Step 3: Verify**

Run: `cd /Users/noonoon/Dev/proxy-claude && python3 -c "import py_compile; py_compile.compile('proxy.py', doraise=True)"`

- [ ] **Step 4: Commit**

```bash
git add proxy.py
git commit -m "fix(proxy): harden strip_system against malformed messages, fix HTTP/1.1 protocol version"
```

---

## Task 3: proxy.py — Bounded concurrency, client socket timeout, request/response limits

**Files:**
- Modify: `proxy.py:Handler._proxy` method
- Modify: `proxy.py:Handler._buffer_response` method
- Modify: `proxy.py:Handler._stream_response` method
- Modify: `proxy.py:Threaded` class (replace entirely)
- Modify: `proxy.py:__main__` block

- [ ] **Step 1: Rewrite Handler._proxy with request body limit, upstream timeouts, and sanitized errors**

Replace the `_proxy` method with:

```python
    def _proxy(self, method):
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_REQUEST_BODY:
            error_msg = json.dumps({"error": {"message": "Request body too large", "type": "proxy_error"}}).encode()
            self.send_response(413)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(error_msg)))
            self.end_headers()
            self.wfile.write(error_msg)
            return

        body = self.rfile.read(length) if length else b""

        if method == "POST" and "/v1/messages" in self.path:
            body = strip_system(body)

        conn = http.client.HTTPConnection(LITELLM_HOST, LITELLM_PORT, timeout=CONNECT_TIMEOUT)
        try:
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "content-length", "transfer-encoding")}
            headers["Content-Length"] = str(len(body))
            headers["Host"] = f"{LITELLM_HOST}:{LITELLM_PORT}"

            conn.request(method, self.path, body=body if method == "POST" else None, headers=headers)
            conn.sock.settimeout(READ_TIMEOUT)
            resp = conn.getresponse()

            if _is_streaming(resp):
                self._stream_response(resp, conn)
            else:
                self._buffer_response(resp, conn)
        except Exception as e:
            print(f"Proxy upstream error: {e}", file=sys.stderr, flush=True)
            error_msg = json.dumps({"error": {"message": "Upstream proxy error", "type": "proxy_error"}}).encode()
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(error_msg)))
                self.end_headers()
                self.wfile.write(error_msg)
            except Exception:
                pass
        finally:
            conn.close()
```

- [ ] **Step 2: Rewrite _buffer_response with response body cap**

Replace `_buffer_response` with:

```python
    def _buffer_response(self, resp, conn):
        """Forward a non-streaming response after fully buffering it (with size cap)."""
        chunks = []
        total = 0
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_RESPONSE_BODY:
                print(f"Upstream response exceeded {MAX_RESPONSE_BODY} bytes, aborting", file=sys.stderr, flush=True)
                error_msg = json.dumps({"error": {"message": "Upstream response too large", "type": "proxy_error"}}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(error_msg)))
                self.end_headers()
                self.wfile.write(error_msg)
                return
            chunks.append(chunk)
        resp_body = b"".join(chunks)

        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "keep-alive", "content-length"):
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)
```

- [ ] **Step 3: Rewrite _stream_response with idle timeout**

Replace `_stream_response` with:

```python
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
        except (AttributeError, TypeError):
            pass  # Best-effort; READ_TIMEOUT still applies

        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):x}\r\n".encode())
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except socket.timeout:
            print(f"Stream idle timeout ({STREAM_IDLE_TIMEOUT}s), aborting", file=sys.stderr, flush=True)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
```

- [ ] **Step 4: Replace Threaded class with bounded concurrency server**

Replace the entire `Threaded` class and `__main__` block with:

```python
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
            except Exception:
                pass
            self.shutdown_request(req)
            return
        self._pool.submit(self._handle, req, addr)

    def _handle(self, req, addr):
        try:
            req.settimeout(SOCKET_TIMEOUT)
            self.finish_request(req, addr)
        except Exception as e:
            print(f"Proxy request error: {e}", file=sys.stderr, flush=True)
        finally:
            self.shutdown_request(req)
            self._semaphore.release()

    def server_close(self):
        super().server_close()
        self._pool.shutdown(wait=False)


if __name__ == "__main__":
    print(f"Proxy :{LISTEN_PORT} -> LiteLLM :{LITELLM_PORT} (workers={MAX_WORKERS})", flush=True)
    BoundedThreadServer(("127.0.0.1", LISTEN_PORT), Handler).serve_forever()
```

- [ ] **Step 5: Verify**

Run: `cd /Users/noonoon/Dev/proxy-claude && python3 -c "import py_compile; py_compile.compile('proxy.py', doraise=True)"`

- [ ] **Step 6: Commit**

```bash
git add proxy.py
git commit -m "feat(proxy): bounded concurrency, request/response limits, upstream timeouts, sanitized errors"
```

---

## Task 4: container.py — FD leak fix and subprocess timeouts

**Files:**
- Modify: `container.py:25-52` (_compose_cmd)
- Modify: `container.py:55-72` (_run)
- Modify: `container.py:75-83` (_docker_running)
- Modify: `container.py:98-107` (_is_proxy_process)
- Modify: `container.py:110-141` (_start_proxy)
- Modify: `container.py:245-265` (get_logs_since, get_logs_tail)

- [ ] **Step 1: Add timeout to _compose_cmd**

In `_compose_cmd`, add `timeout=30` to both `subprocess.run` calls and catch `subprocess.TimeoutExpired`:

```python
def _compose_cmd():
    """Return the docker compose command as a list. Cached after first call."""
    global _cached_compose_cmd
    if _cached_compose_cmd is not None:
        return _cached_compose_cmd
    docker = _docker_bin()
    try:
        result = subprocess.run(
            [docker, "compose", "version"], capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            _cached_compose_cmd = [docker, "compose"]
            return _cached_compose_cmd
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        result = subprocess.run(
            ["docker-compose", "version"], capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            _cached_compose_cmd = ["docker-compose"]
            return _cached_compose_cmd
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    _cached_compose_cmd = ["docker-compose"]
    return _cached_compose_cmd
```

- [ ] **Step 2: Add timeout to _run**

In `_run`, add `timeout=120` to `subprocess.run` (longer for compose operations like `up`, `down`) and catch `subprocess.TimeoutExpired`. For the streaming `Popen` path, no change (it's interactive):

```python
def _run(args, capture=False, stream=False):
    """Run a docker compose command from the project directory."""
    cmd = _compose_cmd() + args
    log.debug("Running: %s", " ".join(cmd))
    try:
        if stream:
            proc = subprocess.Popen(cmd, cwd=DIR)
            proc.wait()
            return proc.returncode == 0, ""
        result = subprocess.run(
            cmd, cwd=DIR, capture_output=capture, text=True, timeout=120
        )
        if capture:
            return result.returncode == 0, result.stdout
        return result.returncode == 0, ""
    except subprocess.TimeoutExpired:
        log.warning("Docker compose command timed out: %s", " ".join(args))
        return False, ""
    except FileNotFoundError:
        print("Error: docker compose is required. Install Docker Desktop or docker-compose.")
        sys.exit(1)
```

- [ ] **Step 3: Add timeout to _docker_running**

```python
def _docker_running():
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            [_docker_bin(), "info"], capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
```

- [ ] **Step 4: Add timeout to _is_proxy_process**

```python
def _is_proxy_process(pid):
    """Check if the given PID is actually our proxy.py process."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0 and PROXY_SCRIPT in result.stdout
    except (OSError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
```

- [ ] **Step 5: Fix FD leak in _start_proxy**

Wrap `log_fh` in try/finally:

```python
def _start_proxy():
    """Start the system message rewriter proxy in the background.

    Returns True if the proxy started successfully, False otherwise.
    """
    _stop_proxy()  # Clean up any stale process
    if not os.path.exists(PROXY_SCRIPT):
        log.debug("proxy.py not found, skipping proxy start")
        return False
    venv_python = os.path.join(DIR, ".venv", "bin", "python")
    python = venv_python if os.path.exists(venv_python) else "python3"
    proxy_log = os.path.join(DIR, ".proxy.log")
    log_fh = open(proxy_log, "a")
    try:
        proc = subprocess.Popen(
            [python, PROXY_SCRIPT, str(PROXY_PORT)],
            cwd=DIR, stdout=log_fh, stderr=log_fh,
        )
    except Exception:
        log_fh.close()
        raise
    log_fh.close()  # Child inherits FD via fork; parent doesn't need it
    with open(PROXY_PID_FILE, "w") as f:
        f.write(str(proc.pid))
    log.debug("Started proxy (pid=%d) on port %d", proc.pid, PROXY_PORT)

    # Brief wait to detect immediate startup failures
    time.sleep(0.5)
    exit_code = proc.poll()
    if exit_code is not None:
        log.warning("Proxy exited immediately with code %d", exit_code)
        try:
            os.unlink(PROXY_PID_FILE)
        except OSError:
            pass
        return False
    return True
```

- [ ] **Step 6: Add timeout to get_logs_since and get_logs_tail**

```python
def get_logs_since(timestamp):
    """Get container logs since a timestamp (RFC3339 format). Returns log text."""
    _check_docker()
    docker = _docker_bin()
    log.debug("Reading logs since %s via %s", timestamp, docker)
    try:
        result = subprocess.run(
            [docker, "logs", CONTAINER_NAME, "--since", timestamp],
            capture_output=True, text=True, cwd=DIR, timeout=30,
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        log.warning("docker logs --since timed out")
        return ""


def get_logs_tail(lines=200):
    """Get last N lines of container logs. Returns log text."""
    docker = _docker_bin()
    log.debug("Reading last %d log lines via %s", lines, docker)
    try:
        result = subprocess.run(
            [docker, "logs", CONTAINER_NAME, "--tail", str(lines)],
            capture_output=True, text=True, cwd=DIR, timeout=30,
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        log.warning("docker logs --tail timed out")
        return ""
```

- [ ] **Step 7: Verify**

Run: `cd /Users/noonoon/Dev/proxy-claude && python3 -c "import py_compile; py_compile.compile('container.py', doraise=True)"`

- [ ] **Step 8: Commit**

```bash
git add container.py
git commit -m "fix(container): close leaked proxy log FD, add subprocess timeouts everywhere"
```

---

## Task 5: config.py — MalformedConfig state corruption guard

**Files:**
- Modify: `config.py:17-33` (_load_yaml)
- Modify: `config.py:48-56` (_save_yaml)

- [ ] **Step 1: Add MalformedConfig class and update _load_yaml**

Add the class before `_load_yaml` and update the function:

```python
class MalformedConfig(dict):
    """Sentinel dict subclass returned when litellm_config.yaml is malformed.

    Read-only callers (list_models, provider_has_models) work transparently.
    _save_yaml refuses to persist this, preventing state corruption.
    """
    pass


def _load_yaml():
    """Load litellm_config.yaml, return full dict."""
    if not os.path.exists(CONFIG_PATH):
        return {"model_list": [], "general_settings": {}}
    try:
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        log.warning("litellm_config.yaml is malformed: %s", e)
        print(f"Warning: litellm_config.yaml is malformed: {e}")
        if os.path.exists(CONFIG_BACKUP):
            print(f"  A backup exists at litellm_config.yaml.bak")
        return MalformedConfig({"model_list": [], "general_settings": {}})
    if "model_list" not in data:
        data["model_list"] = []
    return data
```

- [ ] **Step 2: Add guard to _save_yaml**

```python
def _save_yaml(data):
    """Backup then write litellm_config.yaml atomically."""
    if isinstance(data, MalformedConfig):
        raise ValueError(
            "Refusing to save: config was loaded from a malformed file. "
            "Fix litellm_config.yaml manually or restore from litellm_config.yaml.bak"
        )
    if os.path.exists(CONFIG_PATH):
        shutil.copy2(CONFIG_PATH, CONFIG_BACKUP)
        log.debug("Backed up config to %s", CONFIG_BACKUP)
    _atomic_write(CONFIG_PATH, lambda f: yaml.dump(
        data, f, default_flow_style=False, sort_keys=False
    ))
    log.debug("Wrote config to %s", CONFIG_PATH)
```

- [ ] **Step 3: Verify**

Run: `cd /Users/noonoon/Dev/proxy-claude && python3 -c "import py_compile; py_compile.compile('config.py', doraise=True)"`

- [ ] **Step 4: Commit**

```bash
git add config.py
git commit -m "fix(config): prevent saving malformed config over valid file"
```

---

## Task 6: providers/openai.py — Response validation and honest status messages

**Files:**
- Modify: `providers/openai.py:66-81` (_validate_api_key)
- Modify: `providers/openai.py:83-123` (_validate_browser)
- Modify: `providers/openai.py:207-237` (login polling loop)

- [ ] **Step 1: Harden _validate_api_key**

```python
    def _validate_api_key(self, api_key):
        try:
            resp = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            if resp.status_code == 401:
                return AuthStatus.INVALID, "Invalid OPENAI_API_KEY"
            if resp.status_code != 200:
                return AuthStatus.INVALID, f"OpenAI returned status {resp.status_code}"
            ct = resp.headers.get("Content-Type", "")
            if "application/json" not in ct:
                return AuthStatus.UNREACHABLE, f"OpenAI returned unexpected content-type: {ct}"
            try:
                resp.json()
            except ValueError:
                return AuthStatus.UNREACHABLE, "OpenAI returned invalid JSON"
            return AuthStatus.OK, "Authenticated with OpenAI API key"
        except requests.RequestException as e:
            return AuthStatus.UNREACHABLE, f"Cannot reach OpenAI API: {e}"
```

- [ ] **Step 2: Harden _validate_browser with honest messages**

```python
    def _validate_browser(self):
        """Check browser OAuth auth without making billing API calls.

        Uses two lightweight signals:
        1. Container logs for auth-success patterns (primary)
        2. Proxy GET /v1/models to see if chatgpt/ models are served (no billing)
        """
        running, _ = container.status()
        if not running:
            return AuthStatus.NOT_CONFIGURED, "Container not running — cannot check browser auth"

        # Primary check: look for auth-success patterns in container logs (free)
        logs = container.get_logs_tail(200)
        if re.search(r"(?i)(successfully authenticated|chatgpt.*auth|session.*authenticated|access.token)", logs):
            log.debug("Browser OAuth auth pattern found in logs")
            return AuthStatus.OK, "Browser OAuth appears configured (based on container logs)"

        # Secondary check: query the proxy's model list endpoint (no billing)
        master_key = config.get_env("LITELLM_MASTER_KEY") or "sk-1234"
        chatgpt_configured = [m for m in config.list_models() if m["model"].startswith("chatgpt/")]
        if chatgpt_configured:
            try:
                resp = requests.get(
                    f"http://localhost:{PROXY_PORT}/v1/models",
                    headers={"Authorization": f"Bearer {master_key}"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    ct = resp.headers.get("Content-Type", "")
                    if "application/json" in ct:
                        try:
                            data = resp.json()
                            served_ids = {m.get("id", "") for m in data.get("data", [])}
                            configured_aliases = {m["alias"] for m in chatgpt_configured}
                            if configured_aliases & served_ids:
                                log.debug("Browser OAuth validated — chatgpt models served by proxy")
                                return AuthStatus.OK, "Browser OAuth appears configured (models served by proxy)"
                        except ValueError:
                            log.debug("Proxy /v1/models returned invalid JSON")
                log.debug("Proxy /v1/models returned %d or no chatgpt models served", resp.status_code)
            except (requests.RequestException, ValueError) as e:
                log.debug("Proxy /v1/models check failed: %s", e)

        log.debug("No auth evidence found")
        return AuthStatus.NOT_CONFIGURED, "Not authenticated with OpenAI"
```

- [ ] **Step 3: Harden login polling loop**

In `_login_browser`, replace the `except Exception:` on line 236 with proper exception handling, and add JSON validation to the polling response:

Find and replace this block (around lines 222-237):
```python
            if chatgpt_aliases:
                try:
                    resp = requests.get(
                        f"http://localhost:{PROXY_PORT}/v1/models",
                        headers={"Authorization": f"Bearer {master_key}"},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        served_ids = {m.get("id", "") for m in data.get("data", [])}
                        if chatgpt_aliases & served_ids:
                            print("\n  ✓ Authenticated with OpenAI via browser OAuth!")
                            return True, "Authenticated via browser OAuth"
                    log.debug("Proxy /v1/models returned %d", resp.status_code)
                except Exception:
                    pass
```

Replace with:
```python
            if chatgpt_aliases:
                try:
                    resp = requests.get(
                        f"http://localhost:{PROXY_PORT}/v1/models",
                        headers={"Authorization": f"Bearer {master_key}"},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        ct = resp.headers.get("Content-Type", "")
                        if "application/json" in ct:
                            data = resp.json()
                            served_ids = {m.get("id", "") for m in data.get("data", [])}
                            if chatgpt_aliases & served_ids:
                                print("\n  ✓ Authenticated with OpenAI via browser OAuth!")
                                return True, "Authenticated via browser OAuth"
                    log.debug("Proxy /v1/models returned %d", resp.status_code)
                except (requests.RequestException, ValueError):
                    pass
```

- [ ] **Step 4: Verify**

Run: `cd /Users/noonoon/Dev/proxy-claude && python3 -c "import py_compile; py_compile.compile('providers/openai.py', doraise=True)"`

- [ ] **Step 5: Commit**

```bash
git add providers/openai.py
git commit -m "fix(openai): validate response content-type/JSON, honest browser auth messages"
```

---

## Task 7: providers/ollama.py — Response validation and hardened pull

**Files:**
- Modify: `providers/ollama.py:52-62` (validate)
- Modify: `providers/ollama.py:105-133` (discover_models)
- Modify: `providers/ollama.py:135-164` (pull_model)

- [ ] **Step 1: Harden validate**

```python
    def validate(self):
        host = self.OLLAMA_HOST
        try:
            resp = requests.get(f"{host}/api/tags", timeout=3)
            if resp.status_code != 200:
                return AuthStatus.UNREACHABLE, f"Ollama returned status {resp.status_code}"
            ct = resp.headers.get("Content-Type", "")
            if "json" not in ct:
                return AuthStatus.UNREACHABLE, f"Ollama returned unexpected content-type: {ct}"
            try:
                resp.json()
            except ValueError:
                return AuthStatus.UNREACHABLE, "Ollama returned invalid JSON"
            return AuthStatus.OK, f"Ollama is reachable at {host}"
        except requests.RequestException as e:
            return AuthStatus.UNREACHABLE, f"Ollama is not reachable at {host}: {e}"
```

- [ ] **Step 2: Harden discover_models**

```python
    def discover_models(self):
        """Fetch available models from Ollama. Returns dict of alias -> litellm model string."""
        host = self.OLLAMA_HOST
        try:
            resp = requests.get(f"{host}/api/tags", timeout=5)
            if resp.status_code != 200:
                logger.warning(
                    "Ollama at %s returned HTTP %d — cannot discover models",
                    host, resp.status_code,
                )
                return {}
            try:
                data = resp.json()
            except ValueError:
                logger.warning("Ollama at %s returned invalid JSON", host)
                return {}
            models_list = data.get("models", [])
            if not isinstance(models_list, list):
                logger.warning("Ollama at %s returned non-list 'models' field", host)
                return {}
            models = {}
            for m in models_list:
                if not isinstance(m, dict):
                    continue
                name = m.get("name", "")
                if not isinstance(name, str) or not name:
                    continue
                alias = name.replace(":latest", "")
                models[alias] = f"ollama/{name}"
            return models
        except requests.RequestException as e:
            logger.warning("Could not connect to Ollama at %s: %s", host, e)
            return {}
```

- [ ] **Step 3: Harden pull_model with per-line error handling and idle timeout**

```python
    def pull_model(self, model_name):
        """Pull a model via Ollama REST API. Returns (success, message)."""
        try:
            resp = requests.post(
                f"{self.OLLAMA_HOST}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600,
            )
            if resp.status_code != 200:
                return False, f"Pull failed with status {resp.status_code}"

            # Set idle timeout on underlying socket for iter_lines (best-effort;
            # overall 600s timeout still applies if socket access fails)
            try:
                resp.raw._fp.fp.raw._sock.settimeout(60)
            except (AttributeError, TypeError):
                pass

            last_status = ""
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                    except ValueError:
                        logger.debug("Skipping malformed NDJSON line: %s", line[:100])
                        continue
                    status = data.get("status", "")
                    if "completed" in data and "total" in data:
                        total = data["total"]
                        pct = int(data["completed"] / total * 100) if total > 0 else 0
                        print(f"\r  {status}: {pct}%    ", end="", flush=True)
                    elif status != last_status:
                        print(f"\r  {status}              ", end="", flush=True)
                    last_status = status
            print()
            return True, f"Pulled {model_name}"
        except requests.RequestException as e:
            return False, f"Pull failed: {e}"
```

- [ ] **Step 4: Verify**

Run: `cd /Users/noonoon/Dev/proxy-claude && python3 -c "import py_compile; py_compile.compile('providers/ollama.py', doraise=True)"`

- [ ] **Step 5: Commit**

```bash
git add providers/ollama.py
git commit -m "fix(ollama): validate response schema, harden NDJSON parsing with idle timeout"
```

---

## Task 8: Update .env.example

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Update .env.example with all variables organized by section**

Replace the file contents. All existing variables (`DASHSCOPE_API_KEY`, `LITELLM_MASTER_KEY`, `CHATGPT_EMAIL`/`CHATGPT_PASSWORD`, `LITELLM_LOG`) are preserved. New additions: `OPENAI_API_KEY`, `OLLAMA_HOST`, and all proxy tunables.

```env
# =============================================================================
# LiteLLM Proxy Configuration
# =============================================================================

# --- LiteLLM Core ---
# Master key for authenticating with the LiteLLM proxy
LITELLM_MASTER_KEY=sk-1234

# Log level (DEBUG, INFO, WARNING, ERROR)
LITELLM_LOG=DEBUG

# --- OpenAI ---
# API key for OpenAI platform access (openai/ prefix models)
# Get one at: https://platform.openai.com/api-keys
# OPENAI_API_KEY=your-openai-key-here

# Optional: OpenAI subscription credentials (if not using browser OAuth)
# CHATGPT_EMAIL=
# CHATGPT_PASSWORD=

# --- Alibaba DashScope ---
# API key for DashScope (dashscope/ prefix models)
DASHSCOPE_API_KEY=your-alibaba-key-here

# --- Ollama ---
# Ollama server URL (default: http://localhost:11434)
# OLLAMA_HOST=http://localhost:11434

# --- Reverse Proxy (proxy.py) ---
# These configure the host-side reverse proxy that sits in front of LiteLLM.
# All values have sane defaults and are optional.

# Proxy listen port (default: 2555)
# PROXY_LISTEN_PORT=2555

# Upstream LiteLLM host and port (default: localhost:4000)
# PROXY_LITELLM_HOST=localhost
# PROXY_LITELLM_PORT=4000

# Max concurrent requests (default: 20). Returns 503 when full.
# PROXY_MAX_WORKERS=20

# Max request body size (default: 10MB). Accepts KB, MB, GB suffixes.
# PROXY_MAX_REQUEST_BODY=10MB

# Max buffered response body size (default: 50MB). Accepts KB, MB, GB suffixes.
# PROXY_MAX_RESPONSE_BODY=50MB

# Upstream connect timeout in seconds (default: 10)
# PROXY_CONNECT_TIMEOUT=10

# Upstream read timeout in seconds (default: 300)
# PROXY_READ_TIMEOUT=300

# Streaming idle timeout in seconds — aborts if upstream goes silent (default: 60)
# PROXY_STREAM_IDLE_TIMEOUT=60

# Client socket timeout in seconds (default: 30)
# PROXY_SOCKET_TIMEOUT=30
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: comprehensive .env.example with all proxy tunables"
```

---

## Task 9: Final verification

- [ ] **Step 1: Verify all files compile**

```bash
cd /Users/noonoon/Dev/proxy-claude
python3 -c "
import py_compile
for f in ['proxy.py', 'container.py', 'config.py', 'providers/openai.py', 'providers/ollama.py']:
    py_compile.compile(f, doraise=True)
    print(f'  OK: {f}')
print('All files compile successfully')
"
```

- [ ] **Step 2: Verify proxy starts and stops cleanly**

```bash
cd /Users/noonoon/Dev/proxy-claude
python3 proxy.py &
PROXY_PID=$!
sleep 1
curl -s http://localhost:2555/health || true
kill $PROXY_PID 2>/dev/null
echo "Proxy start/stop OK"
```
