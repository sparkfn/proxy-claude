#!/usr/bin/env python3
"""Host-side runtime helpers for the thin shell wrapper.

This module owns launch/start UX that depends on host capabilities such as:
- Docker Compose log access
- host-visible gateway health endpoints
- interactive launch-time auth prompts

The shell wrapper should only delegate to this module, not implement product
logic directly.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
DEFAULT_GATEWAY_URL = "http://127.0.0.1:2555"
AUTH_URL_RE = re.compile(r"https://auth\.openai\.com/[^\s\"']+")
AUTH_CODE_RE = re.compile(r"Enter code:\s*([A-Z0-9]+-[A-Z0-9]+)")
AUTH_SUCCESS_RE = re.compile(
    r"(?i)(successfully authenticated|chatgpt.*auth|session.*authenticated|access.token)"
)
ENV_PATH = REPO_ROOT / ".env"
ENV_BACKUP = REPO_ROOT / ".env.bak"
ENV_EXAMPLE = REPO_ROOT / ".env.example"


def _strip_quotes(value: str) -> str:
    if len(value) >= 2:
        if (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"):
            return value[1:-1]
    return value


def _ensure_env_file() -> None:
    import shutil, stat
    if ENV_PATH.exists():
        return
    if ENV_EXAMPLE.exists():
        shutil.copy2(ENV_EXAMPLE, ENV_PATH)
    else:
        ENV_PATH.write_text("", encoding="utf-8")
    ENV_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _read_env_lines() -> list[str]:
    _ensure_env_file()
    return ENV_PATH.read_text(encoding="utf-8").splitlines(keepends=True)


def _write_env_lines(lines: list[str]) -> None:
    import shutil, stat, tempfile, os
    if ENV_PATH.exists():
        shutil.copy2(ENV_PATH, ENV_BACKUP)
    fd, tmp = tempfile.mkstemp(dir=str(ENV_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.writelines(lines)
        os.replace(tmp, str(ENV_PATH))
    except Exception:
        os.unlink(tmp)
        raise
    ENV_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _get_env(key: str) -> str | None:
    for line in _read_env_lines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        current_key, _, value = stripped.partition("=")
        if current_key.strip() == key:
            return _strip_quotes(value.strip())
    return None


def _set_env(key: str, value: str) -> None:
    lines = _read_env_lines()
    found = False
    updated: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#") and "=" in stripped:
            current_key, _, _ = stripped.partition("=")
            if current_key.strip() == key:
                updated.append(f"{key}={value}\n")
                found = True
                continue
        updated.append(line)
    if not found:
        if updated and not updated[-1].endswith("\n"):
            updated.append("\n")
        updated.append(f"{key}={value}\n")
    _write_env_lines(updated)


def _ensure_master_key() -> str:
    import secrets
    master_key = _get_env("LITELLM_MASTER_KEY")
    if master_key:
        return master_key
    master_key = secrets.token_hex(16)
    _set_env("LITELLM_MASTER_KEY", master_key)
    return master_key


def _docker_compose_logs(compose_file: str, service: str, tail: int = 30) -> str:
    try:
        return subprocess.check_output(
            [
                "docker",
                "compose",
                "-f",
                compose_file,
                "logs",
                "--tail",
                str(tail),
                service,
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        return exc.output or ""


def _gateway_json(url: str, path: str) -> tuple[int | None, dict | None]:
    req = urllib.request.Request(f"{url}{path}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = resp.read().decode("utf-8")
            try:
                data = json.loads(payload) if payload else None
            except ValueError:
                data = None
            return resp.status, data
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8")
        try:
            data = json.loads(payload) if payload else None
        except ValueError:
            data = None
        return exc.code, data
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, None


def _gateway_post_json(url: str, path: str, payload: dict, headers: dict[str, str]) -> tuple[int | None, str]:
    req = urllib.request.Request(
        f"{url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, ""


def _parse_auth_prompt(logs: str) -> tuple[str | None, str | None]:
    url_match = AUTH_URL_RE.findall(logs or "")
    code_match = AUTH_CODE_RE.findall(logs or "")
    url = url_match[-1] if url_match else None
    code = code_match[-1] if code_match else None
    return url, code


def _print_auth_prompt(url: str, code: str) -> None:
    print("")
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  OpenAI Login Required                              │")
    print("  │                                                     │")
    print(f"  │  1) Visit:  {url:<42}│")
    print(f"  │  2) Enter code:  {code:<36}│")
    print("  │                                                     │")
    print("  └─────────────────────────────────────────────────────┘")
    print("")


def _report_start_status(compose_file: str, gateway_url: str) -> int:
    status, _payload = _gateway_json(gateway_url, "/health/readiness")
    if status == 200:
        print("  ✓ LiteLLM backend is reachable")
    else:
        logs = _docker_compose_logs(compose_file, "litellm", tail=30)
        url, code = _parse_auth_prompt(logs)
        if url and code:
            _print_auth_prompt(url, code)
            print("  ⚠ LiteLLM is running, but upstream auth is still pending.")
            print("    You can finish auth now or continue and do it during launch.")
        else:
            print("  ⚠ LiteLLM is running, but the backend is not yet reachable.")
            print("    Check './proclaude.sh logs' if it does not finish initializing.")

    # Check provider auth status inside the container (timeout 5s — don't block startup)
    try:
        check_output = subprocess.check_output(
            [
                "docker", "compose", "-f", compose_file,
                "exec", "-T", "gateway", "python", "-c",
                "import config, os\n"
                "from providers import all_providers\n"
                "env_data = config.load_env_file(config.ENV_PATH)\n"
                "auth_dir = os.path.join(config.DIR, 'auth')\n"
                "for p in all_providers():\n"
                "    if not p.models: continue\n"
                "    ready, reason = p.check_ready(env_data, auth_dir=auth_dir)\n"
                "    alias = next(iter(p.models))\n"
                "    icon = 'ok' if ready else reason\n"
                "    print(f'{alias}|{p.display_name}|{ready}|{icon}')\n",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        lines = [l.strip() for l in check_output.strip().splitlines() if l.strip()]
        if lines:
            print("")
            print("  Models:")
            for line in lines:
                parts = line.split("|", 3)
                if len(parts) == 4:
                    alias, display, ready_str, reason = parts
                    icon = "✓" if ready_str == "True" else "✗"
                    label = "ready" if ready_str == "True" else reason
                    print("    %s %-16s %-10s %s" % (icon, alias, display, label))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass  # Container not ready or too slow; skip auth report

    return 0


def _wait_for_readiness(gateway_url: str, timeout: int) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        status, _payload = _gateway_json(gateway_url, "/health/readiness")
        if status == 200:
            return True
        remaining = max(0, timeout - int(time.time() - start))
        mins, secs = divmod(remaining, 60)
        print(f"\r  Waiting for login... {mins}:{secs:02d} remaining  ", end="", flush=True)
        time.sleep(3)
    print("")
    return False


def _offer_pending_auth(compose_file: str, gateway_url: str, selected_model: str, timeout: int) -> int:
    status, _payload = _gateway_json(gateway_url, "/health/readiness")
    if status == 200:
        return 0

    logs = _docker_compose_logs(compose_file, "litellm", tail=30)
    url, code = _parse_auth_prompt(logs)
    if not (url and code):
        print("  ⚠ LiteLLM is running, but the backend is not yet reachable.")
        print("    Claude may fail until LiteLLM finishes initializing.")
        return 0

    _print_auth_prompt(url, code)
    print(f"  ⚠ LiteLLM is waiting on upstream authentication before it can serve {selected_model}.")
    auth_now = input("  Authenticate now before launching Claude? [Y/n]: ").strip()
    if auth_now.lower().startswith("n"):
        print("  Proceeding without completing auth.")
        return 0

    if _wait_for_readiness(gateway_url, timeout):
        print("  ✓ LiteLLM is ready")
        return 0

    print("  Proceeding without completed auth.")
    return 0


def _check_proxy_models(gateway_url: str, master_key: str, aliases: set[str]) -> tuple[bool, str | None]:
    status, payload = _gateway_json_with_auth(gateway_url, "/v1/models", master_key)
    if status != 200:
        return False, f"proxy returned HTTP {status}" if status is not None else "proxy unreachable"
    if not isinstance(payload, dict):
        return False, "proxy returned invalid JSON"
    items = payload.get("data")
    if not isinstance(items, list):
        return False, "proxy returned no valid data list"
    served_ids = {item.get("id", "") for item in items if isinstance(item, dict)}
    return bool(served_ids & aliases), None


def _gateway_json_with_auth(url: str, path: str, master_key: str) -> tuple[int | None, dict | None]:
    req = urllib.request.Request(
        f"{url}{path}",
        headers={"Authorization": f"Bearer {master_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = resp.read().decode("utf-8")
            try:
                data = json.loads(payload) if payload else None
            except ValueError:
                data = None
            return resp.status, data
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8")
        try:
            data = json.loads(payload) if payload else None
        except ValueError:
            data = None
        return exc.code, data
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, None


def _configured_chatgpt_models(compose_file: str) -> list[str]:
    """Get chatgpt/ model aliases via docker exec into the gateway container."""
    script = (
        "import json, config; "
        "print(json.dumps([m['alias'] for m in config.list_models() "
        "if m.get('model', '').startswith('chatgpt/')]))"
    )
    try:
        output = subprocess.check_output(
            ["docker", "compose", "-f", compose_file, "exec", "-T",
             "gateway", "python", "-c", script],
            text=True, stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        return []
    try:
        models = json.loads(output)
    except ValueError:
        return []
    if not isinstance(models, list):
        return []
    return [m for m in models if isinstance(m, str)]


def _openai_browser_login(compose_file: str, gateway_url: str, timeout: int) -> int:
    chatgpt_models = _configured_chatgpt_models(compose_file)
    if not chatgpt_models:
        print("  ✗ No chatgpt/ models configured. Add one before using OpenAI browser OAuth.")
        return 1

    master_key = _get_env("LITELLM_MASTER_KEY")
    if not master_key:
        print("  ✗ LITELLM_MASTER_KEY not set. Run './proclaude.sh start' first.")
        return 1

    found, err = _check_proxy_models(gateway_url, master_key, set(chatgpt_models))
    if found:
        print("  ? Browser OAuth may already be active.")
        return 0

    if err:
        print(f"  ⚠ Cannot verify browser auth yet: {err}")

    # Trigger the auth flow. LiteLLM emits the device prompt only after a real request.
    _gateway_post_json(
        gateway_url,
        "/v1/chat/completions",
        {
            "model": chatgpt_models[0],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
        {
            "Authorization": f"Bearer {master_key}",
            "Content-Type": "application/json",
        },
    )

    print("\n  Waiting for login instructions from LiteLLM...")
    login_url = None
    device_code = None
    start = time.time()
    while time.time() - start < 60:
        logs = _docker_compose_logs(compose_file, "litellm", tail=80)
        login_url, device_code = _parse_auth_prompt(logs)
        if login_url:
            break
        print(".", end="", flush=True)
        time.sleep(2)

    if not login_url:
        print("")
        print("  ✗ Could not find the OpenAI device-login URL in LiteLLM logs.")
        print("    Check './proclaude.sh logs litellm' for details.")
        return 1

    _print_auth_prompt(login_url, device_code or "")
    print("  Waiting for login confirmation... (timeout: 5 min)")
    start = time.time()
    while time.time() - start < timeout:
        logs = _docker_compose_logs(compose_file, "litellm", tail=120)
        if AUTH_SUCCESS_RE.search(logs):
            print("  ? Browser OAuth may be active (log pattern detected).")
            return 0
        found, _err = _check_proxy_models(gateway_url, master_key, set(chatgpt_models))
        if found:
            print("  ? Browser OAuth may be active (models detected in proxy).")
            return 0
        remaining = max(0, timeout - int(time.time() - start))
        mins, secs = divmod(remaining, 60)
        print(f"\r  Polling... {mins}:{secs:02d} remaining  ", end="", flush=True)
        time.sleep(3)
    print("")
    print("  ✗ Login timed out after 5 minutes.")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Host runtime helper for proclaude.sh")
    parser.add_argument(
        "--compose-file",
        default=str(DEFAULT_COMPOSE_FILE),
        help="Path to docker-compose.yml",
    )
    parser.add_argument(
        "--gateway-url",
        default=DEFAULT_GATEWAY_URL,
        help="Gateway base URL visible from the host",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("report-start-status")
    sub.add_parser("ensure-master-key")

    auth_parser = sub.add_parser("offer-pending-auth")
    auth_parser.add_argument("--selected-model", default="selected model")
    auth_parser.add_argument("--timeout", type=int, default=300)

    login_parser = sub.add_parser("openai-browser-login")
    login_parser.add_argument("--timeout", type=int, default=300)

    args = parser.parse_args(argv)
    if args.command == "report-start-status":
        return _report_start_status(args.compose_file, args.gateway_url)
    if args.command == "ensure-master-key":
        _ensure_master_key()
        return 0
    if args.command == "offer-pending-auth":
        return _offer_pending_auth(
            args.compose_file,
            args.gateway_url,
            args.selected_model,
            args.timeout,
        )
    if args.command == "openai-browser-login":
        return _openai_browser_login(args.compose_file, args.gateway_url, args.timeout)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
