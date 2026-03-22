import logging
import re
import time
from datetime import datetime, timezone

import requests

import config
import container
from container import PROXY_PORT
from providers.base import BaseProvider, AuthStatus

log = logging.getLogger("litellm-cli.openai")


class OpenAIProvider(BaseProvider):
    name = "openai"
    display_name = "OpenAI"
    auth_types = ["browser_oauth", "api_key"]
    _AUTH_LOG_PATTERN = re.compile(
        r"(?i)(successfully authenticated|chatgpt.*auth|session.*authenticated|access.token)"
    )
    env_vars = {
        "browser_oauth": [],
        "api_key": ["OPENAI_API_KEY"],
    }
    # ChatGPT subscription models (browser OAuth, chatgpt/ prefix)
    models = {
        "gpt-5.4": "chatgpt/gpt-5.4",
        "gpt-5.4-pro": "chatgpt/gpt-5.4-pro",
        "gpt-5.3-codex": "chatgpt/gpt-5.3-codex",
        "gpt-5.3-codex-spark": "chatgpt/gpt-5.3-codex-spark",
        "gpt-5.3-instant": "chatgpt/gpt-5.3-instant",
        "gpt-5.3-chat-latest": "chatgpt/gpt-5.3-chat-latest",
    }

    # OpenAI API key models (openai/ prefix)
    _api_key_models = {
        "gpt-5.4": "openai/gpt-5.4",
        "gpt-5.4-pro": "openai/gpt-5.4-pro",
        "gpt-5.3-instant": "openai/gpt-5.3-instant",
        "o3": "openai/o3",
        "o3-pro": "openai/o3-pro",
        "o4-mini": "openai/o4-mini",
    }

    def get_model_string(self, alias, auth_type=None):
        if auth_type == "api_key":
            return self._api_key_models.get(alias)
        return self.models.get(alias)

    def get_models_for_auth(self, auth_type):
        """Return the model catalog for a given auth type."""
        if auth_type == "api_key":
            return self._api_key_models
        return self.models

    def validate(self):
        # Check API key first
        api_key = config.get_env("OPENAI_API_KEY")
        if api_key:
            log.debug("OPENAI_API_KEY found, validating via API")
            return self._validate_api_key(api_key)
        # Check browser OAuth via container logs
        log.debug("No API key, checking browser OAuth via container logs")
        return self._validate_browser()

    def _validate_api_key(self, api_key):
        try:
            resp = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            if resp.status_code == 401:
                return AuthStatus.INVALID, "Invalid OPENAI_API_KEY"
            if resp.status_code == 403:
                return AuthStatus.INVALID, "OPENAI_API_KEY lacks required permissions (403 Forbidden)"
            if resp.status_code == 429:
                return AuthStatus.UNREACHABLE, "Rate limited (HTTP 429) — credential not verified"
            if resp.status_code >= 500:
                return AuthStatus.UNREACHABLE, f"OpenAI server error (HTTP {resp.status_code}) — key not validated"
            if resp.status_code != 200:
                return AuthStatus.UNREACHABLE, f"OpenAI returned unexpected status {resp.status_code} — key not validated"
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
        if self._AUTH_LOG_PATTERN.search(logs):
            log.debug("Browser OAuth auth pattern found in logs")
            return AuthStatus.UNVERIFIED, "Browser OAuth may be active (log pattern found, but cannot independently verify upstream auth)"

        # Secondary check: query the proxy's model list endpoint (no billing)
        chatgpt_aliases = {m["alias"] for m in config.list_models() if m["model"].startswith("chatgpt/")}
        if chatgpt_aliases:
            found, err = self._check_proxy_models(chatgpt_aliases)
            if found:
                log.debug("Browser OAuth validated — chatgpt models served by proxy")
                return AuthStatus.UNVERIFIED, "Browser OAuth may be active (models configured in proxy, but cannot independently verify upstream auth)"
            if err:
                log.debug("Proxy model check error: %s", err)
                return AuthStatus.UNREACHABLE, f"Cannot verify browser auth (proxy check failed: {err})"

        log.debug("No auth evidence found")
        return AuthStatus.NOT_CONFIGURED, "Not authenticated — no browser OAuth evidence found. Run './litellm.sh login openai' to authenticate."

    def _check_proxy_models(self, chatgpt_aliases):
        """Check if chatgpt models are served by the proxy.

        Returns (found: bool, error: str|None).
        found=True means models detected. error is set on transport/parse failures.
        found=False with error=None means "checked successfully, models not present."
        """
        master_key = config.get_env("LITELLM_MASTER_KEY") or "sk-1234"
        try:
            resp = requests.get(
                f"http://localhost:{PROXY_PORT}/v1/models",
                headers={"Authorization": f"Bearer {master_key}"},
                timeout=10,
            )
            if resp.status_code != 200:
                log.debug("Proxy /v1/models returned %d", resp.status_code)
                return False, f"proxy returned HTTP {resp.status_code}"
            ct = resp.headers.get("Content-Type", "")
            if "application/json" not in ct:
                log.debug("Proxy /v1/models returned unexpected content-type: %s", ct)
                return False, f"unexpected content-type: {ct}"
            data = resp.json()
            if not isinstance(data, dict):
                log.debug("Proxy /v1/models returned non-dict response")
                return False, "non-dict response"
            items = data.get("data")
            if not isinstance(items, list):
                log.debug("Proxy /v1/models response has no valid 'data' list")
                return False, "no valid 'data' list in response"
            served_ids = {m.get("id", "") for m in items if isinstance(m, dict)}
            if chatgpt_aliases & served_ids:
                return True, None
            log.debug("Proxy /v1/models: no chatgpt models in served set")
            return False, None
        except requests.RequestException as e:
            log.debug("Proxy /v1/models request failed: %s", e)
            return False, f"request failed: {e}"
        except ValueError as e:
            log.debug("Proxy /v1/models JSON parse failed: %s", e)
            return False, f"invalid JSON: {e}"

    def login(self, auth_type="browser_oauth"):
        if auth_type == "api_key":
            return self._login_api_key()
        return self._login_browser()

    def _login_api_key(self):
        print(f"\n  Enter your OpenAI API key.")
        print(f"  Get one at: https://platform.openai.com/api-keys\n")
        key = input("  OPENAI_API_KEY: ").strip()
        if not key:
            return AuthStatus.INVALID, "No key entered."
        config.set_env("OPENAI_API_KEY", key)
        status, msg = self.validate()
        return status, msg

    def _login_browser(self):
        """Drive the browser OAuth flow by reading container logs."""
        # Pre-check
        status, msg = self.validate()
        if status == AuthStatus.OK:
            return AuthStatus.OK, f"Already authenticated. {msg}"

        # Ensure container is running
        running, _ = container.status()
        if not running:
            print("  Container not running. Starting it...")
            container.up()
            if not container.wait_healthy(30):
                return AuthStatus.UNREACHABLE, "Container failed to start."

        # Capture timestamp before looking for URL
        since = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        print("\n  Waiting for login instructions from container...")
        login_url = None
        device_code = None
        for attempt in range(30):  # 30 * 2s = 60s to find URL
            logs = container.get_logs_since(since)

            # Also check recent logs if --since returns nothing (timestamp drift)
            if not logs.strip():
                logs = container.get_logs_tail(50)

            # Match OpenAI auth URLs
            urls = re.findall(
                r'https?://(?:auth\.openai\.com|login\.chatgpt\.com|auth0\.openai\.com|chat\.openai\.com)[^\s"\']*',
                logs
            )
            if not urls:
                urls = re.findall(r'https?://[^\s"\']*(?:/codex/|/device|device_code|user_code)[^\s"\']*', logs)

            # Extract device code (e.g. "Enter code: 38KX-M5UES")
            code_match = re.search(r'(?:Enter code|enter.*code)[:\s]+([A-Z0-9]{4,}-[A-Z0-9]{4,})', logs)
            if code_match:
                device_code = code_match.group(1)

            if urls:
                login_url = urls[-1]
                break
            time.sleep(2)
            print(".", end="", flush=True)

        if not login_url:
            return AuthStatus.UNREACHABLE, (
                "Could not find login URL in container logs.\n"
                "  Make sure you have a chatgpt/ model configured and run './litellm.sh logs' to debug."
            )

        print(f"\n")
        print(f"  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  OpenAI Login Required                             │")
        print(f"  │                                                     │")
        print(f"  │  1) Open:  {login_url:<42} │")
        if device_code:
            print(f"  │  2) Enter code:  {device_code:<36} │")
        print(f"  │                                                     │")
        print(f"  │  Waiting for login... (timeout: 5 min)              │")
        print(f"  └─────────────────────────────────────────────────────┘")
        print()

        # Poll for auth success by checking logs AND the proxy model list
        timeout = 300  # 5 minutes
        start = time.time()
        # Determine which chatgpt/ model aliases are configured
        chatgpt_aliases = {m["alias"] for m in config.list_models() if m["model"].startswith("chatgpt/")}
        while time.time() - start < timeout:
            # Check logs for auth patterns
            logs = container.get_logs_since(since)
            if self._AUTH_LOG_PATTERN.search(logs):
                print("\n  ? Browser OAuth may be active (log pattern detected, not independently verified)")
                return AuthStatus.UNVERIFIED, "Browser OAuth may be active (log pattern detected, not independently verified)"

            # Lightweight proxy check — query /v1/models (no billing) to see
            # if chatgpt/ models are now being served after login
            if chatgpt_aliases:
                found, err = self._check_proxy_models(chatgpt_aliases)
                if found:
                    print("\n  ? Browser OAuth may be active (models detected in proxy, not independently verified)")
                    return AuthStatus.UNVERIFIED, "Browser OAuth may be active (models detected in proxy, not independently verified)"
                if err:
                    log.debug("Proxy model check failed during login poll: %s", err)
                    poll_note = f" (proxy: {err})"
                else:
                    poll_note = ""
            else:
                poll_note = ""

            elapsed = int(time.time() - start)
            remaining = timeout - elapsed
            mins, secs = divmod(remaining, 60)
            print(f"\r  Polling... {mins}:{secs:02d} remaining{poll_note}  ", end="", flush=True)
            time.sleep(3)

        print()
        return AuthStatus.UNREACHABLE, "Login timed out after 5 minutes. Run './litellm.sh login openai' to try again."
