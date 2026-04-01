import logging
import re
import time
from datetime import datetime, timezone

import requests

try:
    from .. import config
    from .base import BaseProvider, Status
except ImportError:
    import config
    from providers.base import BaseProvider, Status

log = logging.getLogger("litellm-cli.openai")


class OpenAIProvider(BaseProvider):
    name = "openai"
    display_name = "OpenAI"
    supports_thinking = True
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

    model_limits = {
        "gpt-5.4":             {"context": 1000000, "max_output": 128000},
        "gpt-5.4-pro":         {"context": 1000000, "max_output": 128000},
        "gpt-5.3-codex":       {"context": 1000000, "max_output": 32768},
        "gpt-5.3-codex-spark": {"context": 1000000, "max_output": 32768},
        "gpt-5.3-instant":     {"context": 1000000, "max_output": 32768},
        "gpt-5.3-chat-latest": {"context": 1000000, "max_output": 32768},
        "o3":                  {"context": 200000,  "max_output": 100000},
        "o3-pro":              {"context": 200000,  "max_output": 100000},
        "o4-mini":             {"context": 200000,  "max_output": 100000},
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

    def resolve_thinking_contract(self, alias, litellm_model, litellm_params=None):
        if litellm_model.startswith("chatgpt/"):
            return self._openai_reasoning_contract("chatgpt")
        if litellm_model.startswith("openai/"):
            return self._openai_reasoning_contract("openai")
        return None

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
        except requests.RequestException as e:
            return Status.UNREACHABLE, f"Cannot reach OpenAI API: {e}"
        status, msg = self._classify_response(resp)
        if status == Status.OK:
            return status, "Authenticated with OpenAI API key"
        if status == Status.INVALID and resp.status_code == 403:
            return status, "OPENAI_API_KEY lacks required permissions (403 Forbidden)"
        return status, msg

    def _validate_browser(self):
        """Check browser OAuth auth without making billing API calls.

        Uses two lightweight signals:
        1. Container logs for auth-success patterns (primary)
        2. Proxy GET /v1/models to see if chatgpt/ models are served (no billing)
        """
        import container
        cs, _ = container.status()
        if cs != Status.OK:
            return Status.NOT_CONFIGURED, "Container not running — cannot check browser auth"

        # Primary check: look for auth-success patterns in container logs (free)
        logs = container.get_logs_tail(200)
        if not logs:
            # Containerized mode: can't access LiteLLM logs from gateway
            # Fall back to proxy model check only
            log.debug("No logs available (containerized mode), using proxy check only")
        elif self._AUTH_LOG_PATTERN.search(logs):
            log.debug("Browser OAuth auth pattern found in logs")
            return Status.UNVERIFIED, "Browser OAuth may be active (log pattern found, but cannot independently verify upstream auth)"

        # Secondary check: query the proxy's model list endpoint (no billing)
        chatgpt_aliases = {m["alias"] for m in config.list_models() if m["model"].startswith("chatgpt/")}
        if chatgpt_aliases:
            found, err = self._check_proxy_models(chatgpt_aliases)
            if found:
                log.debug("Browser OAuth validated — chatgpt models served by proxy")
                return Status.UNVERIFIED, "Browser OAuth may be active (models configured in proxy, but cannot independently verify upstream auth)"
            if err:
                log.debug("Proxy model check error: %s", err)
                return Status.UNREACHABLE, f"Cannot verify browser auth (proxy check failed: {err})"

        log.debug("No auth evidence found")
        return Status.NOT_CONFIGURED, "Not authenticated — no browser OAuth evidence found. Run './proclaude.sh login openai' to authenticate."

    def _check_proxy_models(self, chatgpt_aliases):
        """Check if chatgpt models are served by the proxy.

        Returns (found: bool, error: str|None).
        found=True means models detected. error is set on transport/parse failures.
        found=False with error=None means "checked successfully, models not present."
        """
        from container import PROXY_PORT
        master_key = config.get_env("LITELLM_MASTER_KEY")
        if not master_key:
            return False, "LITELLM_MASTER_KEY not set. Run './proclaude.sh start' first."
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

    def login(self, auth_type="browser_oauth", credentials=None):
        if auth_type == "api_key":
            return self._login_api_key(credentials)
        return self._login_browser()

    # Credentials prompt shown by CLI layer, not here
    login_prompts = {
        "api_key": {
            "instructions": "Enter your OpenAI API key.\n  Get one at: https://platform.openai.com/api-keys",
            "fields": [("OPENAI_API_KEY", "OPENAI_API_KEY: ")],
        }
    }

    def _login_api_key(self, credentials=None):
        """Authenticate with API key. Caller must prompt via login_prompts."""
        if credentials is None:
            return Status.INVALID, "No credentials provided. Use login_prompts to collect them."
        key = credentials.get("OPENAI_API_KEY", "")
        if not key:
            return Status.INVALID, "No key entered."
        config.set_env("OPENAI_API_KEY", key)
        return self.validate()

    def _login_browser(self):
        """Drive the browser OAuth flow by reading container logs."""
        import container
        # Pre-check — OK and UNVERIFIED both mean auth is likely working
        status, msg = self.validate()
        if status in (Status.OK, Status.UNVERIFIED):
            return status, f"Already authenticated. {msg}"

        # Ensure container is running
        cs, _ = container.status()
        if cs != Status.OK:
            s, msg = container.up()
            if s != Status.OK:
                return Status.UNREACHABLE, f"Container failed to start: {msg}"
            if not container.wait_healthy(30):
                return Status.UNREACHABLE, "Container not healthy after startup."

        # Capture timestamp before looking for URL
        since = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Quick check: can we access logs?
        test_logs = container.get_logs_tail(10)
        if not test_logs:
            # Containerized mode: can't read LiteLLM container logs
            # The OAuth URL must be found via ./proclaude.sh logs
            return Status.UNREACHABLE, (
                "Browser OAuth requires access to LiteLLM logs.\n"
                "  Run './proclaude.sh logs' in another terminal to find the login URL."
            )

        # Trigger the OAuth flow — LiteLLM only emits the login URL
        # when a request actually hits a chatgpt/ model endpoint
        from container import PROXY_PORT
        chatgpt_model = None
        for m in config.list_models():
            if m["model"].startswith("chatgpt/"):
                chatgpt_model = m["alias"]
                break
        if chatgpt_model:
            master_key = config.get_env("LITELLM_MASTER_KEY")
            if not master_key:
                return Status.NOT_CONFIGURED, "LITELLM_MASTER_KEY not set. Run './proclaude.sh start' first."
            log.debug("Triggering OAuth flow with request to %s", chatgpt_model)
            try:
                requests.post(
                    f"http://localhost:{PROXY_PORT}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {master_key}",
                             "Content-Type": "application/json"},
                    json={"model": chatgpt_model,
                          "messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 1},
                    timeout=5,
                )
            except requests.RequestException as e:
                log.debug("OAuth trigger request failed (expected): %s", e)

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
            return Status.UNREACHABLE, (
                "Could not find login URL in container logs.\n"
                "  Make sure you have a chatgpt/ model configured and run './proclaude.sh logs' to debug."
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
                return Status.UNVERIFIED, "Browser OAuth may be active (log pattern detected, not independently verified)"

            # Lightweight proxy check — query /v1/models (no billing) to see
            # if chatgpt/ models are now being served after login
            if chatgpt_aliases:
                found, err = self._check_proxy_models(chatgpt_aliases)
                if found:
                    print("\n  ? Browser OAuth may be active (models detected in proxy, not independently verified)")
                    return Status.UNVERIFIED, "Browser OAuth may be active (models detected in proxy, not independently verified)"
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
        return Status.UNREACHABLE, "Login timed out after 5 minutes. Run './proclaude.sh login openai' to try again."
