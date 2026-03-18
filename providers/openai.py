import logging
import re
import sys
import time
from datetime import datetime, timezone

import requests

import config
import container
from providers.base import BaseProvider, AuthStatus

log = logging.getLogger("litellm-cli.openai")


class OpenAIProvider(BaseProvider):
    name = "openai"
    display_name = "OpenAI"
    auth_types = ["browser_oauth", "api_key"]
    env_vars = {
        "browser_oauth": [],
        "api_key": ["OPENAI_API_KEY"],
    }
    models = {
        "gpt-5": "chatgpt/gpt-5",
        "gpt-4o": "chatgpt/gpt-4o",
        "gpt-4o-mini": "chatgpt/gpt-4o-mini",
    }

    _api_key_models = {
        "gpt-5": "openai/gpt-5",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
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
            if resp.status_code == 200:
                return AuthStatus.OK, "Authenticated with OpenAI API key"
            if resp.status_code == 401:
                return AuthStatus.INVALID, "Invalid OPENAI_API_KEY"
            return AuthStatus.INVALID, f"OpenAI returned status {resp.status_code}"
        except requests.ConnectionError:
            return AuthStatus.UNREACHABLE, "Cannot reach OpenAI API"
        except requests.Timeout:
            return AuthStatus.UNREACHABLE, "OpenAI API timed out"

    def _validate_browser(self):
        """Check container logs for browser OAuth auth state."""
        running, _ = container.status()
        if not running:
            return AuthStatus.NOT_CONFIGURED, "Container not running — cannot check browser auth"
        logs = container.get_logs_tail(200)
        # Match specific LiteLLM auth success patterns, not bare "authenticated"
        if re.search(r"(?i)(successfully authenticated|chatgpt.*auth|session.*authenticated)", logs):
            log.debug("Browser OAuth auth pattern found in logs")
            return AuthStatus.OK, "Authenticated via browser OAuth"
        log.debug("No auth pattern found in last 200 log lines")
        return AuthStatus.NOT_CONFIGURED, "Not authenticated with OpenAI"

    def login(self, auth_type="browser_oauth"):
        if auth_type == "api_key":
            return self._login_api_key()
        return self._login_browser()

    def _login_api_key(self):
        print(f"\n  Enter your OpenAI API key.")
        print(f"  Get one at: https://platform.openai.com/api-keys\n")
        key = input("  OPENAI_API_KEY: ").strip()
        if not key:
            return False, "No key entered."
        config.set_env("OPENAI_API_KEY", key)
        status, msg = self.validate()
        if status == AuthStatus.OK:
            return True, msg
        return False, msg

    def _login_browser(self):
        """Drive the browser OAuth flow by reading container logs."""
        # Pre-check
        status, msg = self.validate()
        if status == AuthStatus.OK:
            return True, f"Already authenticated. {msg}"

        # Ensure container is running
        running, _ = container.status()
        if not running:
            print("  Container not running. Starting it...")
            container.up()
            if not container.wait_healthy(30):
                return False, "Container failed to start."

        # Capture timestamp before looking for URL
        since = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        print("\n  Waiting for login URL from container...")
        login_url = None
        for attempt in range(30):  # 30 * 2s = 60s to find URL
            logs = container.get_logs_since(since)
            # Match OpenAI device code URLs specifically — not generic docs/GitHub URLs
            urls = re.findall(r'https?://(?:login\.chatgpt\.com|auth0\.openai\.com|chat\.openai\.com)[^\s"\']*', logs)
            if not urls:
                # Fallback: match any URL with device_code or user_code params
                urls = re.findall(r'https?://[^\s"\']*(?:device_code|user_code|device)[^\s"\']*', logs)
            if urls:
                login_url = urls[-1]
                break
            time.sleep(2)
            print(".", end="", flush=True)

        if not login_url:
            return False, (
                "Could not find login URL in container logs.\n"
                "  Make sure you have a chatgpt/ model configured and run './litellm.sh logs' to debug."
            )

        print(f"\n")
        print(f"  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  OpenAI Login Required                             │")
        print(f"  │                                                     │")
        print(f"  │  Open this URL in your browser:                     │")
        print(f"  │  {login_url[:50]:<50} │")
        if len(login_url) > 50:
            print(f"  │  {login_url[50:100]:<50} │")
        print(f"  │                                                     │")
        print(f"  │  Waiting for browser login... (timeout: 5 min)      │")
        print(f"  └─────────────────────────────────────────────────────┘")
        print()

        # Poll for auth success
        timeout = 300  # 5 minutes
        start = time.time()
        while time.time() - start < timeout:
            logs = container.get_logs_since(since)
            if re.search(r"(?i)authenticated", logs):
                print("  ✓ Authenticated with OpenAI via browser OAuth!")
                return True, "Authenticated via browser OAuth"
            elapsed = int(time.time() - start)
            remaining = timeout - elapsed
            mins, secs = divmod(remaining, 60)
            print(f"\r  Polling... {mins}:{secs:02d} remaining  ", end="", flush=True)
            time.sleep(3)

        print()
        return False, "Login timed out after 5 minutes. Run './litellm.sh login openai' to try again."
