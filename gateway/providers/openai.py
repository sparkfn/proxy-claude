import logging
import os

import requests

try:
    from .. import config
    from .base import BaseProvider, Status, is_placeholder
except ImportError:
    import config
    from providers.base import BaseProvider, Status, is_placeholder

log = logging.getLogger("litellm-cli.openai")


class OpenAIProvider(BaseProvider):
    name = "openai"
    display_name = "OpenAI"
    supports_thinking = True

    models = {
        "gpt-5.4": {
            "model": "chatgpt/gpt-5.4",
            "context": 1000000,
            "max_output": 128000,
            "thinking_levels": ("low", "medium", "high", "xhigh"),
        },
    }

    # API key auth uses a different litellm prefix
    _api_key_model_overrides = {
        "gpt-5.4": "openai/gpt-5.4",
    }

    auth = {
        "browser_oauth": {},
        "api_key": {
            "env_vars": ["OPENAI_API_KEY"],
            "instructions": "Enter your OpenAI API key.\n  Get one at: https://platform.openai.com/api-keys",
            "fields": [("OPENAI_API_KEY", "OPENAI_API_KEY: ")],
        },
    }

    def check_ready(self, env_data, auth_dir=None):
        """OpenAI is ready if API key is set OR browser OAuth token exists."""
        api_key = env_data.get("OPENAI_API_KEY", "")
        if api_key and not is_placeholder(api_key):
            return True, ""
        if auth_dir:
            auth_file = os.path.join(str(auth_dir), "chatgpt", "auth.json")
            if os.path.isfile(auth_file):
                try:
                    if os.path.getsize(auth_file) > 2:
                        return True, ""
                except OSError:
                    pass
        instructions = self.auth.get("api_key", {}).get("instructions", "")
        hint = ""
        if "Get one at: " in instructions:
            hint = " (%s)" % instructions.split("Get one at: ")[-1].strip()
        return False, "OPENAI_API_KEY not set and no browser OAuth found%s" % hint

    def get_model_string(self, alias, auth_type=None):
        if auth_type == "api_key":
            return self._api_key_model_overrides.get(alias)
        entry = self.models.get(alias)
        return entry["model"] if isinstance(entry, dict) else entry

    def get_models_for_auth(self, auth_type):
        if auth_type == "api_key":
            return {alias: {"model": override} for alias, override in self._api_key_model_overrides.items()}
        return self.models

    def resolve_thinking_contract(self, alias, litellm_model, litellm_params=None):
        levels = self._get_model_thinking_levels(alias) or self.thinking_levels
        if litellm_model.startswith("chatgpt/"):
            return self._openai_reasoning_contract("chatgpt", levels=levels)
        if litellm_model.startswith("openai/"):
            return self._openai_reasoning_contract("openai", levels=levels)
        return None

    def validate(self):
        api_key = config.get_env("OPENAI_API_KEY")
        if api_key:
            return self._validate_api_key(api_key)
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
        import container
        cs, _ = container.status()
        if cs != Status.OK:
            return Status.UNREACHABLE, (
                "LiteLLM backend is not yet reachable, so browser OAuth cannot be checked from inside "
                "the gateway. Use './proclaude.sh launch claude' or inspect './proclaude.sh logs litellm' on the host."
            )
        chatgpt_aliases = {alias for alias, m in self.models.items()
                          if isinstance(m, dict) and m.get("model", "").startswith("chatgpt/")}
        if chatgpt_aliases:
            found, err = self._check_proxy_models(chatgpt_aliases)
            if found:
                return Status.UNVERIFIED, "Browser OAuth may be active (models configured in proxy, but cannot independently verify upstream auth)"
            if err:
                return Status.UNREACHABLE, f"Cannot verify browser auth (proxy check failed: {err})"
        return Status.NOT_CONFIGURED, "Not authenticated — no browser OAuth evidence found. Run './proclaude.sh login openai' to authenticate."

    def _check_proxy_models(self, chatgpt_aliases):
        from container import PROXY_PORT
        master_key = config.get_env("LITELLM_MASTER_KEY")
        if not master_key:
            return False, "LITELLM_MASTER_KEY not set."
        try:
            resp = requests.get(
                f"http://localhost:{PROXY_PORT}/v1/models",
                headers={"Authorization": f"Bearer {master_key}"},
                timeout=10,
            )
            if resp.status_code != 200:
                return False, f"proxy returned HTTP {resp.status_code}"
            data = resp.json()
            if not isinstance(data, dict):
                return False, "non-dict response"
            items = data.get("data")
            if not isinstance(items, list):
                return False, "no valid 'data' list"
            served_ids = {m.get("id", "") for m in items if isinstance(m, dict)}
            if chatgpt_aliases & served_ids:
                return True, None
            return False, None
        except requests.RequestException as e:
            return False, f"request failed: {e}"
        except ValueError as e:
            return False, f"invalid JSON: {e}"

    def login(self, auth_type="browser_oauth", credentials=None):
        if auth_type == "api_key":
            return self._login_api_key(credentials)
        return self._login_browser()

    def _login_api_key(self, credentials=None):
        if credentials is None:
            return Status.INVALID, "No credentials provided."
        key = credentials.get("OPENAI_API_KEY", "")
        if not key:
            return Status.INVALID, "No key entered."
        config.set_env("OPENAI_API_KEY", key)
        return self.validate()

    def _login_browser(self):
        return Status.INVALID, "OpenAI browser OAuth must be started from './proclaude.sh provider login openai' on the host."
