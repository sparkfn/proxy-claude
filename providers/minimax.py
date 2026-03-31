import logging
import requests
import config
from providers.base import BaseProvider, Status, is_placeholder

log = logging.getLogger("litellm-cli.minimax")


class MiniMaxProvider(BaseProvider):
    name = "minimax"
    display_name = "MiniMax"
    supports_thinking = True
    auth_types = ["api_key"]
    env_vars = {"api_key": ["MINIMAX_API_KEY"]}
    # Use openai/ prefix so LiteLLM translates via OpenAI-compatible endpoint
    # (minimax/ prefix tries the Anthropic-native path which requires a paid plan)
    models = {
        "MiniMax-M2.7": "openai/MiniMax-M2.7",
        "MiniMax-M2.5": "openai/MiniMax-M2.5",
        "MiniMax-Text-01": "openai/MiniMax-Text-01",
    }

    def get_extra_params(self):
        """Extra litellm_params for MiniMax models."""
        return {"api_base": f"{self.API_BASE}/v1", "api_key": "os.environ/MINIMAX_API_KEY"}

    # LiteLLM appends /v1/ internally, so no trailing /v1 here
    API_BASE = "https://api.minimax.io"

    def validate(self):
        api_key = config.get_env("MINIMAX_API_KEY")
        if not api_key or is_placeholder(api_key):
            return Status.NOT_CONFIGURED, "MINIMAX_API_KEY not set"

        probe_model = next(iter(self.models))
        log.debug("Validating MiniMax key with model %s", probe_model)

        try:
            resp = requests.post(
                f"{self.API_BASE}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": probe_model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                timeout=10,
            )
        except requests.RequestException as e:
            return Status.UNREACHABLE, f"Cannot reach MiniMax API: {e}"

        if resp.status_code in (401, 403):
            return Status.INVALID, "Invalid MINIMAX_API_KEY"
        if resp.status_code == 429:
            return Status.UNREACHABLE, "Rate limited by MiniMax API"
        if resp.status_code >= 500:
            return Status.UNREACHABLE, f"MiniMax server error (HTTP {resp.status_code})"
        if resp.status_code != 200:
            log.warning("Unexpected HTTP %s from MiniMax validation", resp.status_code)
            return Status.UNREACHABLE, f"MiniMax returned unexpected status {resp.status_code}"

        content_type = resp.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            log.warning("MiniMax returned non-JSON Content-Type: %s", content_type)
            return Status.UNREACHABLE, "MiniMax returned non-JSON response"

        try:
            body = resp.json()
        except ValueError:
            log.warning("MiniMax returned invalid JSON body")
            return Status.UNREACHABLE, "MiniMax returned unparseable JSON"

        if "error" in body:
            err_msg = body["error"] if isinstance(body["error"], str) else body["error"].get("message", "unknown")
            log.warning("MiniMax 200 with error envelope: %s", err_msg)
            return Status.INVALID, f"MiniMax API error: {err_msg}"

        return Status.OK, "Authenticated with MiniMax"

    login_prompts = {
        "api_key": {
            "instructions": "Enter your MiniMax API key.\n  Get one at: https://platform.minimaxi.com/",
            "fields": [("MINIMAX_API_KEY", "MINIMAX_API_KEY: ")],
        }
    }

    def login(self, auth_type="api_key", credentials=None):
        """Authenticate with provided credentials. Caller must prompt via login_prompts."""
        if credentials is None:
            return Status.INVALID, "No credentials provided. Use login_prompts to collect them."
        key = credentials.get("MINIMAX_API_KEY", "")
        if not key:
            return Status.INVALID, "No key entered."
        config.set_env("MINIMAX_API_KEY", key)
        return self.validate()
