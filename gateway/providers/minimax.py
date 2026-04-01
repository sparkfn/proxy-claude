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
    model_limits = {
        "MiniMax-M2.7":    {"context": 1000000, "max_output": 131072},
        "MiniMax-M2.5":    {"context": 1000000, "max_output": 131072},
        "MiniMax-Text-01": {"context": 1000000, "max_output": 131072},
    }

    def get_extra_params(self):
        """Extra litellm_params for MiniMax models."""
        return {"api_base": f"{self.API_BASE}/v1", "api_key": "os.environ/MINIMAX_API_KEY"}

    def resolve_thinking_contract(self, alias, litellm_model, litellm_params=None):
        if litellm_model.startswith("openai/"):
            return self._openai_reasoning_contract("openai")
        return None

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

        status, _ = self._classify_response(resp)
        if status == Status.OK:
            return status, "Authenticated with MiniMax"
        if status == Status.INVALID:
            return status, f"Invalid MINIMAX_API_KEY"
        return status, _

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
