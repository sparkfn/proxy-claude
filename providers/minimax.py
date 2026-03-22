import requests
import config
from providers.base import BaseProvider, Status, is_placeholder


class MiniMaxProvider(BaseProvider):
    name = "minimax"
    display_name = "MiniMax"
    auth_types = ["api_key"]
    env_vars = {"api_key": ["MINIMAX_API_KEY"]}
    models = {
        "MiniMax-M2.7": "minimax/MiniMax-M2.7",
        "MiniMax-M2.5": "minimax/MiniMax-M2.5",
        "MiniMax-Text-01": "minimax/MiniMax-Text-01",
    }

    def get_extra_params(self):
        """Extra litellm_params for MiniMax models."""
        return {"api_base": self.API_BASE}

    API_BASE = "https://api.minimax.io/v1"

    def validate(self):
        api_key = config.get_env("MINIMAX_API_KEY")
        if not api_key or is_placeholder(api_key):
            return Status.NOT_CONFIGURED, "MINIMAX_API_KEY not set"
        # MiniMax has no /v1/models endpoint, so validate with a 1-token completion
        try:
            resp = requests.post(
                f"{self.API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": "MiniMax-M2.5", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                timeout=10,
            )
            if resp.status_code == 200:
                return Status.OK, "Authenticated with MiniMax"
            if resp.status_code in (401, 403):
                return Status.INVALID, "Invalid MINIMAX_API_KEY"
            return Status.INVALID, f"MiniMax returned status {resp.status_code}"
        except requests.RequestException as e:
            return Status.UNREACHABLE, f"Cannot reach MiniMax API: {e}"

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
