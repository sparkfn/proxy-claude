import logging
try:
    from .. import config
    from .base import BaseProvider, Status
except ImportError:
    import config
    from providers.base import BaseProvider, Status

log = logging.getLogger("litellm-cli.minimax")


class MiniMaxProvider(BaseProvider):
    name = "minimax"
    display_name = "MiniMax"
    supports_thinking = True

    API_BASE = "https://api.minimax.io"

    models = {
        "MiniMax-M2.7": {
            "model": "openai/MiniMax-M2.7",
            "context": 1000000,
            "max_output": 131072,
        },
    }

    auth = {
        "api_key": {
            "env_vars": ["MINIMAX_API_KEY"],
            "instructions": "Enter your MiniMax API key.\n  Get one at: https://platform.minimaxi.com/",
            "fields": [("MINIMAX_API_KEY", "MINIMAX_API_KEY: ")],
        },
    }

    def get_extra_params(self):
        return {"api_base": f"{self.API_BASE}/v1", "api_key": "os.environ/MINIMAX_API_KEY"}

    def resolve_thinking_contract(self, alias, litellm_model, litellm_params=None):
        if litellm_model.startswith("openai/"):
            levels = self._get_model_thinking_levels(alias) or self.thinking_levels
            return self._openai_reasoning_contract("openai", levels=levels)
        return None

    def validate(self):
        probe_alias = next(iter(self.models))
        return self._validate_openai_compatible_api_key(
            env_var="MINIMAX_API_KEY",
            api_base=f"{self.API_BASE}/v1",
            model=probe_alias,
            provider_label="MiniMax",
            success_message="Authenticated with MiniMax",
            invalid_message="Invalid MINIMAX_API_KEY",
        )

    def login(self, auth_type="api_key", credentials=None):
        if credentials is None:
            return Status.INVALID, "No credentials provided."
        key = credentials.get("MINIMAX_API_KEY", "")
        if not key:
            return Status.INVALID, "No key entered."
        config.set_env("MINIMAX_API_KEY", key)
        return self.validate()
