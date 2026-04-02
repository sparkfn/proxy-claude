import logging

try:
    from .. import config
    from .base import BaseProvider, Status
except ImportError:
    import config
    from providers.base import BaseProvider, Status

log = logging.getLogger("litellm-cli.zhipu")


class ZhipuProvider(BaseProvider):
    name = "zhipu"
    display_name = "Z.AI (Zhipu)"
    supports_thinking = True

    API_BASE = "https://api.z.ai/api/coding/paas/v4"

    models = {
        "glm-5.1": {
            "model": "openai/glm-5.1",
            "context": 204800,
            "max_output": 131072,
        },
    }

    auth = {
        "api_key": {
            "env_vars": ["ZAI_API_KEY"],
            "instructions": "Enter your Z.AI API key.\n  Get one at: https://z.ai/manage-apikey/apikey-list",
            "fields": [("ZAI_API_KEY", "ZAI_API_KEY: ")],
        },
    }

    def get_extra_params(self):
        return {"api_base": self.API_BASE, "api_key": "os.environ/ZAI_API_KEY"}

    def resolve_thinking_contract(self, alias, litellm_model, litellm_params=None):
        if litellm_model.startswith("openai/"):
            levels = self._get_model_thinking_levels(alias) or self.thinking_levels
            return self._openai_reasoning_contract("openai", levels=levels)
        return None

    def validate(self):
        probe_alias = next(iter(self.models))
        return self._validate_openai_compatible_api_key(
            env_var="ZAI_API_KEY",
            api_base=self.API_BASE,
            model=probe_alias,
            provider_label="Z.AI",
            success_message="Authenticated with Z.AI",
            invalid_message="Invalid ZAI_API_KEY",
        )

    def login(self, auth_type="api_key", credentials=None):
        if credentials is None:
            return Status.INVALID, "No credentials provided."
        key = credentials.get("ZAI_API_KEY", "")
        if not key:
            return Status.INVALID, "No key entered."
        config.set_env("ZAI_API_KEY", key)
        return self.validate()
