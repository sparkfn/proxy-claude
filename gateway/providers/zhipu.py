import logging

import requests

import config
from providers.base import BaseProvider, Status, is_placeholder

log = logging.getLogger("litellm-cli.zhipu")


class ZhipuProvider(BaseProvider):
    name = "zhipu"
    display_name = "Z.AI (Zhipu)"
    supports_thinking = True
    auth_types = ["api_key"]
    env_vars = {"api_key": ["ZAI_API_KEY"]}
    # OpenAI-compatible — use openai/ prefix so LiteLLM translates
    models = {
        "glm-5.1": "openai/glm-5.1",
        "glm-5": "openai/glm-5",
    }
    model_limits = {
        "glm-5.1": {"context": 204800, "max_output": 131072},
        "glm-5":   {"context": 204800, "max_output": 131072},
    }

    API_BASE = "https://api.z.ai/api/coding/paas/v4"

    def get_extra_params(self):
        """Extra litellm_params for Zhipu models."""
        return {"api_base": self.API_BASE, "api_key": "os.environ/ZAI_API_KEY"}

    def resolve_thinking_contract(self, alias, litellm_model, litellm_params=None):
        if litellm_model.startswith("openai/"):
            return self._openai_reasoning_contract("openai")
        return None

    def validate(self):
        api_key = config.get_env("ZAI_API_KEY")
        if not api_key or is_placeholder(api_key):
            return Status.NOT_CONFIGURED, "ZAI_API_KEY not set"

        probe_model = next(iter(self.models))
        log.debug("Validating Z.AI credentials with model %s", probe_model)

        try:
            resp = requests.post(
                f"{self.API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": probe_model,
                      "messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 1},
                timeout=10,
            )
        except requests.RequestException as e:
            return Status.UNREACHABLE, f"Cannot reach Z.AI API: {e}"

        status, _ = self._classify_response(resp)
        if status == Status.OK:
            return status, "Authenticated with Z.AI"
        return status, _

    login_prompts = {
        "api_key": {
            "instructions": "Enter your Z.AI API key.\n  Get one at: https://z.ai/manage-apikey/apikey-list",
            "fields": [("ZAI_API_KEY", "ZAI_API_KEY: ")],
        }
    }

    def login(self, auth_type="api_key", credentials=None):
        if credentials is None:
            return Status.INVALID, "No credentials provided. Use login_prompts to collect them."
        key = credentials.get("ZAI_API_KEY", "")
        if not key:
            return Status.INVALID, "No key entered."
        config.set_env("ZAI_API_KEY", key)
        return self.validate()
