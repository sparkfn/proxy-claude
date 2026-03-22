import requests
import config
from providers.base import BaseProvider, Status, is_placeholder


class AlibabaProvider(BaseProvider):
    name = "alibaba"
    display_name = "Alibaba (DashScope)"
    auth_types = ["api_key"]
    env_vars = {"api_key": ["DASHSCOPE_API_KEY"]}
    models = {
        "qwen-max": "dashscope/qwen-max",
        "qwen-plus": "dashscope/qwen-plus",
        "qwen-turbo": "dashscope/qwen-turbo",
    }

    API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/models"

    def validate(self):
        api_key = config.get_env("DASHSCOPE_API_KEY")
        if not api_key or is_placeholder(api_key):
            return Status.NOT_CONFIGURED, "DASHSCOPE_API_KEY not set"
        try:
            resp = requests.get(
                self.API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                return Status.OK, "Authenticated with DashScope"
            if resp.status_code == 401:
                return Status.INVALID, "Invalid DASHSCOPE_API_KEY"
            return Status.INVALID, f"DashScope returned status {resp.status_code}"
        except requests.RequestException as e:
            return Status.UNREACHABLE, f"Cannot reach DashScope API: {e}"

    # Credentials prompt shown by CLI layer, not here
    login_prompts = {
        "api_key": {
            "instructions": "Enter your DashScope API key.\n  Get one at: https://dashscope.console.aliyun.com/",
            "fields": [("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY: ")],
        }
    }

    def login(self, auth_type="api_key", credentials=None):
        """Authenticate with provided credentials. Caller must prompt via login_prompts."""
        if credentials is None:
            return Status.INVALID, "No credentials provided. Use login_prompts to collect them."
        key = credentials.get("DASHSCOPE_API_KEY", "")
        if not key:
            return Status.INVALID, "No key entered."
        config.set_env("DASHSCOPE_API_KEY", key)
        return self.validate()
