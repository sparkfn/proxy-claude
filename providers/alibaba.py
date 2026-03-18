import requests
import config
from providers.base import BaseProvider, AuthStatus, is_placeholder


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
            return AuthStatus.NOT_CONFIGURED, "DASHSCOPE_API_KEY not set"
        try:
            resp = requests.get(
                self.API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                return AuthStatus.OK, "Authenticated with DashScope"
            if resp.status_code == 401:
                return AuthStatus.INVALID, "Invalid DASHSCOPE_API_KEY"
            return AuthStatus.INVALID, f"DashScope returned status {resp.status_code}"
        except requests.ConnectionError:
            return AuthStatus.UNREACHABLE, "Cannot reach DashScope API"
        except requests.Timeout:
            return AuthStatus.UNREACHABLE, "DashScope API timed out"

    def login(self, auth_type="api_key"):
        print(f"\n  Enter your DashScope API key.")
        print(f"  Get one at: https://dashscope.console.aliyun.com/\n")
        key = input("  DASHSCOPE_API_KEY: ").strip()
        if not key:
            return False, "No key entered."
        config.set_env("DASHSCOPE_API_KEY", key)
        # Validate the key
        status, msg = self.validate()
        if status == AuthStatus.OK:
            return True, msg
        return False, msg
