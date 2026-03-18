import requests
from providers.base import BaseProvider, AuthStatus


class OllamaProvider(BaseProvider):
    name = "ollama"
    display_name = "Ollama (Local)"
    auth_types = []
    env_vars = {}
    models = {}  # Dynamic — discovered at runtime

    OLLAMA_HOST = "http://localhost:11434"
    DOCKER_HOST = "http://host.docker.internal:11434"

    def validate(self):
        try:
            resp = requests.get(f"{self.OLLAMA_HOST}/api/tags", timeout=3)
            if resp.status_code == 200:
                return AuthStatus.OK, "Ollama is reachable"
            return AuthStatus.UNREACHABLE, f"Ollama returned status {resp.status_code}"
        except requests.ConnectionError:
            return AuthStatus.UNREACHABLE, "Ollama is not running at localhost:11434"
        except requests.Timeout:
            return AuthStatus.UNREACHABLE, "Ollama connection timed out"

    def login(self, auth_type=None):
        # No auth needed — just check reachability
        status, msg = self.validate()
        if status == AuthStatus.OK:
            return True, msg
        return False, msg

    def discover_models(self):
        """Fetch available models from Ollama. Returns dict of alias -> litellm model string."""
        try:
            resp = requests.get(f"{self.OLLAMA_HOST}/api/tags", timeout=5)
            if resp.status_code != 200:
                return {}
            data = resp.json()
            models = {}
            for m in data.get("models", []):
                name = m.get("name", "")
                if name:
                    alias = name.replace(":latest", "")
                    models[alias] = f"ollama/{name}"
            return models
        except (requests.ConnectionError, requests.Timeout):
            return {}

    def get_model_string(self, alias, auth_type=None):
        return f"ollama/{alias}"

    def get_extra_params(self):
        """Return extra litellm_params for Ollama models."""
        return {"api_base": self.DOCKER_HOST}
