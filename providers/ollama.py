import json
import shutil
import subprocess
import requests
from providers.base import BaseProvider, AuthStatus


class OllamaProvider(BaseProvider):
    name = "ollama"
    display_name = "Ollama"
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
        status, msg = self.validate()
        if status != AuthStatus.OK:
            return False, msg

        print(f"  ✓ {msg}")

        # Offer ollama login for cloud model access
        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            print("  ⚠ ollama CLI not found — install it to login for cloud models")
        else:
            choice = input("\n  Login to ollama.com for cloud models? [y/N]: ").strip()
            if choice.lower() == "y":
                print()
                result = subprocess.run([ollama_bin, "login"])
                if result.returncode != 0:
                    return False, "ollama login failed"
                print("  ✓ Logged in to ollama.com")

        # Show available models
        models = self.discover_models()
        if models:
            print(f"\n  Available models ({len(models)}):\n")
            for alias in models:
                print(f"    • {alias}")
        else:
            print("\n  No models found.")

        # Offer to pull
        pull = input("\n  Pull a model? Enter name (or Enter to skip): ").strip()
        if pull:
            print()
            ok, pull_msg = self.pull_model(pull)
            if ok:
                print(f"  ✓ {pull_msg}")
            else:
                print(f"  ✗ {pull_msg}")

        return True, "Ollama ready"

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

    def pull_model(self, model_name):
        """Pull a model via Ollama REST API. Returns (success, message)."""
        try:
            resp = requests.post(
                f"{self.OLLAMA_HOST}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600,
            )
            if resp.status_code != 200:
                return False, f"Pull failed with status {resp.status_code}"

            last_status = ""
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if "completed" in data and "total" in data:
                        total = data["total"]
                        pct = int(data["completed"] / total * 100) if total > 0 else 0
                        print(f"\r  {status}: {pct}%    ", end="", flush=True)
                    elif status != last_status:
                        print(f"\r  {status}              ", end="", flush=True)
                    last_status = status
            print()
            return True, f"Pulled {model_name}"
        except requests.ConnectionError:
            return False, "Ollama is not running — cannot pull"
        except requests.Timeout:
            return False, "Pull timed out"

    def get_model_string(self, alias, auth_type=None):
        return f"ollama/{alias}"

    def get_extra_params(self):
        """Return extra litellm_params for Ollama models."""
        return {"api_base": self.DOCKER_HOST}
