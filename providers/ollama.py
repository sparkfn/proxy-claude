import json
import logging
import os
import shutil
import subprocess
from urllib.parse import urlparse
import requests
from providers.base import BaseProvider, Status

log = logging.getLogger("litellm-cli.ollama")


class OllamaProvider(BaseProvider):
    name = "ollama"
    display_name = "Ollama"
    auth_types = []
    env_vars = {}
    models = {}  # Dynamic — discovered at runtime

    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_DOCKER_HOST = "http://host.docker.internal:11434"

    @property
    def OLLAMA_HOST(self):
        """Host URL used for local API calls (respects OLLAMA_HOST env var)."""
        return os.environ.get("OLLAMA_HOST", self.DEFAULT_HOST)

    @property
    def DOCKER_HOST(self):
        """Host URL used inside Docker containers (respects OLLAMA_HOST env var).

        If OLLAMA_HOST points to localhost/127.0.0.1, replace with
        host.docker.internal but keep the port.  If it points elsewhere,
        use it as-is (likely a remote Ollama instance).
        """
        env_host = os.environ.get("OLLAMA_HOST")
        if not env_host:
            return self.DEFAULT_DOCKER_HOST

        try:
            parsed = urlparse(env_host)
            hostname = parsed.hostname or ""
            if hostname in ("localhost", "127.0.0.1", "::1"):
                port = parsed.port or 11434
                scheme = parsed.scheme or "http"
                return f"{scheme}://host.docker.internal:{port}"
            # Non-localhost — use as-is (remote Ollama)
            return env_host
        except (ValueError, AttributeError) as e:
            log.warning("Failed to parse OLLAMA_HOST '%s', using default: %s", env_host, e)
            return self.DEFAULT_DOCKER_HOST

    def validate(self):
        host = self.OLLAMA_HOST
        try:
            resp = requests.get(f"{host}/api/tags", timeout=3)
            if resp.status_code != 200:
                return Status.UNREACHABLE, f"Ollama returned status {resp.status_code}"
            ct = resp.headers.get("Content-Type", "")
            if "json" not in ct:
                return Status.UNREACHABLE, f"Ollama returned unexpected Content-Type: {ct}"
            try:
                resp.json()
            except ValueError:
                return Status.UNREACHABLE, "Ollama returned invalid JSON"
            return Status.OK, f"Ollama is reachable at {host}"
        except requests.RequestException as e:
            log.warning("Ollama validate failed: %s", e)
            return Status.UNREACHABLE, f"Cannot reach Ollama at {host}: {e}"

    def login(self, auth_type=None, credentials=None):
        """Validate Ollama connectivity. Returns (Status, msg).

        Interactive flows (cloud login, model pull) are driven by the CLI layer
        via ollama_cloud_login() and pull_model().
        """
        return self.validate()

    def ollama_cloud_login(self):
        """Attempt ollama.com cloud login. Returns (Status, msg)."""
        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            return Status.NOT_CONFIGURED, "ollama CLI not found — install it to login for cloud models"
        try:
            result = subprocess.run([ollama_bin, "login"], timeout=120)
        except (OSError, subprocess.TimeoutExpired) as e:
            return Status.UNREACHABLE, f"ollama login failed: {e}"
        if result.returncode != 0:
            return Status.UNREACHABLE, "ollama login failed"
        return Status.OK, "Logged in to ollama.com"

    def discover_models(self):
        """Fetch available models from Ollama.

        Returns dict of alias -> litellm model string, or None on error.
        Empty dict means Ollama is reachable but has no models.
        """
        host = self.OLLAMA_HOST
        try:
            resp = requests.get(f"{host}/api/tags", timeout=5)
            if resp.status_code != 200:
                log.warning(
                    "Ollama at %s returned HTTP %d — cannot discover models",
                    host, resp.status_code,
                )
                return None
            try:
                data = resp.json()
            except ValueError:
                log.warning("Ollama at %s returned invalid JSON", host)
                return None
            models_list = data.get("models", [])
            if not isinstance(models_list, list):
                log.warning(
                    "Ollama at %s returned non-list 'models' field: %s",
                    host, type(models_list).__name__,
                )
                return None
            models = {}
            for m in models_list:
                if not isinstance(m, dict):
                    continue
                name = m.get("name", "")
                if not isinstance(name, str) or not name:
                    continue
                alias = name.replace(":latest", "")
                models[alias] = f"ollama/{name}"
            return models
        except requests.RequestException as e:
            log.warning("Could not reach Ollama at %s: %s", host, e)
            return None

    def pull_model(self, model_name):
        """Pull a model via Ollama REST API. Returns (Status, message)."""
        try:
            resp = requests.post(
                f"{self.OLLAMA_HOST}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600,
            )
            if resp.status_code != 200:
                return Status.FAILED, f"Pull failed with status {resp.status_code}"

            # Set socket idle timeout to detect stalled transfers
            try:
                resp.raw._fp.fp.raw._sock.settimeout(60)
            except (AttributeError, TypeError) as e:
                log.warning("Could not set socket idle timeout for pull stream "
                            "(idle timeout protection is not active): %s", e)

            last_status = ""
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                    except ValueError:
                        log.debug("Skipping malformed NDJSON line: %s", line[:100])
                        continue
                    status = data.get("status", "")
                    if "completed" in data and "total" in data:
                        total = data["total"]
                        pct = int(data["completed"] / total * 100) if total > 0 else 0
                        print(f"\r  {status}: {pct}%    ", end="", flush=True)
                    elif status != last_status:
                        print(f"\r  {status}              ", end="", flush=True)
                    last_status = status
            print()
            return Status.OK, f"Pulled {model_name}"
        except requests.RequestException as e:
            return Status.FAILED, f"Pull failed: {e}"

    def get_model_string(self, alias, auth_type=None):
        return f"ollama/{alias}"

    def get_extra_params(self):
        """Return extra litellm_params for Ollama models."""
        return {"api_base": self.DOCKER_HOST}
