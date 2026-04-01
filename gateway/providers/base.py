from abc import ABC, abstractmethod
from enum import Enum

THINKING_LEVELS = ("low", "medium", "high")


def is_placeholder(value):
    """Detect common placeholder values from .env.example files."""
    if not value:
        return True
    v = value.lower().strip()
    return (
        v.startswith("your-") or
        v.startswith("your_") or
        v.endswith("-here") or
        v.endswith("_here") or
        v in ("changeme", "replace-me", "xxx", "todo", "placeholder")
    )


class Status(Enum):
    """Unified result status for all internal Python operations.

    Used by providers, config, and container. Translated at boundaries:
    - proxy.py translates to HTTP status codes + JSON envelopes
    - cli.py translates to print output + sys.exit codes
    - shell scripts translate to exit codes
    """
    OK = "ok"
    FAILED = "failed"              # Generic operation failure
    NOT_FOUND = "not_found"        # Resource/model not found
    UNVERIFIED = "unverified"      # Cannot confirm, may be ok
    NOT_CONFIGURED = "not_configured"  # Missing setup/config
    INVALID = "invalid"            # Bad input or state
    UNREACHABLE = "unreachable"    # Cannot reach external service


class BaseProvider(ABC):
    name: str = ""
    display_name: str = ""
    supports_thinking: bool = False
    thinking_levels = THINKING_LEVELS
    anthropic_base_url: str = None  # Native Anthropic-compatible endpoint (bypasses LiteLLM)
    native_auth: dict = None  # {"header": "x-api-key", "env": "MINIMAX_API_KEY"} — how proxy.py authenticates native forwarding
    # Per-model token limits: {alias: {"context": int, "max_output": int}}
    # Used by cli.py to set CLAUDE_CODE_MAX_MODEL_TOKENS / MAX_OUTPUT_TOKENS at launch.
    model_limits: dict = {}

    def __init__(self):
        # Instance-level copies to prevent mutable class-level sharing
        if not hasattr(self, '_init_done'):
            self.auth_types = list(self.__class__.auth_types) if hasattr(self.__class__, 'auth_types') else []
            self.env_vars = dict(self.__class__.env_vars) if hasattr(self.__class__, 'env_vars') else {}
            self.models = dict(self.__class__.models) if hasattr(self.__class__, 'models') else {}
            self._init_done = True

    # Override in subclass: {auth_type: {"instructions": str, "fields": [(env_var, prompt)]}}
    # CLI layer reads this to prompt the user, then passes credentials to login().
    login_prompts: dict = {}

    @abstractmethod
    def validate(self) -> tuple:
        """Returns (Status, message_string)."""
        pass

    @abstractmethod
    def login(self, auth_type=None, credentials=None) -> tuple:
        """Returns (Status, message_string).

        credentials: dict of {env_var: value} collected by the CLI layer
        using login_prompts. Returns Status.INVALID if None and credentials are required.
        """
        pass

    def get_model_string(self, alias, auth_type=None):
        """Get the litellm model string for an alias."""
        return self.models.get(alias)

    def get_env_vars_for_auth(self, auth_type):
        """Get list of env var names needed for an auth type."""
        return self.env_vars.get(auth_type, [])

    def detect_auth_type(self):
        """Detect which auth type is currently configured based on env vars.
        Returns the auth_type string or None."""
        import config as cfg
        for auth_type, env_var_list in self.env_vars.items():
            if env_var_list:
                all_set = all(cfg.get_env(v) for v in env_var_list)
                if all_set:
                    return auth_type
        # If no env-var-based auth found, check for auth types with no env vars
        for auth_type, env_var_list in self.env_vars.items():
            if not env_var_list:
                return auth_type
        return self.auth_types[0] if self.auth_types else None

    def resolve_thinking_contract(self, alias, litellm_model, litellm_params=None):
        """Return a verified thinking contract dict for a configured model, or None."""
        return None

    def _openai_reasoning_contract(self, route_family):
        """Thinking contract for OpenAI-compatible chat/completions routes."""
        return {
            "provider": self.name,
            "strategy": "openai_chat_reasoning_effort",
            "route_family": route_family,
            "levels": self.thinking_levels,
            "requires_openai_translation": True,
        }

    def _classify_response(self, resp):
        """Classify an HTTP response from a validation probe.

        Shared logic for all providers: transport errors, HTTP status classification,
        content-type validation, JSON parsing, and 200-with-error-envelope detection.

        Returns (Status, message_string) — caller returns directly.
        """
        if resp.status_code in (401, 403):
            return Status.INVALID, f"Invalid or forbidden credentials (HTTP {resp.status_code})"
        if resp.status_code == 429:
            return Status.UNREACHABLE, "Rate limited — credential not verified"
        if resp.status_code >= 500:
            return Status.UNREACHABLE, f"Provider server error (HTTP {resp.status_code}) — key not validated"
        if resp.status_code != 200:
            return Status.UNREACHABLE, f"Unexpected status {resp.status_code}"

        ct = resp.headers.get("Content-Type", "")
        if "application/json" not in ct:
            return Status.UNREACHABLE, f"Unexpected Content-Type: {ct}"

        try:
            data = resp.json()
        except ValueError:
            return Status.UNREACHABLE, "Non-JSON response"

        if isinstance(data, dict) and "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return Status.INVALID, f"Provider error: {msg}"

        return Status.OK, ""
