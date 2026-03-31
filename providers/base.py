from abc import ABC, abstractmethod
from enum import Enum


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
    anthropic_base_url: str = None  # Native Anthropic-compatible endpoint (bypasses LiteLLM)
    native_auth: dict = None  # {"header": "x-api-key", "env": "MINIMAX_API_KEY"} — how proxy.py authenticates native forwarding

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
