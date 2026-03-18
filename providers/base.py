from abc import ABC, abstractmethod
from enum import Enum


class AuthStatus(Enum):
    OK = "ok"
    NOT_CONFIGURED = "not_configured"
    INVALID = "invalid"
    UNREACHABLE = "unreachable"


class BaseProvider(ABC):
    name: str = ""
    display_name: str = ""

    def __init__(self):
        # Instance-level copies to prevent mutable class-level sharing
        if not hasattr(self, '_init_done'):
            self.auth_types = list(self.__class__.auth_types) if hasattr(self.__class__, 'auth_types') else []
            self.env_vars = dict(self.__class__.env_vars) if hasattr(self.__class__, 'env_vars') else {}
            self.models = dict(self.__class__.models) if hasattr(self.__class__, 'models') else {}
            self._init_done = True

    @abstractmethod
    def validate(self) -> tuple:
        """Returns (AuthStatus, message_string)."""
        pass

    @abstractmethod
    def login(self, auth_type=None) -> tuple:
        """Returns (success: bool, message_string)."""
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
