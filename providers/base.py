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
    auth_types: list = []
    env_vars: dict = {}  # auth_type -> list of env var names
    models: dict = {}    # alias -> litellm model string

    @abstractmethod
    def validate(self) -> tuple:
        """Returns (AuthStatus, message_string)."""
        pass

    @abstractmethod
    def login(self, auth_type: str) -> tuple:
        """Returns (success: bool, message_string)."""
        pass

    def get_model_string(self, alias, auth_type=None):
        """Get the litellm model string for an alias."""
        return self.models.get(alias)

    def get_env_vars_for_auth(self, auth_type):
        """Get list of env var names needed for an auth type."""
        return self.env_vars.get(auth_type, [])
