import logging
from abc import ABC, abstractmethod
from enum import Enum
import requests

log = logging.getLogger("litellm-cli.providers")

THINKING_LEVELS = ("low", "medium", "high")
THINKING_LEVEL_LABELS = {
    "none": "None",
    "minimal": "Minimal",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "Extra high",
}


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
    OK = "ok"
    FAILED = "failed"
    NOT_FOUND = "not_found"
    UNVERIFIED = "unverified"
    NOT_CONFIGURED = "not_configured"
    INVALID = "invalid"
    UNREACHABLE = "unreachable"


class BaseProvider(ABC):
    """Base class for LLM providers.

    Subclasses declare two dicts as the single source of truth:

        models = {
            "alias": {
                "model": "litellm/model-string",   # required
                "context": 1000000,                 # optional
                "max_output": 128000,               # optional
                "thinking_levels": ("low", ...),    # optional
            },
        }

        auth = {
            "api_key": {
                "env_vars": ["API_KEY_VAR"],
                "instructions": "Enter your key.\\n  Get one at: https://...",
                "fields": [("API_KEY_VAR", "API_KEY_VAR: ")],
            },
        }

    Everything else is derived from these two dicts.
    """
    name: str = ""
    display_name: str = ""
    supports_thinking: bool = False
    thinking_levels = THINKING_LEVELS
    anthropic_base_url: str = None
    native_auth: dict = None

    # --- Single source of truth (override in subclass) ---
    models: dict = {}
    auth: dict = {}

    def __init__(self):
        if not hasattr(self, '_init_done'):
            self.models = dict(self.__class__.models)
            self.auth = dict(self.__class__.auth)
            self._init_done = True

    # --- Derived properties (backward compatible) ---

    @property
    def auth_types(self):
        return list(self.auth.keys())

    @property
    def env_vars(self):
        return {at: cfg.get("env_vars", []) for at, cfg in self.auth.items()}

    @property
    def login_prompts(self):
        result = {}
        for at, cfg in self.auth.items():
            if cfg.get("fields"):
                result[at] = {
                    "instructions": cfg.get("instructions", ""),
                    "fields": cfg["fields"],
                }
        return result

    @property
    def model_limits(self):
        result = {}
        for alias, m in self.models.items():
            if not isinstance(m, dict):
                continue
            limits = {}
            if m.get("context"):
                limits["context"] = m["context"]
            if m.get("max_output"):
                limits["max_output"] = m["max_output"]
            if limits:
                result[alias] = limits
        return result

    # --- Core methods ---

    def check_ready(self, env_data, auth_dir=None):
        """Fast local-only readiness check. No network calls."""
        for auth_type, cfg in self.auth.items():
            env_var_list = cfg.get("env_vars", [])
            if not env_var_list:
                continue
            for var in env_var_list:
                val = env_data.get(var, "")
                if not val or is_placeholder(val):
                    instructions = cfg.get("instructions", "")
                    hint = ""
                    if "Get one at: " in instructions:
                        hint = " (%s)" % instructions.split("Get one at: ")[-1].strip()
                    elif "get one at: " in instructions.lower():
                        hint = " (%s)" % instructions.split("get one at: ")[-1].strip()
                    return False, "%s not set%s" % (var, hint)
            return True, ""
        return True, ""

    @abstractmethod
    def validate(self) -> tuple:
        """Returns (Status, message_string)."""
        pass

    @abstractmethod
    def login(self, auth_type=None, credentials=None) -> tuple:
        """Returns (Status, message_string)."""
        pass

    def get_model_string(self, alias, auth_type=None):
        """Get the litellm model string for an alias."""
        entry = self.models.get(alias)
        if isinstance(entry, dict):
            return entry.get("model")
        return entry

    def get_env_vars_for_auth(self, auth_type):
        cfg = self.auth.get(auth_type, {})
        return cfg.get("env_vars", [])

    def detect_auth_type(self):
        """Detect which auth type is currently configured based on env vars."""
        import config as cfg
        for auth_type, auth_cfg in self.auth.items():
            env_var_list = auth_cfg.get("env_vars", [])
            if env_var_list:
                if all(cfg.get_env(v) for v in env_var_list):
                    return auth_type
        for auth_type, auth_cfg in self.auth.items():
            if not auth_cfg.get("env_vars"):
                return auth_type
        return self.auth_types[0] if self.auth_types else None

    def resolve_thinking_contract(self, alias, litellm_model, litellm_params=None):
        """Return a verified thinking contract dict for a configured model, or None."""
        return None

    def _get_model_thinking_levels(self, alias):
        """Get thinking levels for a model from the unified models dict."""
        entry = self.models.get(alias)
        if isinstance(entry, dict):
            return entry.get("thinking_levels")
        return None

    def _openai_reasoning_contract(self, route_family, *, levels=None, default_level=None):
        resolved_levels = tuple(levels or self.thinking_levels)
        if not resolved_levels:
            raise ValueError("thinking levels must not be empty")
        resolved_default = default_level or ("medium" if "medium" in resolved_levels else resolved_levels[0])
        return {
            "provider": self.name,
            "strategy": "openai_chat_reasoning_effort",
            "route_family": route_family,
            "levels": resolved_levels,
            "default_level": resolved_default,
            "level_labels": {
                level: THINKING_LEVEL_LABELS.get(level, level.replace("_", " ").title())
                for level in resolved_levels
            },
            "requires_openai_translation": True,
        }

    def _classify_response(self, resp):
        if resp.status_code in (401, 403):
            log.warning("%s validation: HTTP %d (invalid credentials)", self.name, resp.status_code)
            return Status.INVALID, f"Invalid or forbidden credentials (HTTP {resp.status_code})"
        if resp.status_code == 429:
            log.warning("%s validation: rate limited (HTTP 429)", self.name)
            return Status.UNREACHABLE, "Rate limited — credential not verified"
        if resp.status_code >= 500:
            log.warning("%s validation: server error (HTTP %d)", self.name, resp.status_code)
            return Status.UNREACHABLE, f"Provider server error (HTTP {resp.status_code}) — key not validated"
        if resp.status_code != 200:
            log.warning("%s validation: unexpected status %d", self.name, resp.status_code)
            return Status.UNREACHABLE, f"Unexpected status {resp.status_code}"

        ct = resp.headers.get("Content-Type", "")
        if "application/json" not in ct:
            log.warning("%s validation: unexpected Content-Type %s", self.name, ct)
            return Status.UNREACHABLE, f"Unexpected Content-Type: {ct}"

        try:
            data = resp.json()
        except ValueError:
            log.warning("%s validation: non-JSON response body", self.name)
            return Status.UNREACHABLE, "Non-JSON response"

        if isinstance(data, dict) and "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            log.warning("%s validation: error envelope in 200 response: %s", self.name, msg)
            return Status.INVALID, f"Provider error: {msg}"

        return Status.OK, ""

    def _require_env_credential(self, env_var):
        import config as cfg
        value = cfg.get_env(env_var)
        if not value or is_placeholder(value):
            return None, (Status.NOT_CONFIGURED, f"{env_var} not set")
        return value, None

    def _validate_openai_compatible_api_key(
        self, *, env_var, api_base, model, provider_label,
        success_message, invalid_message=None, timeout=10,
    ):
        api_key, failure = self._require_env_credential(env_var)
        if failure is not None:
            return failure
        log.debug("Validating %s credentials with model %s", provider_label, model)
        try:
            resp = requests.post(
                f"{api_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                timeout=timeout,
            )
        except requests.RequestException as e:
            return Status.UNREACHABLE, f"Cannot reach {provider_label} API: {e}"
        status, msg = self._classify_response(resp)
        if status == Status.OK:
            return status, success_message
        if status == Status.INVALID and resp.status_code in (401, 403) and invalid_message:
            return status, invalid_message
        return status, msg
