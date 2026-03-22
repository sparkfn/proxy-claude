import logging
import os
import shutil
import stat
import tempfile
import yaml

from providers.base import Status

DIR = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger("litellm-cli.config")
CONFIG_PATH = os.path.join(DIR, "litellm_config.yaml")
CONFIG_BACKUP = CONFIG_PATH + ".bak"
ENV_PATH = os.path.join(DIR, ".env")
ENV_BACKUP = ENV_PATH + ".bak"
ENV_EXAMPLE = os.path.join(DIR, ".env.example")

# --- YAML helpers ---

class MalformedConfig(dict):
    """Sentinel subclass of dict indicating config was loaded from a malformed YAML file."""


def _load_yaml():
    """Load litellm_config.yaml, return full dict."""
    if not os.path.exists(CONFIG_PATH):
        return {"model_list": [], "general_settings": {}}
    try:
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        log.warning("litellm_config.yaml is malformed: %s", e)
        if os.path.exists(CONFIG_BACKUP):
            log.warning("A backup exists at litellm_config.yaml.bak")
        return MalformedConfig({"model_list": [], "general_settings": {}})
    if not isinstance(data, dict):
        log.warning("litellm_config.yaml has invalid structure (expected mapping, got %s)", type(data).__name__)
        if os.path.exists(CONFIG_BACKUP):
            log.warning("A backup exists at litellm_config.yaml.bak")
        return MalformedConfig({"model_list": [], "general_settings": {}})
    if "model_list" not in data:
        data["model_list"] = []
    return data


def _atomic_write(path, content_fn):
    """Write to a temp file then rename atomically."""
    fd, tmp_path = tempfile.mkstemp(dir=DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            content_fn(f)
        os.replace(tmp_path, path)
    except Exception:  # Cleanup-then-reraise: remove temp file before propagating
        os.unlink(tmp_path)
        raise


def _save_yaml(data):
    """Backup then write litellm_config.yaml atomically."""
    if isinstance(data, MalformedConfig):
        raise ValueError(
            "Refusing to save: config was loaded from a malformed file. "
            "Fix litellm_config.yaml manually or restore from litellm_config.yaml.bak"
        )
    if os.path.exists(CONFIG_PATH):
        shutil.copy2(CONFIG_PATH, CONFIG_BACKUP)
        log.debug("Backed up config to %s", CONFIG_BACKUP)
    _atomic_write(CONFIG_PATH, lambda f: yaml.dump(
        data, f, default_flow_style=False, sort_keys=False
    ))
    log.debug("Wrote config to %s", CONFIG_PATH)


def list_models():
    """Return list of dicts: {alias, model, provider_name}."""
    data = _load_yaml()
    results = []
    for entry in data.get("model_list", []):
        alias = entry.get("model_name", "")
        params = entry.get("litellm_params", {})
        model = params.get("model", "")
        provider = _provider_from_model(model, params)
        results.append({"alias": alias, "model": model, "provider": provider})
    return results


def _provider_from_model(model_str, litellm_params=None):
    """Map litellm model prefix to provider name. Returns the raw prefix for unknown providers."""
    prefix = model_str.split("/")[0] if "/" in model_str else ""

    # For openai/ prefix, check api_base to distinguish actual OpenAI from
    # providers using OpenAI-compatible endpoints (e.g. MiniMax)
    if prefix == "openai" and litellm_params:
        api_base = litellm_params.get("api_base", "")
        if "minimax" in api_base:
            return "minimax"

    mapping = {
        "chatgpt": "openai",
        "openai": "openai",
        "minimax": "minimax",
        "ollama": "ollama",
    }
    return mapping.get(prefix, prefix or "unknown")


def add_model(alias, litellm_model, extra_params=None):
    """Add a model to config. Returns (Status, message)."""
    if not alias or not alias.strip():
        return Status.INVALID, "Model alias cannot be empty."
    if not litellm_model or not litellm_model.strip():
        return Status.INVALID, "Model string cannot be empty."
    data = _load_yaml()
    for entry in data["model_list"]:
        if entry.get("model_name") == alias:
            return Status.INVALID, f"Model alias '{alias}' already exists."
    new_entry = {
        "model_name": alias,
        "litellm_params": {"model": litellm_model},
    }
    if extra_params:
        new_entry["litellm_params"].update(extra_params)
    data["model_list"].append(new_entry)
    try:
        _save_yaml(data)
    except OSError as e:
        return Status.FAILED, f"Failed to save config: {e}"
    return Status.OK, f"Added '{alias}' -> {litellm_model}"


def remove_model(alias):
    """Remove a model by alias. Returns (Status, message)."""
    data = _load_yaml()
    original_len = len(data["model_list"])
    data["model_list"] = [
        e for e in data["model_list"] if e.get("model_name") != alias
    ]
    if len(data["model_list"]) == original_len:
        return Status.NOT_FOUND, f"Model '{alias}' not found."
    try:
        _save_yaml(data)
    except OSError as e:
        return Status.FAILED, f"Failed to save config: {e}"
    return Status.OK, f"Removed '{alias}'"


def provider_has_models(provider_name):
    """Check if any remaining models use this provider."""
    for m in list_models():
        if m["provider"] == provider_name:
            return True
    return False


# --- .env helpers ---

def _ensure_env():
    """Ensure .env exists with restrictive permissions."""
    if not os.path.exists(ENV_PATH):
        if os.path.exists(ENV_EXAMPLE):
            shutil.copy2(ENV_EXAMPLE, ENV_PATH)
        else:
            with open(ENV_PATH, "w") as f:
                pass
    # Ensure .env is only readable by owner (contains API keys)
    os.chmod(ENV_PATH, stat.S_IRUSR | stat.S_IWUSR)


def _read_env_lines():
    """Read .env as raw lines."""
    _ensure_env()
    with open(ENV_PATH, "r") as f:
        return f.readlines()


def _write_env_lines(lines):
    """Backup then write .env atomically with restrictive permissions."""
    if os.path.exists(ENV_PATH):
        shutil.copy2(ENV_PATH, ENV_BACKUP)
    _atomic_write(ENV_PATH, lambda f: f.writelines(lines))
    os.chmod(ENV_PATH, stat.S_IRUSR | stat.S_IWUSR)


def _strip_quotes(value):
    """Strip surrounding single or double quotes from a value."""
    if len(value) >= 2:
        if (value[0] == '"' and value[-1] == '"') or \
           (value[0] == "'" and value[-1] == "'"):
            return value[1:-1]
    return value


def load_env_file(path):
    """Parse a .env file into a dict of key->value pairs.

    - Skips blank lines and comment lines (starting with #)
    - Splits on first '=' only (values may contain '=')
    - Strips matching surrounding quote pairs (single or double)
    - Returns empty dict if the file doesn't exist
    """
    result = {}
    if not os.path.exists(path):
        return result
    try:
        with open(path, "r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    continue
                key, _, value = stripped.partition("=")
                key = key.strip()
                value = _strip_quotes(value.strip())
                if key:
                    result[key] = value
    except OSError as e:
        log.warning("Cannot read env file %s: %s", path, e)
    return result


def get_env(key):
    """Get value of an env var from .env. Returns None if not set or commented."""
    for line in _read_env_lines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        k, _, v = stripped.partition("=")
        if k.strip() == key:
            return _strip_quotes(v.strip())
    return None


def set_env(key, value):
    """Set an env var in .env. Updates existing or appends."""
    log.debug("Setting env var: %s", key)
    lines = _read_env_lines()
    found = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#") and "=" in stripped:
            k, _, _ = stripped.partition("=")
            if k.strip() == key:
                new_lines.append(f"{key}={value}\n")
                found = True
                continue
        new_lines.append(line)
    if not found:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append(f"{key}={value}\n")
    _write_env_lines(new_lines)


def remove_env(key):
    """Comment out an env var in .env."""
    log.debug("Removing env var: %s", key)
    lines = _read_env_lines()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#") and "=" in stripped:
            k, _, _ = stripped.partition("=")
            if k.strip() == key:
                new_lines.append(f"# REMOVED: {stripped}\n")
                continue
        new_lines.append(line)
    _write_env_lines(new_lines)
