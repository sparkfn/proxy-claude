# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A configuration-only deployment repo for a [LiteLLM](https://github.com/BerryAI/litellm) proxy server, providing a unified API gateway to multiple LLM providers. The CLI (`litellm.sh`) is a Python-based control plane behind a thin bash wrapper.

## Architecture

```
Client → localhost:2555 (proxy.py) → localhost:4000 (Docker: litellm-proxy) → LLM providers
```

There are two layers of proxying:
1. **proxy.py** — host-side reverse proxy on port 2555 that strips `system` messages from Anthropic `/v1/messages` requests (needed because `chatgpt/` provider rejects system messages and LiteLLM doesn't strip them in the Anthropic-to-Responses translation)
2. **Docker container** — LiteLLM on port 4000, handles actual API translation and routing

### Layer Boundaries

The codebase enforces strict layer boundaries for error handling, logging, and I/O:

- **proxy.py** (HTTP boundary) — uses `logging` module exclusively. Translates internal errors to HTTP JSON envelopes (`_error_response()`). Typed exception handlers map to HTTP status codes (504 timeout, 502 refused/error). One thread safety-net catch with `exc_info=True` traceback. No `print()`.
- **config.py, container.py, providers/** (library layer) — return `(Status, str)` tuples using the unified `Status` enum (`providers/base.py`). No `sys.exit()`. No `print()` except two streaming methods that can't batch-return progress: `openai._login_browser()` and `ollama.pull_model()`. Exceptions either propagate (e.g. `DockerNotFoundError`) or are caught and returned as status.
- **cli.py** (CLI boundary) — owns all user-facing output (`print()`), interactive prompting (`input()`), and process exit (`sys.exit()`). Translates `Status` returns to icons and messages. Catches `DockerNotFoundError` at the top level.
- **litellm.sh** (shell boundary) — delegates .env parsing to `config.load_env_file()` via Python. Translates exit codes to user messages.

### Error Contract

All internal Python functions return `(Status, str)`:

```python
class Status(Enum):
    OK            # Success
    FAILED        # Operation failed
    NOT_FOUND     # Resource not found
    INVALID       # Bad input or state
    NOT_CONFIGURED # Missing setup/config
    UNREACHABLE   # Cannot reach external service
    UNVERIFIED    # Cannot confirm, may be ok
```

- `config.add_model()` → `(Status.OK, msg)` or `(Status.INVALID, msg)`
- `container.up()` → `(Status.OK, msg)` or `(Status.FAILED, msg)` — FAILED if proxy fails
- `provider.validate()` → `(Status, msg)` with specific status per failure mode
- `provider.discover_models()` → `None` on error, `{}` on empty, `dict` on success

### .env Parsing

One canonical parser: `config.load_env_file()`. All consumers route through it:
- `config.py` — uses it directly for `get_env()`, `set_env()`, `remove_env()`
- `container.py` — imports `load_env_file` from config for proxy env overlay
- `litellm.sh` — calls `load_env_file()` via Python to export vars to host env

Format contract: skip blank/comment lines, split on first `=`, strip matching quote pairs.

### Logging Policy

- **Diagnostics** — `logging` module. Logger names: `litellm-proxy` (proxy.py), `litellm-cli.<module>` (everything else). Use `%s` formatting, never f-strings. Levels: DEBUG for tracing, INFO for lifecycle, WARNING for degraded, ERROR for failures.
- **User output** — `print()` in `cli.py` only, except `openai._login_browser()` and `ollama.pull_model()` which stream progress that can't be batch-returned. Providers otherwise return structured data; CLI formats it.
- **Exception**: `_print_counters()` atexit handler in proxy.py uses `print` (logging may be torn down).

Key files:
- **litellm.sh** — thin bash wrapper: manages `.venv/`, loads `.env` via Python, forwards to `cli.py`
- **cli.py** — main CLI entry point (argument routing, interactive wizards, all user I/O)
- **config.py** — reads/writes `litellm_config.yaml` and `.env` (atomic writes with backups, canonical `.env` parser)
- **container.py** — Docker container lifecycle + proxy.py process management
- **proxy.py** — system message rewriter proxy (threaded HTTP server, supports SSE streaming pass-through, `logging`-based observability)
- **providers/** — provider registry with `BaseProvider` ABC and `Status` enum
- **docker-compose.yml** — single-service definition, maps port 4000 (not 2555; proxy.py handles 2555)
- **litellm_config.yaml** — model registry (managed by CLI, uses litellm model string format like `chatgpt/gpt-5.4`, `minimax/MiniMax-M2.7`, `ollama/llama3`)
- **.env** — API keys and master key (managed by CLI, chmod 600)
- **auth/** — mounted into container at `/root/.config/litellm` for browser OAuth persistence
- **data/** — LiteLLM persistent state (mounted into container at `/root/.litellm`)

## Commands

All operations go through the CLI wrapper:

```bash
./litellm.sh start           # Start proxy (port 2555)
./litellm.sh stop            # Stop and remove container
./litellm.sh restart         # Restart container (force-recreate to pick up .env/config changes)
./litellm.sh status          # Container status + per-model auth status
./litellm.sh logs            # Stream container logs (follow mode)

./litellm.sh model add       # Add models (interactive, pick provider first)
./litellm.sh model rm        # Remove configured models
./litellm.sh model list      # List configured models

./litellm.sh provider list   # Show available providers
./litellm.sh provider status # Show auth status per provider
./litellm.sh provider login  # Authenticate with a provider
./litellm.sh provider logout # Remove provider credentials

./litellm.sh launch claude   # Launch Claude Code through the proxy
```

## Provider System

Providers inherit from `BaseProvider` (in `providers/base.py`) and must implement:
- `validate() -> (Status, msg)` — check auth state without side effects
- `login(auth_type, credentials=None) -> (Status, msg)` — authenticate with provided credentials
- `login_prompts` dict — declares what credentials the CLI should collect per auth_type

The CLI layer reads `login_prompts` to prompt the user, then passes a `credentials` dict to `login()`. Providers never call `input()`. Two methods still `print()` streaming progress that can't be batch-returned: `openai._login_browser()` (OAuth polling with countdown) and `ollama.pull_model()` (download progress).

Registered in `providers/__init__.py` via `_register()` — order matters for display.

**Provider-specific details:**
- **OpenAI** — two auth paths: `browser_oauth` (chatgpt/ prefix, uses container logs to find OAuth URL) and `api_key` (openai/ prefix, stored in `.env`). `get_models_for_auth()` returns different model catalogs per auth type. Browser OAuth flow (`_login_browser()`) is an exception to the no-print rule — it streams interactive progress (URL box, polling countdown) that can't be batch-returned.
- **MiniMax** — single `api_key` auth via `MINIMAX_API_KEY` env var. Static model catalog (MiniMax-M2.7, M2.5, Text-01).
- **Ollama** — no auth_types (validates connectivity only). `login()` returns pure status. Interactive flows (cloud login, model pull) are separate methods (`ollama_cloud_login()`, `pull_model()`) driven by `cli.py:_ollama_interactive_login()`. Dynamic model discovery via `/api/tags` — returns `None` on error, `{}` on empty. Models use `api_base: http://host.docker.internal:11434` to reach host Ollama from container. Respects `OLLAMA_HOST` env var.

### Adding a New Provider

1. Create `providers/yourprovider.py` inheriting from `BaseProvider`
2. Implement `validate()` and `login()`, define `auth_types`, `env_vars`, `models`, `login_prompts`
3. `login()` must accept `credentials` dict and never call `input()`. Avoid `print()` — return `(Status, str)` and let the CLI format output. Exception: streaming progress (OAuth polling, download bars) that can't be batch-returned
4. For dynamic catalogs, implement `discover_models()` — return `None` on error, `{}` on empty (see Ollama)
5. Register in `providers/__init__.py`

## Key Details

- Container image: `ghcr.io/berriai/litellm:main-v1.82.4-nightly`
- Master key for proxy auth: set via `LITELLM_MASTER_KEY` in `.env` (default: `sk-1234`)
- Python deps: `pyyaml`, `requests` (managed in `.venv/`, auto-created by litellm.sh)
- Config writes are atomic (temp file + rename) with `.bak` backups
- `container.restart()` uses `--force-recreate` to pick up `.env`/config changes; returns `(Status.FAILED, msg)` if proxy fails
- `container.up()` also starts proxy.py; `container.down()` stops it (PID file at `.proxy.pid`, log at `.proxy.log`). Returns `Status.FAILED` if either container or proxy fails.
- `_check_docker()` raises `DockerNotFoundError` with specific messages: binary not found, probe timeout, daemon not running
- `_stop_proxy()` only unlinks PID file when process confirmed dead
- Ollama models need `extra_hosts: host.docker.internal:host-gateway` in docker-compose to reach host
- `chatgpt/` models require special litellm params: `drop_params: true`, `modify_params: true`, `supports_system_messages: false` — proxy.py strips system messages because LiteLLM doesn't handle this for the Anthropic-to-Responses translation path
- Global `litellm_settings` in config set `drop_params: true` and `modify_params: true` as defaults
- No automated tests. Runtime smoke tests can be run with `python3 -c "import container; print(container.status())"`
