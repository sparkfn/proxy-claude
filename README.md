# LiteLLM Claude Gateway

A CLI-driven gateway that lets you run [Claude Code](https://claude.ai/code) through any LLM provider via [LiteLLM](https://github.com/BerryAI/litellm) proxy.

```
Claude Code → localhost:2555 → LiteLLM Proxy (Docker) → LLM Provider
```

## Supported Providers

| Provider | Auth | Models |
|----------|------|--------|
| **OpenAI** | Browser OAuth or API key | GPT-5.3, GPT-5.4 |
| **Alibaba (DashScope)** | API key | Qwen-Max, Qwen-Plus, Qwen-Turbo |
| **Ollama** | Local (no auth) or `ollama login` for cloud models | Any pulled model + cloud catalog |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/jaaacki/litellm-claude.git
cd litellm-claude

# 2. Add a provider and model
./litellm.sh add

# 3. Start the proxy
./litellm.sh up

# 4. Create a shell alias for Claude Code
./setup-alias.sh
source ~/.zshrc   # or ~/.bashrc

# 5. Use it
claude-gpt-5.4    # or whatever alias you chose
```

## CLI Reference

### Lifecycle

```bash
./litellm.sh up              # Start proxy container (port 2555)
./litellm.sh down            # Stop and remove container
./litellm.sh restart         # Restart container
./litellm.sh status          # Container + per-model auth status
./litellm.sh logs            # Stream container logs
```

### Models

```bash
./litellm.sh add             # Interactive wizard — pick provider, then models
./litellm.sh remove          # Remove configured models
./litellm.sh models          # List configured models
```

### Auth

```bash
./litellm.sh login           # Show auth status for all providers
./litellm.sh login openai    # Authenticate with OpenAI
./litellm.sh login alibaba   # Authenticate with DashScope
./litellm.sh login ollama    # Check Ollama + cloud login + list models + pull
```

### Claude Code

```bash
./litellm.sh claude          # Launch Claude Code through the proxy
./setup-alias.sh             # Create a persistent shell alias
```

## Provider Setup

### OpenAI

```bash
./litellm.sh login openai
# Choose: [1] Browser Login  [2] API Key
```

Browser login opens the OpenAI OAuth flow. API key is stored in `.env`.

### Alibaba (DashScope)

```bash
./litellm.sh login alibaba
# Enter your DashScope API key
```

### Ollama

Ollama runs locally — no API keys needed for local models.

```bash
# Make sure Ollama is running
ollama serve

# Login for cloud models (optional)
./litellm.sh login ollama
# → Offers: ollama login, list models, pull models

# Or add models directly
./litellm.sh add
# → Pick Ollama → choose from discovered models or enter name manually
```

Cloud models (e.g. `glm-5:cloud`, `kimi-k2.5:cloud`) appear automatically after `ollama login`.

## How It Works

The proxy runs as a Docker container with [LiteLLM](https://github.com/BerryAI/litellm), which translates between API formats. Claude Code connects to `localhost:2555`, which routes to the configured provider.

```
litellm.sh          Thin bash wrapper — manages .venv, forwards to cli.py
cli.py              Main CLI — argument routing, interactive wizards
config.py           Reads/writes litellm_config.yaml and .env
container.py        Docker container lifecycle
proxy.py            Local reverse proxy (port 2555 → container :4000)
providers/          Provider registry (OpenAI, Alibaba, Ollama)
setup-alias.sh      Creates shell functions for Claude Code aliases
```

### Files managed by the CLI

| File | Purpose |
|------|---------|
| `litellm_config.yaml` | Model registry — maps aliases to provider model strings |
| `.env` | API keys and master key (git-ignored) |
| `docker-compose.yml` | Container definition |
| `data/` | LiteLLM persistent state (mounted into container) |

## Adding a New Provider

1. Create `providers/yourprovider.py` inheriting from `BaseProvider`
2. Implement `validate()` and `login()`
3. Define `models` dict (static catalog) or `discover_models()` (dynamic)
4. Register in `providers/__init__.py`

## Requirements

- Docker
- Python 3
- Claude Code CLI (`npm install -g @anthropic-ai/claude-code`)
- Ollama (optional, for local/cloud models)
