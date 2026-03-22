# LiteLLM Claude Gateway

A CLI-driven gateway that lets you run [Claude Code](https://claude.ai/code) through any LLM provider via [LiteLLM](https://github.com/BerryAI/litellm) proxy.

```
Claude Code → localhost:2555 → LiteLLM Proxy (Docker) → LLM Provider
```

## Supported Providers

| Provider | Auth | Models |
|----------|------|--------|
| **OpenAI** | Browser OAuth or API key | GPT-5.3, GPT-5.4 |
| **MiniMax** | API key | MiniMax-M2.7, MiniMax-M2.5, MiniMax-Text-01 |
| **Ollama** | Local (no auth) or `ollama login` for cloud models | Any pulled model + cloud catalog |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/jaaacki/litellm-claude.git
cd litellm-claude

# 2. Login to a provider
./litellm.sh provider login openai

# 3. Add a model
./litellm.sh model add

# 4. Start the proxy
./litellm.sh start

# 5. Launch Claude Code
./litellm.sh launch claude
```

## CLI Reference

### Infrastructure

```bash
./litellm.sh start           # Start proxy container (port 2555)
./litellm.sh stop            # Stop and remove container
./litellm.sh restart         # Restart container
./litellm.sh status          # Container + per-model auth status
./litellm.sh logs            # Stream container logs
```

### Models

```bash
./litellm.sh model add       # Add models (interactive, pick provider first)
./litellm.sh model rm        # Remove configured models
./litellm.sh model list      # List configured models
```

### Providers

```bash
./litellm.sh provider list   # Show available providers
./litellm.sh provider status # Show auth status for all providers
./litellm.sh provider login  # Authenticate with a provider
./litellm.sh provider logout # Remove provider credentials
```

### Launch

```bash
./litellm.sh launch claude   # Launch Claude Code through the proxy
```

Flags skip interactive prompts:

```bash
./litellm.sh launch claude --provider openai --model gpt-5.4
./litellm.sh model list --provider ollama
```

## Provider Setup

### OpenAI

```bash
./litellm.sh provider login openai
# Choose: [1] Browser OAuth  [2] API Key
```

Browser login opens the OpenAI OAuth flow. API key is stored in `.env`.

### MiniMax

```bash
./litellm.sh provider login minimax
# Enter your MiniMax API key
```

### Ollama

Ollama runs locally — no API keys needed for local models.

```bash
# Make sure Ollama is running
ollama serve

# Login for cloud models (optional)
./litellm.sh provider login ollama
# → Offers: ollama login, list models, pull models

# Or add models directly
./litellm.sh model add --provider ollama
# → Choose from discovered models or enter name manually
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
providers/          Provider registry (OpenAI, MiniMax, Ollama)
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
