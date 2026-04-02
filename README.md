# proxy-claude

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
git clone https://github.com/sparkfn/proxy-claude.git
cd proxy-claude

# 2. Login to a provider
./proclaude.sh provider login openai

# 3. Add a model
./proclaude.sh model add

# 4. Start the proxy
./proclaude.sh start

# 5. Launch Claude Code
./proclaude.sh launch claude
```

## CLI Reference

### Infrastructure

```bash
./proclaude.sh start           # Start proxy container (port 2555)
./proclaude.sh stop            # Stop and remove container
./proclaude.sh restart         # Restart container
./proclaude.sh status          # Container + per-model auth status
./proclaude.sh logs            # Stream container logs
```

### Models

```bash
./proclaude.sh model add       # Add models (interactive, pick provider first)
./proclaude.sh model rm        # Remove configured models
./proclaude.sh model list      # List configured models
```

### Providers

```bash
./proclaude.sh provider list   # Show available providers
./proclaude.sh provider status # Show auth status for all providers
./proclaude.sh provider login  # Authenticate with a provider
./proclaude.sh provider logout # Remove provider credentials
```

### Launch

```bash
./proclaude.sh launch claude   # Launch Claude Code through the proxy
```

Thinking levels are now strict:

- `--thinking low|medium|high` is accepted only for models whose configured upstream route has a verified thinking contract
- unsupported or unverified models hard-fail instead of silently degrading
- future models inherit thinking support automatically when their configured provider route matches a verified contract

Flags skip interactive prompts:

```bash
./proclaude.sh launch claude --provider openai --model gpt-5.4
./proclaude.sh model list --provider ollama
```

## Provider Setup

### OpenAI

```bash
./proclaude.sh provider login openai
# Choose: [1] Browser OAuth  [2] API Key
```

Browser login opens the OpenAI OAuth flow. API key is stored in `.env`.

### MiniMax

```bash
./proclaude.sh provider login minimax
# Enter your MiniMax API key
```

### Ollama

Ollama runs locally — no API keys needed for local models.

```bash
# Make sure Ollama is running
ollama serve

# Login for cloud models (optional)
./proclaude.sh provider login ollama
# → Offers: ollama login, list models, pull models

# Or add models directly
./proclaude.sh model add --provider ollama
# → Choose from discovered models or enter name manually
```

Cloud models (e.g. `glm-5:cloud`, `kimi-k2.5:cloud`) appear automatically after `ollama login`.

## How It Works

The proxy runs as a Docker container with [LiteLLM](https://github.com/BerryAI/litellm), which translates between API formats. Claude Code connects to `localhost:2555`, which routes to the configured provider.

```
proclaude.sh          Thin bash wrapper — manages .venv, forwards to cli.py
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
