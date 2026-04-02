#!/bin/bash
set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_FILE="$DIR/docker-compose.yml"
GATEWAY_CONTAINER="litellm-gateway"

# --- Helpers (UI only — no business logic) ---

_docker_compose() {
    docker compose -f "$COMPOSE_FILE" "$@"
}

_gateway_exec() {
    local tty_flags=""
    if [ -t 0 ]; then
        tty_flags="-i"
    fi
    # shellcheck disable=SC2086
    docker exec $tty_flags "$GATEWAY_CONTAINER" "$@"
}

_gateway_running() {
    docker inspect -f '{{.State.Running}}' "$GATEWAY_CONTAINER" 2>/dev/null | grep -q true
}

_ensure_running() {
    if ! _gateway_running; then
        echo "  ✗ Gateway container is not running. Run './proclaude.sh start' first."
        exit 1
    fi
}

_ensure_docker() {
    if ! command -v docker &>/dev/null; then
        echo "  ✗ Docker is not installed. Install Docker and try again."
        exit 1
    fi
    if ! docker info &>/dev/null; then
        echo "  ✗ Docker daemon is not running. Start Docker and try again."
        exit 1
    fi
}

_wait_for_gateway() {
    local wait=0
    while ! _gateway_running && [ $wait -lt 15 ]; do
        sleep 1
        wait=$((wait + 1))
    done
    if ! _gateway_running; then
        echo "  ✗ Gateway failed to start. Check './proclaude.sh logs'"
        exit 1
    fi
}

# --- Help ---

_show_help() {
    cat <<'HELP'
LiteLLM Gateway CLI
Usage: ./proclaude.sh <command> [options]

Infrastructure:
  start             Start the gateway and LiteLLM
  stop              Stop all services
  restart           Restart all services
  status            Container and model status
  logs              Stream container logs
  build             Rebuild the gateway image

Models:
  model add         Add models (interactive)
  model rm          Remove a configured model
  model list        List configured models

Providers:
  provider list     Show available providers
  provider status   Show auth status per provider
  provider login    Authenticate with a provider
  provider logout   Remove provider credentials

Launch:
  launch claude     Launch Claude Code through the proxy

Options:
  --verbose, -v     Enable debug logging
HELP
}

# --- Main ---

VERBOSE=""
args=()
for arg in "$@"; do
    if [[ "$arg" == "--verbose" || "$arg" == "-v" ]]; then
        VERBOSE="--verbose"
    else
        args+=("$arg")
    fi
done
set -- "${args[@]+"${args[@]}"}"

if [ $# -eq 0 ] || [[ "$1" == "help" || "$1" == "-h" || "$1" == "--help" ]]; then
    _show_help
    exit 0
fi

_ensure_docker

# Ensure mount-point directories exist (gitignored, must be created at runtime)
mkdir -p "$DIR/auth/chatgpt" "$DIR/data" "$DIR/data/gateway"

CMD="$1"
shift

_ensure_env() {
    # Ensure .env exists and has a master key (pure bash, no Python)
    if [ ! -f "$DIR/.env" ]; then
        if [ -f "$DIR/.env.example" ]; then
            cp "$DIR/.env.example" "$DIR/.env"
        else
            touch "$DIR/.env"
        fi
        chmod 600 "$DIR/.env"
    fi
    if ! grep -q '^LITELLM_MASTER_KEY=.\+' "$DIR/.env" 2>/dev/null; then
        local key
        key=$(openssl rand -hex 16 2>/dev/null || head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
        if grep -q '^LITELLM_MASTER_KEY=' "$DIR/.env" 2>/dev/null; then
            sed -i.bak "s/^LITELLM_MASTER_KEY=.*/LITELLM_MASTER_KEY=$key/" "$DIR/.env"
            rm -f "$DIR/.env.bak"
        else
            echo "LITELLM_MASTER_KEY=$key" >> "$DIR/.env"
        fi
    fi
}

case "$CMD" in
    start)
        _ensure_env
        echo "  Starting services..."
        _docker_compose up -d --build
        _wait_for_gateway
        echo "  ✓ Gateway running on http://localhost:2555"
        _gateway_exec python cli.py start-status $VERBOSE || true
        ;;
    stop)
        _docker_compose down
        echo "  ✓ Services stopped"
        ;;
    restart)
        echo "  Restarting services..."
        _docker_compose up -d --force-recreate --build
        echo "  ✓ Services restarted"
        ;;
    logs)
        _docker_compose logs -f "${@}"
        ;;
    build)
        _docker_compose build
        echo "  ✓ Gateway image rebuilt"
        ;;
    status)
        _ensure_running
        _gateway_exec python cli.py status $VERBOSE
        ;;
    model)
        _ensure_running
        _gateway_exec python cli.py model "$@" $VERBOSE
        ;;
    provider)
        if [ "${1:-}" = "login" ] && [ "${2:-}" = "openai" ]; then
            if ! _gateway_running; then
                _ensure_env
                echo "  Starting services..."
                _docker_compose up -d --build
                _wait_for_gateway
            fi
            shift 2
            # Browser OAuth needs host-side docker log access
            _gateway_exec python cli.py provider openai-browser-trigger $VERBOSE || true
            echo ""
            echo "  Waiting for login instructions from LiteLLM..."
            login_url=""
            for _ in $(seq 1 30); do
                login_url=$(_docker_compose logs --tail 80 litellm 2>&1 | grep -o 'https://auth\.openai\.com/[^ "]*' | tail -1)
                [ -n "$login_url" ] && break
                printf "." >&2
                sleep 2
            done
            if [ -z "$login_url" ]; then
                echo ""
                echo "  ✗ Could not find OpenAI login URL in LiteLLM logs."
                echo "    Check './proclaude.sh logs litellm' for details."
                exit 1
            fi
            device_code=$(_docker_compose logs --tail 80 litellm 2>&1 | grep -o 'Enter code: [A-Z0-9-]*' | tail -1 | sed 's/Enter code: //')
            echo ""
            echo "  ┌─────────────────────────────────────────────────────┐"
            echo "  │  OpenAI Login Required                              │"
            echo "  │                                                     │"
            printf "  │  1) Visit:  %-42s│\n" "$login_url"
            printf "  │  2) Enter code:  %-36s│\n" "$device_code"
            echo "  │                                                     │"
            echo "  └─────────────────────────────────────────────────────┘"
            echo ""
            echo "  Waiting for login confirmation... (timeout: 5 min)"
            for _ in $(seq 1 100); do
                if _docker_compose logs --tail 120 litellm 2>&1 | grep -qi "successfully authenticated\|chatgpt.*auth\|access.token"; then
                    echo "  ✓ Browser OAuth may be active."
                    exit 0
                fi
                sleep 3
            done
            echo "  ✗ Login timed out."
            exit 1
        else
            _ensure_running
            _gateway_exec python cli.py provider "$@" $VERBOSE
        fi
        ;;
    launch)
        if [ "${1:-}" != "claude" ]; then
            echo "  Unknown launch target: ${1:-}"
            echo "  Available: claude"
            exit 1
        fi
        shift

        if ! command -v claude &>/dev/null; then
            echo "  ✗ Claude Code CLI not found. Install it first:"
            echo "    npm install -g @anthropic-ai/claude-code"
            exit 1
        fi

        if ! _gateway_running; then
            echo "  Starting services..."
            _docker_compose up -d --build
            _wait_for_gateway
        fi

        # Container handles: model picker, readiness checks, thinking, telegram,
        # config management. Writes env + command to temp file.
        emit_path="/tmp/claude_env"
        _gateway_exec python cli.py launch claude --emit-env "$emit_path" "$@" $VERBOSE
        # Reset terminal state after interactive docker exec TTY session
        printf '\033[0m' 2>/dev/null || true
        env_output=$(docker exec "$GATEWAY_CONTAINER" cat "$emit_path" 2>/dev/null) || {
            echo "  ✗ Failed to read launch configuration."
            exit 1
        }
        docker exec "$GATEWAY_CONTAINER" rm -f "$emit_path" 2>/dev/null || true

        # Source the env vars and command emitted by the container.
        # Output is trusted (our own container) — safe to eval.
        eval "$env_output"

        # If credentials were entered during launch, restart so LiteLLM picks them up
        if [ "${NEEDS_RESTART:-}" = "1" ]; then
            echo "  Restarting services to apply new credentials..."
            _docker_compose up -d --force-recreate
            _wait_for_gateway
        fi

        # For OpenAI browser OAuth: check if auth is pending
        if [ "${CLAUDE_SELECTED_PROVIDER:-}" = "openai" ]; then
            if ! curl -s -o /dev/null -w '%{http_code}' http://localhost:2555/health/readiness 2>/dev/null | grep -q 200; then
                auth_url=$(_docker_compose logs --tail 30 litellm 2>&1 | grep -o 'https://auth\.openai\.com/[^ "]*' | tail -1)
                if [ -n "$auth_url" ]; then
                    device_code=$(_docker_compose logs --tail 30 litellm 2>&1 | grep -o 'Enter code: [A-Z0-9-]*' | tail -1 | sed 's/Enter code: //')
                    echo ""
                    echo "  ┌─────────────────────────────────────────────────────┐"
                    echo "  │  OpenAI Login Required                              │"
                    echo "  │                                                     │"
                    printf "  │  1) Visit:  %-42s│\n" "$auth_url"
                    printf "  │  2) Enter code:  %-36s│\n" "$device_code"
                    echo "  │                                                     │"
                    echo "  └─────────────────────────────────────────────────────┘"
                    echo ""
                    printf "  Authenticate now before launching? [Y/n]: "
                    read -r auth_choice
                    if [ "${auth_choice:-Y}" != "n" ] && [ "${auth_choice:-Y}" != "N" ]; then
                        echo "  Waiting for login..."
                        for _ in $(seq 1 100); do
                            if curl -s -o /dev/null -w '%{http_code}' http://localhost:2555/health/readiness 2>/dev/null | grep -q 200; then
                                echo "  ✓ LiteLLM is ready"
                                break
                            fi
                            sleep 3
                        done
                    fi
                fi
            fi
        fi

        # Brief wait for LiteLLM backend (avoids 502 on first request)
        ready_wait=0
        while [ $ready_wait -lt 30 ]; do
            if curl -s -o /dev/null -w '%{http_code}' http://localhost:2555/health/readiness 2>/dev/null | grep -q 200; then
                break
            fi
            if [ $ready_wait -eq 0 ]; then
                printf "  Waiting for backend" >&2
            fi
            printf "." >&2
            sleep 1
            ready_wait=$((ready_wait + 1))
        done
        if [ $ready_wait -gt 0 ]; then
            echo "" >&2
        fi

        echo "  Launching Claude Code (${ANTHROPIC_MODEL:-unknown})..."
        # shellcheck disable=SC2086
        exec $LAUNCH_CMD
        ;;
    *)
        echo "  Unknown command: $CMD"
        _show_help
        exit 1
        ;;
esac
