#!/bin/bash
set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_FILE="$DIR/docker-compose.yml"
GATEWAY_CONTAINER="litellm-gateway"
HOST_RUNTIME="$DIR/gateway/host_runtime.py"

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

CMD="$1"
shift

case "$CMD" in
    start)
        python3 "$HOST_RUNTIME" ensure-master-key
        echo "  Starting services..."
        _docker_compose up -d --build
        _wait_for_gateway
        echo "  ✓ Gateway running on http://localhost:2555"
        python3 "$HOST_RUNTIME" --compose-file "$COMPOSE_FILE" report-start-status
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
                echo "  Starting services..."
                _docker_compose up -d --build
                _wait_for_gateway
            fi
            shift 2
            python3 "$HOST_RUNTIME" --compose-file "$COMPOSE_FILE" openai-browser-login "$@"
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

        python3 "$HOST_RUNTIME" \
            --compose-file "$COMPOSE_FILE" \
            offer-pending-auth \
            --selected-model "${ANTHROPIC_MODEL:-selected model}"

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
