#!/bin/bash
set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_FILE="$DIR/docker-compose.yml"
GATEWAY_CONTAINER="litellm-gateway"

# --- .env loader (pure bash, replaces Python config.load_env_file) ---
# Contract: skip blank/comment lines, split on first =, strip matching quote pairs.

_load_env() {
    local env_file="$DIR/.env"
    [ -f "$env_file" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        # Trim leading/trailing whitespace
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        # Skip blank lines and comments
        [[ -z "$line" || "$line" == \#* ]] && continue
        # Must contain =
        [[ "$line" != *=* ]] && continue
        local key="${line%%=*}"
        local value="${line#*=}"
        # Trim leading/trailing whitespace from value
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        # Strip matching surrounding quotes (double or single)
        if [[ ${#value} -ge 2 ]]; then
            if [[ "$value" == \"*\" ]]; then
                value="${value:1:${#value}-2}"
            elif [[ "$value" == \'*\' ]]; then
                value="${value:1:${#value}-2}"
            fi
        fi
        export "$key=$value" 2>/dev/null || echo "  Warning: could not export '$key'" >&2
    done < "$env_file"
}

# --- Helpers ---

_docker_compose() {
    docker compose -f "$COMPOSE_FILE" "$@"
}

_gateway_exec() {
    # Execute a CLI command inside the gateway container.
    # Attach TTY only when both stdin and stdout are terminals (interactive use).
    local tty_flags=""
    if [ -t 0 ] && [ -t 1 ]; then
        tty_flags="-it"
    fi
    # shellcheck disable=SC2086
    docker exec $tty_flags "$GATEWAY_CONTAINER" "$@"
}

_gateway_running() {
    docker inspect -f '{{.State.Running}}' "$GATEWAY_CONTAINER" 2>/dev/null | grep -q true
}

_ensure_running() {
    if ! _gateway_running; then
        echo "  ✗ Gateway container is not running. Run './litellm.sh start' first."
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

_litellm_healthy() {
    # Check if LiteLLM is serving via gateway's internal network
    docker exec "$GATEWAY_CONTAINER" python -c "
import sys, requests
try:
    r = requests.get('http://litellm:4000/health/readiness', timeout=2)
    sys.exit(0 if r.status_code == 200 else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

_show_auth_prompt() {
    # Check litellm logs for pending device code auth. Shows prompt if found.
    # Returns 0 if auth prompt found, 1 if not.
    local logs
    logs=$(_docker_compose logs litellm 2>&1 | tail -30)

    local url code
    url=$(echo "$logs" | grep -oE 'https://auth\.openai\.com/[^ "]+' | tail -1)
    code=$(echo "$logs" | grep -oE 'Enter code: [A-Z0-9]+-[A-Z0-9]+' | tail -1 | sed 's/Enter code: //')

    if [ -n "$url" ] && [ -n "$code" ]; then
        echo ""
        echo "  ┌─────────────────────────────────────────────────────┐"
        echo "  │  OpenAI Login Required                              │"
        echo "  │                                                     │"
        printf "  │  1) Visit:  %-42s│\n" "$url"
        printf "  │  2) Enter code:  %-36s│\n" "$code"
        echo "  │                                                     │"
        echo "  └─────────────────────────────────────────────────────┘"
        echo ""
        return 0
    fi
    return 1
}

_wait_litellm_ready() {
    # Wait for LiteLLM backend to be healthy. Shows auth prompt if needed.
    # Returns 0 on success, 1 on timeout.
    local timeout=${1:-300}
    local auth_shown=false
    local start=$SECONDS

    # Quick check — already healthy?
    if _litellm_healthy; then
        return 0
    fi

    echo "  Waiting for LiteLLM backend..."

    while [ $((SECONDS - start)) -lt "$timeout" ]; do
        if _litellm_healthy; then
            printf "\r%60s\r" ""
            echo "  ✓ LiteLLM is ready"
            return 0
        fi

        # Show auth prompt once if detected
        if [ "$auth_shown" = false ] && _show_auth_prompt; then
            auth_shown=true
        fi

        local elapsed=$((SECONDS - start))
        local remaining=$((timeout - elapsed))
        local mins=$((remaining / 60))
        local secs=$((remaining % 60))
        if [ "$auth_shown" = true ]; then
            printf "\r  Waiting for login... %d:%02d remaining  " "$mins" "$secs"
        else
            printf "\r  Waiting for LiteLLM... %ds  " "$elapsed"
        fi
        sleep 3
    done

    echo ""
    if [ "$auth_shown" = true ]; then
        echo "  ✗ Login timed out. Complete the auth and run again."
    else
        echo "  ✗ LiteLLM did not become ready. Check './litellm.sh logs litellm'"
    fi
    return 1
}

# --- Launch claude (runs on host, no Python needed) ---
# Interactive model/config selection runs inside the gateway container via
# docker exec -it. cli.py --emit-env writes env vars to a temp file inside
# the container. We read them back and exec claude on the host.

_launch_claude() {
    # Check claude binary on host
    if ! command -v claude &>/dev/null; then
        echo "  ✗ Claude Code CLI not found. Install it first:"
        echo "    npm install -g @anthropic-ai/claude-code"
        exit 1
    fi

    # Ensure services are running (auto-start if needed)
    if ! _gateway_running; then
        echo "  Starting services..."
        _docker_compose up -d --build
        # Brief wait for container readiness
        local wait=0
        while ! _gateway_running && [ $wait -lt 15 ]; do
            sleep 1
            wait=$((wait + 1))
        done
        if ! _gateway_running; then
            echo "  ✗ Gateway failed to start. Check './litellm.sh logs'"
            exit 1
        fi
    fi

    # Ensure LiteLLM backend is ready (shows auth prompt if needed)
    if ! _wait_litellm_ready 300; then
        exit 1
    fi

    # Run the interactive launch flow inside the container.
    # cli.py launch claude --emit-env /tmp/claude_env does:
    #   1. Interactive model picker, thinking effort, telegram prompts (via TTY)
    #   2. Writes shell-sourceable env vars to /tmp/claude_env inside container
    local emit_path="/tmp/claude_env"
    _gateway_exec python cli.py launch claude --emit-env "$emit_path" "$@"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        exit $exit_code
    fi

    # Read the emitted env vars from the container
    local env_output
    env_output=$(docker exec "$GATEWAY_CONTAINER" cat "$emit_path" 2>/dev/null) || {
        echo "  ✗ Failed to read launch configuration from container."
        exit 1
    }
    docker exec "$GATEWAY_CONTAINER" rm -f "$emit_path" 2>/dev/null || true

    # Read and validate env vars (only allow export KEY='value' lines)
    while IFS= read -r line; do
        if [[ "$line" =~ ^export\ [A-Za-z_][A-Za-z_0-9]*= ]]; then
            eval "$line"
        else
            echo "  Warning: skipping unexpected line in env output: $line" >&2
        fi
    done <<< "$env_output"

    # Build the claude command
    local cmd=(claude --dangerously-skip-permissions)
    if [ -n "${CLAUDE_CHANNELS:-}" ]; then
        cmd+=(--channels "$CLAUDE_CHANNELS")
    fi
    if [ -n "${CLAUDE_EXTRA_ARGS:-}" ]; then
        # shellcheck disable=SC2206
        cmd+=($CLAUDE_EXTRA_ARGS)
    fi

    echo "  Launching Claude Code (${ANTHROPIC_MODEL:-unknown})..."
    exec "${cmd[@]}"
}

# --- Help ---

_show_help() {
    cat <<'HELP'
LiteLLM Gateway CLI
Usage: ./litellm.sh <command> [options]

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

# Parse --verbose / -v from anywhere in args
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

# Load .env for host-side env vars (master key, etc.)
_load_env

# No args or help
if [ $# -eq 0 ] || [[ "$1" == "help" || "$1" == "-h" || "$1" == "--help" ]]; then
    _show_help
    exit 0
fi

_ensure_docker

CMD="$1"
shift

case "$CMD" in
    start)
        echo "  Starting services..."
        _docker_compose up -d --build
        # Wait for gateway container
        local gw_wait=0
        while ! _gateway_running && [ $gw_wait -lt 15 ]; do
            sleep 1
            gw_wait=$((gw_wait + 1))
        done
        if ! _gateway_running; then
            echo "  ✗ Gateway failed to start. Check './litellm.sh logs'"
            exit 1
        fi
        echo "  ✓ Gateway running on http://localhost:2555"
        # Wait for LiteLLM (shows auth prompt if needed)
        _wait_litellm_ready 300 || true
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
        _ensure_running
        _gateway_exec python cli.py provider "$@" $VERBOSE
        ;;
    launch)
        if [ "${1:-}" = "claude" ]; then
            shift
            _launch_claude "$@" $VERBOSE
        else
            echo "  Unknown launch target: ${1:-}"
            echo "  Available: claude"
            exit 1
        fi
        ;;
    *)
        echo "  Unknown command: $CMD"
        _show_help
        exit 1
        ;;
esac
