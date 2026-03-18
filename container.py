import logging
import os
import subprocess
import sys
import time

DIR = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger("litellm-cli.container")
CONTAINER_NAME = "litellm-proxy"

# Cache compose command after first detection
_cached_compose_cmd = None


def _compose_cmd():
    """Return the docker compose command as a list. Cached after first call."""
    global _cached_compose_cmd
    if _cached_compose_cmd is not None:
        return _cached_compose_cmd
    try:
        result = subprocess.run(
            ["docker", "compose", "version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            _cached_compose_cmd = ["docker", "compose"]
            return _cached_compose_cmd
    except FileNotFoundError:
        pass
    # Verify docker-compose exists before caching
    try:
        result = subprocess.run(
            ["docker-compose", "version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            _cached_compose_cmd = ["docker-compose"]
            return _cached_compose_cmd
    except FileNotFoundError:
        pass
    # Neither found — return docker-compose and let _run handle the error
    return ["docker-compose"]


def _run(args, capture=False, stream=False):
    """Run a docker compose command from the project directory."""
    cmd = _compose_cmd() + args
    log.debug("Running: %s", " ".join(cmd))
    try:
        if stream:
            proc = subprocess.Popen(cmd, cwd=DIR)
            proc.wait()
            return proc.returncode == 0, ""
        result = subprocess.run(
            cmd, cwd=DIR, capture_output=capture, text=True
        )
        if capture:
            return result.returncode == 0, result.stdout
        return result.returncode == 0, ""
    except FileNotFoundError:
        print("Error: docker compose is required. Install Docker Desktop or docker-compose.")
        sys.exit(1)


def _docker_running():
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _check_docker():
    """Exit with message if Docker isn't available."""
    if not _docker_running():
        print("Error: Docker is not running. Start Docker and try again.")
        sys.exit(1)


def up():
    _check_docker()
    ok, _ = _run(["up", "-d"])
    if ok:
        print("Service started on http://localhost:2555")
    return ok


def down():
    _check_docker()
    ok, _ = _run(["down"])
    return ok


def restart():
    """Recreate container to pick up .env and config changes."""
    _check_docker()
    log.debug("Recreating container with --force-recreate to pick up env/config changes")
    ok, _ = _run(["up", "-d", "--force-recreate"])
    return ok


def status():
    """Return (is_running: bool, output: str)."""
    _check_docker()
    ok, output = _run(["ps"], capture=True)
    is_running = "Up" in output or "running" in output.lower()
    return is_running, output


def logs(follow=True):
    _check_docker()
    args = ["logs"]
    if follow:
        args.append("-f")
    _run(args, stream=True)


def get_logs_since(timestamp):
    """Get container logs since a timestamp (RFC3339 format). Returns log text."""
    _check_docker()
    result = subprocess.run(
        ["docker", "logs", CONTAINER_NAME, "--since", timestamp],
        capture_output=True, text=True, cwd=DIR,
    )
    return result.stdout + result.stderr


def get_logs_tail(lines=200):
    """Get last N lines of container logs. Returns log text."""
    result = subprocess.run(
        ["docker", "logs", CONTAINER_NAME, "--tail", str(lines)],
        capture_output=True, text=True, cwd=DIR,
    )
    return result.stdout + result.stderr


def wait_healthy(timeout=30):
    """Poll until container is up or timeout. Returns True if healthy."""
    for _ in range(timeout):
        try:
            ok, output = _run(["ps"], capture=True)
            is_running = "Up" in output or "running" in output.lower()
            if is_running:
                return True
        except SystemExit:
            pass  # Docker temporarily unavailable, keep polling
        time.sleep(1)
    return False
