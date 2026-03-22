import logging
import os
import signal
import subprocess
import sys
import time

from config import load_env_file

DIR = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger("litellm-cli.container")
CONTAINER_NAME = "litellm-proxy"


class DockerNotFoundError(RuntimeError):
    """Raised when docker/docker-compose is not installed."""
    pass


def _docker_bin():
    """Find the real docker binary, bypassing shell aliases/proxies (e.g. rtk)."""
    for path in ["/usr/local/bin/docker", "/usr/bin/docker",
                 os.path.expanduser("~/.docker/bin/docker")]:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return "docker"

# Cache compose command after first detection
_cached_compose_cmd = None


def _compose_cmd():
    """Return the docker compose command as a list. Cached after first call."""
    global _cached_compose_cmd
    if _cached_compose_cmd is not None:
        return _cached_compose_cmd
    docker = _docker_bin()
    try:
        result = subprocess.run(
            [docker, "compose", "version"], capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            _cached_compose_cmd = [docker, "compose"]
            return _cached_compose_cmd
    except FileNotFoundError:
        log.debug("'%s compose' binary not found", docker)
    except subprocess.TimeoutExpired:
        log.warning("'%s compose version' timed out", docker)
    try:
        result = subprocess.run(
            ["docker-compose", "version"], capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            _cached_compose_cmd = ["docker-compose"]
            return _cached_compose_cmd
    except FileNotFoundError:
        log.debug("'docker-compose' binary not found")
    except subprocess.TimeoutExpired:
        log.warning("'docker-compose version' timed out")
    raise DockerNotFoundError(
        "Neither 'docker compose' nor 'docker-compose' found. "
        "Install Docker Desktop or docker-compose."
    )


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
            cmd, cwd=DIR, capture_output=capture, text=True, timeout=120
        )
        if capture:
            return result.returncode == 0, result.stdout
        return result.returncode == 0, ""
    except subprocess.TimeoutExpired:
        log.warning("Docker compose command timed out: %s", " ".join(args))
        return False, ""
    except FileNotFoundError:
        log.error("docker compose is required. Install Docker Desktop or docker-compose.")
        raise DockerNotFoundError(
            "docker compose is required. Install Docker Desktop or docker-compose."
        )


def _docker_running():
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            [_docker_bin(), "info"], capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_docker():
    """Exit with message if Docker isn't available."""
    if not _docker_running():
        print("Error: Docker is not running. Start Docker and try again.")
        sys.exit(1)


PROXY_PID_FILE = os.path.join(DIR, ".proxy.pid")
PROXY_SCRIPT = os.path.join(DIR, "proxy.py")
PROXY_PORT = 2555


def _is_proxy_process(pid):
    """Check if the given PID is actually our proxy.py process."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0 and PROXY_SCRIPT in result.stdout
    except (OSError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _start_proxy():
    """Start the system message rewriter proxy in the background.

    Returns True if the proxy started successfully, False otherwise.
    """
    _stop_proxy()  # Clean up any stale process
    if not os.path.exists(PROXY_SCRIPT):
        log.debug("proxy.py not found, skipping proxy start")
        return False
    venv_python = os.path.join(DIR, ".venv", "bin", "python")
    python = venv_python if os.path.exists(venv_python) else "python3"
    proxy_log = os.path.join(DIR, ".proxy.log")
    try:
        log_fh = open(proxy_log, "a")
    except OSError as e:
        log.warning("Cannot open proxy log %s: %s", proxy_log, e)
        return False
    # Build environment: inherit current env, overlay .env values
    env = os.environ.copy()
    env.update(load_env_file(os.path.join(DIR, ".env")))
    try:
        proc = subprocess.Popen(
            [python, PROXY_SCRIPT, str(PROXY_PORT)],
            cwd=DIR, stdout=log_fh, stderr=log_fh, env=env,
        )
    except Exception:
        log_fh.close()
        raise
    log_fh.close()  # Child inherits FD via fork; parent doesn't need it
    with open(PROXY_PID_FILE, "w") as f:
        f.write(str(proc.pid))
    log.debug("Started proxy (pid=%d) on port %d", proc.pid, PROXY_PORT)

    # Brief wait to detect immediate startup failures
    time.sleep(0.5)
    exit_code = proc.poll()
    if exit_code is not None:
        log.warning("Proxy exited immediately with code %d", exit_code)
        _unlink_pid_file()
        return False
    return True


def _unlink_pid_file():
    """Remove the proxy PID file."""
    try:
        os.unlink(PROXY_PID_FILE)
    except OSError:
        log.debug("Could not remove PID file %s", PROXY_PID_FILE)


def _stop_proxy():
    """Stop the rewriter proxy if running."""
    if not os.path.exists(PROXY_PID_FILE):
        return

    try:
        with open(PROXY_PID_FILE) as f:
            raw = f.read().strip()
    except OSError as e:
        log.warning("Cannot read PID file %s: %s", PROXY_PID_FILE, e)
        return

    try:
        pid = int(raw)
    except ValueError:
        log.warning("Invalid PID in %s: %r", PROXY_PID_FILE, raw)
        _unlink_pid_file()
        return

    if not _is_proxy_process(pid):
        log.debug("PID %d is not a proxy process, cleaning up stale PID file", pid)
        _unlink_pid_file()
        return

    # Send SIGTERM
    try:
        os.kill(pid, signal.SIGTERM)
        log.debug("Sent SIGTERM to proxy (pid=%d)", pid)
    except ProcessLookupError:
        log.debug("Proxy (pid=%d) already gone", pid)
        _unlink_pid_file()
        return
    except OSError as e:
        log.warning("Failed to send SIGTERM to proxy (pid=%d): %s", pid, e)
        return  # Don't unlink — process may still be running

    # Wait for process to die (up to 3 seconds)
    stopped = False
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, OSError):
            stopped = True
            break
        time.sleep(0.1)

    if not stopped:
        try:
            os.kill(pid, signal.SIGKILL)
            log.debug("Sent SIGKILL to proxy (pid=%d)", pid)
            stopped = True
        except ProcessLookupError:
            stopped = True
        except OSError as e:
            log.warning("Failed to SIGKILL proxy (pid=%d): %s", pid, e)

    if stopped:
        _unlink_pid_file()
    else:
        log.warning("Could not confirm proxy (pid=%d) stopped; PID file retained", pid)


def _proxy_running():
    """Check if the proxy process is alive."""
    if not os.path.exists(PROXY_PID_FILE):
        return False
    try:
        with open(PROXY_PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # Check if alive
        # Verify it's actually a proxy.py process (guards against PID recycling)
        return _is_proxy_process(pid)
    except (ProcessLookupError, ValueError, OSError):
        return False


def up():
    _check_docker()
    ok, _ = _run(["up", "-d"])
    if ok:
        proxy_ok = _start_proxy()
        if proxy_ok:
            print(f"Service started on http://localhost:{PROXY_PORT}")
        else:
            print(f"Warning: Reverse proxy failed to start. "
                  f"Container is running but proxy on port {PROXY_PORT} is not available.")
    return ok


def down():
    _check_docker()
    _stop_proxy()
    ok, _ = _run(["down"])
    return ok


def restart():
    """Recreate container to pick up .env and config changes."""
    _check_docker()
    log.debug("Recreating container with --force-recreate to pick up env/config changes")
    ok, _ = _run(["up", "-d", "--force-recreate"])
    if ok:
        proxy_ok = _start_proxy()
        if not proxy_ok:
            print(f"Warning: Reverse proxy failed to start on port {PROXY_PORT}.")
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
    docker = _docker_bin()
    log.debug("Reading logs since %s via %s", timestamp, docker)
    try:
        result = subprocess.run(
            [docker, "logs", CONTAINER_NAME, "--since", timestamp],
            capture_output=True, text=True, cwd=DIR, timeout=30,
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        log.warning("docker logs --since timed out")
        return ""


def get_logs_tail(lines=200):
    """Get last N lines of container logs. Returns log text."""
    docker = _docker_bin()
    log.debug("Reading last %d log lines via %s", lines, docker)
    try:
        result = subprocess.run(
            [docker, "logs", CONTAINER_NAME, "--tail", str(lines)],
            capture_output=True, text=True, cwd=DIR, timeout=30,
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        log.warning("docker logs --tail timed out")
        return ""


def wait_healthy(timeout=30):
    """Poll until container is up or timeout. Returns True if healthy."""
    for i in range(timeout):
        # Check Docker availability on first iteration to fail fast
        if i == 0 and not _docker_running():
            log.debug("Docker not running, aborting wait_healthy")
            return False
        try:
            ok, output = _run(["ps"], capture=True)
            is_running = "Up" in output or "running" in output.lower()
            if is_running:
                return True
        except DockerNotFoundError:
            return False
        time.sleep(1)
    return False
