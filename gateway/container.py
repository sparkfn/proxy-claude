"""Container status utilities.

When running inside the gateway container, status checks use direct HTTP
to the litellm service. Docker lifecycle is managed by proclaude.sh on the host.
"""
import logging
import os

import requests

from providers.base import Status

log = logging.getLogger("litellm-cli.container")

PROXY_PORT = 2555
LITELLM_HOST = os.environ.get("PROXY_LITELLM_HOST", "litellm")
LITELLM_PORT = int(os.environ.get("PROXY_LITELLM_PORT", "4000"))


def status():
    """Check if LiteLLM backend is reachable. Returns (Status, message)."""
    try:
        resp = requests.get(
            f"http://{LITELLM_HOST}:{LITELLM_PORT}/health/readiness",
            timeout=5,
        )
        if resp.status_code == 200:
            return Status.OK, "LiteLLM is running"
        return Status.UNREACHABLE, f"LiteLLM returned status {resp.status_code}"
    except requests.RequestException as e:
        log.warning("LiteLLM health check failed: %s", e)
        return Status.UNREACHABLE, f"Cannot reach LiteLLM at {LITELLM_HOST}:{LITELLM_PORT}"


def get_logs_tail(lines=200):
    """Get recent container logs. Returns empty string inside container.

    Logs are accessed via './proclaude.sh logs' from the host.
    """
    return ""


def get_logs_since(timestamp):
    """Get container logs since timestamp. Returns empty string inside container."""
    return ""


def up():
    """Stub — container lifecycle is managed by proclaude.sh on the host."""
    log.warning("container.up() called inside container — lifecycle managed by proclaude.sh")
    return Status.FAILED, "Start services with './proclaude.sh start' on the host"


def wait_healthy(timeout=30):
    """Check if LiteLLM is reachable (replaces Docker health polling)."""
    import time as _time
    for _ in range(timeout):
        s, _ = status()
        if s == Status.OK:
            return True
        _time.sleep(1)
    return False


def logs():
    """Direct user to host-side log access. Returns (Status, message)."""
    return Status.FAILED, "View logs from host: ./proclaude.sh logs"
