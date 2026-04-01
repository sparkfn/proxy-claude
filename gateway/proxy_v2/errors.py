"""Canonical proxy error helpers extracted from gateway/proxy.py."""

from dataclasses import dataclass
import json
import logging

log = logging.getLogger("litellm-proxy.v2.errors")


@dataclass(eq=True)
class ProxyError(Exception):
    """Structured proxy error for V2 translation modules."""

    status_code: int
    message: str
    error_type: str = "proxy_error"
    code: str | None = None
    retryable: bool = False
    upstream_status: int | None = None
    provider: str | None = None
    details: dict | None = None

    def __post_init__(self):
        super().__init__(self.message)

    def public_error(self):
        return {"error": {"message": self.message, "type": self.error_type}}

    def to_response(self):
        return error_response(self.status_code, self.message, self.error_type)


def error_response(status_code, message, error_type="proxy_error"):
    """Build a JSON error body and return (status_code, body_bytes)."""
    return status_code, json.dumps(
        {"error": {"message": message, "type": error_type}}
    ).encode("utf-8")


def map_upstream_status(status_code):
    """Map an upstream HTTP status to the canonical proxy envelope tuple."""
    if status_code in (401, 403):
        return 502, "Provider authentication failed", "auth_error"
    if status_code == 429:
        return 429, "Provider rate limited — retry later", "upstream_error"
    if status_code >= 500:
        return 502, "Provider temporarily unavailable", "upstream_error"
    return 502, "Provider request failed", "upstream_error"
