"""Incremental SSE parsing for V2 translated streams."""

from dataclasses import dataclass

try:
    from gateway.proxy_v2.errors import ProxyError
except ImportError:
    from proxy_v2.errors import ProxyError

_ALLOWED_FIELDS = frozenset({"event", "data", "id", "retry"})


@dataclass(eq=True)
class SSEFrame:
    event: str | None
    data: str
    id: str | None
    retry: int | None


@dataclass(eq=True)
class SSEEvent:
    event: str | None
    data: str


class SSEParser:
    def __init__(self):
        self._buffer = b""
        self._event = None
        self._data_lines = []
        self._event_id = None
        self._retry = None

    def feed(self, chunk):
        if not isinstance(chunk, (bytes, bytearray)):
            raise ProxyError(502, "SSE chunk must be bytes", "upstream_error", code="invalid_sse_chunk")
        self._buffer += bytes(chunk)
        frames = []
        while b"\n" in self._buffer:
            raw_line, self._buffer = self._buffer.split(b"\n", 1)
            if raw_line.endswith(b"\r"):
                raw_line = raw_line[:-1]
            frames.extend(self._process_line(raw_line))
        return frames

    def finish(self):
        if self._buffer:
            raise ProxyError(502, "Truncated SSE frame", "upstream_error", code="truncated_sse_frame")
        if self._event is not None or self._data_lines or self._event_id is not None or self._retry is not None:
            raise ProxyError(502, "Unterminated SSE event", "upstream_error", code="unterminated_sse_event")
        return []

    def _process_line(self, raw_line):
        if not raw_line:
            frame = self._flush_event()
            return [frame] if frame is not None else []

        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProxyError(502, "Malformed SSE encoding", "upstream_error", code="invalid_sse_encoding") from exc

        if line.startswith(":"):
            return []

        field_name, field_value = self._parse_field(line)
        if field_name not in _ALLOWED_FIELDS:
            raise ProxyError(502, "Unsupported SSE field '%s'" % field_name, "upstream_error", code="invalid_sse_field")
        if field_name == "event":
            self._event = field_value
        elif field_name == "data":
            self._data_lines.append(field_value)
        elif field_name == "id":
            self._event_id = field_value
        elif field_name == "retry":
            if field_value:
                try:
                    self._retry = int(field_value)
                except ValueError as exc:
                    raise ProxyError(502, "Malformed SSE retry field", "upstream_error", code="invalid_sse_retry") from exc
        return []

    def _parse_field(self, line):
        if ":" in line:
            field_name, field_value = line.split(":", 1)
            if field_value.startswith(" "):
                field_value = field_value[1:]
        else:
            field_name, field_value = line, ""
        if not field_name or any(ch.isspace() for ch in field_name):
            raise ProxyError(502, "Malformed SSE field line", "upstream_error", code="invalid_sse_field")
        return field_name, field_value

    def _flush_event(self):
        if self._event is None and not self._data_lines and self._event_id is None and self._retry is None:
            return None
        frame = SSEFrame(
            event=self._event,
            data="\n".join(self._data_lines),
            id=self._event_id,
            retry=self._retry,
        )
        self._event = None
        self._data_lines = []
        self._event_id = None
        self._retry = None
        return frame


class SSEDecoder:
    """Compatibility wrapper matching the earlier V2 event-layer tests."""

    def __init__(self):
        self._parser = SSEParser()

    def feed(self, chunk):
        return [
            SSEEvent(event=frame.event, data=frame.data)
            for frame in self._parser.feed(chunk)
        ]
