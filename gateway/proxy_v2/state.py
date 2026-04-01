"""Semantic state machine for V2 translated streams."""

from dataclasses import dataclass
import json
import logging

try:
    from gateway.proxy_v2.contracts import map_openai_finish_reason
    from gateway.proxy_v2.events import (
        Abort,
        IncompleteMessageError,
        MessageStart,
        MessageStop,
        OpenAIChunk,
        TextDelta,
        ToolCallArgsDelta,
        ToolCallComplete,
        ToolCallStart,
        ToolUseArgsDelta,
        ToolUseStart,
        TransportAbortError,
        UsageDelta,
    )
except ImportError:
    from proxy_v2.contracts import map_openai_finish_reason
    from proxy_v2.events import (
        Abort,
        IncompleteMessageError,
        MessageStart,
        MessageStop,
        OpenAIChunk,
        TextDelta,
        ToolCallArgsDelta,
        ToolCallComplete,
        ToolCallStart,
        ToolUseArgsDelta,
        ToolUseStart,
        TransportAbortError,
        UsageDelta,
    )

log = logging.getLogger("litellm-proxy.v2.state")


@dataclass
class _ToolCallBuffer:
    index: int
    tool_call_id: str = ""
    name: str = ""
    arguments_buffer: str = ""
    started: bool = False
    completed: bool = False


class TranslationState:
    def __init__(self):
        self._started = False
        self._stopped = False
        self._aborted = False
        self._message_id = ""
        self._model = ""
        self._input_tokens = 0
        self._output_tokens = 0
        self._visible_text = []
        self._final_stop_reason = None
        self._tool_calls = {}

    def apply_chunk(self, chunk):
        if self._stopped or self._aborted:
            return []

        if chunk.error is not None:
            return self.abort("upstream_error", message=_error_message(chunk.error))

        events = []
        events.extend(self._ensure_started(chunk))
        events.extend(self._apply_usage(chunk.usage or {}))
        events.extend(self._apply_text(chunk.delta or {}))

        tool_events, tool_abort = self._apply_tool_calls(chunk.delta or {})
        events.extend(tool_events)
        if tool_abort is not None:
            return events + self.abort(tool_abort)

        if chunk.finish_reason:
            if self._has_incomplete_tool_calls():
                log.error("finish_reason=%s arrived before tool arguments completed", chunk.finish_reason)
                return events + self.abort("incomplete_tool_args")
            self._stopped = True
            self._final_stop_reason = map_openai_finish_reason(chunk.finish_reason)
            events.append(MessageStop(
                stop_reason=self._final_stop_reason,
                output_tokens=self._output_tokens,
            ))
        return events

    def finish_eof(self):
        if self._stopped or self._aborted:
            return []
        return self.abort("upstream_eof_no_finish")

    def abort(self, reason, *, message=None):
        if self._stopped or self._aborted:
            return []
        self._aborted = True
        return [Abort(reason=reason, message=message)]

    def _ensure_started(self, chunk):
        if self._started:
            return []
        self._started = True
        self._message_id = chunk.chunk_id or "msg_translated"
        self._model = chunk.model or ""
        self._input_tokens = int((chunk.usage or {}).get("prompt_tokens", 0) or 0)
        return [MessageStart(
            message_id=self._message_id,
            model=self._model,
            input_tokens=self._input_tokens,
        )]

    def _apply_usage(self, usage):
        if not usage:
            return []
        input_tokens = int(usage.get("prompt_tokens", self._input_tokens) or 0)
        output_tokens = int(usage.get("completion_tokens", self._output_tokens) or 0)
        if input_tokens == self._input_tokens and output_tokens == self._output_tokens:
            return []
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        return [UsageDelta(input_tokens=input_tokens, output_tokens=output_tokens)]

    def _apply_text(self, delta):
        text = ""
        if "content" in delta:
            text = delta.get("content") or ""
        elif delta.get("reasoning_content"):
            text = delta.get("reasoning_content") or ""
        if not text:
            return []
        self._visible_text.append(text)
        return [TextDelta(text=text)]

    def _apply_tool_calls(self, delta):
        events = []
        for tool_call in delta.get("tool_calls", []) or []:
            if not isinstance(tool_call, dict):
                return events, "malformed_tool_call_delta"
            index = int(tool_call.get("index", 0) or 0)
            function = tool_call.get("function") or {}
            if not isinstance(function, dict):
                return events, "malformed_tool_call_delta"

            buffer = self._tool_calls.get(index)
            if buffer is None:
                buffer = _ToolCallBuffer(index=index)
                self._tool_calls[index] = buffer

            if tool_call.get("id"):
                buffer.tool_call_id = tool_call["id"]
            if function.get("name"):
                buffer.name = function["name"]
            if not buffer.started and (buffer.tool_call_id or buffer.name):
                buffer.started = True
                events.append(ToolCallStart(index=index, tool_call_id=buffer.tool_call_id, name=buffer.name))
                if buffer.arguments_buffer:
                    events.append(ToolCallArgsDelta(index=index, partial_json=buffer.arguments_buffer))

            arguments_delta = function.get("arguments")
            if arguments_delta:
                buffer.arguments_buffer += arguments_delta
                if buffer.started:
                    events.append(ToolCallArgsDelta(index=index, partial_json=arguments_delta))
                parsed_input, complete = _parse_tool_arguments(buffer.arguments_buffer)
                if complete:
                    buffer.completed = True
                    events.append(ToolCallComplete(index=index, input=parsed_input))
        return events, None

    def _has_incomplete_tool_calls(self):
        return any(
            (buf.started or buf.arguments_buffer) and not buf.completed
            for buf in self._tool_calls.values()
        )

    def to_anthropic_message(self):
        if self._aborted:
            raise IncompleteMessageError("stream aborted before an explicit message stop")
        if not self._stopped:
            raise IncompleteMessageError("stream ended without an explicit message stop")

        content = []
        if self._visible_text:
            content.append({"type": "text", "text": "".join(self._visible_text)})
        for index in sorted(self._tool_calls):
            tool_state = self._tool_calls[index]
            if not tool_state.completed:
                continue
            content.append({
                "type": "tool_use",
                "id": tool_state.tool_call_id,
                "name": tool_state.name,
                "input": json.loads(tool_state.arguments_buffer),
            })
        return {
            "id": self._message_id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": self._model,
            "stop_reason": self._final_stop_reason,
            "stop_sequence": None,
        }


def _parse_tool_arguments(arguments_buffer):
    try:
        parsed = json.loads(arguments_buffer)
    except (TypeError, ValueError):
        return None, False
    if not isinstance(parsed, dict):
        return None, False
    return parsed, True


def _error_message(error):
    if isinstance(error, dict):
        return error.get("message", "Unknown upstream error")
    return str(error)


class OpenAIStreamState:
    """Compatibility semantic accumulator for the earlier V2 event tests."""

    def __init__(self, message_id, model):
        self.message_id = message_id
        self.model = model
        self.visible_text = ""
        self._translation_state = TranslationState()
        self._stop_reason = None
        self._aborted = False

    def consume_chunk(self, chunk):
        if self._aborted:
            raise TransportAbortError("stream already aborted")
        normalized_events = self._translation_state.apply_chunk(_chunk_from_payload(
            chunk,
            message_id=self.message_id,
            model=self.model,
        ))
        return self._convert_events(normalized_events)

    def note_transport_done(self):
        if not self._stop_reason:
            raise IncompleteMessageError("stream ended without an explicit message stop")

    def abort(self, reason):
        self._aborted = True
        self._translation_state.abort(reason)
        raise TransportAbortError(reason)

    def to_anthropic_message(self):
        message = self._translation_state.to_anthropic_message()
        message["id"] = self.message_id
        message["model"] = self.model
        return message

    def _convert_events(self, normalized_events):
        emitted = []
        for event in normalized_events:
            if isinstance(event, TextDelta):
                self.visible_text += event.text
                emitted.append(event)
            elif isinstance(event, ToolCallStart):
                emitted.append(ToolUseStart(index=event.index, tool_id=event.tool_call_id, name=event.name))
            elif isinstance(event, ToolCallArgsDelta):
                emitted.append(ToolUseArgsDelta(index=event.index, partial_json=event.partial_json))
            elif isinstance(event, MessageStop):
                self._stop_reason = event.stop_reason
                emitted.append(MessageStop(event.stop_reason))
            elif isinstance(event, Abort):
                self._aborted = True
                raise TransportAbortError(event.reason)
        return emitted


def _chunk_from_payload(payload, *, message_id, model):
    choice = ((payload.get("choices") or [{}])[0] or {})
    return OpenAIChunk(
        chunk_id=payload.get("id", message_id),
        model=payload.get("model", model),
        usage=payload.get("usage") or {},
        delta=choice.get("delta") or {},
        finish_reason=choice.get("finish_reason"),
        error=payload.get("error"),
    )
