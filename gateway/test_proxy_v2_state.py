import unittest

try:
    from gateway.proxy_v2.events import (
        Abort,
        MessageStart,
        MessageStop,
        OpenAIChunk,
        TextDelta,
        ToolCallArgsDelta,
        ToolCallComplete,
        ToolCallStart,
    )
    from gateway.proxy_v2.state import TranslationState
except ImportError:
    from proxy_v2.events import (
        Abort,
        MessageStart,
        MessageStop,
        OpenAIChunk,
        TextDelta,
        ToolCallArgsDelta,
        ToolCallComplete,
        ToolCallStart,
    )
    from proxy_v2.state import TranslationState


class ProxyV2TranslationStateTests(unittest.TestCase):
    def test_text_chunk_emits_message_start_text_and_stop(self):
        state = TranslationState()

        events = state.apply_chunk(OpenAIChunk(
            chunk_id="chatcmpl_1",
            model="demo-model",
            usage={"prompt_tokens": 3, "completion_tokens": 0},
            delta={"content": "Hello"},
            finish_reason=None,
            error=None,
        ))
        self.assertIsInstance(events[0], MessageStart)
        self.assertIsInstance(events[1], TextDelta)
        self.assertEqual("Hello", events[1].text)

        events = state.apply_chunk(OpenAIChunk(
            chunk_id="chatcmpl_1",
            model="demo-model",
            usage={"prompt_tokens": 3, "completion_tokens": 1},
            delta={},
            finish_reason="stop",
            error=None,
        ))
        self.assertIsInstance(events[-1], MessageStop)
        self.assertEqual("end_turn", events[-1].stop_reason)

    def test_tool_call_completes_only_after_valid_json(self):
        state = TranslationState()

        first_events = state.apply_chunk(OpenAIChunk(
            chunk_id="chatcmpl_tool",
            model="demo-model",
            usage={},
            delta={
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "lookup_weather", "arguments": "{\"city\""},
                }]
            },
            finish_reason=None,
            error=None,
        ))
        self.assertTrue(any(isinstance(evt, ToolCallStart) for evt in first_events))
        self.assertTrue(any(isinstance(evt, ToolCallArgsDelta) for evt in first_events))
        self.assertFalse(any(isinstance(evt, ToolCallComplete) for evt in first_events))

        second_events = state.apply_chunk(OpenAIChunk(
            chunk_id="chatcmpl_tool",
            model="demo-model",
            usage={},
            delta={
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": ": \"Singapore\"}"},
                }]
            },
            finish_reason="tool_calls",
            error=None,
        ))
        self.assertTrue(any(isinstance(evt, ToolCallComplete) for evt in second_events))
        self.assertTrue(any(isinstance(evt, MessageStop) for evt in second_events))

    def test_translation_state_renders_final_anthropic_message(self):
        state = TranslationState()
        state.apply_chunk(OpenAIChunk(
            chunk_id="chatcmpl_final",
            model="demo-model",
            usage={"prompt_tokens": 1, "completion_tokens": 0},
            delta={"content": "Done"},
            finish_reason=None,
            error=None,
        ))
        state.apply_chunk(OpenAIChunk(
            chunk_id="chatcmpl_final",
            model="demo-model",
            usage={"prompt_tokens": 1, "completion_tokens": 1},
            delta={},
            finish_reason="stop",
            error=None,
        ))

        self.assertEqual(
            {
                "id": "chatcmpl_final",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Done"}],
                "model": "demo-model",
                "stop_reason": "end_turn",
                "stop_sequence": None,
            },
            state.to_anthropic_message(),
        )

    def test_finish_eof_without_finish_reason_aborts(self):
        state = TranslationState()
        state.apply_chunk(OpenAIChunk(
            chunk_id="chatcmpl_eof",
            model="demo-model",
            usage={},
            delta={"content": "Partial"},
            finish_reason=None,
            error=None,
        ))

        events = state.finish_eof()
        self.assertEqual([Abort(reason="upstream_eof_no_finish", message=None)], events)

    def test_abort_shutdown_is_terminal(self):
        state = TranslationState()
        events = state.abort("server_shutdown")
        self.assertEqual([Abort(reason="server_shutdown", message=None)], events)
        self.assertEqual([], state.abort("server_shutdown"))

    def test_upstream_error_chunk_aborts_without_message_stop(self):
        state = TranslationState()
        events = state.apply_chunk(OpenAIChunk(
            chunk_id="chatcmpl_err",
            model="demo-model",
            usage={},
            delta={},
            finish_reason=None,
            error={"message": "upstream exploded"},
        ))
        self.assertEqual([Abort(reason="upstream_error", message="upstream exploded")], events)


if __name__ == "__main__":
    unittest.main()
