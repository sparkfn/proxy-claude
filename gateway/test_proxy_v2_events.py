import json
import unittest

try:
    from gateway.proxy_v2 import events, sse, state
except ImportError:
    from proxy_v2 import events, sse, state


class ProxyV2SSEDecoderTests(unittest.TestCase):
    def test_decoder_reassembles_incremental_sse_frames(self):
        decoder = sse.SSEDecoder()

        self.assertEqual([], decoder.feed(b"event: message\ndata: {\"id\":"))

        parsed = decoder.feed(b"\"chunk_1\"}\n\ndata: [DONE]\n\n")

        self.assertEqual(
            [
                sse.SSEEvent(event="message", data='{"id":"chunk_1"}'),
                sse.SSEEvent(event=None, data="[DONE]"),
            ],
            parsed,
        )


class ProxyV2SemanticStateTests(unittest.TestCase):
    def test_text_deltas_accumulate_and_require_explicit_message_stop(self):
        stream_state = state.OpenAIStreamState(message_id="msg_1", model="demo-model")

        first_events = stream_state.consume_chunk({
            "choices": [{"delta": {"content": "Hel"}}],
        })
        second_events = stream_state.consume_chunk({
            "choices": [{"delta": {"content": "lo"}}],
        })

        self.assertEqual([events.TextDelta("Hel")], first_events)
        self.assertEqual([events.TextDelta("lo")], second_events)
        self.assertEqual("Hello", stream_state.visible_text)

        with self.assertRaisesRegex(events.IncompleteMessageError, "explicit message stop"):
            stream_state.to_anthropic_message()

        stop_events = stream_state.consume_chunk({
            "choices": [{"delta": {}, "finish_reason": "stop"}],
        })

        self.assertEqual([events.MessageStop("end_turn")], stop_events)
        self.assertEqual(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
                "model": "demo-model",
                "stop_reason": "end_turn",
                "stop_sequence": None,
            },
            stream_state.to_anthropic_message(),
        )

    def test_tool_call_start_and_argument_deltas_accumulate(self):
        stream_state = state.OpenAIStreamState(message_id="msg_2", model="demo-model")

        first_events = stream_state.consume_chunk({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "function": {
                            "name": "lookup_weather",
                            "arguments": "{\"city\":\"Sing",
                        },
                    }],
                },
            }],
        })
        second_events = stream_state.consume_chunk({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "apore\"}"},
                    }],
                },
            }],
        })
        stop_events = stream_state.consume_chunk({
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
        })

        self.assertEqual(
            [
                events.ToolUseStart(index=0, tool_id="call_1", name="lookup_weather"),
                events.ToolUseArgsDelta(index=0, partial_json='{"city":"Sing'),
            ],
            first_events,
        )
        self.assertEqual(
            [events.ToolUseArgsDelta(index=0, partial_json='apore"}')],
            second_events,
        )
        self.assertEqual([events.MessageStop("tool_use")], stop_events)
        self.assertEqual(
            {
                "id": "msg_2",
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "lookup_weather",
                    "input": {"city": "Singapore"},
                }],
                "model": "demo-model",
                "stop_reason": "tool_use",
                "stop_sequence": None,
            },
            stream_state.to_anthropic_message(),
        )

    def test_done_without_finish_reason_is_rejected(self):
        stream_state = state.OpenAIStreamState(message_id="msg_3", model="demo-model")
        stream_state.consume_chunk({"choices": [{"delta": {"content": "partial"}}]})

        with self.assertRaisesRegex(events.IncompleteMessageError, "explicit message stop"):
            stream_state.note_transport_done()

    def test_transport_abort_cannot_be_finalized_as_success(self):
        stream_state = state.OpenAIStreamState(message_id="msg_4", model="demo-model")
        stream_state.consume_chunk({"choices": [{"delta": {"content": "partial"}}]})

        with self.assertRaisesRegex(events.TransportAbortError, "server_shutdown"):
            stream_state.abort("server_shutdown")

        with self.assertRaisesRegex(events.IncompleteMessageError, "explicit message stop"):
            stream_state.to_anthropic_message()


if __name__ == "__main__":
    unittest.main()
