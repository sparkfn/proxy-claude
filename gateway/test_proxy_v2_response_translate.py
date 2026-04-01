import json
import unittest

try:
    from gateway.proxy_v2.response_translate import serialize_anthropic_message, translate_openai_response
except ImportError:
    from proxy_v2.response_translate import serialize_anthropic_message, translate_openai_response


class ProxyV2ResponseTranslateTests(unittest.TestCase):
    def test_translate_openai_response_translates_tool_calls(self):
        translated = translate_openai_response({
            "id": "resp_1",
            "model": "demo-model",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "lookup_weather",
                            "arguments": "{\"city\": \"Singapore\"}",
                        },
                    }],
                },
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        })
        self.assertEqual("tool_use", translated["stop_reason"])
        self.assertEqual("tool_use", translated["content"][0]["type"])

    def test_translate_openai_response_fails_closed_on_malformed_tool_args(self):
        translated = translate_openai_response({
            "id": "resp_2",
            "model": "demo-model",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": "Calling tool",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "lookup_weather", "arguments": "{oops"},
                    }],
                },
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        })
        self.assertEqual("end_turn", translated["stop_reason"])
        self.assertEqual("text", translated["content"][-1]["type"])

    def test_translate_openai_response_fails_closed_if_any_tool_call_is_malformed(self):
        translated = translate_openai_response({
            "id": "resp_2b",
            "model": "demo-model",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "lookup_weather",
                                "arguments": "{\"city\": \"Singapore\"}",
                            },
                        },
                        {
                            "id": "call_2",
                            "function": {"name": "lookup_time", "arguments": "{oops"},
                        },
                    ],
                },
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        })
        self.assertEqual("end_turn", translated["stop_reason"])
        self.assertEqual(
            [{"type": "text", "text": "[Tool call failed: malformed arguments for lookup_time]"}],
            translated["content"],
        )

    def test_serialize_anthropic_message_returns_json_bytes(self):
        message = {
            "id": "resp_3",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "demo-model",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        encoded = serialize_anthropic_message(message)
        self.assertEqual(message, json.loads(encoded))


if __name__ == "__main__":
    unittest.main()
