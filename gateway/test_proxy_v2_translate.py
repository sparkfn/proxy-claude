import json
import unittest

try:
    from gateway.proxy_v2 import translate
except ImportError:
    from proxy_v2 import translate


class ProxyV2TranslateRequestTests(unittest.TestCase):
    def test_anthropic_to_openai_translates_system_and_messages(self):
        payload = {
            "model": "demo-model",
            "system": [
                {"type": "text", "text": "System rules"},
                {"type": "text", "text": "More rules", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
            "metadata": {"user_id": "user-123"},
            "max_tokens": 256,
            "temperature": 0.2,
            "top_p": 0.8,
            "stop_sequences": ["DONE"],
            "stream": True,
        }

        translated = json.loads(translate.anthropic_to_openai_request(payload))

        self.assertEqual("demo-model", translated["model"])
        self.assertEqual(
            [
                {"role": "system", "content": "System rules\nMore rules"},
                {"role": "user", "content": "Hello"},
            ],
            translated["messages"],
        )
        self.assertEqual("user-123", translated["user"])
        self.assertEqual(256, translated["max_tokens"])
        self.assertEqual(256, translated["max_completion_tokens"])
        self.assertEqual(0.2, translated["temperature"])
        self.assertEqual(0.8, translated["top_p"])
        self.assertEqual(["DONE"], translated["stop"])
        self.assertTrue(translated["stream"])
        self.assertEqual({"include_usage": True}, translated["stream_options"])

    def test_anthropic_to_openai_preserves_tool_use_tool_result_ordering(self):
        payload = {
            "model": "demo-model",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I will call a tool."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup_weather",
                            "input": {"city": "Singapore"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "sunny"}],
                        },
                        {"type": "text", "text": "What next?"},
                    ],
                },
            ],
        }

        translated = json.loads(translate.anthropic_to_openai_request(payload))

        self.assertEqual(
            {
                "role": "assistant",
                "content": "I will call a tool.",
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "arguments": "{\"city\": \"Singapore\"}",
                        },
                    }
                ],
            },
            translated["messages"][0],
        )
        self.assertEqual(
            {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"},
            translated["messages"][1],
        )
        self.assertEqual(
            {"role": "user", "content": "What next?"},
            translated["messages"][2],
        )

    def test_anthropic_to_openai_translates_image_blocks(self):
        payload = {
            "model": "demo-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "ZmFrZS1pbWFnZQ==",
                            },
                        },
                    ],
                }
            ],
        }

        translated = json.loads(translate.anthropic_to_openai_request(payload))

        self.assertEqual(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,ZmFrZS1pbWFnZQ=="},
                    },
                ],
            },
            translated["messages"][0],
        )

    def test_validate_anthropic_messages_request_rejects_missing_messages(self):
        with self.assertRaisesRegex(ValueError, "messages field is required"):
            translate.validate_anthropic_messages_request({"model": "demo-model"})

    def test_anthropic_to_openai_request_preserves_legacy_value_error_shape(self):
        with self.assertRaisesRegex(ValueError, "messages field is required"):
            translate.anthropic_to_openai_request({"model": "demo-model"})


class ProxyV2TranslateResponseTests(unittest.TestCase):
    def test_openai_to_anthropic_fails_closed_on_malformed_tool_arguments(self):
        upstream = {
            "id": "resp_1",
            "model": "demo-model",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "Calling tool",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "lookup_weather", "arguments": "{oops"},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }

        with self.assertLogs("litellm-proxy", level="WARNING") as captured:
            translated = json.loads(
                translate.openai_to_anthropic_response(json.dumps(upstream).encode("utf-8"))
            )

        self.assertEqual("end_turn", translated["stop_reason"])
        self.assertEqual(
            [
                {"type": "text", "text": "Calling tool"},
                {
                    "type": "text",
                    "text": "[Tool call failed: malformed arguments for lookup_weather]",
                },
            ],
            translated["content"],
        )
        self.assertIn("Malformed tool arguments from upstream", "\n".join(captured.output))

    def test_openai_to_anthropic_uses_reasoning_content_fallback_and_strips_think_tags(self):
        upstream = {
            "id": "resp_2",
            "model": "demo-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "reasoning_content": "<think>private</think>\nVisible answer",
                    },
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

        translated = json.loads(
            translate.openai_to_anthropic_response(json.dumps(upstream).encode("utf-8"))
        )

        self.assertEqual("end_turn", translated["stop_reason"])
        self.assertEqual([{"type": "text", "text": "Visible answer"}], translated["content"])

    def test_strip_think_tags_leaves_non_think_text_unchanged(self):
        self.assertEqual("plain text", translate.strip_think_tags("plain text"))


if __name__ == "__main__":
    unittest.main()
