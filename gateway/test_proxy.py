import sys
import unittest
import types

sys.argv = [sys.argv[0]]

# Stub yaml before any module imports it (proxy imports config which imports yaml)
yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = lambda *a, **k: {}
sys.modules["yaml"] = yaml_stub

import proxy


class ProxyErrorMappingTests(unittest.TestCase):
    """Tests for canonical error-mapping infrastructure."""

    def test_map_upstream_status_auth_errors(self):
        for status in (401, 403):
            code, msg, err_type = proxy._map_upstream_status(status)
            self.assertEqual(code, 502)
            self.assertEqual(err_type, "auth_error")

    def test_map_upstream_status_rate_limit(self):
        code, msg, err_type = proxy._map_upstream_status(429)
        self.assertEqual(code, 429)
        self.assertEqual(err_type, "upstream_error")
        self.assertIn("rate limited", msg.lower())

    def test_map_upstream_status_server_error(self):
        for status in (500, 502, 503, 504):
            code, msg, err_type = proxy._map_upstream_status(status)
            self.assertEqual(code, 502)
            self.assertEqual(err_type, "upstream_error")

    def test_map_upstream_status_other_4xx(self):
        code, msg, err_type = proxy._map_upstream_status(400)
        self.assertEqual(code, 502)
        self.assertEqual(err_type, "upstream_error")

    def test_error_response_format(self):
        code, body = proxy._error_response(502, "test error", "upstream_error")
        self.assertEqual(code, 502)
        import json
        parsed = json.loads(body)
        self.assertEqual(parsed["error"]["message"], "test error")
        self.assertEqual(parsed["error"]["type"], "upstream_error")


class ProxyValidationTests(unittest.TestCase):
    """Tests for request validation."""

    def test_validate_messages_rejects_missing_model(self):
        err = proxy._validate_messages({"messages": [{"role": "user", "content": "hi"}]})
        self.assertIsNotNone(err)
        self.assertIn("model", err.lower())

    def test_validate_messages_rejects_non_list_messages(self):
        err = proxy._validate_messages({"model": "gpt-5.4", "messages": "not a list"})
        self.assertIsNotNone(err)

    def test_validate_messages_rejects_empty_messages(self):
        err = proxy._validate_messages({"model": "gpt-5.4", "messages": []})
        self.assertIsNotNone(err)

    def test_validate_messages_accepts_valid_request(self):
        err = proxy._validate_messages({
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertIsNone(err)


class ProxyThinkingContractTests(unittest.TestCase):
    def test_build_route_state_handles_empty_entries(self):
        """_build_route_state must not fail on empty input."""
        import providers as p_module
        orig_all = p_module.all_providers
        p_module.all_providers = lambda: []
        try:
            route_state = proxy._build_route_state([])
            self.assertEqual(route_state["translated"], set())
            self.assertEqual(route_state["all_models"], [])
            self.assertEqual(route_state["native"], {})
            self.assertEqual(route_state["thinking_contracts"], {})
        finally:
            p_module.all_providers = orig_all

    def test_apply_verified_thinking_contract_injects_reasoning_effort(self):
        openai_body = {"model": "gpt-5.4", "messages": []}
        thinking_contract = {
            "strategy": "openai_chat_reasoning_effort",
            "route_family": "chatgpt",
            "provider": "openai",
            "levels": ("low", "medium", "high"),
        }

        proxy._apply_verified_thinking_contract(openai_body, thinking_contract, "high")

        self.assertEqual("high", openai_body["reasoning_effort"])

    def test_require_verified_thinking_contract_rejects_unverified_model(self):
        with self.assertRaisesRegex(ValueError, "Thinking effort is not supported"):
            proxy._require_verified_thinking_contract(
                "llama3",
                "high",
                thinking_contracts={},
            )

    def test_upstream_error_stop_reason_is_distinct(self):
        self.assertEqual(proxy._UPSTREAM_ERROR_STOP, "upstream_error")
        # Must be distinct from normal finish reasons
        self.assertNotEqual(proxy._UPSTREAM_ERROR_STOP, proxy._map_finish_reason("stop"))


if __name__ == "__main__":
    unittest.main()
