import os
import tempfile
import types
import unittest
from unittest import mock

# Keep yaml import behavior deterministic for modules that may import config.
yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = lambda *a, **k: {}
import sys
sys.modules.setdefault("yaml", yaml_stub)
sys.argv = [sys.argv[0]]

try:
    from gateway.proxy_v2 import contracts, errors, routes
except ImportError:
    from proxy_v2 import contracts, errors, routes


class ProxyV2ContractsTests(unittest.TestCase):
    def test_map_openai_finish_reason_matches_current_contract(self):
        self.assertEqual("end_turn", contracts.map_openai_finish_reason("stop"))
        self.assertEqual("tool_use", contracts.map_openai_finish_reason("tool_calls"))
        self.assertEqual("max_tokens", contracts.map_openai_finish_reason("length"))

    def test_resolve_terminal_stop_reason_preserves_only_contract_stop_reasons(self):
        self.assertEqual(
            contracts.UPSTREAM_ERROR_STOP,
            contracts.resolve_terminal_stop_reason(contracts.UPSTREAM_ERROR_STOP),
        )
        self.assertEqual("end_turn", contracts.resolve_terminal_stop_reason("end_turn"))
        self.assertEqual("tool_use", contracts.resolve_terminal_stop_reason("tool_use"))
        self.assertEqual("max_tokens", contracts.resolve_terminal_stop_reason("max_tokens"))

    def test_resolve_terminal_stop_reason_rejects_transport_abort_reason(self):
        with self.assertRaisesRegex(ValueError, "non-contract terminal reason"):
            contracts.resolve_terminal_stop_reason("server_shutdown")

    def test_resolve_legacy_terminal_stop_reason_preserves_v1_shutdown_behavior(self):
        self.assertEqual(
            "end_turn",
            contracts.resolve_legacy_terminal_stop_reason("server_shutdown"),
        )
        self.assertEqual(
            contracts.UPSTREAM_ERROR_STOP,
            contracts.resolve_legacy_terminal_stop_reason(contracts.UPSTREAM_ERROR_STOP),
        )


class ProxyV2RoutesTests(unittest.TestCase):
    def test_build_route_state_handles_empty_entries(self):
        dependencies = types.SimpleNamespace(
            provider_registry=lambda: [],
            provider_from_model=lambda model, params: "",
            thinking_contract_resolver=lambda entry: None,
        )
        route_state = routes.build_route_state([], dependencies=dependencies)
        self.assertEqual(route_state["translated"], set())
        self.assertEqual(route_state["all_models"], [])
        self.assertEqual(route_state["native"], {})
        self.assertEqual(route_state["thinking_contracts"], {})

    def test_resolve_config_path_prefers_local_then_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proxy_dir = os.path.join(tmpdir, "gateway")
            os.makedirs(proxy_dir)
            local_cfg = os.path.join(proxy_dir, "litellm_config.yaml")
            parent_cfg = os.path.join(tmpdir, "litellm_config.yaml")

            with open(parent_cfg, "w", encoding="utf-8") as f:
                f.write("model_list: []\n")
            self.assertEqual(parent_cfg, routes.resolve_config_path(proxy_dir))

            with open(local_cfg, "w", encoding="utf-8") as f:
                f.write("model_list: []\n")
            self.assertEqual(local_cfg, routes.resolve_config_path(proxy_dir))

    def test_build_route_state_populates_translated_native_and_thinking_contracts(self):
        provider = types.SimpleNamespace(
            name="demo",
            anthropic_base_url="https://demo.example.com/anthropic",
            native_auth={"env": "DEMO_API_KEY", "header": "x-demo-key"},
            models={"demo-native": "demo/native-model"},
        )
        entries = [{
            "model_name": "demo-native",
            "litellm_params": {"model": "openai/demo-model"},
        }]

        dependencies = types.SimpleNamespace(
            provider_registry=lambda: [provider],
            provider_from_model=lambda model, params: "demo",
            thinking_contract_resolver=lambda entry: {
                "provider": "demo",
                "strategy": "openai_chat_reasoning_effort",
                "levels": ("low", "medium", "high"),
                "requires_openai_translation": True,
            },
        )
        route_state = routes.build_route_state(entries, dependencies=dependencies)

        self.assertEqual({"demo-native"}, route_state["translated"])
        self.assertEqual(["demo-native"], route_state["all_models"])
        self.assertEqual(
            {
                "host": "demo.example.com",
                "port": 443,
                "path": "/anthropic",
                "api_key_env": "DEMO_API_KEY",
                "auth_header": "x-demo-key",
            },
            route_state["native"]["demo-native"],
        )
        self.assertEqual(
            ("low", "medium", "high"),
            route_state["thinking_contracts"]["demo-native"]["levels"],
        )

    def test_build_route_state_supports_dependency_injection(self):
        provider = types.SimpleNamespace(
            name="demo",
            anthropic_base_url="https://demo.example.com/anthropic",
            native_auth={"env": "DEMO_API_KEY", "header": "x-demo-key"},
            models={"demo-native": "demo/native-model"},
        )
        route_state = routes.build_route_state(
            [{"model_name": "demo-native", "litellm_params": {"model": "openai/demo-model"}}],
            provider_registry=lambda: [provider],
            provider_from_model=lambda model, params: "demo",
            thinking_contract_resolver=lambda entry: None,
        )
        self.assertEqual({"demo-native"}, set(route_state.native_routes.keys()))

    def test_build_route_state_supports_dependency_bundle(self):
        provider = types.SimpleNamespace(
            name="demo",
            anthropic_base_url="https://demo.example.com/anthropic",
            native_auth={"env": "DEMO_API_KEY", "header": "x-demo-key"},
            models={"demo-native": "demo/native-model"},
        )
        dependencies = types.SimpleNamespace(
            provider_registry=lambda: [provider],
            provider_from_model=lambda model, params: "demo",
            thinking_contract_resolver=lambda entry: None,
        )
        route_state = routes.build_route_state(
            [{"model_name": "demo-native", "litellm_params": {"model": "openai/demo-model"}}],
            dependencies=dependencies,
        )
        self.assertEqual(set(), route_state.translated_models)
        self.assertEqual(["demo-native"], route_state.all_models)
        self.assertEqual("demo.example.com", route_state.native_routes["demo-native"].host)

    def test_routes_module_does_not_capture_config_or_providers_at_import_time(self):
        self.assertFalse(hasattr(routes, "_config"))
        self.assertFalse(hasattr(routes, "_providers"))

    def test_build_route_state_uses_lazy_default_dependency_loader(self):
        provider = types.SimpleNamespace(
            name="demo",
            anthropic_base_url="https://demo.example.com/anthropic",
            native_auth={"env": "DEMO_API_KEY", "header": "x-demo-key"},
            models={"demo-native": "demo/native-model"},
        )

        def fake_import(module_name):
            if module_name == "gateway.config":
                return types.SimpleNamespace(
                    _provider_from_model=lambda model, params: "demo",
                    resolve_thinking_contract=lambda entry: None,
                )
            if module_name == "gateway.providers":
                return types.SimpleNamespace(all_providers=lambda: [provider])
            raise ImportError(module_name)

        with mock.patch.object(routes.importlib, "import_module", side_effect=fake_import):
            route_state = routes.build_route_state([{"model_name": "demo-native", "litellm_params": {"model": "openai/demo-model"}}])

        self.assertEqual("demo.example.com", route_state.native_routes["demo-native"].host)


class ProxyV2ErrorsTests(unittest.TestCase):
    def test_map_upstream_status_matches_legacy_contract(self):
        self.assertEqual(
            (502, "Provider authentication failed", "auth_error"),
            errors.map_upstream_status(401),
        )
        self.assertEqual(
            (429, "Provider rate limited — retry later", "upstream_error"),
            errors.map_upstream_status(429),
        )
        self.assertEqual(
            (502, "Provider temporarily unavailable", "upstream_error"),
            errors.map_upstream_status(503),
        )

    def test_error_response_builds_proxy_json_envelope(self):
        status_code, body = errors.error_response(502, "boom", "upstream_error")
        self.assertEqual(502, status_code)
        self.assertEqual(
            b'{"error": {"message": "boom", "type": "upstream_error"}}',
            body,
        )


if __name__ == "__main__":
    unittest.main()
