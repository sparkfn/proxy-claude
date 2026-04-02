import os
import json
import subprocess
import tempfile
import unittest
from unittest import mock

import cli
import config
from providers.base import Status


class ThinkingContractTests(unittest.TestCase):
    def test_resolve_thinking_contract_for_chatgpt_model(self):
        model = {
            "alias": "gpt-5.4",
            "provider": "openai",
            "model": "chatgpt/gpt-5.4",
            "litellm_params": {"model": "chatgpt/gpt-5.4"},
        }

        contract = config.resolve_thinking_contract(model)

        self.assertIsNotNone(contract)
        self.assertEqual("openai_chat_reasoning_effort", contract["strategy"])
        self.assertEqual("chatgpt", contract["route_family"])
        self.assertEqual(("low", "medium", "high", "xhigh"), contract["levels"])
        self.assertEqual("xhigh", contract["levels"][-1])

    def test_resolve_thinking_contract_for_openai_compatible_model(self):
        model = {
            "alias": "MiniMax-M2.7",
            "provider": "minimax",
            "model": "openai/MiniMax-M2.7",
            "litellm_params": {
                "model": "openai/MiniMax-M2.7",
                "api_base": "https://api.minimax.io/v1",
            },
        }

        contract = config.resolve_thinking_contract(model)

        self.assertIsNotNone(contract)
        self.assertEqual("openai_chat_reasoning_effort", contract["strategy"])
        self.assertEqual("openai", contract["route_family"])
        self.assertEqual("minimax", contract["provider"])
        self.assertEqual(("low", "medium", "high"), contract["levels"])


class LaunchClaudeModelSelectionTests(unittest.TestCase):
    def test_proclaude_launcher_exists(self):
        self.assertTrue(os.path.exists("/Users/noonoon/Dev/proclaude.sh"))

    def test_proclaude_runs_without_changing_calling_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zsh_command = (
                'source /Users/noonoon/.zshrc >/dev/null 2>&1; '
                'proclaude --emit-env /tmp/proclaude-test-env >/tmp/proclaude.out 2>/tmp/proclaude.err || true; '
                'pwd'
            )
            result = subprocess.run(
                ["zsh", "-ic", zsh_command],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(os.path.realpath(tmpdir), os.path.realpath(result.stdout.strip()))

    @mock.patch("cli.os.execvp", side_effect=SystemExit(0))
    @mock.patch("shutil.which", return_value="/usr/local/bin/claude")
    @mock.patch("cli._eprint")
    @mock.patch("builtins.print")
    @mock.patch("providers.get_provider")
    @mock.patch("providers.all_providers")
    @mock.patch("config.load_env_file", return_value={"OPENAI_API_KEY": "sk-test"})
    @mock.patch("config.list_models", return_value=[])
    @mock.patch("config.add_model", return_value=(Status.OK, "Added"))
    @mock.patch("config.ensure_master_key", return_value="sk-test")
    @mock.patch("container.status", return_value=(Status.OK, "ok"))
    def test_launch_skips_configured_ollama_model_when_not_available(
        self,
        _container_status,
        _ensure_master_key,
        _add_model,
        _list_models,
        _load_env,
        all_providers,
        get_provider,
        _eprint,
        _print,
        _which,
        _execvp,
    ):
        openai_provider = mock.Mock()
        openai_provider.name = "openai"
        openai_provider.display_name = "OpenAI"
        openai_provider.models = {"gpt-5": "chatgpt/gpt-5"}
        openai_provider.model_limits = {}
        openai_provider.auth_types = ["browser_oauth", "api_key"]
        openai_provider.login_prompts = {}
        openai_provider.check_ready.return_value = (True, "")
        openai_provider.detect_auth_type.return_value = "browser_oauth"
        openai_provider.get_models_for_auth = None
        openai_provider.validate.return_value = (Status.OK, "Authenticated")
        openai_provider.resolve_thinking_contract.return_value = None

        ollama_provider = mock.Mock()
        ollama_provider.name = "ollama"
        ollama_provider.display_name = "Ollama"
        ollama_provider.models = {"llama3": "ollama/llama3"}
        ollama_provider.model_limits = {}
        ollama_provider.auth_types = []
        ollama_provider.login_prompts = {}
        ollama_provider.check_ready.return_value = (False, "Ollama not reachable")
        ollama_provider.get_models_for_auth = None
        ollama_provider.validate.return_value = (Status.UNREACHABLE, "Not reachable")

        all_providers.return_value = [openai_provider, ollama_provider]
        get_provider.side_effect = lambda n: openai_provider if n == "openai" else ollama_provider

        cli._init_registry()

        with self.assertRaises(SystemExit):
            cli.cmd_launch_claude(model_flag="gpt-5", telegram=False)

        _execvp.assert_called_once()
        self.assertEqual("gpt-5", cli.os.environ["ANTHROPIC_MODEL"])

    @mock.patch("cli.os.execvp", side_effect=SystemExit(0))
    @mock.patch("shutil.which", return_value="/usr/local/bin/claude")
    @mock.patch("builtins.print")
    @mock.patch("providers.get_provider")
    @mock.patch("providers.all_providers")
    @mock.patch("config.load_env_file", return_value={})
    @mock.patch("config.list_models", return_value=[{"alias": "llama3", "model": "ollama/llama3"}])
    @mock.patch("config.ensure_master_key", return_value="sk-test")
    @mock.patch("container.status", return_value=(Status.OK, "ok"))
    def test_launch_hard_fails_when_thinking_is_requested_without_verified_contract(
        self,
        _container_status,
        _ensure_master_key,
        _list_models,
        _load_env,
        all_providers,
        get_provider,
        _print,
        _which,
        _execvp,
    ):
        ollama_provider = mock.Mock()
        ollama_provider.name = "ollama"
        ollama_provider.display_name = "Ollama"
        ollama_provider.models = {"llama3": "ollama/llama3"}
        ollama_provider.model_limits = {}
        ollama_provider.auth_types = []
        ollama_provider.login_prompts = {}
        ollama_provider.check_ready.return_value = (True, "")
        ollama_provider.get_models_for_auth = None
        ollama_provider.validate.return_value = (Status.OK, "Ollama is reachable")
        ollama_provider.resolve_thinking_contract.return_value = None

        all_providers.return_value = [ollama_provider]
        get_provider.return_value = ollama_provider

        with self.assertRaises(SystemExit):
            cli.cmd_launch_claude(model_flag="llama3", thinking="high", telegram=False)

        output = "\n".join(" ".join(str(arg) for arg in call.args) for call in _print.call_args_list)
        self.assertIn("Thinking effort is not supported", output)
        _execvp.assert_not_called()

    @mock.patch("cli.input", return_value="")
    @mock.patch("shutil.which", return_value="/usr/local/bin/claude")
    @mock.patch("builtins.print")
    @mock.patch("providers.get_provider")
    @mock.patch("providers.all_providers")
    @mock.patch("config.load_env_file", return_value={})
    @mock.patch("config.list_models", return_value=[])
    @mock.patch("container.status", return_value=(Status.OK, "ok"))
    def test_launch_fails_before_exec_when_selected_provider_is_not_authenticated(
        self,
        _container_status,
        _list_models,
        _load_env,
        all_providers,
        get_provider,
        _print,
        _which,
        _input,
    ):
        zhipu_provider = mock.Mock()
        zhipu_provider.name = "zhipu"
        zhipu_provider.display_name = "Z.AI"
        zhipu_provider.models = {"glm-5.1": "openai/glm-5.1"}
        zhipu_provider.model_limits = {}
        zhipu_provider.auth_types = ["api_key"]
        zhipu_provider.login_prompts = {"api_key": {"instructions": "Enter key", "fields": [("ZAI_API_KEY", "ZAI_API_KEY: ")]}}
        zhipu_provider.check_ready.return_value = (False, "ZAI_API_KEY not set")
        zhipu_provider.get_models_for_auth = None
        zhipu_provider.validate.return_value = (Status.NOT_CONFIGURED, "ZAI_API_KEY not set")

        all_providers.return_value = [zhipu_provider]
        get_provider.return_value = zhipu_provider

        with self.assertRaises(SystemExit):
            cli.cmd_launch_claude(model_flag="glm-5.1", telegram=False)

        output = "\n".join(" ".join(str(arg) for arg in call.args) for call in _print.call_args_list)
        self.assertIn("glm-5.1 is not ready", output)

    @mock.patch("providers.get_provider")
    @mock.patch("providers.all_providers")
    @mock.patch("config.load_env_file", return_value={"OPENAI_API_KEY": "sk-test"})
    @mock.patch("config.list_models", return_value=[{"alias": "gpt-5.4", "model": "chatgpt/gpt-5.4"}])
    @mock.patch("config.resolve_thinking_contract")
    @mock.patch("config.ensure_master_key", return_value="sk-test")
    @mock.patch("container.status", return_value=(Status.OK, "ok"))
    def test_launch_emit_env_accepts_xhigh_for_gpt_5_4(
        self,
        _container_status,
        _ensure_master_key,
        resolve_thinking,
        _list_models,
        _load_env,
        all_providers,
        get_provider,
    ):
        openai_provider = mock.Mock()
        openai_provider.name = "openai"
        openai_provider.display_name = "OpenAI"
        openai_provider.models = {"gpt-5.4": "chatgpt/gpt-5.4"}
        openai_provider.model_limits = {}
        openai_provider.auth_types = ["browser_oauth", "api_key"]
        openai_provider.login_prompts = {}
        openai_provider.check_ready.return_value = (True, "")
        openai_provider.detect_auth_type.return_value = "browser_oauth"
        openai_provider.get_models_for_auth = None
        openai_provider.validate.return_value = (Status.UNVERIFIED, "Browser OAuth may be active")
        openai_provider.resolve_thinking_contract.return_value = {
            "provider": "openai",
            "strategy": "openai_chat_reasoning_effort",
            "route_family": "chatgpt",
            "levels": ("low", "medium", "high", "xhigh"),
            "default_level": "medium",
            "level_labels": {"low": "Low", "medium": "Medium", "high": "High", "xhigh": "Extra high"},
        }

        all_providers.return_value = [openai_provider]
        get_provider.return_value = openai_provider
        resolve_thinking.return_value = {
            "provider": "openai",
            "strategy": "openai_chat_reasoning_effort",
            "levels": ("low", "medium", "high", "xhigh"),
            "default_level": "medium",
            "level_labels": {"low": "Low", "medium": "Medium", "high": "High", "xhigh": "Extra high"},
        }

        with tempfile.NamedTemporaryFile() as tmp:
            with tempfile.NamedTemporaryFile() as state_tmp:
                with mock.patch.dict(os.environ, {"PROXY_MODEL_ALIAS_STATE": state_tmp.name}, clear=False):
                    cli.MODEL_ALIAS_STATE_FILE = os.environ["PROXY_MODEL_ALIAS_STATE"]
                    cli.cmd_launch_claude(model_flag="gpt-5.4", thinking="xhigh", telegram=False, emit_env=tmp.name)
                    with open(tmp.name, "r") as f:
                        emitted = f.read()
                    with open(state_tmp.name, "r") as f:
                        state = json.load(f)

        self.assertIn("x-thinking-effort: xhigh", emitted)
        self.assertEqual("gpt-5.4", state["selected_model"])
        self.assertEqual("gpt-5.4", state["anthropic_defaults"]["haiku"])
        self.assertEqual("gpt-5.4", state["anthropic_defaults"]["sonnet"])
        self.assertEqual("gpt-5.4", state["anthropic_defaults"]["opus"])


if __name__ == "__main__":
    unittest.main()
