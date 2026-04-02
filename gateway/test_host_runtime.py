import io
import os
import subprocess
import sys
import unittest
from unittest import mock

import host_runtime


class HostRuntimeTests(unittest.TestCase):
    def test_module_import_does_not_require_host_pyyaml(self):
        completed = subprocess.run(
            [sys.executable, "-c", "import host_runtime"],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_parse_auth_prompt_extracts_url_and_code(self):
        logs = """
        some line
        Open this URL: https://auth.openai.com/codex/device
        Enter code: ABCD-EFGH
        """
        url, code = host_runtime._parse_auth_prompt(logs)
        self.assertEqual("https://auth.openai.com/codex/device", url)
        self.assertEqual("ABCD-EFGH", code)

    @mock.patch("host_runtime._docker_compose_logs", return_value="https://auth.openai.com/codex/device\nEnter code: TEST-CODE")
    @mock.patch("host_runtime._gateway_json", return_value=(503, {"status": "unreachable"}))
    def test_report_start_status_reports_pending_auth(self, _gateway_json, _logs):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            rc = host_runtime._report_start_status("docker-compose.yml", "http://127.0.0.1:2555")
        self.assertEqual(0, rc)
        output = stdout.getvalue()
        self.assertIn("OpenAI Login Required", output)
        self.assertIn("upstream auth is still pending", output)

    @mock.patch("host_runtime._docker_compose_logs", return_value="")
    @mock.patch("host_runtime._gateway_json", return_value=(503, {"status": "unreachable"}))
    def test_offer_pending_auth_degrades_cleanly_without_auth_prompt(self, _gateway_json, _logs):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            rc = host_runtime._offer_pending_auth(
                "docker-compose.yml",
                "http://127.0.0.1:2555",
                "gpt-5.4",
                300,
            )
        self.assertEqual(0, rc)
        self.assertIn("Claude may fail until LiteLLM finishes initializing", stdout.getvalue())

    @mock.patch("host_runtime.time.sleep", return_value=None)
    @mock.patch("host_runtime._docker_compose_logs", side_effect=[
        "https://auth.openai.com/codex/device\nEnter code: TEST-CODE",
        "successfully authenticated",
    ])
    @mock.patch("host_runtime._gateway_post_json", return_value=(500, ""))
    @mock.patch("host_runtime._check_proxy_models", side_effect=[(False, "proxy unreachable"), (False, None)])
    @mock.patch("host_runtime._get_env", return_value="sk-test")
    @mock.patch("host_runtime._configured_chatgpt_models", return_value=["gpt-5.4"])
    def test_openai_browser_login_uses_host_side_flow(
        self,
        _configured_chatgpt_models,
        _get_env,
        _check_proxy_models,
        _post_json,
        _logs,
        _sleep,
    ):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            rc = host_runtime._openai_browser_login("docker-compose.yml", "http://127.0.0.1:2555", 300)
        self.assertEqual(0, rc)
        output = stdout.getvalue()
        self.assertIn("OpenAI Login Required", output)
        self.assertIn("Browser OAuth may be active", output)


if __name__ == "__main__":
    unittest.main()
