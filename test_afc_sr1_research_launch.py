import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parent / "scripts" / "afc-sr1-research.sh"


class AfcSr1ResearchLaunchTests(unittest.TestCase):
    def run_script(self, env_text: str = "", local_env_text: str = ""):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            scripts.mkdir()
            script = scripts / SCRIPT.name
            script.write_bytes(SCRIPT.read_bytes())
            script.chmod(0o755)
            (root / ".env").write_text(env_text, encoding="utf-8")
            (root / ".env.local").write_text(local_env_text, encoding="utf-8")

            fake_bin = root / "bin"
            fake_bin.mkdir()
            invocation = root / "uvicorn-invocation"
            uvicorn = fake_bin / "uvicorn"
            uvicorn.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$FAKE_UVICORN_OUTPUT\"\n",
                encoding="utf-8",
            )
            uvicorn.chmod(0o755)
            environment = {
                **os.environ,
                "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
                "FAKE_UVICORN_OUTPUT": str(invocation),
            }
            result = subprocess.run(
                [str(script)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            arguments = (
                invocation.read_text(encoding="utf-8").splitlines()
                if invocation.exists()
                else None
            )
            return result, arguments

    def test_missing_gate_fails_before_uvicorn(self):
        result, arguments = self.run_script()
        self.assertEqual(result.returncode, 2)
        self.assertIsNone(arguments)
        self.assertIn("reader gate: disabled", result.stdout)
        self.assertIn("placement gate: disabled", result.stdout)

    def test_env_local_overrides_env_and_failure_remains_pre_bind(self):
        result, arguments = self.run_script(
            "AFC_SR1_TR2_READER_ENABLED=true\n"
            "AFC_SR1_TS0_CHILD_PLACEMENT_ENABLED=true\n",
            "AFC_SR1_TS0_CHILD_PLACEMENT_ENABLED=off\n",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIsNone(arguments)
        self.assertIn("reader gate: enabled", result.stdout)
        self.assertIn("placement gate: disabled", result.stdout)

    def test_truthy_values_launch_without_reload(self):
        for value in ("1", "true", "yes", "on"):
            with self.subTest(value=value):
                result, arguments = self.run_script(
                    f"AFC_SR1_TR2_READER_ENABLED={value}\n"
                    f"AFC_SR1_TS0_CHILD_PLACEMENT_ENABLED={value.upper()}\n"
                    "AFC_SR1_RESEARCH_HOST=127.0.0.2\n"
                    "AFC_SR1_RESEARCH_PORT=8123\n"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    arguments,
                    ["main:app", "--host", "127.0.0.2", "--port", "8123"],
                )
                self.assertNotIn("--reload", arguments)
                self.assertIn("listen: 127.0.0.2:8123", result.stdout)


if __name__ == "__main__":
    unittest.main()
