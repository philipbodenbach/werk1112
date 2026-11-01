"""Real CLI/PTY optional setup regression; no network, packages or model weights.

Build werk first, then run with python -m unittest discover -s scripts/tests
-p test_vision_setup.py. WERK_TEST_BIN may select another local build.
"""
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get("WERK_TEST_BIN", ROOT / "target/debug/werk"))


def gguf(architecture):
    result = b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
    for key, value in [("general.architecture", architecture), ("general.name", "test VLM")]:
        key, value = key.encode(), value.encode()
        result += struct.pack("<Q", len(key)) + key + struct.pack("<IQ", 8, len(value)) + value
    return result


@unittest.skipUnless(BINARY.is_file(), "build werk before CLI setup tests")
class VisionSetup(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="werk-vision-cli-")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source = root / "source"
        source.mkdir()
        (source / "model.gguf").write_bytes(gguf("qwen35"))
        (source / "mmproj-F16.gguf").write_bytes(gguf("clip"))
        self.store = root / "store"
        self.command = [str(BINARY), "--model-home", str(self.store)]
        self.env = dict(os.environ, TERM="xterm-256color")
        self.env.pop("WERK_LOG_FILE", None)
        result = self.run_cli("import", str(source), "--name", "vlm", "--text-only", "--offline")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.manifest = self.store / "models/vlm/manifest.json"
        data = json.loads(self.manifest.read_text())
        data.pop("vision_setup", None)  # Fresh undecided setup, not a previous No.
        self.manifest.write_text(json.dumps(data))

    def run_cli(self, *args, input=""):
        return subprocess.run(self.command + list(args), input=input, text=True,
                              capture_output=True, env=self.env, timeout=10)

    def state(self):
        return json.loads(self.manifest.read_text())["vision_setup"]["enabled"]

    def interactive(self, answer, args=None, marker=b"default: Skip", before_answer=None):
        if os.name != "posix":
            self.skipTest("PTY input requires POSIX")
        import pty
        master, slave = pty.openpty()
        process = subprocess.Popen(self.command + (args or ["artifacts", "vision", "vlm", "--offline"]),
                                   stdin=slave, stdout=slave, stderr=slave, env=self.env)
        os.close(slave)
        try:
            output = b""
            deadline = time.monotonic() + 8
            while marker not in output:
                self.assertIsNone(process.poll(), output.decode(errors="replace"))
                self.assertLess(time.monotonic(), deadline, output.decode(errors="replace"))
                if select.select([master], [], [], .05)[0]:
                    try:
                        output += os.read(master, 8192)
                    except OSError:
                        self.fail("CLI exited before setup prompt:\n" + output.decode(errors="replace"))
            if before_answer:
                before_answer(output.decode(errors="replace"))
            os.write(master, answer.encode())
            return process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            os.close(master)

    def test_interactive_yes_and_recorded_choice(self):
        self.assertEqual(self.interactive("y\n"), 0)
        self.assertTrue(self.state())
        result = self.run_cli("artifacts", "vision", "vlm", "--offline")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.state())

    def test_interactive_default_skip(self):
        self.assertEqual(self.interactive("\n"), 0)
        self.assertFalse(self.state())

    def test_cancel_does_not_change_manifest(self):
        before = self.manifest.read_bytes()
        self.assertNotEqual(self.interactive("q\n"), 0)
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_cancelled_import_leaves_new_model_text_only(self):
        result = self.interactive("q\n", ["import", str(Path(self.temp.name) / "source"),
                                         "--name", "cancelled", "--offline"])
        self.assertNotEqual(result, 0)
        saved = json.loads((self.store / "models/cancelled/manifest.json").read_text())
        self.assertFalse(saved["vision_setup"]["enabled"])

    def test_piped_yes_is_not_interactive_consent(self):
        result = self.run_cli("artifacts", "vision", "vlm", "--offline", input="y\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("default: Skip", result.stderr)
        self.assertFalse(self.state())

    def setup_runtime(self, vision=True):
        binary = self.store / "backends/llama-cpu/llama-server"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("#!/bin/sh\necho --mmproj\n" if vision else "#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        # Isolate discovery from any developer-installed runtimes.
        self.env["WERK_LLAMA_SERVER_CPU"] = str(binary)
        self.env["WERK_LLAMA_SERVER"] = str(binary)
        data = json.loads(self.manifest.read_text())
        data["vision_setup"] = {"enabled": False, "backend": "llama-server"}
        self.manifest.write_text(json.dumps(data))

    def test_complete_setup_yes_activates_all_local_components(self):
        self.setup_runtime()
        code = self.interactive("y\n", ["--backend", "cpu", "setup", "vlm", "--offline"],
                                marker=b"Set up all")
        self.assertEqual(code, 0)
        self.assertTrue(self.state())
        # Subsequent starts/checks need no approval or downloads.
        result = self.run_cli("--backend", "cpu", "setup", "vlm", "--offline")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("confirmation", result.stderr)

    def test_complete_setup_later_and_cancel_preserve_model(self):
        self.setup_runtime()
        before = self.manifest.read_bytes()
        for answer, expected in [("\n", 0), ("q\n", 1)]:
            result = self.interactive(answer, ["--backend", "cpu", "setup", "vlm", "--offline"],
                                      marker=b"Set up all")
            self.assertEqual(result == 0, expected == 0)
            self.assertEqual(before, self.manifest.read_bytes())

    def test_complete_setup_does_not_take_piped_input_as_consent(self):
        self.setup_runtime()
        before = self.manifest.read_bytes()
        result = self.run_cli("--backend", "cpu", "setup", "vlm", "--offline", input="y\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("needs confirmation", result.stderr)
        self.assertEqual(before, self.manifest.read_bytes())

    def test_complete_setup_explicit_approval_works_without_terminal(self):
        self.setup_runtime()
        result = self.run_cli("--auto-install-backends", "--backend", "cpu", "setup", "vlm", "--offline")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.state())

    def test_run_offers_vision_and_backend_upgrade_together(self):
        self.setup_runtime(vision=False)
        before = self.manifest.read_bytes()
        def check(output):
            self.assertIn("llama-cpu", output)
            self.assertIn("Vision:", output)
            self.assertIn("tool calls", output)
        result = self.interactive("q\n", ["--backend", "cpu", "run", "vlm", "hello"],
                                  marker=b"Set up all", before_answer=check)
        self.assertNotEqual(result, 0)
        self.assertEqual(before, self.manifest.read_bytes())

    def test_pull_finishes_before_backend_setup_dialog(self):
        # Stub only the repository transport: no network or Git commits.
        repo = Path(self.temp.name) / "git-source"
        repo.mkdir()
        (repo / "model.gguf").write_bytes(gguf("qwen35"))
        bindir = Path(self.temp.name) / "bin"
        bindir.mkdir()
        git = bindir / "git"
        git.write_text("#!/usr/bin/env python3\nimport shutil, sys\n"
                       + f"source = {str(repo)!r}\n"
                       + "if sys.argv[1] == 'clone': shutil.copytree(source, sys.argv[-1])\n"
                       + "elif 'rev-parse' in sys.argv: print('a' * 40)\n"
                       + "elif 'ls-remote' in sys.argv: print('a' * 40 + '\\tHEAD')\n"
                       + "elif '--version' in sys.argv: print('git version 2.40.0')\n"
                       + "else: sys.exit(1)\n")
        git.chmod(0o755)
        self.env["PATH"] = str(bindir) + os.pathsep + self.env["PATH"]
        self.env["WERK_LLAMA_SERVER_CPU"] = str(repo / "missing-runtime")
        self.env["WERK_LLAMA_SERVER"] = str(repo / "missing-runtime")
        def check(output):
            self.assertIn("Pulled", output)
            saved = self.store / "models/pulled/manifest.json"
            self.assertTrue(saved.is_file())
            self.assertTrue((saved.parent / "source-revision.json").is_file())
        result = self.interactive("\n", ["--backend", "cpu", "pull", "test/setup-fixture",
                                         "--name", "pulled", "--text-only"],
                                  marker=b"Set up all", before_answer=check)
        self.assertEqual(result, 0)
        self.assertTrue((self.store / "models/pulled/files/model.gguf").is_file())

    def test_offline_and_unsupported_installers_fail_before_setup(self):
        for args in [
            ["pull", "test/model", "--vision", "--offline"],
            ["backend", "install", "llama-cpu", "--vision", "--offline"],
            ["artifacts", "vision", "vlm", "--vision", "--download", "--offline"],
            ["backend", "install", "media", "--vision"],
            ["backend", "install", "onnx-cpu", "--vision"],
        ]:
            before = self.manifest.read_bytes()
            result = self.run_cli(*args)
            self.assertNotEqual(result.returncode, 0, args)
            self.assertEqual(self.manifest.read_bytes(), before)
            self.assertFalse((self.store / "backends").exists())


if __name__ == "__main__":
    unittest.main()
