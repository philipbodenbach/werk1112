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

    def interactive(self, answer, args=None):
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
            while b"default: Skip" not in output:
                self.assertIsNone(process.poll(), output.decode(errors="replace"))
                self.assertLess(time.monotonic(), deadline, output.decode(errors="replace"))
                if select.select([master], [], [], .05)[0]:
                    output += os.read(master, 8192)
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
