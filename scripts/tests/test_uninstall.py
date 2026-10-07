"""Test removal and data retention using only disposable user directories."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "uninstall.sh"


class UninstallTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.home = self.root / "home"
        self.env = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("WERK_", "XDG_"))
        }
        self.env["HOME"] = str(self.home)
        self.binary = self.home / ".local/bin/werk"
        self.store = self.home / ".local/share/werk1112"
        self.keys = self.home / ".config/werk1112/api-keys.toml"

    def write(self, path, contents="fixture"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)

    def seed(self):
        self.write(self.binary)
        self.write(self.store / "models/example.gguf")
        self.write(self.keys)
        self.neighbor = self.binary.with_name("another-tool")
        self.write(self.neighbor)

    def uninstall(self, answers=""):
        # Match the documented sh -c invocation, leaving stdin for prompts.
        result = subprocess.run(
            ["sh", "-c", SCRIPT.read_text()], env=self.env, input=answers,
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Werk1112 successfully removed.", result.stdout)
        self.assertFalse(self.binary.exists())
        self.assertFalse(self.binary.is_symlink())
        return result

    def test_default_paths_keep_data_on_empty_answer_or_eof(self):
        for answers in ["", "\n\n", "n\nN\n"]:
            with self.subTest(answers=answers):
                self.seed()
                result = self.uninstall(answers)
                self.assertIn("Model store kept.", result.stdout)
                self.assertIn("API keys kept.", result.stdout)
                self.assertEqual((self.store / "models/example.gguf").read_text(), "fixture")
                self.assertEqual(self.keys.read_text(), "fixture")
                self.assertEqual(self.neighbor.read_text(), "fixture")

    def test_removes_data_only_when_each_prompt_is_confirmed(self):
        for answers, keep_models, keep_keys in [
            ("y\nn\n", False, True), ("n\nyes\n", True, False),
            ("YES\nY\n", False, False),
        ]:
            with self.subTest(answers=answers):
                self.seed()
                self.uninstall(answers)
                self.assertEqual(self.store.exists(), keep_models)
                self.assertEqual(self.keys.exists(), keep_keys)
                self.assertEqual(self.neighbor.read_text(), "fixture")

    def test_custom_paths_with_spaces_take_precedence(self):
        self.seed()
        default_binary, default_store, default_keys = self.binary, self.store, self.keys
        self.binary = self.root / "custom bin/werk"
        self.store = self.root / "custom models"
        self.keys = self.root / "custom config/keys.toml"
        self.env.update(WERK_INSTALL_DIR=str(self.binary.parent),
                        WERK_HOME=str(self.store), WERK_API_KEYS=str(self.keys))
        self.seed()
        self.uninstall("y\ny\n")
        self.assertFalse(self.store.exists())
        self.assertFalse(self.keys.exists())
        self.assertTrue(default_binary.exists())
        self.assertTrue(default_store.exists())
        self.assertTrue(default_keys.exists())

    def test_xdg_paths_take_precedence_over_home_defaults(self):
        self.seed()
        default_store, default_keys = self.store, self.keys
        data_home, config_home = self.root / "data", self.root / "config"
        self.store = data_home / "werk1112"
        self.keys = config_home / "werk1112/api-keys.toml"
        self.env.update(XDG_DATA_HOME=str(data_home), XDG_CONFIG_HOME=str(config_home))
        self.seed()
        self.uninstall("y\ny\n")
        self.assertFalse(self.store.exists())
        self.assertFalse(self.keys.exists())
        self.assertTrue(default_store.exists())
        self.assertTrue(default_keys.exists())

    def test_repeated_uninstall_succeeds(self):
        self.seed()
        self.uninstall("y\ny\n")
        result = self.uninstall()
        self.assertIn("Werk1112 is not installed.", result.stdout)

    def test_removes_binary_symlink_without_deleting_target(self):
        target = self.root / "original-werk"
        self.write(target)
        self.binary.parent.mkdir(parents=True)
        for exists in [True, False]:
            with self.subTest(target_exists=exists):
                if not exists:
                    target.unlink()
                self.binary.symlink_to(target)
                self.uninstall()
                self.assertEqual(target.exists(), exists)


if __name__ == "__main__":
    unittest.main()
