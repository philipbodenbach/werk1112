"""Model-free generator contracts: profiles, discovery parsing and read-only checks."""

from contextlib import redirect_stdout
import fnmatch
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("cli_reference", ROOT / "scripts/generate-cli-reference.py")
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)
HELP = """Example CLI

Usage: werk [OPTIONS]

Options:
      --backend <BACKEND>  Backend [default: auto] [possible values: auto, cpu]
      --key <KEY>          Credential [env: WERK_API_KEY=fixture-secret]
  -h, --help               Print help

A trailing operational note.
"""


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="werk-generator-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "Cargo.toml").write_text('version = "1.2.3"\n')
        tasks = self.root / "src/capabilities/task.rs"
        tasks.parent.mkdir(parents=True)
        tasks.write_text('Example => "example"\nimpl InferenceTask {}')
        self.canonical = self.root / "docs/reference"
        self.canonical.mkdir(parents=True)
        (self.canonical / "tasks-and-formats.md").write_text('| `example` | Example |\n')
        for name, value in [("ROOT", self.root), ("CANONICAL_OUTPUT", self.canonical)]:
            patcher = patch.object(REFERENCE, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def discover(self, profile="linux-neutral", help_text=HELP, version="werk 1.2.3"):
        def run(command, **kwargs):
            args = command[1:]
            self.assertNotIn("WERK_API_KEY", kwargs["env"])
            self.assertNotIn("HF_TOKEN", kwargs["env"])
            if args == ["--version"]:
                output = version
            elif args == ["--help"]:
                output = help_text
            else:
                self.assertEqual(args, ["--model-home", str(self.root / "store"),
                                       "--no-auto-install-backends", "parameters",
                                       "--task", "example", "--json"])
                output = json.dumps({"parameters": []})
            return subprocess.CompletedProcess(command, 0, stdout=output)

        with patch.object(REFERENCE.subprocess, "run", side_effect=run), \
                patch.dict(os.environ, WERK_API_KEY="fixture-secret", HF_TOKEN="fixture-token"):
            return REFERENCE.generate(self.root / "werk", self.root / "store", profile)

    def test_discovery_redacts_environment_and_keeps_after_help_separate(self):
        pages, nodes, tasks = self.discover()
        self.assertEqual(tasks, ["example"])
        self.assertEqual(len(pages), 5)
        self.assertNotIn("fixture-secret", str(pages))
        self.assertEqual(nodes[0]["after"], "A trailing operational note.")
        self.assertEqual(nodes[0]["options"][-1]["description"], "Print help")

    def test_each_generated_page_identifies_its_profile_and_version(self):
        for profile in REFERENCE.PROFILES:
            with self.subTest(profile=profile):
                pages, _, _ = self.discover(profile)
                for content in pages.values():
                    self.assertIn(REFERENCE.PROFILES[profile][0], content)
                    self.assertIn("werk 1.2.3", content)
                    other = "discovery" if profile == "linux-neutral" else "linux-neutral"
                    self.assertNotIn(REFERENCE.PROFILES[other][0], content)

    def test_burn_feature_discovery_cannot_be_mislabeled_canonical(self):
        burn_help = HELP.replace("auto, cpu]", "auto, cpu, burn]")
        with self.assertRaisesRegex(ValueError, "noncanonical feature build"):
            self.discover(help_text=burn_help)
        pages, _, _ = self.discover("discovery", help_text=burn_help)
        self.assertIn("auto, cpu, burn", pages["cli-options.md"])

    def test_version_mismatch_requires_rebuilding_checkout(self):
        with self.assertRaisesRegex(ValueError, "expected werk 1.2.3, got werk 1.2.2"):
            self.discover(version="werk 1.2.2")

    def test_child_options_with_shadowed_globals_are_preserved(self):
        root = REFERENCE.parse_help([], HELP)
        child = REFERENCE.parse_help(["example"], HELP.replace("Backend [default", "Media backend [default"))
        self.assertEqual([o["flag"] for o in REFERENCE.local_options(child, root)], ["--backend"])

    def test_profile_guards_precede_any_discovery_or_output(self):
        alias = self.root / "canonical-alias"
        alias.symlink_to(self.canonical, target_is_directory=True)
        for check in ([], ["--check"]):
            for destination in (self.canonical, alias):
                with self.subTest(check=check, destination=destination), \
                        patch.object(sys, "argv", ["generator", "werk", "--profile", "discovery",
                                                   "--output-dir", str(destination), *check]), \
                        patch.object(REFERENCE, "discovery_binary") as discover, \
                        patch("sys.stderr", new_callable=io.StringIO):
                    with self.assertRaises(SystemExit) as error:
                        REFERENCE.main()
                    self.assertEqual(error.exception.code, 2)
                    discover.assert_not_called()

    def test_canonical_profile_requires_linux_without_packaging_target(self):
        with patch.object(sys, "platform", "win32"):
            with self.assertRaises(ValueError):
                REFERENCE.validate_profile("linux-neutral", self.canonical, None)
        with patch.object(sys, "platform", "linux"):
            with self.assertRaises(ValueError):
                REFERENCE.validate_profile("linux-neutral", self.canonical, "aarch64-apple-darwin")
            REFERENCE.validate_profile("linux-neutral", self.canonical, None)
        REFERENCE.validate_profile("discovery", self.root / "output", "aarch64-apple-darwin")

    def invoke_main(self, output, check=False):
        argv = ["generator", "fresh binary", "--profile", "discovery", "--output-dir", str(output)]
        if check:
            argv.append("--check")
        with patch.object(sys, "argv", argv), \
                patch.object(REFERENCE, "generate", return_value=({"cli-options.md": "fresh\n"}, [], ["example"])):
            REFERENCE.main()

    def test_generation_and_current_check_leave_canonical_files_untouched(self):
        marker = self.canonical / "cli-options.md"
        marker.write_text("canonical\n")
        output = self.root / "separate output"
        with redirect_stdout(io.StringIO()):
            self.invoke_main(output)
            timestamp = (output / "cli-options.md").stat().st_mtime_ns
            self.invoke_main(output, check=True)
        self.assertEqual((output / "cli-options.md").read_text(), "fresh\n")
        self.assertEqual((output / "cli-options.md").stat().st_mtime_ns, timestamp)
        self.assertEqual(marker.read_text(), "canonical\n")

    def test_stale_and_missing_check_fail_with_diff_and_reusable_command_without_writes(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                output = self.root / ("stale output" if existing else "missing output")
                destination = output / "cli-options.md"
                if existing:
                    output.mkdir()
                    destination.write_text("stale\n")
                    timestamp = destination.stat().st_mtime_ns
                stdout = io.StringIO()
                with redirect_stdout(stdout), self.assertRaises(SystemExit) as error:
                    self.invoke_main(output, check=True)
                self.assertIn("Reference drift: cli-options.md", str(error.exception))
                self.assertIn("+++ cli-options.md (generated)", stdout.getvalue())
                self.assertIn("+fresh", stdout.getvalue())
                regenerate = shlex.split(str(error.exception).splitlines()[-1])
                self.assertEqual(regenerate[2:], ["fresh binary", "--profile", "discovery",
                                                 "--output-dir", str(output)])
                self.assertEqual(destination.exists(), existing)
                if existing:
                    self.assertEqual(destination.read_text(), "stale\n")
                    self.assertEqual(destination.stat().st_mtime_ns, timestamp)
                else:
                    self.assertFalse(output.exists())


class WorkflowCoverageTests(unittest.TestCase):
    def test_pull_request_paths_cover_modular_sources_and_generator_inputs(self):
        workflow = (ROOT / ".github/workflows/cli-reference.yml").read_text()
        paths = workflow.split("  pull_request:\n    paths:\n", 1)[1].split("  workflow_dispatch:", 1)[0]
        patterns = re.findall(r'^      - "([^"]+)"$', paths, re.M)
        changed_files = ["src/cli.rs", "src/cli/top.rs", "src/media_cli.rs",
                         "src/capabilities/task.rs", "src/new_module/schema.rs", "runtime/helper.py",
                         "Cargo.toml", "Cargo.lock", ".cargo/config.toml", "build.rs",
                         "scripts/generate-cli-reference.py", "scripts/tests/test_cli_reference.py",
                         "docs/reference/cli-options.md", ".github/workflows/cli-reference.yml"]
        for path in changed_files:
            with self.subTest(path=path):
                self.assertTrue(any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns))


if __name__ == "__main__":
    unittest.main()
