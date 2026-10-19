"""Build/reference ordering and cross-target discovery without a release build."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("cli_reference", ROOT / "scripts/generate-cli-reference.py")
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)
HOST = "x86_64-unknown-linux-gnu"


class DiscoveryBinaryTests(unittest.TestCase):
    def test_native_build_reuses_artifact_without_another_cargo_build(self):
        version = subprocess.CompletedProcess([], 0, stdout=f"host: {HOST}\n")
        with patch.object(REFERENCE.subprocess, "run", return_value=version) as run:
            binary = Path("target/custom/werk")
            self.assertEqual(REFERENCE.discovery_binary(binary, HOST), binary.resolve())
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], ["rustc", "-vV"])

    def test_cross_build_uses_cargo_reported_host_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "custom target directory" / "werk"
            binary.parent.mkdir()
            binary.touch()
            events = [
                {"reason": "compiler-artifact", "target": {"name": "other", "kind": ["lib"]}},
                {"reason": "compiler-artifact", "target": {"name": "werk", "kind": ["bin"]},
                 "executable": str(binary), "fresh": True},
                {"reason": "build-finished", "success": True},
            ]
            results = [subprocess.CompletedProcess([], 0, stdout=f"host: {HOST}\n"),
                       subprocess.CompletedProcess([], 0, stdout="\n".join(map(json.dumps, events)))]
            with patch.object(REFERENCE.subprocess, "run", side_effect=results) as run:
                selected = REFERENCE.discovery_binary(Path("foreign/werk.exe"), "x86_64-pc-windows-msvc")
                self.assertEqual(selected, binary.resolve())
                command = run.call_args.args[0]
                self.assertEqual(command[:2], ["cargo", "build"])
                self.assertIn("--locked", command)
                self.assertIn("--no-default-features", command)
                self.assertEqual(command[command.index("--target") + 1], HOST)
                self.assertEqual(command[command.index("--target-dir") + 1],
                                 str(ROOT / "target/cli-reference-host"))

    def test_host_build_failure_never_falls_back_to_foreign_artifact(self):
        results = [subprocess.CompletedProcess([], 0, stdout=f"host: {HOST}\n"),
                   subprocess.CalledProcessError(101, ["cargo", "build"])]
        with patch.object(REFERENCE.subprocess, "run", side_effect=results):
            with self.assertRaises(subprocess.CalledProcessError):
                REFERENCE.discovery_binary(Path("foreign/werk"), "aarch64-apple-darwin")

    def test_missing_host_executable_is_an_error(self):
        results = [subprocess.CompletedProcess([], 0, stdout=f"host: {HOST}\n"),
                   subprocess.CompletedProcess([], 0, stdout='{"reason":"build-finished"}\n')]
        with patch.object(REFERENCE.subprocess, "run", side_effect=results):
            with self.assertRaisesRegex(ValueError, "exactly one built werk"):
                REFERENCE.discovery_binary(Path("foreign/werk"), "aarch64-apple-darwin")


@unittest.skipUnless(os.name == "posix" and shutil.which("bash") and shutil.which("tar"),
                     "shell packaging test requires POSIX bash and tar")
class PackagingReferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="werk-build-reference-")
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name) / "repo with spaces"
        scripts = self.repo / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(ROOT / "scripts/package-release.sh", scripts / "package-release.sh")
        (self.repo / "Cargo.toml").write_text('[package]\nversion = "1.2.3"\n')
        (self.repo / "README.md").write_text("before generation\n")
        (self.repo / "LICENSE").write_text("license fixture\n")
        assets = self.repo / "utils/observability"
        assets.mkdir(parents=True)
        for name in ["README.md", "prometheus.yml", "grafana-dashboard.json"]:
            (assets / name).write_text("fixture\n")
        self.canonical = self.repo / "docs/reference/cli-options.md"
        self.canonical.parent.mkdir(parents=True)
        self.canonical.write_text("canonical reference fixture\n")
        self.trace = self.repo / "trace.jsonl"
        bin_dir = self.repo / "fake-bin"
        bin_dir.mkdir()
        cargo = bin_dir / "cargo"
        cargo.write_text("""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
with open(os.environ['BUILD_TRACE'], 'a') as trace:
    trace.write(json.dumps(['cargo', *sys.argv[1:]]) + '\\n')
if os.environ.get('CARGO_FAIL'):
    sys.exit(17)
binary = Path('target/x86_64-unknown-linux-gnu/release/werk')
binary.parent.mkdir(parents=True, exist_ok=True)
binary.write_text('fresh binary fixture')
""")
        cargo.chmod(0o755)
        (scripts / "generate-cli-reference.py").write_text("""import json, os, sys
from pathlib import Path
assert Path(sys.argv[1]).read_text() == 'fresh binary fixture'
with open(os.environ['BUILD_TRACE'], 'a') as trace:
    trace.write(json.dumps(['reference', *sys.argv[1:]]) + '\\n')
output = Path(sys.argv[sys.argv.index('--output-dir') + 1])
assert output.resolve() != Path('docs/reference').resolve()
(output / 'cli-options.md').write_text('noncanonical discovery')
if os.environ.get('REFERENCE_FAIL'):
    sys.exit(23)
Path('README.md').write_text('reference generated before staging')
""")
        self.env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", BUILD_TRACE=str(self.trace))
        self.archive = self.repo / "releases/werk1112-v1.2.3-linux-x86_64.tar.gz"
        self.archive.parent.mkdir()
        self.archive.write_bytes(b"previous release")
        self.staged = self.repo / "target/package/linux/previous"
        self.staged.parent.mkdir(parents=True)
        self.staged.write_text("previous staging")

    def run_package(self, **env):
        return subprocess.run(["bash", str(self.repo / "scripts/package-release.sh"), "linux"],
                              cwd=self.repo.parent, env=dict(self.env, **env),
                              text=True, capture_output=True, timeout=30)

    def test_generation_runs_after_build_before_packaging(self):
        result = self.run_package()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.trace.read_text().splitlines()]
        self.assertEqual(calls[0], ["cargo", "build-linux"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][:-1],
                         ["reference", "target/x86_64-unknown-linux-gnu/release/werk",
                          "--target", HOST, "--profile", "discovery", "--output-dir"])
        self.assertFalse(Path(calls[1][-1]).exists())
        self.assertEqual(self.canonical.read_text(), "canonical reference fixture\n")
        with tarfile.open(self.archive) as archive:
            self.assertEqual(archive.extractfile("README.md").read(),
                             b"reference generated before staging")
        self.assertTrue(Path(str(self.archive) + ".sha256").is_file())

    def test_build_failure_does_not_generate_or_replace_archive(self):
        result = self.run_package(CARGO_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        calls = [json.loads(line) for line in self.trace.read_text().splitlines()]
        self.assertEqual(calls, [["cargo", "build-linux"]])
        self.assertEqual(self.archive.read_bytes(), b"previous release")
        self.assertTrue(self.staged.exists())

    def test_generation_failure_preserves_archive_and_staging(self):
        result = self.run_package(REFERENCE_FAIL="1")
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.trace.read_text().splitlines()]
        self.assertFalse(Path(calls[-1][-1]).exists())
        self.assertEqual(self.canonical.read_text(), "canonical reference fixture\n")
        self.assertEqual(self.archive.read_bytes(), b"previous release")
        self.assertTrue(self.staged.exists())


if __name__ == "__main__":
    unittest.main()
