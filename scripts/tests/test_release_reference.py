"""Exercise release reference generation without Git mutations or publishing."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/release.yml").read_text()
REFERENCE_STEP = "Build and refresh the CLI reference"


def step_script(name):
    step = WORKFLOW.split(f"      - name: {name}\n", 1)[1].split("\n      - name:", 1)[0]
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


class ReleaseReferenceOrderTests(unittest.TestCase):
    def test_build_follows_version_preparation_and_precedes_release_pr(self):
        names = ["Prepare versions, README, documentation and release notes",
                 REFERENCE_STEP, "Create the release preparation PR",
                 "Merge the release preparation PR", "Validate and tag the merged release"]
        positions = [WORKFLOW.index(f"      - name: {name}\n") for name in names]
        self.assertEqual(positions, sorted(positions))

    def test_reference_check_is_required_before_any_tag_mutation(self):
        script = step_script("Validate and tag the merged release")
        check = "python scripts/generate-cli-reference.py target/cli-reference-neutral/debug/werk --check"
        self.assertTrue(script.startswith(check + "\n"))
        self.assertLess(script.index(check), script.index("git tag -a"))
        self.assertLess(script.index(check), script.index("git push"))


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires POSIX bash")
class ReleaseReferenceStepTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="werk-release-reference-")
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        (self.repo / "Cargo.toml").write_text('version = "1.2.4"\n')
        scripts = self.repo / "scripts"
        scripts.mkdir()
        self.reference = self.repo / "reference.md"
        self.reference.write_text("Generated for werk 1.2.3\n")
        self.trace = self.repo / "trace.jsonl"
        bin_dir = self.repo / "fake-bin"
        bin_dir.mkdir()

        def executable(name, source):
            path = bin_dir / name
            path.write_text(source)
            path.chmod(0o755)

        executable("python", f"#!/bin/sh\nexec {shlex.quote(sys.executable)} \"$@\"\n")
        executable("cargo", """#!/usr/bin/env python
import json, os, re, sys
from pathlib import Path
with open('trace.jsonl', 'a') as trace:
    trace.write(json.dumps(['cargo', *sys.argv[1:]]) + '\\n')
if os.environ.get('REFERENCE_BUILD_FAIL'):
    sys.exit(17)
version = re.search(r'version = "([^"]+)"', Path('Cargo.toml').read_text())[1]
binary = Path(os.environ['CARGO_TARGET_DIR']) / 'debug/werk'
binary.parent.mkdir(parents=True, exist_ok=True)
binary.write_text('werk ' + version)
""")
        executable("git", """#!/usr/bin/env python
import os, sys
assert sys.argv[1:] == ['show-ref', '--verify', '--quiet', 'refs/tags/v1.2.4'], sys.argv
sys.exit(0 if os.environ.get('REFERENCE_TAG_EXISTS') else 1)
""")
        (scripts / "generate-cli-reference.py").write_text("""import json, os, sys
from pathlib import Path
with open('trace.jsonl', 'a') as trace:
    trace.write(json.dumps(['reference', *sys.argv[1:]]) + '\\n')
if os.environ.get('REFERENCE_GENERATION_FAIL'):
    sys.exit(23)
expected = 'Generated for ' + Path(sys.argv[1]).read_text() + '\\n'
reference = Path('reference.md')
if '--check' in sys.argv:
    sys.exit(0 if reference.read_text() == expected else 1)
reference.write_text(expected)
""")
        self.env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", RELEASE_TAG="v1.2.4",
                        CARGO_TARGET_DIR="target/cli-reference-neutral")

    def run_step(self, **env):
        return subprocess.run(["bash", "-e", "-o", "pipefail", "-c", step_script(REFERENCE_STEP)],
                              cwd=self.repo, env=dict(self.env, **env), capture_output=True,
                              text=True, timeout=30)

    def calls(self):
        return [json.loads(line) for line in self.trace.read_text().splitlines()]

    def test_new_release_regenerates_using_the_prepared_version(self):
        result = self.run_step()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.reference.read_text(), "Generated for werk 1.2.4\n")
        self.assertEqual(self.calls(), [
            ["cargo", "+stable", "build", "--locked", "--bin", "werk", "--no-default-features"],
            ["reference", "target/cli-reference-neutral/debug/werk"],
            ["reference", "target/cli-reference-neutral/debug/werk", "--check"],
        ])

    def test_build_failure_leaves_reference_untouched(self):
        before = self.reference.read_bytes()
        result = self.run_step(REFERENCE_BUILD_FAIL="1")
        self.assertEqual(result.returncode, 17)
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(self.reference.read_bytes(), before)

    def test_generation_failure_fails_the_release_step(self):
        result = self.run_step(REFERENCE_GENERATION_FAIL="1")
        self.assertEqual(result.returncode, 23)

    def test_existing_tag_with_stale_reference_fails_without_rewriting(self):
        before = self.reference.read_bytes()
        result = self.run_step(REFERENCE_TAG_EXISTS="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.reference.read_bytes(), before)
        self.assertEqual(self.calls()[-1], ["reference", "target/cli-reference-neutral/debug/werk", "--check"])

    def test_existing_tag_with_current_reference_passes_without_rewriting(self):
        self.reference.write_text("Generated for werk 1.2.4\n")
        before = self.reference.stat().st_mtime_ns
        result = self.run_step(REFERENCE_TAG_EXISTS="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.reference.stat().st_mtime_ns, before)
        self.assertEqual(self.calls()[-1], ["reference", "target/cli-reference-neutral/debug/werk", "--check"])


if __name__ == "__main__":
    unittest.main()
