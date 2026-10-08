"""Exercise the release workflow's shell steps against disposable Git remotes."""

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/release.yml").read_text()


def step_script(name):
    step = WORKFLOW.split(f"      - name: {name}\n", 1)[1].split("\n      - name:", 1)[0]
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


class ReleaseWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.remote = self.root / "origin.git"
        self.git("init", "--bare", str(self.remote))
        self.git("init", "-b", "main")
        self.git("config", "user.name", "Release test")
        self.git("config", "user.email", "release@example.invalid")
        (self.repo / "version").write_text("1.0.0\n")
        self.git("add", ".")
        self.git("commit", "-m", "Initial product")
        self.base = self.git("rev-parse", "HEAD")
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "origin", "main")
        (self.repo / "version").write_text("1.0.1\n")
        self.branch = f"release/auto-v1.0.1-{self.base[:12]}"
        self.output = self.root / "output"
        self.calls = self.root / "gh-calls"
        executable = self.root / "gh"
        executable.write_text("""#!/bin/bash
set -eu
echo "$*" >> "$GH_CALLS"
case "$1 $2" in
  'pr list') ;;
  'pr create') echo 'https://example.invalid/owner/repo/pull/1' ;;
  'pr view') echo 1 ;;
  *) echo "Unexpected gh command: $*" >&2; exit 1 ;;
esac
""")
        executable.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{self.root}:{os.environ['PATH']}",
                        GH_CALLS=str(self.calls), DEFAULT_BRANCH="main",
                        RELEASE_TAG="v1.0.1", RUNNER_TEMP=str(self.root),
                        GITHUB_OUTPUT=str(self.output))

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo,
                                       text=True, stderr=subprocess.PIPE).strip()

    def run_step(self, name, **env):
        return subprocess.run(["bash", "-e", "-o", "pipefail", "-c", step_script(name)],
                              cwd=self.repo, env=dict(self.env, **env),
                              text=True, capture_output=True)

    def prepare(self):
        result = self.run_step("Create the release preparation PR")
        self.assertEqual(result.returncode, 0, result.stderr)
        return self.git("rev-parse", "HEAD")

    def test_generated_commit_skips_duplicate_ci_and_emits_exact_head(self):
        head = self.prepare()
        self.assertEqual(self.git("log", "-1", "--format=%s"),
                         "release(): v1.0.1 [skip ci]")
        self.assertEqual(self.git("rev-parse", f"origin/{self.branch}"), head)
        self.assertEqual(self.git("rev-parse", "origin/main"), self.base)
        self.assertEqual(self.output.read_text(), f"number=1\nhead={head}\nbase={self.base}\n")
        self.assertIn("pr create", self.calls.read_text())

    def test_retry_reuses_identical_prepared_commit(self):
        head = self.prepare()
        self.git("reset", "--hard", self.base)
        (self.repo / "version").write_text("1.0.1\n")
        self.assertEqual(self.prepare(), head)

    def test_retry_rejects_different_release_branch_tree(self):
        self.prepare()
        self.git("reset", "--hard", self.base)
        (self.repo / "version").write_text("different preparation\n")
        self.calls.write_text("")
        result = self.run_step("Create the release preparation PR")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Existing release branch differs", result.stdout)
        self.assertEqual(self.calls.read_text(), "")

    def test_changed_default_branch_blocks_merge(self):
        head = self.prepare()
        self.git("push", "origin", "HEAD:main")
        self.calls.write_text("")
        result = self.run_step("Merge the release preparation PR", RELEASE_PR="1",
                               RELEASE_SHA=head, BASE_SHA=self.base)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Default branch advanced", result.stdout)
        self.assertEqual(self.calls.read_text(), "")

    def test_release_requires_no_extra_credentials_and_keeps_merge_guards(self):
        self.assertIn("GH_TOKEN: ${{ github.token }}", WORKFLOW)
        self.assertNotIn("secrets.", WORKFLOW)
        self.assertNotIn("create-github-app-token", WORKFLOW)
        self.assertNotIn("wait-release-checks", WORKFLOW)
        self.assertNotIn("continue-on-error", WORKFLOW)
        merge = step_script("Merge the release preparation PR")
        self.assertNotIn("--admin", merge.split("gh pr merge", 1)[1])
        self.assertIn('--match-head-commit "$RELEASE_SHA"', merge)
        self.assertIn('test "$(git rev-parse HEAD^1)" = "$BASE_SHA"', merge)
        self.assertIn('test "$(git rev-parse HEAD^2)" = "$RELEASE_SHA"', merge)
        self.assertIn('test "$(git rev-parse \'HEAD^{tree}\')"', merge)
        self.assertIn('args+=(--draft)', step_script("Create GitHub release"))
        self.assertIn('python scripts/prepare-release.py check',
                      step_script("Validate and tag the merged release"))


if __name__ == "__main__":
    unittest.main()
