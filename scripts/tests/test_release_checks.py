"""Release gates must fail closed without publishing anything."""

import importlib.util
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("release_checks", ROOT / "scripts/wait-release-checks.py")
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)
SHA = "a" * 40


def runs(status="completed", conclusion="success"):
    return [dict(id=index, path=path, head_sha=SHA, event="pull_request",
                 status=status, conclusion=conclusion, run_attempt=1,
                 html_url=f"https://example.invalid/runs/{index}")
            for index, path in enumerate(sorted(GATE.REQUIRED_WORKFLOWS), 1)]


class ReleaseCheckTests(unittest.TestCase):
    def test_missing_or_partial_ci_never_passes(self):
        self.assertEqual(len(GATE.evaluate_runs([], SHA)), 3)
        self.assertEqual(len(GATE.evaluate_runs(runs()[:1], SHA)), 2)
        self.assertTrue(GATE.evaluate_runs(runs("queued", None), SHA))
        self.assertEqual(GATE.evaluate_runs(runs(), SHA), [])

    def test_wrong_commit_and_push_success_cannot_validate_release_pr(self):
        wrong_sha = [dict(run, head_sha="b" * 40) for run in runs()]
        push = [dict(run, event="push") for run in runs()]
        self.assertEqual(len(GATE.evaluate_runs(wrong_sha + push, SHA)), 3)

    def test_failure_timeout_cancel_and_skipped_workflows_block(self):
        for conclusion in ["failure", "cancelled", "timed_out", "skipped", "neutral", None]:
            with self.subTest(conclusion=conclusion), self.assertRaises(RuntimeError):
                GATE.evaluate_runs(runs(conclusion=conclusion), SHA)

    def test_approval_is_actionable_and_never_green(self):
        for status, conclusion in [("waiting", None), ("action_required", None),
                                   ("completed", "action_required")]:
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, "App token"):
                GATE.evaluate_runs(runs(status, conclusion), SHA)

    def test_latest_attempt_supersedes_old_result_in_either_api_order(self):
        old = runs(conclusion="failure")
        newer = [dict(run, run_attempt=2) for run in runs()]
        self.assertEqual(GATE.evaluate_runs(old + newer, SHA), [])
        self.assertEqual(GATE.evaluate_runs(newer + old, SHA), [])
        running = [dict(run, id=run["id"] + 100, status="in_progress", conclusion=None)
                   for run in runs()]
        self.assertTrue(GATE.evaluate_runs(runs() + running, SHA))

    def test_additional_failed_workflow_is_not_ignored(self):
        extra = dict(runs()[0], id=100, path=".github/workflows/additional.yml", conclusion="failure")
        with self.assertRaisesRegex(RuntimeError, "additional"):
            GATE.evaluate_runs(runs() + [extra], SHA)

    def query(self, snapshots, *, sha=SHA, state="OPEN", checks=()):
        snapshots = iter(snapshots)

        def query(*args):
            if args[0] == "pr":
                return [dict(statusCheckRollup=list(checks))]
            if "/pulls/" in args[1]:
                return [dict(head=dict(sha=sha), state=state.lower(), merged=state == "MERGED")]
            return next(snapshots)
        return query

    def test_waits_for_delayed_registration_then_completion(self):
        sleeps = []
        GATE.wait_for_checks("owner/repo", 29, SHA,
                             query=self.query([[], runs("in_progress", None), runs()]),
                             clock=lambda: 0, sleep=sleeps.append)
        self.assertEqual(sleeps, [15, 15])

    def test_timeout_with_zero_checks_is_failure(self):
        times = iter([0, 10])
        with self.assertRaisesRegex(RuntimeError, "Timed out"):
            GATE.wait_for_checks("owner/repo", 29, SHA, timeout=1,
                                 query=self.query([[]]), clock=lambda: next(times),
                                 sleep=lambda _: self.fail("must not sleep after timeout"))

    def test_head_change_and_closed_pr_block(self):
        for kwargs in [dict(sha="b" * 40), dict(state="CLOSED")]:
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                GATE.wait_for_checks("owner/repo", 29, SHA, query=self.query([], **kwargs))

    def test_failed_external_status_blocks_even_when_workflows_pass(self):
        for check in [dict(context="external", state="FAILURE"),
                      dict(name="external", status="COMPLETED", conclusion="FAILURE")]:
            with self.subTest(check=check), self.assertRaisesRegex(RuntimeError, "external"):
                GATE.wait_for_checks("owner/repo", 29, SHA,
                                     query=self.query([runs()], checks=[check]))

    def test_merged_prepared_pr_can_be_verified_on_retry(self):
        GATE.wait_for_checks("owner/repo", 29, SHA,
                             query=self.query([runs()], state="MERGED"))

    def test_api_failure_does_not_allow_merge(self):
        def failing_query(*args):
            raise subprocess.CalledProcessError(1, "gh")
        with self.assertRaises(subprocess.CalledProcessError):
            GATE.wait_for_checks("owner/repo", 29, SHA, query=failing_query)

    def test_workflow_has_no_token_fallback_or_merge_bypass(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text()
        create = workflow.index("- name: Create the release preparation PR")
        wait = workflow.index("- name: Wait for release PR validation")
        merge = workflow.index("- name: Merge the validated release PR")
        tag = workflow.index("- name: Validate and tag the merged release")
        self.assertLess(create, wait)
        self.assertLess(wait, merge)
        self.assertLess(merge, tag)
        self.assertIn("GH_TOKEN: ${{ steps.release-app.outputs.token }}", workflow[create:wait])
        self.assertNotIn("|| github.token", workflow)
        self.assertNotIn("gh pr merge --admin", workflow)
        self.assertIn('--match-head-commit "$RELEASE_SHA"', workflow[merge:tag])
        self.assertIn('test "$(git rev-parse HEAD^1)" = "$BASE_SHA"', workflow[merge:tag])
        self.assertNotIn("continue-on-error", workflow)


if __name__ == "__main__":
    unittest.main()
