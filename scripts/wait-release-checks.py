#!/usr/bin/env python3
"""Wait for real PR validation of a release head, without merging or publishing."""

import argparse
import json
import os
import re
import subprocess
import time


# Release preparation changes Cargo and both integration packages, so all three
# path-filtered workflows must appear. An empty check list is never success.
REQUIRED_WORKFLOWS = frozenset({
    ".github/workflows/observability.yml",
    ".github/workflows/n8n.yml",
    ".github/workflows/comfyui-registry.yml",
})


def evaluate_runs(runs, sha):
    """Return pending reasons; reject failures of the latest run of each workflow."""
    latest = {}
    for run in runs:
        if run.get("head_sha") != sha or run.get("event") != "pull_request":
            continue
        path = run["path"].split("@", 1)[0]
        previous = latest.get(path)
        # Reruns share an ID; newer run attempts supersede earlier attempts.
        key = (run["id"], run.get("run_attempt", 1))
        if previous is None or key > (previous["id"], previous.get("run_attempt", 1)):
            latest[path] = run
    pending = [f"not started: {path}" for path in sorted(REQUIRED_WORKFLOWS - latest.keys())]
    for path, run in sorted(latest.items()):
        status = run.get("status")
        conclusion = run.get("conclusion")
        if status in {"action_required", "waiting"} or conclusion == "action_required":
            raise RuntimeError(f"{path} requires approval: {run.get('html_url', '')}. "
                               "The release PR must be created/reopened with the Release App token.")
        if status != "completed":
            pending.append(f"{status}: {path}")
        elif conclusion != "success":
            raise RuntimeError(f"{path}: {conclusion}: {run.get('html_url', '')}. "
                               "Fix the check and rerun it before retrying Release.")
    return pending


def gh_json_lines(*args):
    result = subprocess.run(
        ["gh", *args], check=True, text=True, capture_output=True, timeout=90,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def wait_for_checks(repo, pr, sha, timeout=2700, interval=15, *,
                    query=gh_json_lines, clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + timeout
    last_message = None
    while True:
        data = query("api", f"repos/{repo}/pulls/{pr}", "--jq", "@json")[0]
        if data["head"]["sha"] != sha:
            raise RuntimeError("Release PR head changed during validation; restart Release.")
        if data["state"] != "open" and not data.get("merged"):
            raise RuntimeError("Release PR was closed without merging; restart Release.")
        runs = query("api", f"repos/{repo}/actions/runs?event=pull_request&head_sha={sha}&per_page=100",
                     "--paginate", "--jq", ".workflow_runs[] | @json")
        pending = evaluate_runs(runs, sha)
        # Also honor non-Actions checks/statuses exposed by the PR. Repository
        # branch rules are enforced again by gh pr merge (no --admin bypass).
        rollup = query("pr", "view", str(pr), "--repo", repo,
                       "--json", "statusCheckRollup", "--jq", "@json")[0]
        for check in rollup.get("statusCheckRollup") or []:
            name = check.get("name") or check.get("context") or "PR check"
            if "status" in check:
                if check["status"] != "COMPLETED":
                    pending.append(f"pending: {name}")
                elif check.get("conclusion") not in {"SUCCESS", "NEUTRAL", "SKIPPED"}:
                    raise RuntimeError(f"{name}: {check.get('conclusion')}")
            elif check.get("state") == "PENDING":
                pending.append(f"pending: {name}")
            elif check.get("state") != "SUCCESS":
                raise RuntimeError(f"{name}: {check.get('state')}")
        if not pending:
            print(f"Release PR #{pr}: all validation passed for {sha}.", flush=True)
            return
        message = "; ".join(pending)
        if message != last_message:
            print(f"Waiting for release PR #{pr}: {message}", flush=True)
            last_message = message
        if clock() >= deadline:
            raise RuntimeError(f"Timed out waiting for release validation: {message}")
        sleep(min(interval, max(0, deadline - clock())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr", type=int)
    parser.add_argument("sha")
    parser.add_argument("--repo", default=os.environ.get("GH_REPO"))
    parser.add_argument("--timeout", type=int, default=2700)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        parser.error("sha must be a full commit SHA")
    if not args.repo or not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
        parser.error("--repo or GH_REPO must identify owner/repository")
    if args.pr <= 0 or args.timeout <= 0:
        parser.error("pr and timeout must be positive")
    try:
        wait_for_checks(args.repo, args.pr, args.sha, timeout=args.timeout)
    except (RuntimeError, subprocess.SubprocessError, ValueError, KeyError) as error:
        parser.exit(1, f"Release validation blocked: {error}\n")


if __name__ == "__main__":
    main()
