#!/usr/bin/env bash
# .github/scripts/ci-failfast-watcher.sh
#
# Background watcher for self-hosted runners: polls GitHub API every 15s to
# check whether any sibling job in the same workflow run has a REAL test/lint
# failure.  If so, kills the foreground process group and exits 1.
#
# Infrastructure failures (checkout, install, setup) are ignored — those
# should be retried by ci-auto-retry.yaml, not used to cancel healthy jobs.
#
# Usage:  ci-failfast-watcher.sh <command> [args...]
#
# Requires GH_TOKEN, GITHUB_REPOSITORY, GITHUB_RUN_ID in environment.
set -euo pipefail

check_sibling_failed() {
  python3 - <<'PY'
import json, os, re, sys, urllib.request

token = os.environ["GH_TOKEN"]
repo  = os.environ["GITHUB_REPOSITORY"]
run_id = os.environ["GITHUB_RUN_ID"]

# Conclusions that indicate a job finished badly
bad = {"failure", "timed_out"}

# Step name patterns for REAL test/lint failures.
# Only these warrant killing sibling jobs.
TEST_STEP_RE = re.compile(
    r"Run .* tests|Run .* coverage|Run .* suite|"
    r"Lint with pylint|Assert NPU|Verify MPS|"
    r"Run .*card HCCL",
    re.IGNORECASE,
)

req = urllib.request.Request(
    f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
    headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    },
)
try:
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.load(resp)
except Exception:
    sys.exit(1)  # network error → don't kill

for job in data.get("jobs", []):
    if job.get("conclusion") not in bad:
        continue
    # Check which steps actually failed
    failed_steps = [
        s.get("name", "")
        for s in job.get("steps", [])
        if s.get("conclusion") == "failure"
    ]
    # Job failed with no completed steps (runner allocation) → infra, skip
    if not failed_steps:
        continue
    # Only kill if a real test/lint step failed
    if any(TEST_STEP_RE.search(name) for name in failed_steps):
        print(f"[failfast] sibling real failure: {job.get('name')} "
              f"(steps: {failed_steps})", file=sys.stderr)
        sys.exit(0)
    # Infra failure (checkout, install, etc.) → ignore
    print(f"[failfast] sibling infra failure (ignored): {job.get('name')} "
          f"(steps: {failed_steps})", file=sys.stderr)

sys.exit(1)
PY
}

# Run the real command in its own process group so we can kill the whole tree.
setsid "$@" &
cmd_pid=$!

while kill -0 "$cmd_pid" 2>/dev/null; do
  if check_sibling_failed; then
    echo "[failfast] Killing current step (PID $cmd_pid) due to sibling failure" >&2
    kill -TERM -- "-$cmd_pid" 2>/dev/null || true
    sleep 3
    kill -KILL -- "-$cmd_pid" 2>/dev/null || true
    wait "$cmd_pid" 2>/dev/null || true
    exit 1
  fi
  sleep 15
done

wait "$cmd_pid"
