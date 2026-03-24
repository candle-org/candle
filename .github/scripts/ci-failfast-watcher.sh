#!/usr/bin/env bash
# .github/scripts/ci-failfast-watcher.sh
#
# Background watcher for self-hosted runners: polls GitHub API every 15s to
# check whether any sibling job in the same workflow run has failed.  If so,
# kills the foreground process group and exits 1.
#
# Usage:  ci-failfast-watcher.sh <command> [args...]
#
# Requires GH_TOKEN, GITHUB_REPOSITORY, GITHUB_RUN_ID in environment.
set -euo pipefail

check_sibling_failed() {
  python3 - <<'PY'
import json, os, sys, urllib.request
token = os.environ["GH_TOKEN"]
repo  = os.environ["GITHUB_REPOSITORY"]
run_id = os.environ["GITHUB_RUN_ID"]
bad = {"failure", "cancelled", "timed_out", "startup_failure"}
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
    if job.get("conclusion") in bad:
        print(f"[failfast] sibling failed: {job.get('name')}", file=sys.stderr)
        sys.exit(0)
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
