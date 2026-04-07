from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"


def load_workflow(name):
    return yaml.safe_load((WORKFLOWS_DIR / name).read_text(encoding="utf-8"))


def test_pytorch_compat_workflow_syncs_shared_reference_checkout():
    workflow = load_workflow("pytorch-tests.yaml")
    pr_steps = workflow["jobs"]["pr-gate"]["steps"]
    commands = "\n".join(step.get("run", "") for step in pr_steps)
    assert "python compat/reference/sync.py --source pytorch" in commands
    assert "python compat/reference/scan.py --output compat/reference/reports/source-status.json" in commands
    assert "python compat/reference/diff.py summary" in commands


def test_openi_910b_workflow_emits_layered_reports():
    workflow = load_workflow("openi-910b-pr.yml")
    run_job = workflow["jobs"]["run-910b"]
    assert "openi-910b" in run_job["runs-on"]
    commands = "\n".join(step.get("run", "") for step in run_job["steps"])
    assert "env-summary.json" in commands
    assert "mechanism-regressions.json" in commands
    assert "workload-regressions.json" in commands
    assert "python compat/reference/diff.py npu-report" in commands
