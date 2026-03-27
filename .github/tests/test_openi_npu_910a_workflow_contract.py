import pathlib

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/openi-npu-910a.yml"

REQUIRED_INPUTS = [
    "repo_url",
    "ref",
    "image_id",
    "image_name",
    "spec_id",
    "cluster",
    "compute_source",
    "has_internet",
    "timeout_minutes",
    "reuse_timeout_minutes",
    "keep_task_on_failure",
]

REQUIRED_SUBCOMMANDS = [
    "ensure-task",
    "wait-task",
    "prepare-remote",
    "run-910a-suite",
    "fetch-artifacts",
    "cleanup-task",
]


def _workflow_on(workflow):
    return workflow.get("on") or workflow.get(True) or {}


@pytest.fixture(scope="module")
def workflow():
    assert WORKFLOW_PATH.exists(), f"Expected workflow file at {WORKFLOW_PATH}"
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_workflow_file_exists():
    assert WORKFLOW_PATH.exists()


def test_workflow_dispatch_enabled(workflow):
    assert "workflow_dispatch" in _workflow_on(workflow)


def test_workflow_dispatch_inputs_present(workflow):
    inputs = (_workflow_on(workflow).get("workflow_dispatch") or {}).get("inputs") or {}
    missing = [name for name in REQUIRED_INPUTS if name not in inputs]
    assert not missing, f"Missing workflow_dispatch inputs: {missing}"


def test_workflow_defaults_to_4card_910a_spec(workflow):
    inputs = (_workflow_on(workflow).get("workflow_dispatch") or {}).get("inputs") or {}
    assert inputs.get("spec_id", {}).get("default") == "340"


def test_at_least_one_job_runs_on_ubuntu_latest(workflow):
    jobs = workflow.get("jobs") or {}
    assert jobs, "Workflow must define at least one job"
    for job in jobs.values():
        runs_on = job.get("runs-on")
        if runs_on == "ubuntu-latest":
            return
        if isinstance(runs_on, list) and "ubuntu-latest" in runs_on:
            return
    pytest.fail("Expected at least one job to run on ubuntu-latest")


def test_job_timeout_minutes_uses_workflow_input(workflow):
    jobs = workflow.get("jobs") or {}
    job = jobs.get("openi-910a") or {}
    assert job.get("timeout-minutes") == "${{ fromJSON(inputs.timeout_minutes) }}"


def test_workflow_has_individual_openi_ci_steps(workflow):
    jobs = workflow.get("jobs") or {}
    combined_runs = "\n".join(
        step.get("run", "")
        for job in jobs.values()
        for step in job.get("steps", [])
    )
    missing = [
        subcommand
        for subcommand in REQUIRED_SUBCOMMANDS
        if f"python .github/scripts/openi_ci.py {subcommand}" not in combined_runs
    ]
    assert not missing, f"Missing openi_ci.py subcommands: {missing}"
    assert "python .github/scripts/openi_ci.py create-task" not in combined_runs


def test_workflow_passes_reuse_timeout_to_ensure_task(workflow):
    jobs = workflow.get("jobs") or {}
    combined_runs = "\n".join(
        step.get("run", "")
        for job in jobs.values()
        for step in job.get("steps", [])
    )
    assert '--reuse-timeout-minutes "${{ inputs.reuse_timeout_minutes }}"' in combined_runs


def test_workflow_uploads_openi_artifacts(workflow):
    jobs = workflow.get("jobs") or {}
    for job in jobs.values():
        for step in job.get("steps", []):
            if "upload-artifact" not in step.get("uses", ""):
                continue
            path = (step.get("with") or {}).get("path", "")
            if ".artifacts/openi-910a/" in str(path) or ".artifacts/openi-910a" in str(path):
                return
    pytest.fail("Expected an upload-artifact step for .artifacts/openi-910a/")


def test_cleanup_step_always_runs(workflow):
    jobs = workflow.get("jobs") or {}
    for job in jobs.values():
        for step in job.get("steps", []):
            run = step.get("run", "")
            if "python .github/scripts/openi_ci.py cleanup-task" not in run:
                continue
            assert step.get("if") == "always()"
            return
    pytest.fail("Expected cleanup-task step guarded by if: always()")
