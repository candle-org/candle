import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks.pipeline_npu import run as pipeline_run
from benchmarks.pipeline_npu import worker as pipeline_worker
from benchmarks.pipeline_npu.run import annotate_pipeline_ratios, pipeline_ratio_failures


def test_annotate_pipeline_ratios_adds_candle_ratio():
    rows = [
        {"framework": "torch_npu", "case_id": "A1", "mode": "eager", "median_ms": 10.0, "status": "ok"},
        {"framework": "candle", "case_id": "A1", "mode": "eager", "median_ms": 8.0, "status": "ok"},
    ]

    annotate_pipeline_ratios(rows)

    candle = next(row for row in rows if row["framework"] == "candle")
    assert candle["median_ratio"] == 0.8


def test_pipeline_ratio_failures_require_candle_to_be_faster_than_torch_npu():
    rows = [
        {"framework": "torch_npu", "case_id": "A1", "mode": "eager", "median_ms": 10.0, "status": "ok", "median_ratio": 1.0},
        {"framework": "candle", "case_id": "A1", "mode": "eager", "median_ms": 10.5, "status": "ok", "median_ratio": 1.05},
    ]

    failures = pipeline_ratio_failures(rows, case_ids=["A1"], mode="eager", max_ratio=0.99)

    assert failures == ["A1/eager: candle median_ratio 1.05x > 0.99x"]


def test_worker_unknown_case_emits_structured_error_row(capsys, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker",
            "--framework",
            "candle",
            "--cases",
            "BAD",
            "--mode",
            "eager",
            "--warmup",
            "0",
            "--iters",
            "1",
        ],
    )

    pipeline_worker.main()

    rows = json.loads(capsys.readouterr().out)
    assert rows == [
        {
            "framework": "candle",
            "case_id": "BAD",
            "mode": "eager",
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "op_count": 0,
            "status": "error: unknown case: BAD",
        }
    ]


def test_spawn_worker_runs_from_repo_root_with_repo_and_src_on_pythonpath(monkeypatch):
    captured = {}

    def fake_run(cmd, capture_output, text, env, check, cwd):
        captured["cmd"] = cmd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["env"] = env
        captured["check"] = check
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps([{"framework": "candle", "case_id": "A1"}]), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = SimpleNamespace(
        python=sys.executable,
        cases="A1",
        mode="eager",
        device="npu",
        warmup=0,
        iters=1,
    )

    rows = pipeline_run._spawn_worker("candle", args)

    repo_root = pipeline_run._repo_root()
    pythonpath = captured["env"]["PYTHONPATH"].split(os.pathsep)
    assert rows == [{"framework": "candle", "case_id": "A1"}]
    assert captured["cwd"] == repo_root
    assert pythonpath[0:2] == [repo_root, os.path.join(repo_root, "src")]


def test_spawn_worker_nonzero_exit_raises(monkeypatch):
    def fake_run(cmd, capture_output, text, env, check, cwd):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = SimpleNamespace(
        python=sys.executable,
        cases="A1",
        mode="eager",
        device="npu",
        warmup=0,
        iters=1,
    )

    with pytest.raises(RuntimeError, match="candle worker failed"):
        pipeline_run._spawn_worker("candle", args)


def test_spawn_worker_malformed_json_raises(monkeypatch):
    def fake_run(cmd, capture_output, text, env, check, cwd):
        return subprocess.CompletedProcess(cmd, 0, stdout="not-json", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = SimpleNamespace(
        python=sys.executable,
        cases="A1",
        mode="eager",
        device="npu",
        warmup=0,
        iters=1,
    )

    with pytest.raises(RuntimeError, match="failed to parse candle worker JSON"):
        pipeline_run._spawn_worker("candle", args)
