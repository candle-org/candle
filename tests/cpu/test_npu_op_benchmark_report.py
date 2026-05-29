import json
import subprocess
import sys

import pytest

from benchmarks.op_benchmark_npu import run as op_run
from benchmarks.op_benchmark_npu.report import generate_report, ratio_failures
from benchmarks.op_benchmark_npu.run import _output_stream


def test_generate_report_includes_distribution_columns():
    candle = [{
        "framework": "candle",
        "op": "add",
        "mode": "fwd",
        "dtype": "fp16",
        "scenario": "infer",
        "median_ms": 2.0,
        "p10_ms": 1.5,
        "p90_ms": 2.5,
        "status": "ok",
        "median_ratio": 2.0,
    }]
    torch_ref = [{
        "framework": "torch_npu",
        "op": "add",
        "mode": "fwd",
        "dtype": "fp16",
        "scenario": "infer",
        "median_ms": 1.0,
        "p10_ms": 0.8,
        "p90_ms": 1.2,
        "status": "ok",
        "median_ratio": 1.0,
    }]

    report = generate_report(candle, torch_ref, ["add"], ["fp16"], ["infer"], ["fwd"])

    assert "| Op | candle median | torch_npu median | ratio | candle p10/p90 | torch_npu p10/p90 | impact |" in report
    assert "| add | 2.0000 | 1.0000 | 2.00x | 1.5000/2.5000 | 0.8000/1.2000 | 1.0000 |" in report


def test_ratio_failures_reports_slow_op():
    candle = [{
        "framework": "candle",
        "op": "add",
        "mode": "fwd",
        "dtype": "fp16",
        "scenario": "infer",
        "median_ms": 2.0,
        "status": "ok",
        "median_ratio": 2.0,
    }]
    torch_ref = [{
        "framework": "torch_npu",
        "op": "add",
        "mode": "fwd",
        "dtype": "fp16",
        "scenario": "infer",
        "median_ms": 1.0,
        "status": "ok",
        "median_ratio": 1.0,
    }]

    failures = ratio_failures(
        candle,
        torch_ref,
        op_names=["add"],
        dtype_keys=["fp16"],
        scen_keys=["infer"],
        mode_keys=["fwd"],
        max_ratio=1.0,
    )
    assert failures == ["add/fwd/fp16/infer: candle median_ratio 2.00x > 1.00x"]


def test_generate_report_does_not_mutate_input_rows():
    candle = [{
        "framework": "candle",
        "op": "add",
        "mode": "fwd",
        "dtype": "fp16",
        "scenario": "infer",
        "median_ms": 2.0,
        "status": "ok",
    }]
    torch_ref = [{
        "framework": "torch_npu",
        "op": "add",
        "mode": "fwd",
        "dtype": "fp16",
        "scenario": "infer",
        "median_ms": 1.0,
        "status": "ok",
    }]

    generate_report(candle, torch_ref, ["add"], ["fp16"], ["infer"], ["fwd"])

    assert "median_ratio" not in candle[0]
    assert "median_ratio" not in torch_ref[0]


def test_ratio_failures_does_not_mutate_input_rows():
    candle = [{
        "framework": "candle",
        "op": "add",
        "mode": "fwd",
        "dtype": "fp16",
        "scenario": "infer",
        "median_ms": 2.0,
        "status": "ok",
    }]
    torch_ref = [{
        "framework": "torch_npu",
        "op": "add",
        "mode": "fwd",
        "dtype": "fp16",
        "scenario": "infer",
        "median_ms": 1.0,
        "status": "ok",
    }]

    ratio_failures(
        candle,
        torch_ref,
        op_names=["add"],
        dtype_keys=["fp16"],
        scen_keys=["infer"],
        mode_keys=["fwd"],
        max_ratio=1.0,
    )

    assert "median_ratio" not in candle[0]
    assert "median_ratio" not in torch_ref[0]

def test_output_stream_routes_human_output_to_stderr_for_json_stdout():
    assert _output_stream("-") == "stderr"
    assert _output_stream(None) == "stdout"
    assert _output_stream("results.json") == "stdout"



def _op_args():
    class Args:
        warmup = 1
        iters = 1
        ops = "add"
        scenario = "infer"
        dtype = "fp16"
        mode = "fwd"

    return Args()


def test_run_worker_returns_structured_failure_for_nonzero_exit(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 2, stdout="", stderr="boom")

    monkeypatch.setattr(op_run.subprocess, "run", fake_run)

    rows = op_run._run_worker("candle", _op_args())

    assert rows
    for row in rows:
        assert row["framework"] == "candle"
        assert row["op"] == "add"
        assert row["mode"] == "fwd"
        assert row["dtype"] == "fp16"
        assert row["scenario"] == "infer"
        assert "worker exit 2" in row["status"]
        assert row["mean_ms"] == 0.0
        assert row["median_ms"] == 0.0
        assert row["p10_ms"] == 0.0
        assert row["p90_ms"] == 0.0

    generate_report(rows, [], ["add"], ["fp16"], ["infer"], ["fwd"])


def test_run_worker_returns_structured_failure_for_malformed_json(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="not-json", stderr="worker warning")

    monkeypatch.setattr(op_run.subprocess, "run", fake_run)

    rows = op_run._run_worker("torch", _op_args())

    assert rows[0]["framework"] == "torch_npu"
    assert rows[0]["status"].startswith("error: malformed worker JSON:")
    assert rows[0]["stderr"] == "worker warning"
    assert rows[0]["stdout"] == "not-json"


def test_main_exits_nonzero_for_worker_failure_without_ratio_gate(capsys, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "op_benchmark_npu.run",
            "--ops",
            "add",
            "--scenario",
            "infer",
            "--dtype",
            "fp16",
            "--mode",
            "fwd",
            "--json-output",
            "-",
        ],
    )
    monkeypatch.setattr(
        op_run,
        "_run_worker",
        lambda framework, args: [
            {
                "framework": "candle" if framework == "candle" else "torch_npu",
                "op": "add",
                "mode": "fwd",
                "dtype": "fp16",
                "scenario": "infer",
                "median_ms": 0.0,
                "status": "error: worker exit 4" if framework == "candle" else "ok",
            }
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        op_run.main()

    assert exc_info.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["failures"] == ["add/fwd/fp16/infer/candle: error: worker exit 4"]
