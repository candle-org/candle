import json
import sys

import pytest

import benchmarks.perf_candle_vs_torch_npu as _bench_mod
from benchmarks.perf_candle_vs_torch_npu import _annotate_ratios, _ratio_failures, _spawn_worker


def test_annotate_ratios_adds_total_ratio():
    results = [
        {
            "framework": "torch_npu",
            "case": "xfmr",
            "fwd_ms_median": 2.0,
            "bwd_ms_median": 3.0,
            "total_ms_median": 5.0,
        },
        {
            "framework": "candle",
            "case": "xfmr",
            "fwd_ms_median": 1.5,
            "bwd_ms_median": 2.0,
            "total_ms_median": 3.5,
        },
    ]

    _annotate_ratios(results)

    candle = next(row for row in results if row["framework"] == "candle")
    assert candle["fwd_ratio"] == 0.75
    assert candle["bwd_ratio"] == 2.0 / 3.0
    assert candle["total_ratio"] == 0.7


def test_ratio_failures_checks_total_ratio():
    results = [
        {
            "framework": "torch_npu",
            "case": "xfmr",
            "fwd_ms_median": 2.0,
            "bwd_ms_median": 3.0,
            "total_ms_median": 5.0,
        },
        {
            "framework": "candle",
            "case": "xfmr",
            "fwd_ms_median": 2.0,
            "bwd_ms_median": 3.0,
            "total_ms_median": 5.2,
            "fwd_ratio": 1.0,
            "bwd_ratio": 1.0,
            "total_ratio": 1.04,
        },
    ]

    failures = _ratio_failures(
        results,
        cases=["xfmr"],
        max_fwd_ratio=1.0,
        max_bwd_ratio=1.0,
        max_total_ratio=0.99,
    )

    assert failures == ["xfmr: total ratio 1.04x > 0.99x"]


def test_spawn_worker_normalizes_error_json(monkeypatch):
    monkeypatch.setattr(
        _bench_mod.subprocess,
        "check_output",
        lambda *a, **kw: b'{"error": "NPU not available under candle"}\n',
    )
    row = _spawn_worker("candle", "xfmr", iters=1, warmup=1, dtype_name="float16")
    assert row["framework"] == "candle"
    assert row["case"] == "xfmr"
    assert row["error"] == "NPU not available under candle"
    # _annotate_ratios must not raise KeyError
    _annotate_ratios([row])
    # _ratio_failures reports the missing reference row without crashing
    failures = _ratio_failures([row], ["xfmr"], 1.0, 1.0, 0.99)
    assert failures == ["xfmr: missing candle or torch_npu result for ratio gate"]

def test_json_stdout_mode_routes_human_output_to_stderr(capsys, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perf_candle_vs_torch_npu.py",
            "--cases",
            "mlp",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--json-output",
            "-",
        ],
    )
    monkeypatch.setattr(
        _bench_mod,
        "_spawn_worker",
        lambda framework, case, *args, **kwargs: {
            "framework": framework,
            "case": case,
            "fwd_ms_median": 1.0,
            "bwd_ms_median": 1.0,
            "total_ms_median": 2.0,
            "fwd_ms_min": 1.0,
            "bwd_ms_min": 1.0,
            "total_ms_min": 2.0,
            "fwd_ms_p10": 1.0,
            "bwd_ms_p10": 1.0,
            "total_ms_p10": 2.0,
            "fwd_ms_p90": 1.0,
            "bwd_ms_p90": 1.0,
            "total_ms_p90": 2.0,
        },
    )

    _bench_mod.main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["cases"] == ["mlp"]
    assert captured.err
    assert "# Perf bench: candle vs torch_npu" in captured.err
    assert "running..." in captured.err
    assert "algo" in captured.err


def test_spawn_worker_nonzero_exit_returns_error_row_with_tail(monkeypatch):
    def raise_called_process_error(*args, **kwargs):
        raise _bench_mod.subprocess.CalledProcessError(
            3,
            ["python", "worker"],
            output=b"trace line\njson? nope\n",
        )

    monkeypatch.setattr(_bench_mod.subprocess, "check_output", raise_called_process_error)

    row = _spawn_worker("candle", "xfmr", iters=1, warmup=1, dtype_name="float16")

    assert row["framework"] == "candle"
    assert row["case"] == "xfmr"
    assert row["error"] == "worker exit 3"
    assert row["tail"] == ["trace line", "json? nope"]


def test_spawn_worker_malformed_json_returns_error_row(monkeypatch):
    monkeypatch.setattr(
        _bench_mod.subprocess,
        "check_output",
        lambda *a, **kw: b"worker started\n{bad json}\nnot-json\n",
    )

    row = _spawn_worker("candle", "xfmr", iters=1, warmup=1, dtype_name="float16")

    assert row == {
        "framework": "candle",
        "case": "xfmr",
        "error": "no json from worker",
    }


def test_main_exits_nonzero_when_worker_error_occurs_without_ratio_gate(capsys, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perf_candle_vs_torch_npu.py",
            "--cases",
            "mlp",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--json-output",
            "-",
        ],
    )

    def fake_spawn(framework, case, *args, **kwargs):
        if framework == "candle":
            return {"framework": framework, "case": case, "error": "worker exit 7"}
        return {
            "framework": framework,
            "case": case,
            "fwd_ms_median": 1.0,
            "bwd_ms_median": 1.0,
            "total_ms_median": 2.0,
            "fwd_ms_min": 1.0,
            "bwd_ms_min": 1.0,
            "total_ms_min": 2.0,
            "fwd_ms_p10": 1.0,
            "bwd_ms_p10": 1.0,
            "total_ms_p10": 2.0,
            "fwd_ms_p90": 1.0,
            "bwd_ms_p90": 1.0,
            "total_ms_p90": 2.0,
        }

    monkeypatch.setattr(_bench_mod, "_spawn_worker", fake_spawn)

    with pytest.raises(SystemExit) as exc_info:
        _bench_mod.main()

    assert exc_info.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["failures"] == ["mlp/candle: worker exit 7"]
