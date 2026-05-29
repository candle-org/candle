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
