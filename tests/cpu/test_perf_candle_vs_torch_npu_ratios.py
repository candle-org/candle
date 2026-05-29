from benchmarks.perf_candle_vs_torch_npu import _annotate_ratios, _ratio_failures


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
