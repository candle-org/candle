from benchmarks.npu_perf_gates import (
    annotate_ratio_rows,
    collect_ratio_failures,
    summarize_samples,
)


def test_summarize_samples_reports_distribution_fields():
    summary = summarize_samples([4.0, 1.0, 3.0, 2.0, 5.0])

    assert summary == {
        "sample_count": 5,
        "mean_ms": 3.0,
        "median_ms": 3.0,
        "min_ms": 1.0,
        "max_ms": 5.0,
        "p10_ms": 1.0,
        "p90_ms": 5.0,
        "p95_ms": 5.0,
    }


def test_summarize_samples_even_count_uses_true_median_and_nearest_rank_percentiles():
    summary = summarize_samples([1.0, 100.0])

    assert summary["median_ms"] == 50.5
    assert summary["p10_ms"] == 1.0
    assert summary["p90_ms"] == 100.0
    assert summary["p95_ms"] == 100.0


def test_summarize_samples_empty_list_returns_zero_distribution():
    summary = summarize_samples([])

    assert summary == {
        "sample_count": 0,
        "mean_ms": 0.0,
        "median_ms": 0.0,
        "min_ms": 0.0,
        "max_ms": 0.0,
        "p10_ms": 0.0,
        "p90_ms": 0.0,
        "p95_ms": 0.0,
    }


def test_annotate_ratio_rows_adds_candle_ratio_against_torch_npu():
    rows = [
        {"framework": "torch_npu", "case": "add", "median_ms": 2.0, "status": "ok"},
        {"framework": "candle", "case": "add", "median_ms": 3.0, "status": "ok"},
    ]

    annotate_ratio_rows(rows, key_fields=("case",), metric="median_ms")

    candle = next(row for row in rows if row["framework"] == "candle")
    torch_ref = next(row for row in rows if row["framework"] == "torch_npu")
    assert candle["median_ratio"] == 1.5
    assert torch_ref["median_ratio"] == 1.0


def test_collect_ratio_failures_inclusive_allows_equal_ratio():
    rows = [
        {"framework": "torch_npu", "case": "add", "median_ms": 1.0, "status": "ok"},
        {"framework": "candle", "case": "add", "median_ms": 1.0, "status": "ok", "median_ratio": 1.0},
    ]

    failures = collect_ratio_failures(
        rows,
        key_fields=("case",),
        expected_keys=[("add",)],
        max_ratio=1.0,
        ratio_field="median_ratio",
        inclusive=True,
    )

    assert failures == []


def test_collect_ratio_failures_strict_rejects_equal_ratio():
    rows = [
        {"framework": "torch_npu", "case": "add", "median_ms": 1.0, "status": "ok"},
        {"framework": "candle", "case": "add", "median_ms": 1.0, "status": "ok", "median_ratio": 1.0},
    ]

    failures = collect_ratio_failures(
        rows,
        key_fields=("case",),
        expected_keys=[("add",)],
        max_ratio=1.0,
        ratio_field="median_ratio",
        inclusive=False,
    )

    assert failures == ["add: candle median_ratio 1.00x >= 1.00x"]


def test_collect_ratio_failures_reports_missing_errors_and_slow_rows():
    rows = [
        {"framework": "torch_npu", "case": "add", "median_ms": 2.0, "status": "ok"},
        {"framework": "candle", "case": "add", "median_ms": 3.0, "status": "ok", "median_ratio": 1.5},
        {"framework": "candle", "case": "mul", "median_ms": 0.0, "status": "error: boom"},
    ]

    failures = collect_ratio_failures(
        rows,
        key_fields=("case",),
        expected_keys=[("add",), ("mul",), ("div",)],
        max_ratio=1.0,
        ratio_field="median_ratio",
    )

    assert failures == [
        "add: candle median_ratio 1.50x > 1.00x",
        "mul/candle: error: boom",
        "mul: missing torch_npu result",
        "div: missing candle result",
        "div: missing torch_npu result",
    ]
