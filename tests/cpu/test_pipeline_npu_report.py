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
