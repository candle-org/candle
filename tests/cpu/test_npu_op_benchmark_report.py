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
