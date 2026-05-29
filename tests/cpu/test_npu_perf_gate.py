from benchmarks.npu_perf_gate import build_commands


def test_build_commands_writes_l0_l1_l2_artifacts_under_output_dir():
    commands = build_commands(
        output_dir="results/npu_perf_gate/test",
        warmup=1,
        iters=2,
        dtype="fp16",
        cases="xfmr",
        pipeline_cases="A1,B1s",
    )

    assert commands[0][-5:] == ["--max-ratio", "1.0", "--json-output", "results/npu_perf_gate/test/l0_ops.json", "--fail-on-ratio"]
    assert "benchmarks.op_benchmark_npu.run" in commands[0]
    assert "benchmarks.pipeline_npu.run" in commands[1]
    assert "benchmarks/perf_candle_vs_torch_npu.py" in commands[2]
    assert "results/npu_perf_gate/test/l1_pipeline.json" in commands[1]
    assert "results/npu_perf_gate/test/l2_models.json" in commands[2]
