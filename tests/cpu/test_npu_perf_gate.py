import os

import pytest

import benchmarks.npu_perf_gate as npu_perf_gate
from benchmarks.npu_perf_gate import build_commands




def test_l2_command_uses_absolute_script_path():
    commands = build_commands(
        output_dir="results/npu_perf_gate/test",
        warmup=1,
        iters=2,
        dtype="fp16",
        cases="xfmr",
        pipeline_cases="A1,B1s",
    )
    l2_cmd = commands[2]
    script_arg = next(
        (arg for arg in l2_cmd if arg.endswith("benchmarks/perf_candle_vs_torch_npu.py")),
        None,
    )
    assert script_arg is not None, "L2 command must reference perf_candle_vs_torch_npu.py"
    assert os.path.isabs(script_arg), f"L2 script path must be absolute, got: {script_arg}"


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
    assert any(arg.endswith("benchmarks/perf_candle_vs_torch_npu.py") for arg in commands[2])
    assert "results/npu_perf_gate/test/l1_pipeline.json" in commands[1]
    assert "results/npu_perf_gate/test/l2_models.json" in commands[2]


def test_main_runs_all_commands_and_exits_1_when_any_child_fails(monkeypatch, tmp_path):
    commands = [["cmd0"], ["cmd1"], ["cmd2"]]
    returncodes = [2, 0, 3]
    seen_commands = []

    class Result:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(command, check):
        assert check is False
        seen_commands.append(command)
        return Result(returncodes[len(seen_commands) - 1])

    monkeypatch.setattr(npu_perf_gate, "build_commands", lambda **_: commands)
    monkeypatch.setattr(npu_perf_gate.subprocess, "run", fake_run)
    monkeypatch.setattr(
        npu_perf_gate.sys,
        "argv",
        ["npu_perf_gate", "--output-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as exc_info:
        npu_perf_gate.main()

    assert exc_info.value.code == 1
    assert seen_commands == commands


def test_main_exits_cleanly_when_all_child_commands_pass(monkeypatch, tmp_path):
    commands = [["cmd0"], ["cmd1"], ["cmd2"]]
    seen_commands = []

    class Result:
        returncode = 0

    def fake_run(command, check):
        assert check is False
        seen_commands.append(command)
        return Result()

    monkeypatch.setattr(npu_perf_gate, "build_commands", lambda **_: commands)
    monkeypatch.setattr(npu_perf_gate.subprocess, "run", fake_run)
    monkeypatch.setattr(
        npu_perf_gate.sys,
        "argv",
        ["npu_perf_gate", "--output-dir", str(tmp_path)],
    )

    npu_perf_gate.main()

    assert seen_commands == commands
