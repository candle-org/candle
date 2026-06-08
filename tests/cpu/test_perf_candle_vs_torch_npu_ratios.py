import json
import sys

import pytest

import benchmarks.perf_candle_vs_torch_npu as _bench_mod
from benchmarks.perf_candle_vs_torch_npu import _annotate_ratios, _ratio_failures, _spawn_worker


class _FakeModule:
    def to(self, *args, **kwargs):
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _FakeTensor:
    def __init__(self, name):
        self.name = name


class _FakeFunctional:
    def __init__(self, calls):
        self.calls = calls

    def scaled_dot_product_attention(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return _FakeTensor("sdpa_out")

    def gelu(self, x):
        return x


class _FakeMultiheadAttention(_FakeModule):
    calls = []

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs

    def forward(self, *args, **kwargs):
        type(self).calls.append({"args": args, "kwargs": kwargs})
        return _FakeTensor("mha_out"), kwargs.get("need_weights")


class _FakeNN:
    Module = _FakeModule
    MultiheadAttention = _FakeMultiheadAttention

    def __init__(self, functional):
        self.functional = functional


class _FakeTorch:
    float16 = object()

    def __init__(self):
        self.sdpa_calls = []
        self.nn = _FakeNN(_FakeFunctional(self.sdpa_calls))

    def randn(self, *shape, **kwargs):
        return _FakeTensor((shape, kwargs))


def test_sdpa_cases_are_registered():
    assert "sdpa" in _bench_mod.CASES
    assert "mha_sdpa" in _bench_mod.CASES
    assert "xfmr_sdpa" in _bench_mod.CASES
    assert "xfmr_sdpa_large" in _bench_mod.CASES


def test_main_accepts_sdpa_case(capsys, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perf_candle_vs_torch_npu.py",
            "--cases",
            "sdpa",
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

    payload = json.loads(capsys.readouterr().out)
    assert payload["cases"] == ["sdpa"]


def test_sdpa_builder_calls_functional_scaled_dot_product_attention():
    fake_torch = _FakeTorch()

    model, inputs = _bench_mod.build_sdpa(fake_torch, "npu:0", fake_torch.float16)
    out = model(*inputs)

    assert out.name == "sdpa_out"
    assert len(fake_torch.sdpa_calls) == 1
    assert fake_torch.sdpa_calls[0]["kwargs"]["dropout_p"] == 0.0


def test_mha_sdpa_builder_forces_need_weights_false():
    fake_torch = _FakeTorch()
    _FakeMultiheadAttention.calls = []

    model, inputs = _bench_mod.build_mha_sdpa(fake_torch, "npu:0", fake_torch.float16)
    model(*inputs)

    assert len(_FakeMultiheadAttention.calls) == 1
    assert _FakeMultiheadAttention.calls[0]["kwargs"]["need_weights"] is False


def test_spawn_worker_passes_graph_step_to_worker_command(monkeypatch):
    seen = {}

    def fake_check_output(cmd, **kwargs):
        del kwargs
        seen["cmd"] = cmd
        return b'{"framework": "candle", "case": "sdpa", "total_ms_median": 1.0}\n'

    monkeypatch.setattr(_bench_mod.subprocess, "check_output", fake_check_output)

    _spawn_worker("candle", "sdpa", iters=1, warmup=0, dtype_name="float16", graph_step=True)

    assert "--graph-step" in seen["cmd"]


def test_main_propagates_graph_step_to_spawn_worker(capsys, monkeypatch):
    seen = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perf_candle_vs_torch_npu.py",
            "--cases",
            "sdpa",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--json-output",
            "-",
            "--graph-step",
        ],
    )

    def fake_spawn(framework, case, *args, **kwargs):
        del args
        seen.append((framework, case, kwargs.get("graph_step")))
        return {
            "framework": framework,
            "case": case,
            "graph_step": kwargs.get("graph_step"),
            "fwd_ms_median": None,
            "bwd_ms_median": None,
            "total_ms_median": 2.0,
        }

    monkeypatch.setattr(_bench_mod, "_spawn_worker", fake_spawn)

    _bench_mod.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["graph_step"] is True
    assert seen == [("candle", "sdpa", True), ("torch_npu", "sdpa", True)]


def test_main_can_compare_candle_graph_step_against_torch_npu_eager(capsys, monkeypatch):
    seen = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perf_candle_vs_torch_npu.py",
            "--cases",
            "sdpa",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--json-output",
            "-",
            "--candle-graph-step",
        ],
    )

    def fake_spawn(framework, case, *args, **kwargs):
        del args
        seen.append((framework, case, kwargs.get("graph_step")))
        return {
            "framework": framework,
            "case": case,
            "graph_step": kwargs.get("graph_step"),
            "fwd_ms_median": None if kwargs.get("graph_step") else 1.0,
            "bwd_ms_median": None if kwargs.get("graph_step") else 1.0,
            "total_ms_median": 1.0,
        }

    monkeypatch.setattr(_bench_mod, "_spawn_worker", fake_spawn)

    _bench_mod.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["graph_step"] is False
    assert payload["candle_graph_step"] is True
    assert seen == [("candle", "sdpa", True), ("torch_npu", "sdpa", False)]


def test_worker_mode_honors_candle_graph_step_for_candle(capsys, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perf_candle_vs_torch_npu.py",
            "--worker",
            "--framework",
            "candle",
            "--case",
            "sdpa",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--candle-graph-step",
        ],
    )

    def fake_run_worker(framework, case, iters, warmup, dtype, **kwargs):
        seen.update(
            framework=framework,
            case=case,
            iters=iters,
            warmup=warmup,
            dtype=dtype,
            graph_step=kwargs.get("graph_step"),
        )
        return {"framework": framework, "case": case, "graph_step": kwargs.get("graph_step")}

    monkeypatch.setattr(_bench_mod, "run_worker", fake_run_worker)

    _bench_mod.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["graph_step"] is True
    assert seen["graph_step"] is True


def test_graph_step_worker_uses_npu_graph_replay(monkeypatch):
    calls = {"forward": 0, "backward": 0, "replay": 0, "capture": 0}

    class FakeLoss:
        grad = None

        def sum(self):
            return self

        def backward(self):
            calls["backward"] += 1

    class FakeModel:
        def parameters(self):
            return ()

        def __call__(self, *args):
            del args
            calls["forward"] += 1
            return FakeLoss()

    class FakeGraph:
        def replay(self):
            calls["replay"] += 1

    class FakeGraphContext:
        def __enter__(self):
            calls["capture"] += 1

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    class FakeNpu:
        def is_available(self):
            return True

        def synchronize(self):
            pass

        def NPUGraph(self):
            return FakeGraph()

        def graph(self, graph):
            assert isinstance(graph, FakeGraph)
            return FakeGraphContext()

    class FakeTorch:
        float16 = object()
        npu = FakeNpu()

    monkeypatch.setitem(
        _bench_mod.CASES,
        "fake_graph",
        lambda torch, device, dtype: (FakeModel(), (FakeLoss(),)),
    )
    monkeypatch.setattr(_bench_mod, "_import_framework", lambda framework: FakeTorch())

    result = _bench_mod.run_worker(
        "candle",
        "fake_graph",
        iters=3,
        warmup=0,
        dtype_name="float16",
        graph_step=True,
    )

    assert result["graph_step"] is True
    assert result["fwd_ms_median"] is None
    assert result["bwd_ms_median"] is None
    assert result["total_ms_median"] >= 0.0
    assert calls == {"forward": 1, "backward": 1, "replay": 3, "capture": 1}


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


def test_graph_step_ratios_are_total_only():
    results = [
        {
            "framework": "torch_npu",
            "case": "xfmr_sdpa",
            "graph_step": True,
            "fwd_ms_median": None,
            "bwd_ms_median": None,
            "total_ms_median": 4.0,
        },
        {
            "framework": "candle",
            "case": "xfmr_sdpa",
            "graph_step": True,
            "fwd_ms_median": None,
            "bwd_ms_median": None,
            "total_ms_median": 3.0,
        },
    ]

    _annotate_ratios(results)

    candle = next(row for row in results if row["framework"] == "candle")
    assert candle["total_ratio"] == 0.75
    failures = _ratio_failures(results, ["xfmr_sdpa"], 1.0, 1.0, 0.99)
    assert failures == []


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
