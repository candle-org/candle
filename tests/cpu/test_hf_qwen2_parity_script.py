"""Unit tests for the HuggingFace Qwen2 parity harness."""

import inspect
import math
from types import SimpleNamespace

import pytest

import benchmarks.hf_qwen2_parity as _qwen2


def test_tiny_qwen2_config_is_small_and_gqa_enabled():
    cfg = _qwen2.tiny_qwen2_config_kwargs()

    assert cfg["vocab_size"] == 128
    assert cfg["hidden_size"] == 64
    assert cfg["intermediate_size"] == 128
    assert cfg["num_hidden_layers"] == 2
    assert cfg["num_attention_heads"] == 4
    assert cfg["num_key_value_heads"] == 2
    assert cfg["attention_dropout"] == 0.0
    assert cfg["use_cache"] is True


def test_worker_args_forward_common_options():
    args = SimpleNamespace(
        mode="forward",
        cache_mode="dynamic",
        device="npu:0",
        dtype="float16",
        seed=123,
        iters=7,
        warmup=3,
        out_dir="/tmp/out",
        accuracy_atol=0.01,
        accuracy_rtol=0.02,
        local_files_only=True,
        pretrained_path=None,
    )

    worker_args = _qwen2._base_worker_args(args, "candle", "/tmp/out")

    assert worker_args[:3] == [_qwen2.__file__, "--worker", "--framework"]
    assert "candle" in worker_args
    assert "--mode" in worker_args
    assert "forward" in worker_args
    assert "--cache-mode" in worker_args
    assert "dynamic" in worker_args
    assert "--device" in worker_args
    assert "npu:0" in worker_args
    assert worker_args[worker_args.index("--iters") + 1] == "7"
    assert worker_args[worker_args.index("--warmup") + 1] == "3"
    assert "--local-files-only" in worker_args


def test_candle_env_prepends_src_but_torch_npu_env_does_not(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/existing")

    candle_env = _qwen2._worker_env("candle")
    torch_env = _qwen2._worker_env("torch_npu")

    assert candle_env["USE_CANDLE"] == "1"
    assert candle_env["PYTHONPATH"].split(":")[0].endswith("/src")
    assert torch_env.get("USE_CANDLE") != "1"
    assert not torch_env["PYTHONPATH"].split(":")[0].endswith("/src")
    assert torch_env["PYTHONPATH"].split(":")[0] == _qwen2._REPO_ROOT


def test_torch_npu_launch_uses_distributed_run():
    cmd = _qwen2._torch_npu_launch_command(
        "/python",
        _qwen2.__file__,
        worker_args=["--worker", "--framework", "torch_npu"],
    )

    assert cmd[:3] == ["/python", "-m", "torch.distributed.run"]
    assert "--nproc_per_node=1" in cmd
    assert _qwen2.__file__ in cmd
    assert cmd[-3:] == ["--worker", "--framework", "torch_npu"]


def test_candle_worker_applies_transformers_compat_patches():
    source = inspect.getsource(_qwen2._prepare_framework)

    assert "compat.transformers.conftest" in source
    assert "apply_all_patches()" in source


def test_transformers_compat_provides_dynamo_utils_stub():
    import compat.transformers.conftest as compat_conf

    source = inspect.getsource(compat_conf._apply_module_stubs)
    assert "torch._dynamo.utils" in source
    assert "is_compile_supported" in source


def test_dynamic_cache_loss_uses_only_step_labels():
    source = inspect.getsource(_qwen2._make_forward_step)

    assert "current_tokens = 2 if args.mode in" in source
    assert 'step_kwargs["input_ids"] = input_ids[:, -current_tokens:]' in source
    assert 'step_kwargs["labels"] = labels[:, -current_tokens:]' in source


def test_dynamic_forward_step_recomputes_past_key_values_each_call():
    calls = []

    class _Tensor:
        def __getitem__(self, item):
            return ("slice", item)

    class _Out:
        def __init__(self, tag):
            self.past_key_values = ("past", tag)

    class _Model:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return _Out(len(calls))

    args = SimpleNamespace(mode="train-step", cache_mode="dynamic")
    forward_once = _qwen2._make_forward_step(_Model(), args, _Tensor(), _Tensor(), _Tensor())

    forward_once()
    forward_once()

    prefix_calls = [call for call in calls if "past_key_values" not in call]
    step_calls = [call for call in calls if "past_key_values" in call]
    assert len(prefix_calls) == 2
    assert len(step_calls) == 2
    assert step_calls[0]["past_key_values"] == ("past", 1)
    assert step_calls[1]["past_key_values"] == ("past", 3)


def test_summarize_samples_reports_latency_distribution():
    summary = _qwen2._summarize_samples([3.0, 1.0, 2.0, 5.0, 4.0])

    assert summary["min_ms"] == 1.0
    assert summary["median_ms"] == 3.0
    assert summary["p10_ms"] == 1.0
    assert summary["p90_ms"] == 5.0


def test_time_iterations_runs_warmup_and_measured_steps():
    calls = []
    syncs = []
    ticks = iter([10.0, 10.5, 20.0, 21.0])

    timing = _qwen2._time_iterations(
        lambda: calls.append("step"),
        lambda: syncs.append("sync"),
        warmup=2,
        iters=2,
        timer=lambda: next(ticks),
    )

    assert len(calls) == 4
    assert len(syncs) == 5
    assert timing["samples_ms"] == [500.0, 1000.0]
    assert timing["median_ms"] == 750.0


def test_time_forward_backward_iterations_reports_split_samples():
    calls = []
    syncs = []
    ticks = iter([10.0, 10.25, 10.75, 20.0, 20.40, 21.00])

    timing = _qwen2._time_forward_backward_iterations(
        lambda: calls.append("forward"),
        lambda: calls.append("backward"),
        lambda: syncs.append("sync"),
        warmup=1,
        iters=2,
        timer=lambda: next(ticks),
    )

    assert calls == ["forward", "backward", "forward", "backward", "forward", "backward"]
    assert timing["forward"]["samples_ms"] == pytest.approx([250.0, 400.0])
    assert timing["backward"]["samples_ms"] == pytest.approx([500.0, 600.0])
    assert timing["total"]["samples_ms"] == pytest.approx([750.0, 1000.0])
    assert timing["total"]["median_ms"] == pytest.approx(875.0)


def test_run_worker_payload_includes_timing(monkeypatch, tmp_path):
    class _Torch:
        pass

    monkeypatch.setattr(_qwen2, "_prepare_framework", lambda framework, device: _Torch)
    monkeypatch.setattr(_qwen2, "_device_obj", lambda torch, framework, device: "device")
    monkeypatch.setattr(_qwen2, "_build_model", lambda torch, args, device: object())
    monkeypatch.setattr(_qwen2, "_run_model", lambda torch, model, args, device: ({"loss": 1.0}, {"step": {"median_ms": 2.0}}))

    args = SimpleNamespace(
        framework="candle",
        mode="forward",
        cache_mode="none",
        device="npu:0",
        dtype="float16",
        out_dir=str(tmp_path),
    )

    _qwen2._run_worker(args)
    row = _qwen2._read_worker_result(str(tmp_path), "candle")

    assert row["metrics"] == {"loss": 1.0}
    assert row["timing"] == {"step": {"median_ms": 2.0}}


def test_timing_annotation_records_ratio_separately_from_accuracy():
    candle = {
        "framework": "candle",
        "status": "ok",
        "metrics": {"loss": 1.0},
        "timing": {"step": {"median_ms": 4.0}},
    }
    torch_ref = {
        "framework": "torch_npu",
        "status": "ok",
        "metrics": {"loss": 1.0},
        "timing": {"step": {"median_ms": 2.0}},
    }

    timing_failures = _qwen2.annotate_timing([candle, torch_ref])
    accuracy_failures = _qwen2.annotate_accuracy([candle, torch_ref], atol=0.0, rtol=0.0)

    assert timing_failures == []
    assert accuracy_failures == []
    assert candle["timing"]["step"]["ratio"] == 2.0
    assert "step" not in candle["accuracy"]


def test_accuracy_comparison_records_diffs_and_passes_within_tolerance():
    candle = {
        "framework": "candle",
        "status": "ok",
        "metrics": {
            "loss": 1.0,
            "nan_loss": math.nan,
            "logits_values": [1.0, 2.0, 3.0],
            "grad_names": ["model.embed_tokens.weight"],
            "grad_values": [0.1, 0.2],
        },
    }
    torch_ref = {
        "framework": "torch_npu",
        "status": "ok",
        "metrics": {
            "loss": 1.00001,
            "nan_loss": math.nan,
            "logits_values": [1.0, 2.00002, 2.99999],
            "grad_names": ["model.embed_tokens.weight"],
            "grad_values": [0.10001, 0.20001],
        },
    }

    failures = _qwen2.annotate_accuracy([candle, torch_ref], atol=1e-3, rtol=1e-3)

    assert failures == []
    assert candle["accuracy"]["loss_abs_diff"] < 1e-3
    assert candle["accuracy"]["nan_loss_abs_diff"] == 0.0
    assert candle["accuracy"]["logits_values_max_abs_diff"] < 1e-3
    assert candle["accuracy"]["grad_names_match"] is True
    assert candle["accuracy"]["grad_values_max_abs_diff"] < 1e-3


def test_fill_tiny_qwen2_parameters_is_deterministic_and_name_based():
    class _Param:
        def __init__(self):
            self.shape = (2, 3)
            self.device = "cpu"
            self.dtype = "float32"
            self.data = None

    class _Torch:
        float32 = "float32"

        @staticmethod
        def arange(stop, device=None, dtype=None):
            del device, dtype
            return _Tensor(range(stop))

    class _Tensor:
        def __init__(self, values):
            self.values = [float(value) for value in values]

        def reshape(self, shape):
            return self

        def __add__(self, other):
            return _Tensor(value + other for value in self.values)

        def __mul__(self, other):
            return _Tensor(value * other for value in self.values)

        def to(self, dtype):
            del dtype
            return self

    class _Model:
        def __init__(self):
            self.params = [("embed.weight", _Param()), ("lm_head.weight", _Param())]

        def named_parameters(self):
            return list(self.params)

    first = _Model()
    second = _Model()

    _qwen2._fill_tiny_qwen2_parameters(_Torch, first, dtype="float32")
    _qwen2._fill_tiny_qwen2_parameters(_Torch, second, dtype="float32")

    first_values = [param.data.values for _, param in first.named_parameters()]
    second_values = [param.data.values for _, param in second.named_parameters()]
    assert first_values == second_values
    assert first_values[0] != first_values[1]


def test_accuracy_comparison_fails_for_nan_mismatch():
    candle = {
        "framework": "candle",
        "status": "ok",
        "metrics": {"loss": math.nan},
    }
    torch_ref = {
        "framework": "torch_npu",
        "status": "ok",
        "metrics": {"loss": 1.0},
    }

    failures = _qwen2.annotate_accuracy([candle, torch_ref], atol=1e-3, rtol=1e-3)

    assert failures
    assert "loss" in failures[0]


def test_accuracy_comparison_fails_for_large_logit_diff():
    candle = {
        "framework": "candle",
        "status": "ok",
        "metrics": {"loss": 1.0, "logits_values": [1.0, 4.0]},
    }
    torch_ref = {
        "framework": "torch_npu",
        "status": "ok",
        "metrics": {"loss": 1.0, "logits_values": [1.0, 2.0]},
    }

    failures = _qwen2.annotate_accuracy([candle, torch_ref], atol=1e-3, rtol=1e-3)

    assert failures
    assert "logits_values" in failures[0]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--framework", "bad"], "invalid choice"),
        (["--mode", "bad"], "invalid choice"),
        (["--cache-mode", "bad"], "invalid choice"),
    ],
)
def test_parser_rejects_invalid_choices(argv, message):
    parser = _qwen2._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(argv)
