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
    source = inspect.getsource(_qwen2._run_model)

    assert "current_tokens = 2 if args.mode in" in source
    assert 'kwargs["input_ids"] = input_ids[:, -current_tokens:]' in source
    assert 'kwargs["labels"] = labels[:, -current_tokens:]' in source


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
