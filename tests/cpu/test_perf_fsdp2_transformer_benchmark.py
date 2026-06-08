"""Static/unit tests for the two-card FSDP2 Transformer benchmark."""

import inspect
import json
import sys
from types import SimpleNamespace

import benchmarks.perf_fsdp2_transformer_candle_vs_torch_npu as _bench_mod


def test_transformer_fsdp2_cases_are_registered():
    assert "xfmr_fsdp2_small" in _bench_mod.CASES
    assert "xfmr_fsdp2_medium" in _bench_mod.CASES

    small = _bench_mod.CASES["xfmr_fsdp2_small"]
    assert small.layers >= 1
    assert small.batch_per_rank == 2
    assert small.seq == 128
    assert small.hidden == 512
    assert small.heads == 8

    medium = _bench_mod.CASES["xfmr_fsdp2_medium"]
    assert medium.layers >= small.layers
    assert medium.seq >= small.seq
    assert medium.hidden >= small.hidden


def test_transformer_model_uses_mha_sdpa_path():
    source = inspect.getsource(_bench_mod._build_transformer_model)

    assert "MultiheadAttention" in source
    assert "batch_first=True" in source
    assert "dropout=0.0" in source
    assert "need_weights=False" in source
    assert "functional.gelu" in source


def test_candle_worker_uses_composable_fsdp_and_valid_mesh():
    source = inspect.getsource(_bench_mod._run_candle_worker)

    assert "from candle.distributed._composable.fsdp import fully_shard" in source
    assert "from candle.distributed.device_mesh import DeviceMesh" in source
    assert 'DeviceMesh("npu", (world_size,))' in source
    assert "list(range(world_size))" not in source


def test_torch_npu_worker_uses_torch_fsdp2_and_device_mesh():
    source = inspect.getsource(_bench_mod._run_torch_npu_worker)

    assert "import torch_npu" in source
    assert "from torch.distributed._composable.fsdp import fully_shard" in source
    assert "from torch.distributed.device_mesh import init_device_mesh" in source
    assert 'init_device_mesh("npu", (world_size,))' in source
    assert "torch.npu.set_device" in source


def test_torch_npu_launch_uses_configured_python_distributed_run():
    cmd = _bench_mod._torch_npu_launch_command(
        "/path/to/python",
        "/repo/bench.py",
        world_size=2,
        worker_args=["--worker", "--framework", "torch_npu"],
    )

    assert cmd[:3] == ["/path/to/python", "-m", "torch.distributed.run"]
    assert "--standalone" in cmd
    assert "--nproc_per_node=2" in cmd
    assert "/repo/bench.py" in cmd
    assert cmd[-3:] == ["--worker", "--framework", "torch_npu"]


def test_profile_candle_flag_is_forwarded_to_candle_workers_only():
    args = SimpleNamespace(
        iters=1,
        warmup=0,
        dtype="float16",
        lr=0.001,
        seed=123,
        profile_candle=True,
    )

    candle_args = _bench_mod._base_worker_args(args, "candle", "xfmr_fsdp2_small", "/tmp/out")
    torch_args = _bench_mod._base_worker_args(args, "torch_npu", "xfmr_fsdp2_small", "/tmp/out")

    assert "--profile-candle" in candle_args
    assert "--profile-candle" not in torch_args


def test_aggregate_results_include_fsdp2_metrics_and_ratios():
    candle = _bench_mod._aggregate_rank_results(
        "candle",
        "xfmr_fsdp2_small",
        [
            {
                "rank": 0,
                "world_size": 2,
                "batch_per_rank": 2,
                "global_batch": 4,
                "fwd_ms_samples": [2.0, 2.0],
                "bwd_ms_samples": [3.0, 3.0],
                "optim_ms_samples": [1.0, 1.0],
                "total_ms_samples": [6.0, 6.0],
            },
            {
                "rank": 1,
                "world_size": 2,
                "batch_per_rank": 2,
                "global_batch": 4,
                "fwd_ms_samples": [2.5, 2.5],
                "bwd_ms_samples": [3.5, 3.5],
                "optim_ms_samples": [1.0, 1.0],
                "total_ms_samples": [7.0, 7.0],
            },
        ],
    )
    torch_ref = dict(candle)
    torch_ref.update(
        framework="torch_npu",
        fwd_ms_median=5.0,
        bwd_ms_median=7.0,
        optim_ms_median=2.0,
        total_ms_median=14.0,
        samples_per_second=4.0 / 0.014,
    )

    _bench_mod._annotate_ratios([candle, torch_ref])

    assert candle["fwd_ms_median"] == 2.5
    assert candle["bwd_ms_median"] == 3.5
    assert candle["optim_ms_median"] == 1.0
    assert candle["total_ms_median"] == 7.0
    assert candle["samples_per_second"] == 4.0 / 0.007
    assert candle["fwd_ratio"] == 0.5
    assert candle["bwd_ratio"] == 0.5
    assert candle["optim_ratio"] == 0.5
    assert candle["total_ratio"] == 0.5
    assert candle["throughput_ratio"] == 2.0


def test_fsdp_post_backward_uses_grouped_reduce_scatter_path():
    source = _bench_mod.Path(_bench_mod._REPO_ROOT).joinpath(
        "src/candle/distributed/_composable/fsdp/_fsdp_state.py"
    ).read_text(encoding="utf-8")
    body = source.split("def _post_backward_all", 1)[1].split("def _lazy_init_root", 1)[0]

    assert "param_group.reduce_scatter_grads()" in body
    assert "fsdp_param.reduce_scatter_grad(grad)" not in body


def test_aggregate_results_preserves_candle_profile_summary():
    result = _bench_mod._aggregate_rank_results(
        "candle",
        "xfmr_fsdp2_small",
        [
            {
                "rank": 0,
                "world_size": 2,
                "batch_per_rank": 2,
                "global_batch": 4,
                "fwd_ms_samples": [2.0],
                "bwd_ms_samples": [3.0],
                "optim_ms_samples": [1.0],
                "total_ms_samples": [6.0],
                "profile": {"fsdp.unshard": {"count": 2, "total_ms": 4.0}},
            },
            {
                "rank": 1,
                "world_size": 2,
                "batch_per_rank": 2,
                "global_batch": 4,
                "fwd_ms_samples": [2.5],
                "bwd_ms_samples": [3.5],
                "optim_ms_samples": [1.0],
                "total_ms_samples": [7.0],
                "profile": {"fsdp.unshard": {"count": 2, "total_ms": 6.0}},
            },
        ],
    )

    assert result["profile"]["fsdp.unshard"]["count_max"] == 2
    assert result["profile"]["fsdp.unshard"]["total_ms_max"] == 6.0
    assert result["profile"]["fsdp.unshard"]["total_ms_by_rank"] == [4.0, 6.0]


def test_npu_flat_unshard_stays_disabled_until_native_pack_exists():
    from candle.distributed._composable.fsdp._fsdp_param_group import (
        _should_use_flat_buffer,
    )

    class Device:
        def __init__(self, device_type):
            self.type = device_type

    assert not _should_use_flat_buffer(2, 2, Device("npu"))
    assert _should_use_flat_buffer(2, 2, Device("cpu"))
    assert not _should_use_flat_buffer(1, 2, Device("cpu"))
    assert not _should_use_flat_buffer(2, 1, Device("cpu"))


def test_npu_flat_reduce_scatter_stays_disabled_until_native_pack_exists():
    from candle.distributed._composable.fsdp._fsdp_param_group import (
        _should_use_flat_reduce_scatter,
    )

    class Device:
        def __init__(self, device_type):
            self.type = device_type

    assert not _should_use_flat_reduce_scatter(True, Device("npu"))
    assert _should_use_flat_reduce_scatter(True, Device("cpu"))
    assert not _should_use_flat_reduce_scatter(False, Device("npu"))


def test_npu_sgd_step_uses_scalar_lr_without_full_tensor_fill():
    source = _bench_mod.Path(_bench_mod._REPO_ROOT).joinpath(
        "src/candle/_backends/npu/ops/optim.py"
    ).read_text(encoding="utf-8")
    body = source.split("def _sgd_step_op", 1)[1].split("def _adagrad_step_op", 1)[0]

    assert "fast_sgd_step" in body
    assert "_scalar_to_npu_tensor(lr, param)" not in body


def test_main_json_stdout_payload_has_expected_fields(capsys, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perf_fsdp2_transformer_candle_vs_torch_npu.py",
            "--cases",
            "xfmr_fsdp2_small",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--world-size",
            "2",
            "--json-output",
            "-",
        ],
    )

    def fake_spawn(framework, case, args):
        del args
        return {
            "framework": framework,
            "case": case,
            "world_size": 2,
            "global_batch": 4,
            "batch_per_rank": 2,
            "fwd_ms_median": 1.0,
            "bwd_ms_median": 2.0,
            "optim_ms_median": 0.5,
            "total_ms_median": 3.5,
            "samples_per_second": 4.0 / 0.0035,
        }

    monkeypatch.setattr(_bench_mod, "_spawn_framework", fake_spawn)

    _bench_mod.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["cases"] == ["xfmr_fsdp2_small"]
    assert payload["world_size"] == 2
    assert payload["iters"] == 1
    assert payload["warmup"] == 0
    assert payload["results"][0]["framework"] == "candle"
    assert payload["results"][0]["batch_per_rank"] == 2
    assert "samples_per_second" in payload["results"][0]
