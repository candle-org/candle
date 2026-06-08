"""Public FSDP API tests.

These tests lock the public `candle.distributed.fsdp` surface while keeping
execution single-process / CPU-friendly.  They intentionally exercise the
public namespace rather than the internal `_composable.fsdp` path.
"""

from dataclasses import fields
import inspect

import pytest

import candle as torch
import candle.nn as nn
from candle.distributed.tensor.dtensor import DTensor


class MockMesh:
    """Single-process mock mesh for public FSDP API tests."""
    def __init__(self):
        self.device_type = "cpu"
        self._mesh_shape = (1,)
        self.mesh_dim_names = ("shard",)
        self._dim_groups = [None]

    def get_group(self, dim=0):
        return None

    def size(self, dim=0):
        return 1

    @property
    def ndim(self):
        return 1


class TinyMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


FSDP2_PUBLIC_EXPORTS = {
    "CPUOffloadPolicy",
    "FSDPModule",
    "fully_shard",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "register_fsdp_forward_method",
    "UnshardHandle",
    "share_comm_ctx",
}

FSDP2_MODULE_METHODS = {
    "reshard",
    "unshard",
    "set_is_last_backward",
    "set_requires_gradient_sync",
    "set_requires_all_reduce",
    "set_reshard_after_forward",
    "set_reshard_after_backward",
    "set_modules_to_forward_prefetch",
    "set_modules_to_backward_prefetch",
    "set_custom_all_gather",
    "set_custom_reduce_scatter",
    "set_all_reduce_hook",
    "set_post_optim_event",
    "set_reduce_scatter_divide_factor",
    "set_gradient_divide_factor",
    "set_force_sum_reduction_for_comms",
    "set_unshard_in_backward",
    "set_allocate_memory_from_process_group_for_comm",
    "_set_unshard_async_op",
    "_get_fsdp_state",
    "_apply",
}


def test_public_fsdp_exports_match_torch_fsdp2_surface():
    """Candle should expose PyTorch FSDP2's public Python names."""
    import candle.distributed._composable.fsdp as composable_fsdp
    import candle.distributed.fsdp as public_fsdp

    for name in FSDP2_PUBLIC_EXPORTS:
        assert hasattr(public_fsdp, name), name
        assert hasattr(composable_fsdp, name), name


@pytest.mark.parametrize("name", sorted(FSDP2_PUBLIC_EXPORTS))
def test_public_fsdp_export_modules_match_torch_namespace(name):
    """FSDP2 exports should identify as torch.distributed.fsdp-compatible."""
    import candle.distributed.fsdp as public_fsdp

    obj = getattr(public_fsdp, name)
    if hasattr(obj, "__module__"):
        assert obj.__module__ == "torch.distributed.fsdp"


def test_fully_shard_signature_matches_torch_fsdp2():
    """fully_shard should accept the same Python-side args as PyTorch FSDP2."""
    from candle.distributed.fsdp import fully_shard
    from torch.distributed.fsdp import fully_shard as torch_fully_shard

    candle_params = inspect.signature(fully_shard).parameters
    torch_params = inspect.signature(torch_fully_shard).parameters
    expected = [
        "module",
        "mesh",
        "reshard_after_forward",
        "shard_placement_fn",
        "mp_policy",
        "offload_policy",
        "ignored_params",
    ]

    assert list(candle_params) == expected
    assert list(torch_params) == expected
    for name in expected[1:]:
        assert candle_params[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert candle_params["mesh"].default is None
    assert candle_params["reshard_after_forward"].default is None
    assert candle_params["shard_placement_fn"].default is None
    assert candle_params["ignored_params"].default is None


@pytest.mark.parametrize(
    "policy_name",
    ["MixedPrecisionPolicy", "OffloadPolicy", "CPUOffloadPolicy"],
)
def test_policy_dataclass_fields_match_torch_fsdp2(policy_name):
    """FSDP2 policy dataclasses should mirror PyTorch fields/defaults."""
    import candle.distributed.fsdp as candle_fsdp
    import torch.distributed.fsdp as torch_fsdp

    candle_cls = getattr(candle_fsdp, policy_name)
    torch_cls = getattr(torch_fsdp, policy_name)
    candle_fields = [(f.name, f.default) for f in fields(candle_cls)]
    torch_fields = [(f.name, f.default) for f in fields(torch_cls)]

    assert candle_fields == torch_fields


@pytest.mark.parametrize("method_name", sorted(FSDP2_MODULE_METHODS))
def test_fsdp_module_methods_match_torch_fsdp2(method_name):
    """FSDPModule should expose PyTorch FSDP2's public method surface."""
    from candle.distributed.fsdp import FSDPModule
    from torch.distributed.fsdp import FSDPModule as TorchFSDPModule

    assert hasattr(TorchFSDPModule, method_name), method_name
    assert hasattr(FSDPModule, method_name), method_name


@pytest.mark.parametrize("name", ["Comm", "AllGather", "ReduceScatter"])
def test_composable_fsdp_comm_interfaces_are_importable(name):
    """PyTorch 2.11 custom communication interfaces should be importable."""
    import candle.distributed._composable.fsdp as composable_fsdp

    assert hasattr(composable_fsdp, name)


def test_public_fully_shard_import_and_basic_use():
    """Public fully_shard import should work and shard module params."""
    from candle.distributed.fsdp import fully_shard

    model = TinyMLP(8)
    mesh = MockMesh()

    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    assert isinstance(model.fc1.weight, DTensor)
    assert isinstance(model.fc2.weight, DTensor)
    assert hasattr(model, "_fsdp_state")


def test_public_fully_shard_matches_internal_callable():
    """Public fully_shard should re-export the composable implementation."""
    from candle.distributed.fsdp import fully_shard as public_fully_shard
    from candle.distributed._composable.fsdp import fully_shard as internal_fully_shard

    assert public_fully_shard is internal_fully_shard


def test_public_fully_shard_forward_works():
    """Forward should work when using the public fsdp namespace."""
    from candle.distributed.fsdp import fully_shard

    model = TinyMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    x = torch.randn(2, 8)
    out = model(x)
    assert out.shape == (2, 8)


def test_public_fully_sharded_data_parallel_still_raises():
    """Legacy FSDP wrapper remains unavailable until explicitly implemented."""
    from candle.distributed.fsdp import FullyShardedDataParallel

    model = TinyMLP(8)
    try:
        FullyShardedDataParallel(model)
        assert False, "expected RuntimeError for legacy FSDP wrapper"
    except RuntimeError as exc:
        assert "FSDP is not available" in str(exc)


def test_fully_shard_accepts_mesh_none_for_single_process_api_parity():
    """PyTorch FSDP2 accepts mesh=None; Candle should too for local tests."""
    from candle.distributed.fsdp import fully_shard

    model = nn.Linear(4, 2)
    result = fully_shard(model, mesh=None)

    assert result is model
    assert isinstance(model.weight, DTensor)


def test_fully_shard_ignored_params_remain_unmanaged():
    """ignored_params should leave exact parameter objects unsharded."""
    from candle.distributed.fsdp import fully_shard

    model = TinyMLP(4)
    ignored_weight = model.fc2.weight
    ignored = {ignored_weight}
    fully_shard(model, mesh=MockMesh(), ignored_params=ignored)

    assert isinstance(model.fc1.weight, DTensor)
    assert model.fc2.weight is ignored_weight
    assert not isinstance(model.fc2.weight, DTensor)


def test_fully_shard_rejects_cpu_offload_policy_explicitly():
    """CPU offload is import-compatible but not silently enabled for Candle."""
    from candle.distributed.fsdp import CPUOffloadPolicy, fully_shard

    model = nn.Linear(4, 2)
    with pytest.raises(NotImplementedError, match="CPU offload"):
        fully_shard(model, mesh=MockMesh(), offload_policy=CPUOffloadPolicy())


def test_register_fsdp_forward_method_wraps_custom_forward():
    """Custom forward methods should run through FSDP pre/post hooks."""
    from candle.distributed.fsdp import fully_shard, register_fsdp_forward_method

    class Projector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, x):
            return self.linear(x)

        def project(self, x):
            return self.linear(x)

    model = Projector()
    fully_shard(model, mesh=MockMesh())
    register_fsdp_forward_method(model, "project")

    out = model.project(torch.randn(3, 4))

    assert out.shape == (3, 2)
    assert isinstance(model.linear.weight, DTensor)


def test_register_fsdp_forward_method_noops_for_non_fsdp_module():
    """PyTorch helper is safe to call before FSDP wrapping."""
    from candle.distributed.fsdp import register_fsdp_forward_method

    model = nn.Linear(4, 2)
    register_fsdp_forward_method(model, "forward")

    assert callable(model.forward)


def test_share_comm_ctx_validates_and_shares_context():
    """share_comm_ctx should validate modules and share the state context."""
    from candle.distributed.fsdp import fully_shard, share_comm_ctx

    left = nn.Linear(4, 4)
    right = nn.Linear(4, 2)
    fully_shard(left, mesh=MockMesh())
    fully_shard(right, mesh=MockMesh())

    assert left._fsdp_state._comm_ctx is not right._fsdp_state._comm_ctx
    share_comm_ctx([left, right])
    assert left._fsdp_state._comm_ctx is right._fsdp_state._comm_ctx

    with pytest.raises(ValueError, match="FSDPModule"):
        share_comm_ctx([left, nn.Linear(2, 2)])


def test_root_without_own_state_unshard_reshard_apply_to_child_states():
    """Root FSDPModule APIs should work when only children own FSDP state."""
    from candle.distributed.fsdp import fully_shard

    model = TinyMLP(4)
    fully_shard(model.fc1, mesh=MockMesh())
    fully_shard(model.fc2, mesh=MockMesh())
    fully_shard(model, mesh=MockMesh())

    assert model._fsdp_state is None
    model.unshard()
    assert not isinstance(model.fc1.weight, DTensor)
    assert not isinstance(model.fc2.weight, DTensor)
    model.reshard()
    assert isinstance(model.fc1.weight, DTensor)
    assert isinstance(model.fc2.weight, DTensor)


def test_custom_comm_setters_fail_explicitly_until_runtime_uses_them():
    """Unsupported communication customizations should not be silently stored."""
    from candle.distributed._composable.fsdp import AllGather, ReduceScatter
    from candle.distributed.fsdp import fully_shard

    class MyAllGather(AllGather):
        pass

    class MyReduceScatter(ReduceScatter):
        pass

    model = nn.Linear(4, 2)
    fully_shard(model, mesh=MockMesh())

    with pytest.raises(NotImplementedError, match="custom all-gather"):
        model.set_custom_all_gather(MyAllGather())
    with pytest.raises(NotImplementedError, match="custom reduce-scatter"):
        model.set_custom_reduce_scatter(MyReduceScatter())
    with pytest.raises(NotImplementedError, match="all-reduce hook"):
        model.set_all_reduce_hook(lambda tensor: None)
    with pytest.raises(NotImplementedError, match="reduce-scatter divide"):
        model.set_reduce_scatter_divide_factor(2.0)
    with pytest.raises(NotImplementedError, match="gradient divide"):
        model.set_gradient_divide_factor(2.0)
    with pytest.raises(NotImplementedError, match="force-sum reduction"):
        model.set_force_sum_reduction_for_comms(True)

    model.set_reduce_scatter_divide_factor(1.0)
    model.set_gradient_divide_factor(1.0)
    model.set_force_sum_reduction_for_comms(False)


def test_mixed_precision_cast_forward_inputs_flag_controls_input_casting():
    """cast_forward_inputs should match PyTorch FSDP2 policy behavior."""
    from candle.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    class CaptureDtype(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)
            self.seen_dtype = None

        def forward(self, x):
            self.seen_dtype = x.dtype
            return self.linear(x)

    enabled = CaptureDtype()
    fully_shard(
        enabled,
        mesh=MockMesh(),
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.float16,
            cast_forward_inputs=True,
        ),
    )
    enabled(torch.randn(3, 4, dtype=torch.float32))
    assert enabled.seen_dtype == torch.float16

    disabled = CaptureDtype()
    fully_shard(
        disabled,
        mesh=MockMesh(),
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.float16,
            cast_forward_inputs=False,
        ),
    )
    disabled(torch.randn(3, 4, dtype=torch.float32))
    assert disabled.seen_dtype == torch.float32
