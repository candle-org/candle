"""Integration tests for FSDP2 forward + backward (single-process, world_size=1)."""
import candle as torch
import candle.nn as nn
from candle.distributed.tensor.dtensor import DTensor


class MockMesh:
    def __init__(self):
        self.device_type = "cpu"
        self._mesh_shape = (1,)
        self.mesh_dim_names = ("shard",)
        self._dim_groups = [None]
    def get_group(self, dim=0): return None
    def size(self, dim=0): return 1
    @property
    def ndim(self): return 1


class SimpleMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def test_fsdp_forward_backward():
    """Full forward + backward pass with fully_shard should work."""
    from candle.distributed._composable.fsdp import fully_shard

    model = SimpleMLP(8)
    mesh = MockMesh()

    # Apply FSDP bottom-up
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # Forward
    x = torch.randn(4, 8, requires_grad=True)
    out = model(x)
    assert out.shape == (4, 8)

    # Backward
    loss = out.sum()
    loss.backward()

    # Gradients should exist on sharded params
    for name, param in model.named_parameters():
        local = param.to_local() if isinstance(param, DTensor) else param
        assert local.grad is not None, f"No gradient for {name}"


def test_fsdp_multiple_forward_backward():
    """Multiple forward/backward cycles should work without state corruption."""
    from candle.distributed._composable.fsdp import fully_shard

    model = SimpleMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    for i in range(3):
        # Zero gradients manually
        for name, param in model.named_parameters():
            local = param.to_local() if isinstance(param, DTensor) else param
            if local.grad is not None:
                local.grad = None

        x = torch.randn(4, 8, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

    # Should complete without error


def test_fsdp_params_are_dtensor_between_iterations():
    """Between forward passes, params should be back in sharded (DTensor) state."""
    from candle.distributed._composable.fsdp import fully_shard

    model = SimpleMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # First forward/backward
    x = torch.randn(4, 8, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # After backward, non-root params should be resharded
    assert isinstance(model.fc1.weight, DTensor), "fc1.weight should be resharded"
    assert isinstance(model.fc2.weight, DTensor), "fc2.weight should be resharded"


def test_fsdp_optimizer_step():
    """optimizer.step() should update sharded parameters via DTensor grad proxy."""
    from candle.distributed._composable.fsdp import fully_shard
    from candle.optim import SGD

    model = SimpleMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    optimizer = SGD(model.parameters(), lr=0.1)

    # Snapshot weights before step
    w1_before = model.fc1.weight.to_local().detach().clone()
    w2_before = model.fc2.weight.to_local().detach().clone()

    # Forward + backward
    x = torch.randn(4, 8, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Verify grads are accessible via DTensor.grad
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} should be DTensor"
        assert param.grad is not None, f"{name}.grad should not be None"

    # Optimizer step
    optimizer.step()

    # Weights should have changed
    w1_after = model.fc1.weight.to_local()
    w2_after = model.fc2.weight.to_local()
    w1_diff = float((w1_after - w1_before).abs().sum())
    w2_diff = float((w2_after - w2_before).abs().sum())
    assert w1_diff > 0, "fc1.weight should have been updated"
    assert w2_diff > 0, "fc2.weight should have been updated"


def test_fsdp_unused_parameter():
    """Unused parameters should get zero gradients after finalize_backward."""
    from candle.distributed._composable.fsdp import fully_shard

    class ModelWithUnused(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(8, 8)
            self.unused = nn.Linear(8, 8)  # never used in forward
        def forward(self, x):
            return self.used(x)

    model = ModelWithUnused()
    mesh = MockMesh()
    fully_shard(model.used, mesh=mesh)
    fully_shard(model.unused, mesh=mesh)
    fully_shard(model, mesh=mesh)

    x = torch.randn(2, 8, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # used params should have non-zero grads
    used_local = model.used.weight.to_local() if isinstance(model.used.weight, DTensor) else model.used.weight
    assert used_local.grad is not None, "used.weight should have grad"

    # unused params: finalize_backward should flush them
    model.unused._fsdp_state.finalize_backward()

    unused_local = model.unused.weight.to_local() if isinstance(model.unused.weight, DTensor) else model.unused.weight
    assert unused_local.grad is not None, "unused.weight should have zero grad after finalize"
    assert float(unused_local.grad.abs().sum()) == 0.0, "unused.weight grad should be zero"

    # Both modules should be resharded
    assert isinstance(model.unused.weight, DTensor), "unused.weight should be resharded"


def test_fsdp_no_sync():
    """no_sync() should skip reduce-scatter and accumulate gradients."""
    from candle.distributed._composable.fsdp import fully_shard

    model = SimpleMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # Accumulation step inside no_sync
    with model.no_sync():
        x1 = torch.randn(4, 8, requires_grad=True)
        out1 = model(x1)
        loss1 = out1.sum()
        loss1.backward()

    # After no_sync backward, params should still be resharded
    assert isinstance(model.fc1.weight, DTensor), "fc1.weight should be resharded after no_sync"

    # Sync step (outside no_sync) — reduce-scatter happens
    x2 = torch.randn(4, 8, requires_grad=True)
    out2 = model(x2)
    loss2 = out2.sum()
    loss2.backward()

    # Gradients should exist on sharded params
    for name, param in model.named_parameters():
        local = param.to_local() if isinstance(param, DTensor) else param
        assert local.grad is not None, f"No gradient for {name} after sync step"


def test_fsdp_clip_grad_norm():
    """FSDP-aware clip_grad_norm_ should clip sharded gradients correctly."""
    from candle.distributed._composable.fsdp import fully_shard, clip_grad_norm_

    model = SimpleMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    x = torch.randn(4, 8, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Compute norm before clipping
    total_norm = clip_grad_norm_(model, max_norm=0.1)
    assert float(total_norm) > 0, "Total norm should be positive"

    # After clipping, recompute norm — should be <= max_norm + epsilon
    norm_after = clip_grad_norm_(model, max_norm=1e10)  # large max_norm = no-op
    assert float(norm_after) <= 0.1 + 1e-4, f"Clipped norm {float(norm_after)} exceeds max_norm 0.1"


# ======================================================================
# Phase 3 tests
# ======================================================================


def test_mixed_precision_forward_backward():
    """MixedPrecision: params cast to float16 during forward, grads reduced in float32."""
    from candle.distributed._composable.fsdp import (
        fully_shard, MixedPrecisionPolicy,
    )

    mp = MixedPrecisionPolicy(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    model = SimpleMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh, mp_policy=mp)
    fully_shard(model.fc2, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)

    x = torch.randn(4, 8, requires_grad=True)
    out = model(x)

    # output_dtype should cast output to float32
    assert out.dtype == torch.float32, f"Expected float32 output, got {out.dtype}"
    assert out.shape == (4, 8)

    loss = out.sum()
    loss.backward()

    # Gradients should exist and be in the original param dtype (float32 shards)
    for name, param in model.named_parameters():
        local = param.to_local() if isinstance(param, DTensor) else param
        assert local.grad is not None, f"No gradient for {name}"


def test_mixed_precision_param_dtype_casting():
    """Verify params are cast to param_dtype during forward."""
    from candle.distributed._composable.fsdp import (
        fully_shard, MixedPrecisionPolicy,
    )

    mp = MixedPrecisionPolicy(param_dtype=torch.float16)
    model = nn.Linear(8, 4)
    mesh = MockMesh()
    fully_shard(model, mesh=mesh, mp_policy=mp)

    # During forward, the hook unshards and casts to float16
    # We can verify by checking the param dtype inside forward
    observed_dtypes = []

    orig_forward = model.forward

    def capturing_forward(x):
        observed_dtypes.append(model.weight.dtype)
        return orig_forward(x)

    model.forward = capturing_forward

    x = torch.randn(2, 8)
    model(x)

    assert len(observed_dtypes) == 1
    assert observed_dtypes[0] == torch.float16, (
        f"Expected float16 during forward, got {observed_dtypes[0]}"
    )


def test_summon_full_params_read():
    """summon_full_params should expose full-sized params for reading."""
    from candle.distributed._composable.fsdp import fully_shard

    model = SimpleMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # Params should be sharded (DTensor) outside context
    assert isinstance(model.fc1.weight, DTensor)

    with model.summon_full_params():
        # Inside context, params should be unsharded (full-sized)
        w = model.fc1.weight
        assert w.shape == (16, 8), f"Expected (16, 8), got {w.shape}"
        assert not isinstance(w, DTensor), "Should be plain tensor inside summon"

    # After exit, params should be resharded
    assert isinstance(model.fc1.weight, DTensor)


def test_summon_full_params_writeback():
    """summon_full_params with writeback should persist modifications."""
    from candle.distributed._composable.fsdp import fully_shard

    model = nn.Linear(8, 4)
    mesh = MockMesh()
    fully_shard(model, mesh=mesh)

    with model.summon_full_params(writeback=True):
        # Zero out all weights
        model.weight.data = torch.zeros(4, 8)

    # After writeback, the shard should contain zeros
    local = model.weight.to_local() if isinstance(model.weight, DTensor) else model.weight
    assert float(local.abs().sum()) == 0.0, "Writeback should have zeroed the shard"


def test_summon_full_params_with_grads():
    """summon_full_params(with_grads=True) should expose gradients."""
    from candle.distributed._composable.fsdp import fully_shard

    model = nn.Linear(8, 4)
    mesh = MockMesh()
    fully_shard(model, mesh=mesh)

    # Run forward/backward to get gradients
    x = torch.randn(2, 8, requires_grad=True)
    out = model(x)
    out.sum().backward()

    with model.summon_full_params(with_grads=True):
        assert model.weight.grad is not None, "Grad should be available inside summon"


def test_state_dict_full_roundtrip():
    """Full state dict save/load round-trip."""
    from candle.distributed._composable.fsdp import (
        fully_shard, get_model_state_dict, set_model_state_dict,
    )

    model = SimpleMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # Get full state dict
    sd = get_model_state_dict(model, type="full")
    assert "fc1.weight" in sd
    assert "fc2.weight" in sd
    # Full state dict should have unsharded shapes
    assert sd["fc1.weight"].shape == (16, 8), f"Expected (16, 8), got {sd['fc1.weight'].shape}"

    # Modify model weights
    with model.summon_full_params(writeback=True):
        model.fc1.weight.data = torch.zeros(16, 8)

    # Load original state dict back
    set_model_state_dict(model, sd, type="full")

    # Verify restoration
    sd2 = get_model_state_dict(model, type="full")
    diff = float((sd2["fc1.weight"] - sd["fc1.weight"]).abs().sum())
    assert diff < 1e-6, f"State dict round-trip failed, diff={diff}"


def test_state_dict_sharded_roundtrip():
    """Sharded state dict save/load round-trip."""
    from candle.distributed._composable.fsdp import (
        fully_shard, get_model_state_dict, set_model_state_dict,
    )

    model = nn.Linear(8, 4)
    mesh = MockMesh()
    fully_shard(model, mesh=mesh)

    sd = get_model_state_dict(model, type="sharded")
    assert "weight" in sd

    # Modify and restore
    orig_weight = sd["weight"].detach()
    with model.summon_full_params(writeback=True):
        model.weight.data = torch.zeros(4, 8)

    set_model_state_dict(model, {"weight": orig_weight, "bias": sd["bias"]}, type="sharded")

    sd2 = get_model_state_dict(model, type="sharded")
    diff = float((sd2["weight"] - orig_weight).abs().sum())
    assert diff < 1e-6, f"Sharded state dict round-trip failed, diff={diff}"


def test_shard_grad_op_strategy():
    """reshard_after_forward=False (SHARD_GRAD_OP) keeps params unsharded after forward."""
    from candle.distributed._composable.fsdp import fully_shard

    model = SimpleMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh, reshard_after_forward=False)
    fully_shard(model.fc2, mesh=mesh, reshard_after_forward=False)
    fully_shard(model, mesh=mesh, reshard_after_forward=False)

    x = torch.randn(4, 8, requires_grad=True)
    out = model(x)
    assert out.shape == (4, 8)

    # After forward with reshard_after_forward=False, params should still be unsharded
    # (the post_forward hook skips reshard)
    w = model.fc1.weight
    assert not isinstance(w, DTensor), (
        "With SHARD_GRAD_OP, params should stay unsharded after forward"
    )

    # Backward should still work
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        local = param.to_local() if isinstance(param, DTensor) else param
        assert local.grad is not None, f"No gradient for {name}"
