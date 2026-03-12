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
