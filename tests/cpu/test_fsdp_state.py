"""Tests for FSDPState hook orchestration (single-process)."""
import candle as torch
import candle.nn as nn
from candle.distributed.tensor.dtensor import DTensor


class MockMeshInfo:
    def __init__(self):
        self.shard_mesh_rank = 0
        self.shard_mesh_size = 1
        self.shard_process_group = None
        class _M:
            def size(self, dim=0): return 1
            @property
            def ndim(self): return 1
        self.mesh = _M()


def _apply_fsdp_single_rank(module):
    """Apply FSDP to a module for single-rank testing."""
    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam
    from candle.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
    from candle.distributed._composable.fsdp._fsdp_state import FSDPState

    mesh_info = MockMeshInfo()
    params = [(name, param) for name, param in module.named_parameters(recurse=False)]
    fsdp_params = [FSDPParam(p, module, name, mesh_info) for name, p in params]

    # Override unshard to single-rank version
    for fp in fsdp_params:
        fp.unshard = fp._unshard_single_rank

    group = FSDPParamGroup(fsdp_params, module, mesh_info)
    state = FSDPState(module, group, mesh_info, reshard_after_forward=True)
    module._fsdp_state = state
    return state


def test_fsdp_state_pre_forward_unshards():
    """Pre-forward hook should unshard parameters."""
    m = nn.Linear(8, 4)
    _apply_fsdp_single_rank(m)
    assert isinstance(m.weight, DTensor)
    x = torch.randn(2, 8)
    out = m(x)
    assert out.shape == (2, 4)


def test_fsdp_state_post_forward_reshards():
    """Post-forward hook should reshard parameters (restore DTensor)."""
    m = nn.Linear(8, 4)
    state = _apply_fsdp_single_rank(m)
    x = torch.randn(2, 8)
    out = m(x)
    assert isinstance(m.weight, DTensor), "Expected weight to be resharded to DTensor after forward"


def test_fsdp_nested_parent_detection():
    """fully_shard() should correctly detect parent/child FSDP relationships."""
    from candle.distributed._composable.fsdp import fully_shard

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

    class ThreeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 8)
            self.head = nn.Linear(8, 1)  # root-owned param
        def forward(self, x):
            return self.head(self.fc2(self.fc1(x)))

    model = ThreeLayer()
    mesh = MockMesh()

    # Bottom-up: shard children first, then root
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)  # root has head's params

    # Children should have _fsdp_has_parent set
    assert getattr(model.fc1, '_fsdp_has_parent', False), "fc1 should have _fsdp_has_parent"
    assert getattr(model.fc2, '_fsdp_has_parent', False), "fc2 should have _fsdp_has_parent"

    # Root should NOT have _fsdp_has_parent
    assert not getattr(model, '_fsdp_has_parent', False)

    # Trigger lazy init via forward
    x = torch.randn(2, 8)
    out = model(x)

    # Root should be detected as root
    assert model._fsdp_state._is_root is True
    # Children should be detected as non-root
    assert model.fc1._fsdp_state._is_root is False
    assert model.fc2._fsdp_state._is_root is False
