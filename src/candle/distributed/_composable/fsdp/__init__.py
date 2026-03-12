"""FSDP2 composable API -- fully_shard().

Usage (bottom-up):
    mesh = DeviceMesh("npu", (world_size,))
    fully_shard(model.encoder, mesh=mesh)
    fully_shard(model.decoder, mesh=mesh)
    fully_shard(model, mesh=mesh)  # root
"""
from contextlib import contextmanager

from ._fsdp_common import FSDPMeshInfo
from ._fsdp_param import FSDPParam
from ._fsdp_param_group import FSDPParamGroup
from ._fsdp_state import FSDPState


class FSDPModule:
    """Mixin injected into module's MRO by fully_shard()."""

    @property
    def fsdp_state(self):
        return self._fsdp_state

    def set_reshard_after_forward(self, value):
        self._fsdp_state.reshard_after_forward = value

    def set_modules_to_forward_prefetch(self, modules):
        pass  # MVP no-op

    def set_modules_to_backward_prefetch(self, modules):
        pass  # MVP no-op

    @contextmanager
    def no_sync(self):
        """Skip gradient reduce-scatter for gradient accumulation.

        Gradients are accumulated on the unsharded params. The final
        backward (outside no_sync) will reduce-scatter the accumulated
        gradients.
        """
        # Collect all FSDP states in the subtree
        states = []
        for mod in self.modules():
            st = getattr(mod, '_fsdp_state', None)
            if st is not None:
                states.append(st)
        old_values = [st._sync_gradients for st in states]
        for st in states:
            st._sync_gradients = False
        try:
            yield
        finally:
            for st, old in zip(states, old_values):
                st._sync_gradients = old


class _MockMeshInfo:
    """MeshInfo for single-process / mock scenarios (world_size=1)."""
    def __init__(self, mesh):
        self.mesh = mesh
        self.shard_mesh_size = mesh.size(0)
        self.shard_process_group = (
            mesh.get_group(0)
            if hasattr(mesh, '_dim_groups') and mesh._dim_groups
            else None
        )
        self.shard_mesh_rank = 0
        if (self.shard_process_group is not None
                and hasattr(self.shard_process_group, 'rank')):
            self.shard_mesh_rank = self.shard_process_group.rank()


def fully_shard(module, *, mesh, reshard_after_forward=None):
    """Apply FSDP2 to a module. PyTorch-compatible composable API."""
    # 1. Build mesh info
    try:
        mesh_info = FSDPMeshInfo(mesh)
    except (RuntimeError, AttributeError):
        mesh_info = _MockMeshInfo(mesh)

    # Mark child FSDP modules as non-root (must happen before early return)
    for child in module.modules():
        if child is module:
            continue
        if hasattr(child, '_fsdp_state') and child._fsdp_state is not None:
            child._fsdp_has_parent = True

    # 2. Collect directly-owned parameters (exclude child fully_shard params)
    managed_params = _get_managed_params(module)
    if not managed_params:
        module._fsdp_state = None
        _inject_fsdp_mixin(module)
        return module

    # 3. Create FSDPParam for each parameter
    fsdp_params = [
        FSDPParam(param, owner, name, mesh_info)
        for name, param, owner in managed_params
    ]

    # For single-rank, override unshard to skip collectives
    if mesh_info.shard_mesh_size == 1:
        for fp in fsdp_params:
            fp.unshard = fp._unshard_single_rank

    # 4. Group parameters
    param_group = FSDPParamGroup(fsdp_params, module, mesh_info)

    # 5. Reshard strategy
    if reshard_after_forward is None:
        reshard_after_forward = True

    # 6. Create FSDPState and register hooks
    state = FSDPState(module, param_group, mesh_info, reshard_after_forward)
    module._fsdp_state = state

    # 7. Inject FSDPModule mixin
    _inject_fsdp_mixin(module)

    return module


def _inject_fsdp_mixin(module):
    """Dynamically inject FSDPModule into the module's class hierarchy."""
    cls = type(module)
    if FSDPModule not in cls.__mro__:
        new_cls = type(f"FSDP_{cls.__name__}", (FSDPModule, cls), {})
        module.__class__ = new_cls


def _get_managed_params(module):
    """Collect params owned by *module*, excluding those managed by child FSDP."""
    child_fsdp_params = set()
    for child in module.modules():
        if child is module:
            continue
        if hasattr(child, '_fsdp_state') and child._fsdp_state is not None:
            for p in child.parameters():
                child_fsdp_params.add(id(p))
    managed = []
    for name, param in module.named_parameters():
        if id(param) not in child_fsdp_params:
            # Use the leaf name (last component) for _set_param_on_module
            leaf_name = name.split('.')[-1]
            # Find the owning submodule
            parts = name.split('.')
            owner = module
            for part in parts[:-1]:
                owner = getattr(owner, part)
            managed.append((leaf_name, param, owner))
    return managed
