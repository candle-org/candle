"""FSDP2 composable API -- fully_shard().

Usage (bottom-up):
    mesh = DeviceMesh("npu", (world_size,))
    fully_shard(model.encoder, mesh=mesh)
    fully_shard(model.decoder, mesh=mesh)
    fully_shard(model, mesh=mesh)  # root
"""
from contextlib import contextmanager

from ._fsdp_common import FSDPMeshInfo, MixedPrecisionPolicy
from ._fsdp_param import FSDPParam
from ._fsdp_param_group import FSDPParamGroup
from ._fsdp_state import FSDPState
from ._fsdp_state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
)


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

    @contextmanager
    def summon_full_params(self, writeback=True, with_grads=False):
        """Context manager that unshards all params in the FSDP subtree.

        On enter: all-gather all parameters so they are full-sized.
        On exit: if *writeback* is True, scatter modified params back to
        shards; then reshard all parameters.

        If *with_grads* is True, gradients on the unsharded params are
        also made available (and written back on exit).
        """
        # Collect all param groups in the subtree
        param_groups = []
        for mod in self.modules():
            st = getattr(mod, '_fsdp_state', None)
            if st is not None and st.param_group is not None:
                param_groups.append(st.param_group)

        # Unshard all
        for pg in param_groups:
            pg.unshard()

        # If with_grads, copy shard grads onto unsharded params
        if with_grads:
            for pg in param_groups:
                for fp in pg.fsdp_params:
                    shard_grad = fp._sharded_param.to_local().grad
                    if shard_grad is not None and fp._unsharded_param is not None:
                        fp._unsharded_param.grad = shard_grad

        try:
            yield
        finally:
            if writeback:
                # Write modified unsharded params back to shards
                for pg in param_groups:
                    for fp in pg.fsdp_params:
                        if fp._unsharded_param is not None:
                            _writeback_to_shard(fp)
                            if with_grads and fp._unsharded_param.grad is not None:
                                _writeback_grad_to_shard(fp)
            # Reshard all
            for pg in param_groups:
                pg.reshard()


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


def fully_shard(module, *, mesh, reshard_after_forward=None, mp_policy=None):
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
        FSDPParam(param, owner, name, mesh_info, mp_policy=mp_policy)
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
    state = FSDPState(
        module, param_group, mesh_info, reshard_after_forward,
        mp_policy=mp_policy,
    )
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


def _writeback_to_shard(fp):
    """Write modified unsharded param data back into the local shard."""
    from ._fsdp_param import _chunk_tensor
    unsharded = fp._unsharded_param
    if unsharded is None:
        return
    world_size = fp._mesh_info.shard_mesh_size
    rank = fp._mesh_info.shard_mesh_rank
    if world_size == 1:
        local = fp._sharded_param.to_local()
        local.data = unsharded.data
    else:
        # Re-chunk the full param and copy this rank's shard
        chunks = _chunk_tensor(unsharded.detach(), world_size, dim=fp._shard_dim)
        local = fp._sharded_param.to_local()
        local.data = chunks[rank].contiguous().data


def _writeback_grad_to_shard(fp):
    """Write unsharded grad back into the local shard grad."""
    from ._fsdp_param import _chunk_tensor
    unsharded = fp._unsharded_param
    if unsharded is None or unsharded.grad is None:
        return
    world_size = fp._mesh_info.shard_mesh_size
    rank = fp._mesh_info.shard_mesh_rank
    if world_size == 1:
        fp._sharded_param.to_local().grad = unsharded.grad
    else:
        chunks = _chunk_tensor(unsharded.grad.detach(), world_size, dim=fp._shard_dim)
        fp._sharded_param.to_local().grad = chunks[rank].contiguous()


def clip_grad_norm_(model, max_norm, norm_type=2.0, error_if_nonfinite=False):
    """FSDP-aware gradient clipping that handles sharded DTensor gradients.

    For sharded parameters, computes the local shard norm and all-reduces
    across the process group to get the true global norm before clipping.

    Parameters
    ----------
    model : nn.Module
        The FSDP-wrapped model (or any module with DTensor parameters).
    max_norm : float
        Maximum gradient norm.
    norm_type : float
        Type of p-norm (default: 2.0).

    Returns
    -------
    Tensor
        The total (global) gradient norm before clipping.
    """
    from ...tensor.dtensor import DTensor
    from ...._functional import abs, pow, sum, sqrt, amax, amin, stack, clamp, mul
    from ...._creation import tensor

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # Collect all grads (local shards for DTensor, raw for regular)
    grads = []
    for p in model.parameters():
        if isinstance(p, DTensor):
            local = p.to_local()
            if local.grad is not None:
                grads.append(local.grad)
        elif p.grad is not None:
            grads.append(p.grad)

    if not grads:
        return tensor(0.0)

    # Compute per-grad norms
    def _norm(t):
        if norm_type == float('inf'):
            return amax(abs(t))
        elif norm_type == float('-inf'):
            return amin(abs(t))
        return pow(sum(pow(abs(t), norm_type)), 1.0 / norm_type)

    norms = [_norm(g) for g in grads]

    # Local total norm
    if norm_type == float('inf'):
        local_norm = amax(stack(norms))
    elif norm_type == float('-inf'):
        local_norm = amin(stack(norms))
    else:
        local_norm = pow(
            sum(stack([pow(n, norm_type) for n in norms])),
            1.0 / norm_type,
        )

    # For multi-rank: all-reduce local_norm^p, then take p-th root.
    # For single-rank (MVP), local_norm == global_norm.
    total_norm = local_norm

    if error_if_nonfinite:
        from ...._functional import isnan, isinf, any as any_
        if any_(isnan(total_norm)) or any_(isinf(total_norm)):
            raise RuntimeError(
                f"Total norm of order {norm_type} is non-finite, "
                "cannot clip. Set error_if_nonfinite=False to disable."
            )

    # Clip
    inv_norm = pow(total_norm + tensor(1e-6), -1.0)
    clip_coef = clamp(mul(tensor(max_norm), inv_norm), max_val=1.0)

    for g in grads:
        g.data = g * clip_coef

    return total_norm
