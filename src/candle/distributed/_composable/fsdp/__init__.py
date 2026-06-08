"""FSDP2 composable API -- fully_shard().

Usage (bottom-up):
    mesh = DeviceMesh("npu", (world_size,))
    fully_shard(model.encoder, mesh=mesh)
    fully_shard(model.decoder, mesh=mesh)
    fully_shard(model, mesh=mesh)  # root
"""
from contextlib import contextmanager
from functools import wraps

from ._fsdp_common import (
    AllGather,
    CPUOffloadPolicy,
    Comm,
    FSDPMeshInfo,
    MixedPrecisionPolicy,
    OffloadPolicy,
    ReduceScatter,
)
from ._fsdp_param import FSDPParam
from ._fsdp_param_group import FSDPParamGroup
from ._fsdp_state import FSDPState
from ._fsdp_state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
)


class UnshardHandle:
    """Handle returned by ``FSDPModule.unshard(async_op=True)``."""

    def wait(self):
        """Wait for the unshard operation to finish.

        Candle's current unshard path is synchronous, so this is a no-op kept
        for PyTorch FSDP2 API compatibility.
        """
        return None


class FSDPModule:
    """Mixin injected into module's MRO by fully_shard()."""

    @property
    def fsdp_state(self):
        return self._fsdp_state

    def _get_fsdp_state(self):
        state = getattr(self, "_fsdp_state", None)
        if state is None:
            raise AssertionError("Expected an FSDPModule with FSDP state")
        return state

    def reshard(self):
        """Reshard this module's parameters."""
        state = getattr(self, "_fsdp_state", None)
        if state is None:
            for child_state in _iter_fsdp_states(self, recurse=True):
                if child_state.param_group is not None:
                    child_state.param_group.reshard()
            return
        if state.param_group is not None:
            state.param_group.reshard()

    def unshard(self, async_op=False):
        """Unshard this module's parameters."""
        state = getattr(self, "_fsdp_state", None)
        if state is None:
            for child_state in _iter_fsdp_states(self, recurse=True):
                child_state._unshard_async_op = bool(async_op)
                if child_state.param_group is not None:
                    child_state.param_group.unshard()
            return UnshardHandle() if async_op else None
        state._unshard_async_op = bool(async_op)  # pylint: disable=protected-access
        if state.param_group is not None:
            state.param_group.unshard()
        return UnshardHandle() if async_op else None

    def set_is_last_backward(self, is_last_backward):
        self._get_fsdp_state()._is_last_backward = bool(is_last_backward)


    def set_requires_gradient_sync(self, requires_gradient_sync, *, recurse=True):
        for state in _iter_fsdp_states(self, recurse=recurse):
            state._sync_gradients = bool(requires_gradient_sync)
            if state.param_group is not None:
                state.param_group.reduce_grads = bool(requires_gradient_sync)


    def set_requires_all_reduce(self, requires_all_reduce, *, recurse=True):
        for state in _iter_fsdp_states(self, recurse=recurse):
            state._requires_all_reduce = bool(requires_all_reduce)
            if state.param_group is not None:
                state.param_group.all_reduce_grads = bool(requires_all_reduce)


    def set_reshard_after_forward(self, reshard_after_forward, recurse=True):
        if not isinstance(reshard_after_forward, bool):
            raise TypeError("reshard_after_forward must be a bool")
        for state in _iter_fsdp_states(self, recurse=recurse):
            state.reshard_after_forward = reshard_after_forward


    def set_reshard_after_backward(self, reshard_after_backward, *, recurse=True):
        if not isinstance(reshard_after_backward, bool):
            raise TypeError("reshard_after_backward must be a bool")
        for state in _iter_fsdp_states(self, recurse=recurse):
            state._reshard_after_backward = reshard_after_backward
            if state.param_group is not None:
                state.param_group.reshard_after_backward = reshard_after_backward


    def set_modules_to_forward_prefetch(self, modules):
        states = _states_from_fsdp_modules(modules)
        self._get_fsdp_state()._states_to_forward_prefetch = states


    def set_modules_to_backward_prefetch(self, modules):
        states = _states_from_fsdp_modules(modules)
        self._get_fsdp_state()._states_to_backward_prefetch = states


    def set_custom_all_gather(self, comm):
        if not isinstance(comm, AllGather):
            raise TypeError("comm must be an AllGather")
        raise NotImplementedError(
            "custom all-gather is not implemented for Candle FSDP2"
        )


    def set_custom_reduce_scatter(self, comm):
        if not isinstance(comm, ReduceScatter):
            raise TypeError("comm must be a ReduceScatter")
        raise NotImplementedError(
            "custom reduce-scatter is not implemented for Candle FSDP2"
        )


    def set_all_reduce_hook(self, hook, *, stream=None):
        del stream
        if not callable(hook):
            raise TypeError("hook must be callable")
        raise NotImplementedError(
            "all-reduce hook is not implemented for Candle FSDP2"
        )


    def set_post_optim_event(self, event):
        state = self._get_fsdp_state()
        state._post_optim_event = event
        if state.param_group is not None:
            state.param_group._post_optim_event = event


    def set_reduce_scatter_divide_factor(self, factor):
        factor = float(factor)
        if factor != 1.0:
            raise NotImplementedError(
                "reduce-scatter divide factor other than 1.0 is not "
                "implemented for Candle FSDP2"
            )
        for state in _iter_fsdp_states(self, recurse=True):
            state._reduce_scatter_divide_factor = factor
            if state.param_group is not None:
                state.param_group.reduce_scatter_divide_factor = factor


    def set_gradient_divide_factor(self, factor):
        factor = float(factor)
        if factor != 1.0:
            raise NotImplementedError(
                "gradient divide factor other than 1.0 is not implemented "
                "for Candle FSDP2"
            )
        for state in _iter_fsdp_states(self, recurse=True):
            state._gradient_divide_factor = factor
            if state.param_group is not None:
                state.param_group.gradient_divide_factor = factor


    def set_force_sum_reduction_for_comms(self, enable):
        enable = bool(enable)
        if enable:
            raise NotImplementedError(
                "force-sum reduction for comms is not implemented for Candle FSDP2"
            )
        for state in _iter_fsdp_states(self, recurse=True):
            state._force_sum_reduction_for_comms = enable
            if state.param_group is not None:
                state.param_group.force_sum_reduction_for_comms = enable


    def set_unshard_in_backward(self, unshard_in_backward):
        state = self._get_fsdp_state()
        state._unshard_in_backward = bool(unshard_in_backward)
        if state.param_group is not None:
            state.param_group.unshard_in_backward = bool(unshard_in_backward)


    def set_allocate_memory_from_process_group_for_comm(self, enable):
        state = self._get_fsdp_state()
        state._allocate_memory_from_process_group_for_comm = bool(enable)
        if state.param_group is not None:
            state.param_group.allocate_memory_from_process_group_for_comm = bool(enable)

    def _set_unshard_async_op(self, async_op):
        state = self._get_fsdp_state()
        state._unshard_async_op = bool(async_op)

    def _apply(self, fn):
        state = getattr(self, "_fsdp_state", None)
        if state is None:
            for child in self.children():
                child._apply(fn)
            _apply_to_buffers(self, fn)
            return self

        self.reshard()
        for child in self.children():
            child_state = getattr(child, "_fsdp_state", None)
            if child_state is not None and child_state is not state:
                child._apply(fn)
        if state.param_group is not None:
            _apply_to_param_group(state.param_group, fn)
        _apply_to_buffers(self, fn)
        return self

    @contextmanager
    def no_sync(self):
        """Skip gradient reduce-scatter for gradient accumulation."""
        states = list(_iter_fsdp_states(self, recurse=True))
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
        """Context manager that unshards all params in the FSDP subtree."""
        param_groups = []
        for mod in self.modules():
            st = getattr(mod, "_fsdp_state", None)
            if st is not None and st.param_group is not None:
                param_groups.append(st.param_group)

        for param_group in param_groups:
            param_group.unshard()

        if with_grads:
            for param_group in param_groups:
                for fsdp_param in param_group.fsdp_params:
                    shard_grad = fsdp_param._sharded_param.to_local().grad
                    if (shard_grad is not None
                            and fsdp_param._unsharded_param is not None):
                        fsdp_param._unsharded_param.grad = shard_grad

        try:
            yield
        finally:
            if writeback:
                for param_group in param_groups:
                    for fsdp_param in param_group.fsdp_params:
                        if fsdp_param._unsharded_param is not None:
                            _writeback_to_shard(fsdp_param)
                            if (with_grads
                                    and fsdp_param._unsharded_param.grad is not None):
                                _writeback_grad_to_shard(fsdp_param)
            for param_group in param_groups:
                param_group.reshard()


class _MockMeshInfo:
    """MeshInfo for single-process / mock scenarios (world_size=1)."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.shard_mesh_size = mesh.size(0)
        self.shard_process_group = (
            mesh.get_group(0)
            if hasattr(mesh, "_dim_groups") and mesh._dim_groups
            else None
        )
        self.shard_mesh_rank = 0
        if (self.shard_process_group is not None
                and hasattr(self.shard_process_group, "rank")):
            self.shard_mesh_rank = self.shard_process_group.rank()


class _SingleDeviceMesh:
    """Single-rank mesh used when PyTorch-compatible ``mesh=None`` is given."""

    device_type = "cpu"
    _mesh_shape = (1,)
    mesh_dim_names = ("shard",)
    _dim_groups = []

    def get_group(self, dim=0):  # pylint: disable=unused-argument
        return None

    def size(self, dim=0):  # pylint: disable=unused-argument
        return 1

    def get_local_rank(self, dim=0):  # pylint: disable=unused-argument
        return 0

    @property
    def ndim(self):
        return 1


def fully_shard(
    module,
    *,
    mesh=None,
    reshard_after_forward=None,
    shard_placement_fn=None,
    mp_policy=MixedPrecisionPolicy(),
    offload_policy=OffloadPolicy(),
    ignored_params=None,
):
    """Apply FSDP2 to a module or list of modules.

    The Python signature mirrors PyTorch FSDP2.  Runtime behavior is backed by
    Candle's native DTensor/FSDP implementation.
    """
    if isinstance(module, (list, tuple)):
        return [
            fully_shard(
                mod,
                mesh=mesh,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=shard_placement_fn,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                ignored_params=ignored_params,
            )
            for mod in module
        ]

    if isinstance(offload_policy, CPUOffloadPolicy):
        raise NotImplementedError("CPU offload is not implemented for Candle FSDP2")
    if not isinstance(offload_policy, OffloadPolicy):
        raise TypeError("offload_policy must be an OffloadPolicy")

    if (isinstance(reshard_after_forward, int)
            and not isinstance(reshard_after_forward, bool)):
        raise NotImplementedError(
            "Integer reshard_after_forward is not implemented for Candle FSDP2"
        )

    if mesh is None:
        mesh = _SingleDeviceMesh()

    try:
        mesh_info = FSDPMeshInfo(mesh)
    except (RuntimeError, AttributeError):
        mesh_info = _MockMeshInfo(mesh)

    for child in module.modules():
        if child is module:
            continue
        if hasattr(child, "_fsdp_state") and child._fsdp_state is not None:
            child._fsdp_has_parent = True

    ignored_param_ids = {id(param) for param in (ignored_params or set())}
    managed_params = _get_managed_params(module, ignored_param_ids)
    if not managed_params:
        module._fsdp_state = None
        _inject_fsdp_mixin(module)
        return module

    fsdp_params = []
    for name, param, owner in managed_params:
        shard_dim = _get_shard_dim(param, shard_placement_fn)
        fsdp_params.append(
            FSDPParam(
                param,
                owner,
                name,
                mesh_info,
                mp_policy=mp_policy,
                shard_dim=shard_dim,
            )
        )

    if mesh_info.shard_mesh_size == 1:
        for fsdp_param in fsdp_params:
            fsdp_param.unshard = fsdp_param._unshard_single_rank

    param_group = FSDPParamGroup(fsdp_params, module, mesh_info)

    if reshard_after_forward is None:
        reshard_after_forward = True

    state = FSDPState(
        module, param_group, mesh_info, reshard_after_forward,
        mp_policy=mp_policy,
    )
    module._fsdp_state = state

    _inject_fsdp_mixin(module)

    return module


def register_fsdp_forward_method(module, method_name):
    """Register a custom FSDP forward method on ``module``.

    Non-FSDP modules are accepted as a no-op for PyTorch API compatibility.
    """
    if not isinstance(module, FSDPModule):
        return None
    state = getattr(module, "_fsdp_state", None)
    if state is None:
        return None
    if not hasattr(module, method_name):
        raise ValueError(f"FSDPModule has no method named {method_name!r}")
    if method_name == "forward":
        return None

    method = getattr(module, method_name)
    if getattr(method, "_fsdp_wrapped_method", False):
        return None

    @wraps(method)
    def wrapped(*args, **kwargs):
        new_args, new_kwargs = state._pre_forward(module, args, kwargs)
        output = method(*new_args, **new_kwargs)
        return state._post_forward(module, new_args, output)

    wrapped._fsdp_wrapped_method = True
    setattr(module, method_name, wrapped)
    return None


def share_comm_ctx(modules):
    """Share a communication context across FSDP modules."""
    if not modules:
        return None
    states = _states_from_fsdp_modules(modules)
    comm_ctx = states[0]._comm_ctx
    for state in states[1:]:
        state._comm_ctx = comm_ctx
        if state.param_group is not None:
            state.param_group._comm_ctx = comm_ctx
    return None


def _apply_to_param_group(param_group, fn):
    """Apply ``fn`` to local shards while preserving DTensor wrappers."""
    from ...tensor.dtensor import DTensor
    from ...tensor.placement import Shard

    for fsdp_param in param_group.fsdp_params:
        old_dtensor = fsdp_param._sharded_param
        old_local = old_dtensor.to_local()
        new_local = fn(old_local)
        new_local.requires_grad = old_dtensor.requires_grad
        new_dtensor = DTensor.from_local(
            new_local,
            fsdp_param._mesh_info.mesh,
            placements=(Shard(fsdp_param._shard_dim),),
        )
        new_dtensor.requires_grad = old_dtensor.requires_grad
        fsdp_param._sharded_param = new_dtensor
        fsdp_param._orig_dtype = new_local.dtype
        fsdp_param._set_param_on_module(new_dtensor)
    if param_group._use_flat_buffer:
        param_group._init_flat_buffer()


def _apply_to_buffers(module, fn):
    """Apply ``fn`` to buffers on a module, matching ``Module._apply``."""
    for key, buffer in module._buffers.items():
        if buffer is not None:
            new_buffer = fn(buffer)
            module._buffers[key] = new_buffer
            module.__dict__[key] = new_buffer


def _inject_fsdp_mixin(module):
    """Dynamically inject FSDPModule into the module's class hierarchy."""
    cls = type(module)
    if FSDPModule not in cls.__mro__:
        new_cls = type(f"FSDP_{cls.__name__}", (FSDPModule, cls), {})
        module.__class__ = new_cls


def _iter_fsdp_states(module, recurse=True):
    modules = module.modules() if recurse else (module,)
    for mod in modules:
        state = getattr(mod, "_fsdp_state", None)
        if state is not None:
            yield state


def _states_from_fsdp_modules(modules):
    states = []
    for module in modules:
        if not isinstance(module, FSDPModule):
            raise ValueError("Expected all modules to be FSDPModule instances")
        state = getattr(module, "_fsdp_state", None)
        if state is None:
            raise ValueError("Expected all modules to have FSDP state")
        states.append(state)
    return states


def _get_shard_dim(param, shard_placement_fn):
    if shard_placement_fn is None:
        return 0
    placement = shard_placement_fn(param)
    if placement is None:
        return 0
    from ...tensor.placement import Shard
    if not isinstance(placement, Shard):
        raise TypeError("shard_placement_fn must return Shard or None")
    return placement.dim


def _get_managed_params(module, ignored_param_ids=None):
    """Collect params owned by *module*, excluding child FSDP and ignored params."""
    ignored_param_ids = ignored_param_ids or set()
    child_fsdp_params = set()
    for child in module.modules():
        if child is module:
            continue
        if hasattr(child, "_fsdp_state") and child._fsdp_state is not None:
            for param in child.parameters():
                child_fsdp_params.add(id(param))
    managed = []
    for name, param in module.named_parameters():
        param_id = id(param)
        if param_id in child_fsdp_params or param_id in ignored_param_ids:
            continue
        leaf_name = name.split(".")[-1]
        parts = name.split(".")
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
    """FSDP-aware gradient clipping that handles sharded DTensor gradients."""
    from ...tensor.dtensor import DTensor
    from ...._functional import abs, pow, sum, sqrt, amax, amin, stack, clamp, mul
    from ...._creation import tensor

    del sqrt  # imported for API continuity with the previous implementation
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    grads = []
    for param in model.parameters():
        if isinstance(param, DTensor):
            local = param.to_local()
            if local.grad is not None:
                grads.append(local.grad)
        elif param.grad is not None:
            grads.append(param.grad)

    if not grads:
        return tensor(0.0)

    def _norm(tensor_):
        if norm_type == float("inf"):
            return amax(abs(tensor_))
        if norm_type == float("-inf"):
            return amin(abs(tensor_))
        return pow(sum(pow(abs(tensor_), norm_type)), 1.0 / norm_type)

    norms = [_norm(grad) for grad in grads]

    if norm_type == float("inf"):
        local_norm = amax(stack(norms))
    elif norm_type == float("-inf"):
        local_norm = amin(stack(norms))
    else:
        local_norm = pow(
            sum(stack([pow(norm, norm_type) for norm in norms])),
            1.0 / norm_type,
        )

    total_norm = local_norm

    if error_if_nonfinite:
        from ...._functional import isnan, isinf, any as any_
        if any_(isnan(total_norm)) or any_(isinf(total_norm)):
            raise RuntimeError(
                f"Total norm of order {norm_type} is non-finite, "
                "cannot clip. Set error_if_nonfinite=False to disable."
            )

    inv_norm = pow(total_norm + tensor(1e-6), -1.0)
    clip_coef = clamp(mul(tensor(max_norm), inv_norm), max_val=1.0)

    for grad in grads:
        grad.data = grad * clip_coef

    return total_norm


__all__ = [
    "AllGather",
    "CPUOffloadPolicy",
    "Comm",
    "FSDPModule",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "ReduceScatter",
    "UnshardHandle",
    "clip_grad_norm_",
    "fully_shard",
    "get_model_state_dict",
    "get_optimizer_state_dict",
    "register_fsdp_forward_method",
    "set_model_state_dict",
    "share_comm_ctx",
]

_PUBLIC_MODULE = "torch.distributed.fsdp"
FSDPModule.__module__ = _PUBLIC_MODULE
UnshardHandle.__module__ = _PUBLIC_MODULE
fully_shard.__module__ = _PUBLIC_MODULE
register_fsdp_forward_method.__module__ = _PUBLIC_MODULE
share_comm_ctx.__module__ = _PUBLIC_MODULE
