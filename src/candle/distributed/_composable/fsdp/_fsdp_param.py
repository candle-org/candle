"""FSDPParam -- per-parameter shard lifecycle management."""
from enum import Enum, auto
from ....distributed.tensor.dtensor import DTensor
from ....distributed.tensor.placement import Shard
from .... import distributed as dist


class ShardedState(Enum):
    SHARDED = auto()
    UNSHARDED = auto()


class FSDPParam:
    """Manages the shard/unshard lifecycle for a single parameter.

    On construction, the original parameter is chunked along dim-0 and
    wrapped as a DTensor that is stored back on the module.  ``unshard()``
    reconstructs the full parameter via all-gather; ``reshard()`` restores
    the DTensor view.
    """

    def __init__(self, param, module, param_name, mesh_info):
        self._module = module
        self._param_name = param_name
        self._mesh_info = mesh_info
        self._sharded_state = ShardedState.SHARDED
        self._orig_shape = param.shape
        self._orig_dtype = param.dtype
        self._shard_dim = 0

        self._sharded_param = self._init_shard(param)
        self._unsharded_param = None

        # Replace on module
        self._set_param_on_module(self._sharded_param)

    # ------------------------------------------------------------------
    # Shard initialisation
    # ------------------------------------------------------------------

    def _init_shard(self, param):
        """Chunk *param* along ``self._shard_dim`` and wrap as DTensor."""
        rank = self._mesh_info.shard_mesh_rank
        world_size = self._mesh_info.shard_mesh_size
        orig_requires_grad = param.requires_grad
        if world_size == 1:
            self._padded_dim_size = 0  # no padding needed
            local_shard = param.detach()
        else:
            dim = self._shard_dim
            dim_size = param.shape[dim]
            chunk_size = (dim_size + world_size - 1) // world_size
            padded_size = chunk_size * world_size
            self._padded_dim_size = padded_size - dim_size  # 0 if already divisible

            if self._padded_dim_size > 0:
                from ...._creation import zeros
                pad_shape = list(param.shape)
                pad_shape[dim] = self._padded_dim_size
                padded = _cat_tensors(
                    [param.detach(), zeros(*pad_shape, dtype=param.dtype, device=param.device)],
                    dim=dim,
                )
            else:
                padded = param.detach()

            chunks = _chunk_tensor(padded, world_size, dim=dim)
            local_shard = chunks[rank].contiguous()
        dt = DTensor.from_local(
            local_shard,
            self._mesh_info.mesh,
            placements=(Shard(self._shard_dim),),
        )
        dt.requires_grad = orig_requires_grad
        return dt

    # ------------------------------------------------------------------
    # Unshard / reshard
    # ------------------------------------------------------------------

    def unshard(self):
        """Reconstruct the full parameter via all-gather."""
        if self._sharded_state == ShardedState.UNSHARDED:
            return
        local_tensor = self._sharded_param.to_local()
        world_size = self._mesh_info.shard_mesh_size
        if world_size == 1:
            self._unsharded_param = local_tensor
        else:
            from ...._creation import empty
            shard_size = local_tensor.shape[self._shard_dim]
            full_size = list(local_tensor.shape)
            full_size[self._shard_dim] = shard_size * world_size
            output = empty(
                *full_size,
                dtype=local_tensor.dtype,
                device=local_tensor.device,
            )
            pg = self._mesh_info.shard_process_group
            dist.all_gather_into_tensor(output, local_tensor, group=pg)
            # Strip padding to restore original shape
            if self._padded_dim_size > 0:
                from ...._functional import narrow
                orig_dim_size = self._orig_shape[self._shard_dim]
                output = narrow(output, self._shard_dim, 0, orig_dim_size)
            self._unsharded_param = output
        self._unsharded_param.requires_grad = self._sharded_param.requires_grad
        self._set_param_on_module(self._unsharded_param)
        self._sharded_state = ShardedState.UNSHARDED

    def _unshard_single_rank(self):
        """Unshard for single-rank testing (no collective needed)."""
        if self._sharded_state == ShardedState.UNSHARDED:
            return
        local = self._sharded_param.to_local()
        # Strip padding to restore original shape
        if self._padded_dim_size > 0:
            from ...._functional import narrow
            orig_dim_size = self._orig_shape[self._shard_dim]
            local = narrow(local, self._shard_dim, 0, orig_dim_size)
        self._unsharded_param = local
        self._unsharded_param.requires_grad = self._sharded_param.requires_grad
        self._set_param_on_module(self._unsharded_param)
        self._sharded_state = ShardedState.UNSHARDED

    def reshard(self):
        """Restore the sharded DTensor on the module."""
        if self._sharded_state == ShardedState.SHARDED:
            return
        self._set_param_on_module(self._sharded_param)
        self._unsharded_param = None
        self._sharded_state = ShardedState.SHARDED

    # ------------------------------------------------------------------
    # Gradient reduce-scatter
    # ------------------------------------------------------------------

    def reduce_scatter_grad(self, grad=None):
        """Reduce-scatter the unsharded gradient back to the shard.

        Parameters
        ----------
        grad : Tensor, optional
            The full (unsharded) gradient.  When called from a post-backward
            hook, *grad* is the gradient received by the hook.  When *None*,
            falls back to ``self._unsharded_param.grad``.
        """
        if grad is None:
            if self._unsharded_param is None:
                return
            grad = self._unsharded_param.grad
        if grad is None:
            return
        world_size = self._mesh_info.shard_mesh_size
        if world_size == 1:
            self._sharded_param.to_local().grad = grad
            return
        # Pad gradient if the original dim was not divisible by world_size
        if self._padded_dim_size > 0:
            from ...._creation import zeros
            pad_shape = list(grad.shape)
            pad_shape[self._shard_dim] = self._padded_dim_size
            grad = _cat_tensors(
                [grad, zeros(*pad_shape, dtype=grad.dtype, device=grad.device)],
                dim=self._shard_dim,
            )
        from ...._creation import empty
        shard_shape = self._sharded_param.to_local().shape
        reduced_grad = empty(
            *shard_shape, dtype=grad.dtype, device=grad.device
        )
        pg = self._mesh_info.shard_process_group
        dist.reduce_scatter_tensor(reduced_grad, grad, group=pg)
        self._sharded_param.to_local().grad = reduced_grad

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_param_on_module(self, tensor):
        """Replace the parameter on the module via both __dict__ and
        _parameters so that ``module.weight`` and
        ``module._parameters['weight']`` stay consistent."""
        self._module.__dict__[self._param_name] = tensor
        if self._param_name in self._module._parameters:
            self._module._parameters[self._param_name] = tensor


def _chunk_tensor(tensor, num_chunks, dim=0):
    """Split *tensor* into *num_chunks* pieces along *dim*."""
    from ...._functional import split
    size = tensor.shape[dim]
    chunk_size = (size + num_chunks - 1) // num_chunks
    return split(tensor, chunk_size, dim=dim)


def _cat_tensors(tensors, dim=0):
    """Concatenate *tensors* along *dim*."""
    from ...._functional import cat
    return cat(tensors, dim=dim)
