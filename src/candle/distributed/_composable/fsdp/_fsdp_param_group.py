"""FSDPParamGroup -- batched communication for parameter groups."""
from .... import distributed as dist

try:
    from ....distributed._fsdp_fastpath import (
        build_flat_shard_offsets as _cy_flat_offsets,
        pack_shards_to_flat as _cy_pack_shards,
        unpack_flat_to_shards as _cy_unpack_shards,
        build_param_owner_map as _cy_owner_map,
    )
    _HAVE_FASTPATH = True
except ImportError:  # pragma: no cover
    _HAVE_FASTPATH = False


class FSDPParamGroup:
    """Manages a group of FSDPParam instances with batched lifecycle.

    When ``use_flat_buffer=True`` (the default for world_size > 1),
    all local shards are packed into a single contiguous flat buffer so
    that unshard / reduce-scatter each require only **one** collective
    instead of one per parameter.
    """

    def __init__(self, fsdp_params, module, mesh_info):
        self.fsdp_params = fsdp_params
        self.module = module
        self.mesh_info = mesh_info
        self._is_unsharded = False

        world_size = mesh_info.shard_mesh_size
        self._use_flat_buffer = world_size > 1 and len(fsdp_params) > 1

        # Build owner map: param_name -> group_index (0 for all params in this
        # single group).  Uses Cython fastpath when available.
        param_names = [fp._param_name for fp in fsdp_params]
        if _HAVE_FASTPATH:
            self.param_owner_map = _cy_owner_map([param_names])
        else:
            self.param_owner_map = {name: 0 for name in param_names}

        if self._use_flat_buffer:
            self._init_flat_buffer()

    # ------------------------------------------------------------------
    # Flat-buffer initialisation
    # ------------------------------------------------------------------

    def _init_flat_buffer(self):
        """Pack all local shards into one flat buffer and record offsets."""
        from ...._creation import zeros

        self._shard_shapes = []
        for fp in self.fsdp_params:
            local = fp._sharded_param.to_local()
            self._shard_shapes.append(local.shape)

        if _HAVE_FASTPATH:
            # Use Cython fastpath for offset arithmetic
            self._shard_offsets = _cy_flat_offsets(
                [list(s) for s in self._shard_shapes]
            )
            total_numel = self._shard_offsets[-1][1] if self._shard_offsets else 0
        else:
            self._shard_offsets = []
            total_numel = 0
            for fp in self.fsdp_params:
                local = fp._sharded_param.to_local()
                n = local.numel()
                self._shard_offsets.append((total_numel, total_numel + n))
                total_numel += n

        first_local = self.fsdp_params[0]._sharded_param.to_local()
        dtype = first_local.dtype
        device = first_local.device
        self._total_shard_numel = total_numel

        # Flat shard buffer (one per rank)
        self._flat_shard = zeros(total_numel, dtype=dtype, device=device)
        self._copy_shards_to_flat()

        # Flat full buffer (world_size * shard_numel)
        world_size = self.mesh_info.shard_mesh_size
        self._flat_full = zeros(
            total_numel * world_size, dtype=dtype, device=device
        )

    def _copy_shards_to_flat(self):
        """Copy each param's local shard into the flat buffer.

        When the Cython fastpath is available, the inner copy loop is
        executed in ``pack_shards_to_flat`` (no Python per-element overhead).
        The Cython helper operates on Python lists; we convert the flat tensor
        to a list, invoke the helper, then write back -- bridging the tensor
        world to the Cython typed-list interface.
        The fallback slice-assignment path is preserved for environments
        where the extension has not been compiled.
        """
        if _HAVE_FASTPATH:
            shards = [
                list(fp._sharded_param.to_local().reshape(-1))
                for fp in self.fsdp_params
            ]
            flat_list = list(self._flat_shard)
            _cy_pack_shards(shards, flat_list, self._shard_offsets)
            # Write the updated list back into the tensor
            for i, val in enumerate(flat_list):
                self._flat_shard[i] = val
        else:
            for fp, (start, end) in zip(self.fsdp_params, self._shard_offsets):
                local = fp._sharded_param.to_local()
                self._flat_shard[start:end] = local.reshape(-1)

    def _copy_flat_to_shard_grads(self, flat_src):
        """Write gradient data from *flat_src* back to each param shard's grad.

        When the Cython fastpath is available, the inner copy loop is
        executed in ``unpack_flat_to_shards`` (no Python per-element overhead).
        The Cython helper operates on Python lists; we convert flat_src to a
        list and pre-allocate shard lists, invoke the helper, then convert each
        shard list back to a tensor and reshape -- bridging the tensor world to
        the Cython typed-list interface.
        The fallback slice-reshape path is preserved for environments where
        the extension has not been compiled.
        """
        if _HAVE_FASTPATH:
            from ...._creation import zeros
            # Convert flat_src tensor to list for the Cython helper
            flat_list = list(flat_src)
            # Pre-allocate mutable shard lists
            shard_lists = [
                [0.0] * (end - start)
                for start, end in self._shard_offsets
            ]
            _cy_unpack_shards(flat_list, shard_lists, self._shard_offsets)
            for fp, shard_list, shape in zip(
                self.fsdp_params, shard_lists, self._shard_shapes
            ):
                from ...._tensor import Tensor
                flat_g = zeros(len(shard_list), dtype=flat_src.dtype,
                               device=flat_src.device)
                for i, val in enumerate(shard_list):
                    flat_g[i] = val
                fp._sharded_param.to_local().grad = flat_g.reshape(*shape)
        else:
            for i, fp in enumerate(self.fsdp_params):
                start, end = self._shard_offsets[i]
                shape = self._shard_shapes[i]
                fp._sharded_param.to_local().grad = flat_src[start:end].reshape(*shape)

    # ------------------------------------------------------------------
    # Batched shard lifecycle
    # ------------------------------------------------------------------

    def unshard(self):
        """Unshard all parameters in the group."""
        if self._is_unsharded:
            return
        if self._use_flat_buffer:
            self._unshard_flat()
        else:
            for p in self.fsdp_params:
                p.unshard()
        self._is_unsharded = True

    def _unshard_flat(self):
        """Single all-gather on the flat buffer, then scatter to params."""
        self._copy_shards_to_flat()
        pg = self.mesh_info.shard_process_group
        dist.all_gather_into_tensor(self._flat_full, self._flat_shard, group=pg)

        # Scatter gathered data back to individual params
        world_size = self.mesh_info.shard_mesh_size
        for i, fp in enumerate(self.fsdp_params):
            start, end = self._shard_offsets[i]
            numel = end - start
            # Reconstruct full param: gather chunks from each rank's region
            chunks = []
            for rank in range(world_size):
                rank_offset = rank * self._total_shard_numel
                chunk = self._flat_full[rank_offset + start:rank_offset + end]
                chunks.append(chunk)
            from ...._functional import cat, narrow
            full_flat = cat(chunks, dim=0)
            # Reshape to full param shape
            full_shape = list(fp._orig_shape)
            full_param = full_flat.reshape(*full_shape)
            # Strip padding if needed
            if fp._padded_dim_size > 0:
                orig_dim = fp._orig_shape[fp._shard_dim]
                full_param = narrow(full_param, fp._shard_dim, 0, orig_dim)
            full_param.requires_grad = fp._sharded_param.requires_grad
            fp._unsharded_param = full_param
            fp._set_param_on_module(full_param)
            fp._sharded_state = fp._sharded_state.__class__.UNSHARDED

    def reshard(self):
        """Reshard all parameters in the group."""
        if not self._is_unsharded:
            return
        for p in self.fsdp_params:
            p.reshard()
        self._is_unsharded = False

    def reduce_scatter_grads(self):
        """Reduce-scatter gradients for all parameters in the group."""
        if self._use_flat_buffer:
            self._reduce_scatter_flat()
        else:
            for p in self.fsdp_params:
                p.reduce_scatter_grad()

    def _reduce_scatter_flat(self):
        """Single reduce-scatter on flat gradient buffer."""
        from ...._creation import zeros
        from ...._functional import cat

        world_size = self.mesh_info.shard_mesh_size

        # Pack all unsharded grads into flat_full layout
        self._flat_full[:] = 0
        for i, fp in enumerate(self.fsdp_params):
            if fp._unsharded_param is None:
                continue
            grad = fp._unsharded_param.grad
            if grad is None:
                continue
            start, end = self._shard_offsets[i]
            numel = end - start
            # Pad grad if needed
            if fp._padded_dim_size > 0:
                pad_shape = list(grad.shape)
                pad_shape[fp._shard_dim] = fp._padded_dim_size
                grad = cat(
                    [grad, zeros(*pad_shape, dtype=grad.dtype, device=grad.device)],
                    dim=fp._shard_dim,
                )
            grad_flat = grad.reshape(-1)
            # Distribute into rank-strided layout
            for rank in range(world_size):
                rank_offset = rank * self._total_shard_numel
                chunk = grad_flat[rank * numel:(rank + 1) * numel]
                self._flat_full[rank_offset + start:rank_offset + end] = chunk

        # Single reduce-scatter
        pg = self.mesh_info.shard_process_group
        reduced_flat = zeros(
            self._total_shard_numel,
            dtype=self._flat_shard.dtype,
            device=self._flat_shard.device,
        )
        dist.reduce_scatter_tensor(reduced_flat, self._flat_full, group=pg)

        # Scatter reduced shards back to params via Cython unpack or fallback
        self._copy_flat_to_shard_grads(reduced_flat)

    # ------------------------------------------------------------------
    # Module hook helpers
    # ------------------------------------------------------------------

    def pre_forward(self):
        """Call before the module's forward pass."""
        self.unshard()

    def post_forward(self):
        """Call after the module's forward pass."""
        self.reshard()

    def pre_backward(self):
        """Call before the module's backward pass."""
        self.unshard()

    def post_backward(self):
        """Call after the module's backward pass."""
        self.reduce_scatter_grads()
        self.reshard()
