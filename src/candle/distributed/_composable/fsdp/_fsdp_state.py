"""FSDPState -- hook orchestration for FSDP2."""
from ....distributed.tensor.dtensor import DTensor


class FSDPState:
    """Manages FSDP hook lifecycle for a module."""

    def __init__(self, module, param_group, mesh_info, reshard_after_forward):
        self.module = module
        self.param_group = param_group
        self.mesh_info = mesh_info
        self.reshard_after_forward = reshard_after_forward
        self._is_root = None
        self._pre_backward_hook_handles = []
        self._grad_count = 0
        self._total_managed_params = sum(
            1 for fp in param_group.fsdp_params
            if fp._sharded_param.requires_grad
        )

        # Register forward hooks
        self._pre_fwd_handle = module.register_forward_pre_hook(
            self._pre_forward, prepend=True, with_kwargs=True
        )
        self._post_fwd_handle = module.register_forward_hook(
            self._post_forward, prepend=False
        )

    def _pre_forward(self, module, args, kwargs):
        """Pre-forward: all-gather to reconstruct full parameters."""
        if self._is_root is None:
            self._lazy_init_root()
        # Clear stale backward hook handles from previous iteration
        self._pre_backward_hook_handles.clear()
        self.param_group.pre_forward()
        return args, kwargs

    def _post_forward(self, module, args, output):
        """Post-forward: register backward hooks + reshard."""
        self._register_pre_backward_hooks(output)
        self._register_post_backward_hooks()
        if self.reshard_after_forward:
            self.param_group.post_forward()
        return output

    def _register_pre_backward_hooks(self, output):
        tensors = _extract_tensors_from_output(output)
        for t in tensors:
            if t.requires_grad:
                handle = t.register_hook(self._pre_backward)
                self._pre_backward_hook_handles.append(handle)

    def _pre_backward(self, grad):
        if not self.param_group._is_unsharded:
            self.param_group.pre_backward()
        return grad

    def _register_post_backward_hooks(self):
        self._grad_count = 0
        self._pending_grads = {}
        for fsdp_param in self.param_group.fsdp_params:
            unsharded = fsdp_param._unsharded_param
            if unsharded is not None and unsharded.requires_grad:
                unsharded.register_hook(
                    self._make_post_backward_hook(fsdp_param)
                )

    def _make_post_backward_hook(self, fsdp_param):
        def hook(grad):
            # Capture the gradient from the autograd engine and store it
            # keyed by fsdp_param so post_backward can pass it to
            # reduce_scatter_grad.
            self._pending_grads[id(fsdp_param)] = (fsdp_param, grad)
            self._grad_count += 1
            if self._grad_count >= self._total_managed_params:
                self._post_backward_all()
                self._grad_count = 0
            return grad
        return hook

    def _post_backward_all(self):
        """Reduce-scatter captured gradients and reshard."""
        for fsdp_param, grad in self._pending_grads.values():
            fsdp_param.reduce_scatter_grad(grad)
        self._pending_grads.clear()
        self.param_group.reshard()
    def _lazy_init_root(self):
        self._is_root = not _has_parent_fsdp(self.module)
        # Root-level initialization: in a full FSDP tree, fully_shard()
        # controls reshard_after_forward for each module.  _lazy_init_root
        # only determines root status for post-backward finalization.


def _has_parent_fsdp(module):
    """Check if any ancestor module in the FSDP tree is also FSDP-managed.

    Relies on the ``_fsdp_has_parent`` flag set during ``fully_shard()``
    bottom-up traversal.
    """
    return getattr(module, '_fsdp_has_parent', False)


def _extract_tensors_from_output(output):
    from ...._tensor import Tensor
    tensors = []
    if isinstance(output, Tensor):
        tensors.append(output)
    elif isinstance(output, (tuple, list)):
        for item in output:
            tensors.extend(_extract_tensors_from_output(item))
    elif isinstance(output, dict):
        for v in output.values():
            tensors.extend(_extract_tensors_from_output(v))
    return tensors
