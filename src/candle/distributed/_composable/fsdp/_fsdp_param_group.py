"""FSDPParamGroup -- batched communication for parameter groups."""


class FSDPParamGroup:
    """Manages a group of FSDPParam instances with batched lifecycle.

    Provides ``unshard`` / ``reshard`` / ``reduce_scatter_grads`` that
    operate on every parameter in the group, as well as convenience
    ``pre_forward`` / ``post_forward`` / ``pre_backward`` / ``post_backward``
    hooks for module-level integration.
    """

    def __init__(self, fsdp_params, module, mesh_info):
        self.fsdp_params = fsdp_params
        self.module = module
        self.mesh_info = mesh_info
        self._is_unsharded = False

    # ------------------------------------------------------------------
    # Batched shard lifecycle
    # ------------------------------------------------------------------

    def unshard(self):
        """Unshard all parameters in the group."""
        if self._is_unsharded:
            return
        for p in self.fsdp_params:
            p.unshard()
        self._is_unsharded = True

    def reshard(self):
        """Reshard all parameters in the group."""
        if not self._is_unsharded:
            return
        for p in self.fsdp_params:
            p.reshard()
        self._is_unsharded = False

    def reduce_scatter_grads(self):
        """Reduce-scatter gradients for all parameters in the group."""
        for p in self.fsdp_params:
            p.reduce_scatter_grad()

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
