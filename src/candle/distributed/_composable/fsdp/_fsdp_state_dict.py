"""FSDP-aware state_dict helpers for full and sharded checkpointing."""
from ....distributed.tensor.dtensor import DTensor


def _collect_fsdp_states(module):
    """Collect all FSDPState objects in the module subtree."""
    states = []
    for mod in module.modules():
        st = getattr(mod, '_fsdp_state', None)
        if st is not None:
            states.append(st)
    return states


def _collect_param_groups(module):
    """Collect all FSDPParamGroup objects in the module subtree."""
    groups = []
    for st in _collect_fsdp_states(module):
        if st.param_group is not None:
            groups.append(st.param_group)
    return groups


def get_model_state_dict(model, *, type="full"):
    """Get model state dict with FSDP awareness.

    Parameters
    ----------
    model : nn.Module
        FSDP-wrapped model.
    type : str
        ``"full"`` — all-gather all params, return full state dict.
        ``"sharded"`` — return local shard state dict.

    Returns
    -------
    dict
        The state dict.
    """
    if type == "sharded":
        return _get_sharded_model_state_dict(model)
    elif type == "full":
        return _get_full_model_state_dict(model)
    else:
        raise ValueError(f"Unknown state_dict type: {type!r}")


def _get_full_model_state_dict(model):
    """Unshard all params, collect full state dict, reshard."""
    param_groups = _collect_param_groups(model)

    # Unshard all
    for pg in param_groups:
        pg.unshard()

    # Collect state dict from unsharded params
    state_dict = {}
    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            # During unshard, the module attr is replaced with the
            # unsharded tensor, so walk the module tree to get it
            state_dict[name] = _get_module_attr(model, name).detach()
        else:
            state_dict[name] = param.detach()

    # Also collect buffers
    for name, buf in model.named_buffers():
        state_dict[name] = buf.detach()

    # Reshard all
    for pg in param_groups:
        pg.reshard()

    return state_dict


def _get_sharded_model_state_dict(model):
    """Return local shard state dict (each rank saves its own shard)."""
    state_dict = {}
    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            state_dict[name] = param.to_local().detach()
        else:
            state_dict[name] = param.detach()
    for name, buf in model.named_buffers():
        state_dict[name] = buf.detach()
    return state_dict


def set_model_state_dict(model, state_dict, *, type="full"):
    """Load model state dict with FSDP awareness.

    Parameters
    ----------
    model : nn.Module
        FSDP-wrapped model.
    state_dict : dict
        State dict to load.
    type : str
        ``"full"`` — state dict contains full (unsharded) params.
        ``"sharded"`` — state dict contains local shards.
    """
    if type == "sharded":
        _set_sharded_model_state_dict(model, state_dict)
    elif type == "full":
        _set_full_model_state_dict(model, state_dict)
    else:
        raise ValueError(f"Unknown state_dict type: {type!r}")


def _set_full_model_state_dict(model, state_dict):
    """Load full state dict: unshard, overwrite, reshard + re-shard params."""
    param_groups = _collect_param_groups(model)

    # Unshard all
    for pg in param_groups:
        pg.unshard()

    # Overwrite unsharded param data from state dict
    for name, param in model.named_parameters():
        if name not in state_dict:
            continue
        val = state_dict[name]
        # Get the actual module attr (unsharded tensor during unshard)
        attr = _get_module_attr(model, name)
        attr.data = val.data if hasattr(val, 'data') else val

    # Write back to shards and reshard
    for pg in param_groups:
        for fp in pg.fsdp_params:
            if fp._unsharded_param is not None:
                _writeback_to_shard_from_fp(fp)
        pg.reshard()

    # Load buffers directly
    for name, buf in model.named_buffers():
        if name in state_dict:
            buf.data = state_dict[name].data


def _set_sharded_model_state_dict(model, state_dict):
    """Load sharded state dict: directly overwrite local shards."""
    for name, param in model.named_parameters():
        if name not in state_dict:
            continue
        if isinstance(param, DTensor):
            param.to_local().data = state_dict[name].data
        else:
            param.data = state_dict[name].data
    for name, buf in model.named_buffers():
        if name in state_dict:
            buf.data = state_dict[name].data


def _writeback_to_shard_from_fp(fp):
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


def get_optimizer_state_dict(model, optimizer, *, type="full"):
    """Get optimizer state dict with FSDP awareness.

    Parameters
    ----------
    model : nn.Module
        FSDP-wrapped model.
    optimizer : Optimizer
        The optimizer.
    type : str
        ``"full"`` or ``"sharded"``.

    Returns
    -------
    dict
        The optimizer state dict.
    """
    # For sharded, just return the raw optimizer state dict
    # (optimizer already operates on local shards)
    return optimizer.state_dict()


def _get_module_attr(module, dotted_name):
    """Resolve a dotted attribute name like 'fc1.weight' on a module."""
    parts = dotted_name.split('.')
    obj = module
    for part in parts:
        obj = getattr(obj, part)
    return obj
