"""Minimal distributed checkpoint state_dict helpers for candle.

This MVP intentionally provides a narrow surface:
- returns model and optimizer state dictionaries
- restores model and optimizer state dictionaries

It is designed for single-node DDP training recovery flows.
FSDP-awareness is added via the fsdp_type keyword argument.
"""

import candle as torch


def _unwrap_model(model):
    # DDP/DataParallel expose the wrapped module at `.module`.
    return getattr(model, "module", model)


def _is_fsdp_model(model):
    """Return True if *model* (or its inner module) carries FSDP state."""
    base = _unwrap_model(model)
    return hasattr(base, '_fsdp_state')


def get_state_dict(
    model,
    optimizer=None,
    *,
    rank0_only=False,
    rank=None,
    as_payload=False,
    fsdp_type=None,
):
    """Return model and optimizer state dicts.

    Args:
        model: module or parallel wrapper exposing ``state_dict``.
        optimizer: optional optimizer exposing ``state_dict``.
        rank0_only: if True, return payload only for rank 0.
        rank: optional explicit rank used with ``rank0_only``.
        as_payload: if True, return a single checkpoint payload dict.

    Returns:
        Tuple[dict, dict|None]: ``(model_state_dict, optim_state_dict)``.
    """
    if rank0_only:
        current_rank = 0 if rank is None else int(rank)
        if current_rank != 0:
            return None if as_payload else (None, None)
    else:
        current_rank = 0 if rank is None else int(rank)

    base_model = _unwrap_model(model)

    # FSDP-aware model state dict
    if fsdp_type is not None and _is_fsdp_model(model):
        from candle.distributed._composable.fsdp._fsdp_state_dict import (  # pylint: disable=import-outside-toplevel
            get_model_state_dict as _get_model_sd,
        )
        model_state = _get_model_sd(base_model, type=fsdp_type)
    else:
        model_state = base_model.state_dict()

    optim_state = optimizer.state_dict() if optimizer is not None else None
    if as_payload:
        return {
            "model": model_state,
            "optim": optim_state,
            "meta": {
                "rank0_only": bool(rank0_only),
                "rank": int(current_rank),
            },
        }
    return model_state, optim_state


def set_state_dict(
    model,
    optimizer=None,
    *,
    payload=None,
    model_state_dict=None,
    optim_state_dict=None,
    strict=True,
    allow_partial_optimizer=False,
    fsdp_type=None,
):
    """Restore model and optimizer state dicts.

    Args:
        model: module or parallel wrapper exposing ``load_state_dict``.
        optimizer: optional optimizer exposing ``load_state_dict``.
        model_state_dict: state dict for model.
        optim_state_dict: optional state dict for optimizer.
        strict: forwarded to ``model.load_state_dict``.
        fsdp_type: if set, delegate model restore to FSDP-aware helper.
            ``"full"`` or ``"sharded"``.
    """
    if payload is not None:
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError("payload.meta must be a dict")
        if not isinstance(meta.get("rank"), int):
            raise ValueError("payload.meta.rank must be an int")
        model_state_dict = payload.get("model")
        if optim_state_dict is None:
            optim_state_dict = payload.get("optim")

    base_model = _unwrap_model(model)
    restored_optimizer_state_keys_count = 0
    missing_keys = []
    unexpected_keys = []
    loaded_keys_count = 0

    if model_state_dict is not None:
        # FSDP-aware restore
        if fsdp_type is not None and _is_fsdp_model(model):
            from candle.distributed._composable.fsdp._fsdp_state_dict import (  # pylint: disable=import-outside-toplevel
                set_model_state_dict as _set_model_sd,
            )
            _set_model_sd(base_model, model_state_dict, type=fsdp_type)
            loaded_keys_count = len(model_state_dict)
        else:
            expected_keys = set(base_model.state_dict().keys())
            provided_keys = set(model_state_dict.keys())
            incompatible = base_model.load_state_dict(model_state_dict, strict=strict)
            missing_keys = list(getattr(incompatible, "missing_keys", []))
            unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
            if not strict:
                missing_keys = sorted(expected_keys - provided_keys)
                unexpected_keys = sorted(provided_keys - expected_keys)
            loaded_keys_count = len(expected_keys & provided_keys)
    elif payload is None and not strict:
        pass  # optimizer-only restore is fine
    elif payload is None and model_state_dict is None and optim_state_dict is None:
        raise ValueError("model_state_dict must not be None")

    if optimizer is not None and optim_state_dict is not None:
        # Safe optimizer restore: preserve runtime-owned parameter objects and
        # copy only serializable group options/state.
        loaded_groups = optim_state_dict.get("param_groups", [])
        if (not allow_partial_optimizer) and len(loaded_groups) != len(optimizer.param_groups):
            raise ValueError("optimizer param_groups length mismatch")
        if len(loaded_groups) == len(optimizer.param_groups):
            for group, loaded in zip(optimizer.param_groups, loaded_groups):
                for k, v in loaded.items():
                    if k not in ("params", "param_ids"):
                        group[k] = v
        loaded_state = optim_state_dict.get("state", {})
        optimizer.state = {}
        for k, v in loaded_state.items():
            try:
                key = int(k)
            except Exception:
                key = k
            optimizer.state[key] = v
        restored_optimizer_state_keys_count = len(loaded_state)

    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "loaded_keys_count": loaded_keys_count,
        "restored_optimizer_state_keys_count": restored_optimizer_state_keys_count,
    }


def save(
    path,
    model,
    optimizer=None,
    *,
    payload=None,
    rank0_only=False,
    rank=None,
):
    """Save a distributed checkpoint payload to ``path``.

    Returns the payload written to disk, or ``None`` when ``rank0_only`` is
    enabled and current rank is non-zero.
    """
    if payload is None:
        payload = get_state_dict(
            model,
            optimizer=optimizer,
            rank0_only=rank0_only,
            rank=rank,
            as_payload=True,
        )

    if payload is None:
        return None

    torch.save(payload, path)
    return payload


def load(
    path,
    model,
    optimizer=None,
    *,
    payload=None,
    strict=True,
    allow_partial_optimizer=False,
    map_location="cpu",
):
    """Load a distributed checkpoint payload from ``path`` and restore state."""
    if payload is None:
        payload = torch.load(path, map_location=map_location)

    return set_state_dict(
        model,
        optimizer=optimizer,
        payload=payload,
        strict=strict,
        allow_partial_optimizer=allow_partial_optimizer,
    )
