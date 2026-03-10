"""Stub for torch.testing._internal.common_optimizers — optimizer test helpers."""


class OptimizerInfo:
    """Minimal stub for OptimizerInfo."""
    def __init__(self, optim_cls, *, optim_inputs_func=None, **kwargs):
        self.optim_cls = optim_cls
        self.optim_inputs_func = optim_inputs_func


# Empty list — no optimizer infos registered
optim_db = []


def optims(optim_info_iterable, dtypes=None):
    """No-op parametrize decorator for optimizer tests."""
    def decorator(fn):
        return fn
    return decorator


def _get_optim_inputs_including_global_cliquey_kwargs(*args, **kwargs):
    """Stub — returns empty list."""
    return []
