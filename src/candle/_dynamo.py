"""Minimal torch._dynamo stubs for import-time compatibility."""


class _Config:
    suppress_errors = False


config = _Config()


def is_compiling():
    return False


def reset():
    return None


def _identity_decorator(fn=None, **_kwargs):
    if fn is not None:
        return fn
    return lambda real_fn: real_fn


def disable(fn=None, **kwargs):
    return _identity_decorator(fn, **kwargs)


def optimize(fn=None, **kwargs):
    return _identity_decorator(fn, **kwargs)


def allow_in_graph(fn):
    return fn
