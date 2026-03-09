from ._trace import _script_if_tracing

_SCOPE_MSG = "jit scripting is outside 0.1 NPU-first scope"


def is_tracing():
    return False


def is_scripting():
    return False


def script(obj, optimize=None, _frames_up=0, _rcb=None, example_inputs=None):
    if optimize is not None or example_inputs is not None:
        raise NotImplementedError(_SCOPE_MSG)
    return obj


def ignore(drop=False, **kwargs):
    if callable(drop):
        return drop

    def decorator(fn):
        return fn

    return decorator


def unused(fn):
    return fn


def _overload_method(func):
    return func


def _script_if_tracing_wrapper(fn):
    return _script_if_tracing(fn)


def script_if_tracing(fn):
    return _script_if_tracing(fn)
