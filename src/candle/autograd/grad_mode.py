from .._C._grad_mode_state import (
    _STATE as _GRAD_MODE_STATE,
    get_enabled as _get_enabled,
    set_enabled as _set_enabled,
    get_creation_mode as _get_creation_mode,
    set_creation_mode as _set_creation_mode,
)


class GradMode:
    @property
    def enabled(self):
        return _get_enabled()

    @enabled.setter
    def enabled(self, mode):
        _set_enabled(mode)


GradMode = GradMode()


def is_grad_enabled():
    return _get_enabled()


class set_grad_enabled:
    def __init__(self, mode):
        self.mode = bool(mode)
        self._prev = _get_enabled()
        self._prev_creation_mode = _get_creation_mode()
        _set_enabled(self.mode)
        _set_creation_mode(None if self.mode else "no_grad")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _set_enabled(self._prev)
        _set_creation_mode(self._prev_creation_mode)


def _decorate_with_grad_mode(fn, enabled):
    def wrapped(*args, **kwargs):
        with set_grad_enabled(enabled):
            return fn(*args, **kwargs)
    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    wrapped.__module__ = getattr(fn, "__module__", None)
    return wrapped


class _NoGradContext:
    def __enter__(self):
        self._prev = _get_enabled()
        self._prev_creation_mode = _get_creation_mode()
        _set_enabled(False)
        _set_creation_mode("no_grad")
        return self

    def __exit__(self, exc_type, exc, tb):
        _set_enabled(self._prev)
        _set_creation_mode(self._prev_creation_mode)

    def __call__(self, fn):
        return _decorate_with_grad_mode(fn, False)


class _EnableGradContext:
    def __enter__(self):
        self._prev = _get_enabled()
        _set_enabled(True)
        return self

    def __exit__(self, exc_type, exc, tb):
        _set_enabled(self._prev)

    def __call__(self, fn):
        return _decorate_with_grad_mode(fn, True)


def no_grad(func=None):
    ctx = _NoGradContext()
    if func is None:
        return ctx
    return ctx(func)


def enable_grad(func=None):
    ctx = _EnableGradContext()
    if func is None:
        return ctx
    return ctx(func)


class _InferenceModeContext:
    def __init__(self, mode=True):
        self.mode = bool(mode)

    def __enter__(self):
        self._prev = _get_enabled()
        self._prev_creation_mode = _get_creation_mode()
        _set_enabled(not self.mode and self._prev or False)
        if self.mode:
            _set_enabled(False)
            _set_creation_mode("inference_mode")
        return self

    def __exit__(self, exc_type, exc, tb):
        _set_enabled(self._prev)
        _set_creation_mode(self._prev_creation_mode)

    def __call__(self, fn):
        def wrapped(*args, **kwargs):
            with _InferenceModeContext(self.mode):
                return fn(*args, **kwargs)
        wrapped.__name__ = getattr(fn, "__name__", "wrapped")
        wrapped.__doc__ = getattr(fn, "__doc__", None)
        wrapped.__module__ = getattr(fn, "__module__", None)
        return wrapped


def current_creation_mode():
    return _get_creation_mode()


def inference_mode(mode=True):
    return _InferenceModeContext(mode)
