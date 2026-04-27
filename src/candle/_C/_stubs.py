"""torch._C compatibility stubs for candle._C package exports."""
# pylint: disable=no-name-in-module,import-error


def _add_docstr(obj, docstr):
    """Minimal torch._C._add_docstr stub."""
    obj.__doc__ = docstr
    return obj


class _disabled_torch_dispatch_impl:
    """Minimal torch._C._disabled_torch_dispatch_impl context manager."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


_torch_function_enabled = True


class DisableTorchFunctionSubclass:
    """Minimal torch._C.DisableTorchFunctionSubclass context manager."""

    def __init__(self):
        pass

    def __enter__(self):
        global _torch_function_enabled
        self._prev = _torch_function_enabled
        _torch_function_enabled = False
        return self

    def __exit__(self, *args):
        global _torch_function_enabled
        _torch_function_enabled = self._prev


def _has_storage(tensor):
    """Minimal torch._C._has_storage stub."""
    return hasattr(tensor, '_storage') and tensor._storage is not None


def _get_tracing_state():
    """Minimal torch._C._get_tracing_state stub."""
    return None


def _get_privateuse():
    """Minimal torch._C._get_privateuse stub."""
    return "npu"


class _dlpack_exchange_api:
    """Minimal torch._C._dlpack_exchange_api stub."""

    @staticmethod
    def to_dlpack(tensor):
        raise NotImplementedError("DLPack not supported")

    @staticmethod
    def from_dlpack(dlpack):
        raise NotImplementedError("DLPack not supported")


def _to_dlpack(tensor, *args, **kwargs):
    raise NotImplementedError("DLPack not supported")


def _to_dlpack_versioned(tensor, version=None, *args, **kwargs):
    raise NotImplementedError("DLPack not supported")


class _VariableFunctions:
    """Minimal torch._C._VariableFunctions stub."""

    @staticmethod
    def rsub(tensor, other):
        import numpy as np

        if isinstance(other, (int, float, bool, complex, np.integer, np.floating)):
            from candle._functional import mul as _mul

            result = _mul(tensor, -1)
            return result + other
        from candle._functional import sub as _sub
        return _sub(other, tensor)


def _get_PyTorchFileReader():
    from ._stream import PyTorchFileReader
    return PyTorchFileReader


def _get_PyTorchFileWriter():
    from ._stream import PyTorchFileWriter
    return PyTorchFileWriter


def _get_privateuse1_backend_name():
    return "npu"
