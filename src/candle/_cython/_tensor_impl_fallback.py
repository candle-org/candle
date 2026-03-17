"""Pure-Python fallback for _tensor_impl.pyx.

Provides TensorImpl base class and _VersionCounterProxy when Cython is
not available.  The Tensor class inherits from this unconditionally —
when Cython IS available, the .pyx version replaces this module.
"""


class _VersionCounterProxy:
    """Lightweight proxy wrapping TensorImpl._version_value."""
    __slots__ = ("_impl",)

    def __init__(self, impl):
        self._impl = impl

    @property
    def value(self):
        return self._impl._version_value

    @value.setter
    def value(self, v):
        self._impl._version_value = int(v)

    def bump(self):
        self._impl._version_value += 1


class TensorImpl:
    """Pure-Python mirror of the Cython TensorImpl cdef class."""
    __slots__ = (
        "_storage",
        "_shape_tuple", "_stride_tuple",
        "_c_offset", "_c_numel", "_ndim",
        "_device_type", "_device_index", "_device_obj",
        "_dtype_code", "_itemsize", "_dtype_obj",
        "requires_grad", "grad", "grad_fn",
        "_version_value", "_vc_proxy",
        "_base", "_view_meta",
        "_pending", "_retain_grad", "_backward_hooks",
        "__dict__",
    )

    # -- shape --
    @property
    def shape(self):
        return self._shape_tuple

    @shape.setter
    def shape(self, value):
        t = tuple(value)
        self._shape_tuple = t
        self._ndim = len(t)
        numel = 1
        for d in t:
            numel *= d
        self._c_numel = numel

    # -- stride --
    @property
    def stride(self):
        return self._stride_tuple

    @stride.setter
    def stride(self, value):
        self._stride_tuple = value

    # -- offset --
    @property
    def offset(self):
        return self._c_offset

    @offset.setter
    def offset(self, value):
        self._c_offset = int(value)

    # -- device (always from storage, matching PyTorch TensorImpl) --
    @property
    def device(self):
        return self._storage.device

    # -- dtype (always from storage, matching PyTorch TensorImpl) --
    @property
    def dtype(self):
        return self._storage.dtype

    # -- version counter (inlined, views delegate to _base) --
    @property
    def _version_counter(self):
        if self._base is not None:
            return self._base._version_counter
        proxy = self._vc_proxy
        if proxy is not None:
            return proxy
        proxy = _VersionCounterProxy(self)
        self._vc_proxy = proxy
        return proxy

    @_version_counter.setter
    def _version_counter(self, value):
        if isinstance(value, _VersionCounterProxy):
            self._version_value = value._impl._version_value
        else:
            self._version_value = int(getattr(value, "value", 0))
        self._vc_proxy = None

    def _bump_version(self):
        if self._base is not None:
            self._base._bump_version()
        else:
            self._version_value += 1

    # -- storage --
    def storage(self):
        return self._storage

    # -- fast dim/numel --
    def dim(self):
        return self._ndim

    def numel(self):
        return self._c_numel

    def element_size(self):
        return self._itemsize
