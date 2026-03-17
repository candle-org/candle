# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython FastDType — int-code based dtype for fast comparison."""

cdef class FastDType:
    """C-level dtype with int code for fast comparison."""
    cdef public int code
    cdef public int itemsize
    cdef public bint is_floating_point
    cdef public bint is_complex
    cdef public bint is_signed
    cdef public bint _is_quantized
    cdef public str name
    cdef public object _numpy_dtype

    def __init__(self, str name, object numpy_dtype, int itemsize,
                 bint is_floating_point=False, bint is_complex=False,
                 bint is_signed=True, int code=-1):
        self.name = name
        self._numpy_dtype = numpy_dtype
        self.itemsize = itemsize
        self.is_floating_point = is_floating_point
        self.is_complex = is_complex
        self.is_signed = is_signed
        self._is_quantized = False
        self.code = code

    @property
    def is_quantized(self):
        return self._is_quantized

    def __repr__(self):
        return f"torch.{self.name}"

    def __eq__(self, other):
        if isinstance(other, FastDType):
            return self.name == (<FastDType>other).name
        if hasattr(other, "name"):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash(self.name)

    def __reduce__(self):
        return (_reconstruct_dtype, (self.name,))


def _reconstruct_dtype(str name):
    """Reconstruct a FastDType from its name during unpickling."""
    from candle._dtype import from_name
    return from_name(name)
