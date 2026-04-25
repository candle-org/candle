"""Pure-Python fallback for _dtype.pyx — FastDType."""


class FastDType:
    """Pure-Python mirror of Cython FastDType."""
    __slots__ = ("code", "itemsize", "is_floating_point", "is_complex",
                 "is_signed", "_is_quantized", "name", "_numpy_dtype")

    def __init__(self, name, numpy_dtype, itemsize, is_floating_point=False,
                 is_complex=False, is_signed=True, code=-1):
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
            return self.name == other.name
        if hasattr(other, "name"):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash(self.name)
