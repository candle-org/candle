# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython FastDevice — int-based device comparison instead of string ops.

Maintains full API compatibility with the original ``device`` class:
``.index`` returns ``None`` (not -1) when no index was specified.
"""

cdef class FastDevice:
    """C-level device with int type_code for fast comparison."""
    cdef public int type_code    # 0=cpu, 1=npu, 2=cuda, 3=mps, 4=meta
    cdef int _index              # -1 means None
    cdef str _type_str

    def __init__(self, dev=None, index=None):
        if isinstance(dev, FastDevice):
            self.type_code = (<FastDevice>dev).type_code
            self._type_str = (<FastDevice>dev)._type_str
            self._index = (<FastDevice>dev)._index if index is None else int(index)
            return
        # Accept original device class
        if dev is not None and hasattr(dev, "type") and not isinstance(dev, str):
            self._type_str = dev.type
            oi = dev.index if index is None else index
            self._index = -1 if oi is None else int(oi)
            self._assign_type_code()
            return
        cdef str s
        if isinstance(dev, str):
            s = <str>dev
            if ":" in s:
                parts = s.split(":", 1)
                s = parts[0]
                index = int(parts[1])
            self._type_str = s
        elif dev is None:
            self._type_str = "cpu"
        else:
            self._type_str = str(dev)

        self._assign_type_code()
        # Auto-assign index=0 for npu/mps when not specified
        if self._type_str == "npu" and index is None:
            index = 0
        if self._type_str == "mps" and index is None:
            index = 0
        self._index = -1 if index is None else int(index)

    cdef void _assign_type_code(self):
        if self._type_str == "cpu":
            self.type_code = 0
        elif self._type_str == "npu":
            self.type_code = 1
        elif self._type_str == "cuda":
            self.type_code = 2
        elif self._type_str == "mps":
            self.type_code = 3
        elif self._type_str == "meta":
            self.type_code = 4
        else:
            self.type_code = -1

    @property
    def type(self):
        return self._type_str

    @property
    def index(self):
        if self._index == -1:
            return None
        return self._index

    @index.setter
    def index(self, value):
        self._index = -1 if value is None else int(value)

    def __repr__(self):
        if self._index == -1:
            return f"device(type='{self._type_str}')"
        return f"device(type='{self._type_str}', index={self._index})"

    def __str__(self):
        if self._index == -1:
            return self._type_str
        return f"{self._type_str}:{self._index}"

    def __eq__(self, other):
        if isinstance(other, FastDevice):
            return (self.type_code == (<FastDevice>other).type_code
                    and self._index == (<FastDevice>other)._index)
        if hasattr(other, "type"):
            oi = getattr(other, "index", None)
            return (self._type_str == other.type
                    and self._index == (-1 if oi is None else oi))
        return NotImplemented

    def __hash__(self):
        return hash((self._type_str, None if self._index == -1 else self._index))

    def __reduce__(self):
        idx = None if self._index == -1 else self._index
        return (FastDevice, (self._type_str, idx))
