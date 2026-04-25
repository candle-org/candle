"""Pure-Python fallback for _device.pyx — FastDevice."""


class FastDevice:
    """Pure-Python mirror of Cython FastDevice."""
    __slots__ = ("type_code", "_index", "_type_str")

    _TYPE_MAP = {"cpu": 0, "npu": 1, "cuda": 2, "mps": 3, "meta": 4}

    def __init__(self, dev=None, index=None):
        if isinstance(dev, FastDevice):
            self.type_code = dev.type_code
            self._type_str = dev._type_str
            self._index = dev._index if index is None else int(index)
            return
        if dev is not None and hasattr(dev, "type") and not isinstance(dev, str):
            self._type_str = dev.type
            oi = dev.index if index is None else index
            self._index = -1 if oi is None else int(oi)
            self.type_code = self._TYPE_MAP.get(self._type_str, -1)
            return
        if isinstance(dev, str):
            s = dev
            if ":" in s:
                s, idx = s.split(":", 1)
                index = int(idx)
            self._type_str = s
        elif dev is None:
            self._type_str = "cpu"
        else:
            self._type_str = str(dev)

        self.type_code = self._TYPE_MAP.get(self._type_str, -1)
        if self._type_str in ("npu", "mps") and index is None:
            index = 0
        self._index = -1 if index is None else int(index)

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
            return self.type_code == other.type_code and self._index == other._index
        if hasattr(other, "type"):
            oi = getattr(other, "index", None)
            return (self._type_str == other.type
                    and self._index == (-1 if oi is None else oi))
        return NotImplemented

    def __hash__(self):
        return hash((self._type_str, None if self._index == -1 else self._index))
