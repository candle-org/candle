# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython FastNode — C-level autograd node with typed fields.

Accelerates _freeze_next_functions and attribute access on the forward path.
"""

from libc.stdint cimport int64_t


cdef class FastNode:
    """C-level base for autograd Node."""
    cdef public object backward
    cdef public tuple inputs
    cdef public list _saved_tensors_list
    cdef public dict _saved_fields
    cdef public dict _hooks
    cdef public dict _prehooks
    cdef public tuple _next_functions_cache
    cdef public object _metadata
    cdef public str _name
    cdef public object _anomaly_trace
    cdef public object _anomaly_parent
    cdef dict __dict__

    def __init__(self, backward, inputs, *, name=None):
        if backward is not None:
            self.backward = backward
        self.inputs = tuple(inputs)
        self._saved_tensors_list = []
        self._saved_fields = {}
        self._hooks = {}
        self._prehooks = {}
        self._metadata = None
        self._name = name or type(self).__name__
        self._anomaly_trace = None
        self._anomaly_parent = None
        self._next_functions_cache = self._freeze_next_functions()

    cdef tuple _freeze_next_functions(self):
        """Build next_functions tuple — C-speed field access."""
        cdef list result = []
        for inp in self.inputs:
            fn = getattr(inp, "grad_fn", None)
            if fn is not None:
                result.append((fn, 0))
            elif getattr(inp, "requires_grad", False):
                acc = getattr(inp, "_accumulate_grad_node", None)
                if acc is None:
                    from candle.autograd.node import AccumulateGrad
                    acc = AccumulateGrad(inp)
                    inp._accumulate_grad_node = acc
                result.append((acc, 0))
            else:
                result.append((None, 0))
        return tuple(result)
