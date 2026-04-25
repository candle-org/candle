# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython hot-path accelerations for DataLoader.

1. ReorderBuffer: C-array indexed by send_idx, avoids Python dict hash overhead.
2. fast_collate_tensors: skip isinstance checks when all items are Tensors.
"""

cdef class ReorderBuffer:
    """Fixed-size reorder buffer backed by indexed Python lists."""
    cdef list _slots
    cdef list _present
    cdef int _next_idx
    cdef int _capacity

    def __init__(self, int capacity):
        self._capacity = capacity
        self._slots = [None] * capacity
        self._present = [False] * capacity
        self._next_idx = 0

    cpdef void put(self, int idx, object data):
        self._slots[idx % self._capacity] = data
        self._present[idx % self._capacity] = True

    cpdef list drain(self):
        cdef list result = []
        cdef int slot
        while True:
            slot = self._next_idx % self._capacity
            if not self._present[slot]:
                break
            result.append(self._slots[slot])
            self._slots[slot] = None
            self._present[slot] = False
            self._next_idx += 1
        return result

    @property
    def next_idx(self):
        return self._next_idx


def cy_reorder_put(buffer, int idx, object data):
    (<ReorderBuffer>buffer).put(idx, data)


def cy_reorder_drain(buffer):
    return (<ReorderBuffer>buffer).drain()


def cy_fast_collate_tensors(list batch):
    cdef object stack_fn
    from candle._functional import stack as stack_fn
    return stack_fn(batch, dim=0)
