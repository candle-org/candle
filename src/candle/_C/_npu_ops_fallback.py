"""Pure-Python fallback for _npu_ops.pyx."""


def fast_binary_op(a, b, fn, name):
    # Call the original Python slow path directly. Routing back through
    # _binary_op() would re-enter the fast-op dispatch switch and recurse.
    from candle._backends.npu.ops._helpers import _binary_op_slow
    return _binary_op_slow(a, b, fn, name)
