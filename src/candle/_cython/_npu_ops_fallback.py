"""Pure-Python fallback for _npu_ops.pyx."""


def fast_binary_op(a, b, fn, name):
    from candle._backends.npu.ops._helpers import _binary_op
    return _binary_op(a, b, fn, name)
