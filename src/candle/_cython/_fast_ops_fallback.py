"""Pure-Python fallback for _fast_ops.pyx.

When Cython is not available, the original _functional.py implementations
are used directly — this module is only imported to satisfy the conditional
import pattern.
"""

# These are no-ops; the real functions live in _functional.py
# and are only replaced when the Cython extension compiles.
fast_add = None
fast_mul = None
fast_matmul = None
fast_sub = None
fast_div = None
fast_relu = None
fast_neg = None
