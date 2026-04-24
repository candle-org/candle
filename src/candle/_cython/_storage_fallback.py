"""Pure-Python fallback for _storage.pyx.

Re-exports the existing npu_typed_storage_from_ptr.
"""

from candle._storage_compat import npu_typed_storage_from_ptr as cy_npu_storage_from_ptr

__all__ = ["cy_npu_storage_from_ptr"]
