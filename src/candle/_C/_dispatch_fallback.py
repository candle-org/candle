"""Pure-Python fallback for _dispatch.pyx.

Re-exports the existing Python implementations so that import paths
from _cython work regardless of whether Cython is available.
"""

from candle._dispatch.dispatcher import dispatch as cy_dispatch
from candle._dispatch.dispatcher import dispatch_with_keyset as cy_dispatch_with_keyset
from candle._dispatch.dispatcher import _extract_tensors as cy_extract_tensors
from candle._dispatch.dispatcher import _prepare_kwargs as cy_prepare_kwargs
from candle._dispatch.dispatcher import _kernel_for_entry as cy_kernel_for_entry
from candle._dispatch.keys import DispatchKeySet as FastDispatchKeySet
from candle._dispatch.keys import apply_tls_masks as cy_apply_tls_masks
