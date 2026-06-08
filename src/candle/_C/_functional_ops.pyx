# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython hot wrappers for candle._functional.

This module accelerates the most frequently used functional entry points while
preserving the pure-Python fallback functions in ``candle._functional`` for
stable ``__torch_function__`` identities and fallback behavior.
"""

import builtins as _builtins
from libc.stdint cimport int64_t
from candle.autograd.node import AccumulateGrad, SavedTensor, _SavedValue
from candle._C._autograd_node cimport Node as _CyAutogradNode
from candle._C._tensor_impl cimport TensorImpl, cy_make_tensor_from_storage
from candle._C._grad_mode_state cimport get_enabled_fast as _grad_enabled_fast
from candle._C._npu_ops cimport (
    fast_add as _cy_fast_npu_add,
    fast_add_exact as _cy_fast_npu_add_exact,
    fast_mul as _cy_fast_npu_mul,
    fast_mul_exact as _cy_fast_npu_mul_exact,
    fast_matmul as _cy_fast_npu_matmul,
    fast_matmul_exact as _cy_fast_npu_matmul_exact,
    fast_addmm as _cy_fast_npu_addmm,
    fast_layer_norm as _cy_fast_npu_layer_norm,
    fast_layer_norm_backward as _cy_fast_npu_layer_norm_backward,
    fast_sum as _cy_fast_npu_sum,
    fast_mm_mat1_backward as _cy_fast_npu_mm_mat1_backward,
    fast_mm_mat2_backward as _cy_fast_npu_mm_mat2_backward,
    fast_gelu as _cy_fast_npu_gelu,
    fast_gelu_exact as _cy_fast_npu_gelu_exact,
    fast_silu as _cy_fast_npu_silu,
    fast_silu_exact as _cy_fast_npu_silu_exact,
    fast_gelu_backward as _cy_fast_npu_gelu_backward,
    fast_silu_backward as _cy_fast_npu_silu_backward,
    fast_sdpa_flash_attention as _cy_fast_sdpa_flash_attention,
    fast_sdpa_flash_attention_backward as _cy_fast_sdpa_flash_attention_backward,
    fast_sdpa_flash_attention_backward_packed_qkv as _cy_fast_sdpa_flash_attention_backward_packed_qkv,
    fast_packed_qkv_projection_forward as _cy_fast_packed_qkv_projection_forward,
    fast_packed_qkv_projection_backward as _cy_fast_packed_qkv_projection_backward,
)

# Cached reference to base Tensor class
cdef object _BaseTensor = None

# Cached Python callables
cdef object _dispatch_fn = None
cdef object _py_add_fn = None
cdef object _py_mul_fn = None
cdef object _py_matmul_fn = None
cdef object _py_sub_fn = None
cdef object _py_div_fn = None
cdef object _py_relu_fn = None
cdef object _py_neg_fn = None

# NPU fast-path: cached references for direct kernel calls
cdef object _npu_add_fn = None
cdef object _npu_mul_fn = None
cdef object _npu_sub_fn = None
cdef object _npu_div_fn = None
cdef object _npu_silu_fn = None
cdef object _grad_mode_state = None
cdef object _is_functionalize_fn = None
cdef object _current_pipeline_fn = None
cdef object _is_profiler_enabled_fn = None
cdef object _is_autocast_enabled_fn = None
cdef object _npu_fast_backward_keyset = None
cdef object _npu_fast_active_keyset = None
cdef object _npu_silu_backward_fn = None
cdef object _npu_create_graph_fn = None
cdef object _npu_dispatch_mul_fn = None
cdef object _npu_dispatch_silu_backward_fn = None
cdef bint _npu_refs_loaded = False
cdef bint _npu_autograd_refs_loaded = False
cdef bint _npu_profiler_active_flag = False
cdef bint _npu_autocast_active_flag = False
cdef bint _functionalize_active_flag = False
cdef bint _pipeline_active_flag = False
cdef object _npu_add_alpha_saved_value = None
cdef dict _npu_add_saved_fields = None
cdef list _npu_empty_saved_tensors = []
cdef dict _npu_empty_saved_fields = {}
cdef object _saved_hooks_state = None
cdef object _saved_hooks_state_obj = None


def cy_set_npu_profiler_active(bint active):
    """Set the fast-path profiler guard used by NPU eager wrappers."""
    global _npu_profiler_active_flag
    _npu_profiler_active_flag = active


def cy_set_npu_autocast_active(bint active):
    """Set the fast-path NPU autocast guard used by NPU eager wrappers."""
    global _npu_autocast_active_flag
    _npu_autocast_active_flag = active


def cy_set_functionalize_active(bint active):
    """Set the fast-path functionalization guard used by eager wrappers."""
    global _functionalize_active_flag
    _functionalize_active_flag = active


def cy_set_pipeline_active(bint active):
    """Set the fast-path dispatch pipeline guard used by eager wrappers."""
    global _pipeline_active_flag
    _pipeline_active_flag = active


cdef inline void _ensure_base():
    global _BaseTensor
    if _BaseTensor is None:
        from candle._tensor import Tensor
        _BaseTensor = Tensor


cdef inline bint _npu_has_saved_hooks():
    global _saved_hooks_state, _saved_hooks_state_obj
    if _saved_hooks_state is None:
        from candle._C import _hooks_state as _hs
        _saved_hooks_state = _hs
        _saved_hooks_state_obj = _hs._STATE
    return getattr(_saved_hooks_state_obj, 'hooks', None) is not None and bool(getattr(_saved_hooks_state_obj, 'hooks', None))


cdef inline void _ensure_dispatch():
    global _dispatch_fn
    if _dispatch_fn is None:
        from candle._dispatch.dispatcher import dispatch
        _dispatch_fn = dispatch


cdef inline bint _is_base_tensor(object t):
    """True if t is exactly the base Tensor class (not a subclass)."""
    _ensure_base()
    return type(t) is _BaseTensor


cdef inline bint _class_overrides_torch_function(object cls):
    cdef object base
    _ensure_base()
    for base in cls.__mro__:
        if base is _BaseTensor:
            return False
        if "__torch_function__" in getattr(base, "__dict__", {}):
            return True
    return True


cdef bint _check_value(object val):
    cdef object cls
    cdef object item

    if isinstance(val, _BaseTensor) and type(val) is not _BaseTensor:
        cls = type(val)
        if _class_overrides_torch_function(cls):
            return True

    if isinstance(val, (list, tuple)):
        for item in val:
            if _check_value(item):
                return True

    return False


cdef void _collect_types(object val, object types):
    cdef object cls
    cdef object item

    if isinstance(val, _BaseTensor) and type(val) is not _BaseTensor:
        cls = type(val)
        if _class_overrides_torch_function(cls):
            types.add(cls)

    if isinstance(val, (list, tuple)):
        for item in val:
            _collect_types(item, types)


cdef inline void _ensure_originals():
    global _py_add_fn, _py_mul_fn, _py_matmul_fn, _py_sub_fn, _py_div_fn
    global _py_relu_fn, _py_neg_fn

    _ensure_dispatch()

    if _py_add_fn is None:
        from candle._functional import _py_add, _py_mul, _py_matmul, _py_sub, _py_div, _py_relu, _py_neg
        _py_add_fn = _py_add
        _py_mul_fn = _py_mul
        _py_matmul_fn = _py_matmul
        _py_sub_fn = _py_sub
        _py_div_fn = _py_div
        _py_relu_fn = _py_relu
        _py_neg_fn = _py_neg


cdef inline void _ensure_npu_refs():
    """Load NPU op refs and guard state once."""
    global _npu_add_fn, _npu_mul_fn, _npu_sub_fn, _npu_div_fn, _npu_silu_fn
    global _grad_mode_state, _is_functionalize_fn, _current_pipeline_fn
    global _is_profiler_enabled_fn, _is_autocast_enabled_fn, _npu_refs_loaded

    if _npu_refs_loaded:
        return

    from candle._C._npu_ops import fast_add as _nadd  # pylint: disable=import-error,no-name-in-module
    try:
        from candle._C._npu_ops import fast_mul as _nmul  # pylint: disable=import-error,no-name-in-module
    except ImportError:
        from candle._backends.npu.ops import mul as _nmul
    try:
        from candle._C._npu_ops import fast_silu as _nsilu  # pylint: disable=import-error,no-name-in-module
    except ImportError:
        from candle._backends.npu.ops.activation import silu as _nsilu
    try:
        from candle._C._npu_ops import fast_sub as _nsub  # pylint: disable=import-error,no-name-in-module
    except ImportError:
        from candle._backends.npu.ops import sub as _nsub
    try:
        from candle._C._npu_ops import fast_div as _ndiv  # pylint: disable=import-error,no-name-in-module
    except ImportError:
        from candle._backends.npu.ops import div as _ndiv
    from candle.autograd.grad_mode import _GRAD_MODE_STATE as _gms
    from candle._dispatch.functionalize import is_functionalize_enabled as _ife
    from candle._dispatch.pipeline import current_pipeline as _cp
    from candle.profiler.profiler import is_profiler_enabled as _ipe
    from candle.amp.state import is_autocast_enabled as _iae

    _npu_add_fn = _nadd
    _npu_mul_fn = _nmul
    _npu_sub_fn = _nsub
    _npu_div_fn = _ndiv
    _npu_silu_fn = _nsilu
    _grad_mode_state = _gms
    _is_functionalize_fn = _ife
    _current_pipeline_fn = _cp
    _is_profiler_enabled_fn = _ipe
    _is_autocast_enabled_fn = _iae
    _npu_refs_loaded = True


cdef inline void _ensure_npu_autograd_refs():
    """Load refs for the NPU eager autograd fast path once."""
    global _npu_fast_backward_keyset, _npu_fast_active_keyset
    global _npu_silu_backward_fn, _npu_autograd_refs_loaded
    global _npu_create_graph_fn, _npu_dispatch_mul_fn, _npu_dispatch_silu_backward_fn
    if _npu_autograd_refs_loaded:
        return
    from candle._dispatch.keys import DispatchKey
    from candle._C._dispatch import FastDispatchKeySet
    try:
        from candle._C._npu_ops import fast_silu_backward as _nsilu_backward  # pylint: disable=import-error,no-name-in-module
    except ImportError:
        from candle._backends.npu.backward import npu_silu_backward as _nsilu_backward
    from candle._C._autograd_engine import is_create_graph_enabled as _icg
    from candle._dispatch.dispatcher import dispatch as _disp
    _npu_fast_backward_keyset = FastDispatchKeySet(DispatchKey.NPU)
    _npu_fast_active_keyset = FastDispatchKeySet(
        int(DispatchKey.NPU)
        | int(DispatchKey.ADInplaceOrView)
        | int(DispatchKey.Autograd)
        | int(DispatchKey.AutogradNPU)
    )
    _npu_silu_backward_fn = _nsilu_backward
    _npu_create_graph_fn = _icg
    _npu_dispatch_mul_fn = _disp
    _npu_dispatch_silu_backward_fn = _disp
    _npu_autograd_refs_loaded = True


cdef inline bint _profiler_active():
    return _npu_profiler_active_flag


cdef inline bint _exact_base_npu_pair(object a, object b):
    _ensure_base()
    return (
        type(a) is _BaseTensor
        and type(b) is _BaseTensor
        and (<TensorImpl>a)._device_type == 1
        and (<TensorImpl>b)._device_type == 1
        and (<TensorImpl>a)._device_index == (<TensorImpl>b)._device_index
    )


cdef inline bint _exact_base_npu_unary(object a):
    _ensure_base()
    return type(a) is _BaseTensor and (<TensorImpl>a)._device_type == 1


cdef inline bint _tensor_has_strict_contiguous_stride(TensorImpl t) noexcept:
    cdef int i
    cdef int j
    cdef int64_t acc = 1
    for j in range(t._ndim):
        i = t._ndim - 1 - j
        if t._c_stride[i] != acc:
            return False
        acc = acc * t._c_shape[i]
    return True


cdef inline bint _tensor_has_all_zero_stride(TensorImpl t) noexcept:
    cdef int i
    if t._ndim == 0:
        return False
    for i in range(t._ndim):
        if t._c_stride[i] != 0:
            return False
    return True


cdef inline tuple _contiguous_stride_tuple(object shape):
    cdef Py_ssize_t ndim = len(shape)
    cdef list strides = [1] * ndim
    cdef int64_t acc = 1
    cdef Py_ssize_t j
    cdef Py_ssize_t i
    for j in range(ndim):
        i = ndim - 1 - j
        strides[i] = acc
        acc *= <int64_t>shape[i]
    return tuple(strides)


cdef inline int _exact_npu_addmm_hot_state(object input, object mat1, object mat2, object beta, object alpha):
    """Return 1 for inference, 2 for autograd, 0 for fallback."""
    cdef bint grad_on
    cdef bint requires_grad
    cdef TensorImpl bias
    cdef TensorImpl a
    cdef TensorImpl b
    _ensure_base()
    if not (isinstance(input, _BaseTensor) and isinstance(mat1, _BaseTensor) and isinstance(mat2, _BaseTensor)):
        return 0
    if _check_value(input) or _check_value(mat1) or _check_value(mat2):
        return 0
    bias = <TensorImpl>input
    a = <TensorImpl>mat1
    b = <TensorImpl>mat2
    if bias._device_type != 1 or a._device_type != 1 or b._device_type != 1:
        return 0
    if _functionalize_active_flag or _pipeline_active_flag or _npu_autocast_active_flag:
        return 0
    if hasattr(beta, "shape") or hasattr(alpha, "shape"):
        return 0
    if bias._ndim != 1 or a._ndim != 2 or b._ndim != 2:
        return 0
    if a._c_shape[1] != b._c_shape[0] or bias._c_shape[0] != b._c_shape[1]:
        return 0
    if bias._device_index != a._device_index or bias._device_index != b._device_index:
        return 0
    if bias._dtype_code != a._dtype_code or bias._dtype_code != b._dtype_code:
        return 0
    grad_on = _grad_enabled_fast()
    requires_grad = bias.requires_grad or a.requires_grad or b.requires_grad
    if not grad_on or not requires_grad:
        return 1
    if _npu_profiler_active_flag:
        return 0
    if beta != 1 or alpha != 1:
        return 0
    return 2


cdef inline int _exact_npu_linear_hot_state(object input, object weight, object bias):
    """Return 1 for inference, 2 for autograd, 0 for fallback."""
    cdef bint grad_on
    cdef bint requires_grad
    cdef TensorImpl inp
    cdef TensorImpl w
    cdef TensorImpl b
    _ensure_base()
    if bias is None:
        return 0
    if not (isinstance(input, _BaseTensor) and isinstance(weight, _BaseTensor) and isinstance(bias, _BaseTensor)):
        return 0
    if _check_value(input) or _check_value(weight) or _check_value(bias):
        return 0
    inp = <TensorImpl>input
    w = <TensorImpl>weight
    b = <TensorImpl>bias
    if inp._device_type != 1 or w._device_type != 1 or b._device_type != 1:
        return 0
    if _functionalize_active_flag or _pipeline_active_flag or _npu_autocast_active_flag:
        return 0
    if inp._ndim < 2 or w._ndim != 2 or b._ndim != 1:
        return 0
    if inp._c_shape[inp._ndim - 1] != w._c_shape[1] or b._c_shape[0] != w._c_shape[0]:
        return 0
    if inp._device_index != w._device_index or inp._device_index != b._device_index:
        return 0
    if inp._dtype_code != w._dtype_code or inp._dtype_code != b._dtype_code:
        return 0
    if not _tensor_has_strict_contiguous_stride(inp):
        return 0
    grad_on = _grad_enabled_fast()
    requires_grad = inp.requires_grad or w.requires_grad or b.requires_grad
    if not grad_on or not requires_grad:
        return 1
    if _npu_profiler_active_flag:
        return 0
    return 2


cdef inline int _exact_npu_layer_norm_hot_state(object input, object weight, object bias, object normalized_shape):
    """Return 1 for inference, 2 for autograd, 0 for fallback."""
    cdef bint grad_on
    cdef bint requires_grad
    cdef TensorImpl inp
    cdef TensorImpl w
    cdef TensorImpl b
    cdef Py_ssize_t norm_ndim
    cdef Py_ssize_t i
    cdef Py_ssize_t lead
    _ensure_base()
    if not isinstance(input, _BaseTensor):
        return 0
    if _check_value(input):
        return 0
    if weight is None or bias is None:
        return 0
    if not isinstance(weight, _BaseTensor) or not isinstance(bias, _BaseTensor):
        return 0
    if _check_value(weight) or _check_value(bias):
        return 0
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    elif not isinstance(normalized_shape, (tuple, list)):
        return 0
    norm_ndim = len(normalized_shape)
    inp = <TensorImpl>input
    if inp._device_type != 1 or norm_ndim <= 0 or norm_ndim > inp._ndim:
        return 0
    if _functionalize_active_flag or _pipeline_active_flag or _npu_autocast_active_flag:
        return 0
    if not _tensor_has_strict_contiguous_stride(inp):
        return 0
    lead = inp._ndim - norm_ndim
    for i in range(norm_ndim):
        if inp._c_shape[lead + i] != <int64_t>normalized_shape[i]:
            return 0
    requires_grad = inp.requires_grad
    if weight is not None:
        w = <TensorImpl>weight
        if w._device_type != 1 or w._device_index != inp._device_index or w._dtype_code != inp._dtype_code:
            return 0
        if w._ndim != norm_ndim:
            return 0
        for i in range(norm_ndim):
            if w._c_shape[i] != <int64_t>normalized_shape[i]:
                return 0
        requires_grad = requires_grad or w.requires_grad
    if bias is not None:
        b = <TensorImpl>bias
        if b._device_type != 1 or b._device_index != inp._device_index or b._dtype_code != inp._dtype_code:
            return 0
        if b._ndim != norm_ndim:
            return 0
        for i in range(norm_ndim):
            if b._c_shape[i] != <int64_t>normalized_shape[i]:
                return 0
        requires_grad = requires_grad or b.requires_grad
    grad_on = _grad_enabled_fast()
    if not grad_on or not requires_grad:
        return 1
    return 2


cdef inline int _exact_npu_binary_hot_state(object a, object b):
    """Return 1 for inference, 2 for same-shape autograd, 0 for fallback."""
    cdef bint grad_on
    cdef bint requires_grad
    if _functionalize_active_flag or _pipeline_active_flag:
        return 0
    grad_on = _grad_enabled_fast()
    requires_grad = (<TensorImpl>a).requires_grad or (<TensorImpl>b).requires_grad
    if not grad_on or not requires_grad:
        return 1
    if (<TensorImpl>a)._shape_tuple != (<TensorImpl>b)._shape_tuple:
        return 0
    if _npu_profiler_active_flag or _npu_autocast_active_flag:
        return 0
    return 2


cdef inline int _exact_npu_unary_hot_state(object a):
    """Return 1 for inference, 2 for autograd, 0 for fallback."""
    cdef bint grad_on
    if _functionalize_active_flag or _pipeline_active_flag:
        return 0
    grad_on = _grad_enabled_fast()
    if not grad_on or not (<TensorImpl>a).requires_grad:
        return 1
    if _npu_profiler_active_flag or _npu_autocast_active_flag:
        return 0
    return 2


cdef inline bint _is_npu_tensor_pair(object a, object b):
    """True only when both operands are tensors on the NPU device."""
    if isinstance(a, TensorImpl) and isinstance(b, TensorImpl):
        return (<TensorImpl>a)._device_type == 1 and (<TensorImpl>b)._device_type == 1
    cdef object a_dev = getattr(a, "device", None)
    cdef object b_dev = getattr(b, "device", None)
    if a_dev is None or b_dev is None:
        return False
    return a_dev.type == "npu" and b_dev.type == "npu"


cdef inline bint _npu_pair_ready(object a, object b):
    """True if both operands are NPU tensors and global fast-path guards allow bypass."""
    cdef object b_dev
    if isinstance(a, TensorImpl) and isinstance(b, TensorImpl):
        if (<TensorImpl>a)._device_type != 1 or (<TensorImpl>b)._device_type != 1:
            return False
    else:
        b_dev = getattr(b, "device", None)
        if b_dev is None:
            return False
        if a.device.type != "npu" or b_dev.type != "npu":
            return False
    if _functionalize_active_flag:
        return False
    if _pipeline_active_flag:
        return False
    return True


cdef inline bint _npu_unary_ready(object a):
    """True if a single tensor is NPU and global fast-path guards allow bypass."""
    cdef object dev
    if isinstance(a, TensorImpl):
        if (<TensorImpl>a)._device_type != 1:
            return False
    else:
        dev = getattr(a, "device", None)
        if dev is None or dev.type != "npu":
            return False
    if _functionalize_active_flag:
        return False
    if _pipeline_active_flag:
        return False
    return True


cdef inline bint _npu_fast_ok(object a, object b):
    """True if both are NPU tensors and inference can call the NPU kernel directly."""
    cdef bint grad_on
    if not _npu_pair_ready(a, b):
        return False
    grad_on = _grad_enabled_fast()
    if isinstance(a, TensorImpl) and isinstance(b, TensorImpl):
        if grad_on and ((<TensorImpl>a).requires_grad or (<TensorImpl>b).requires_grad):
            return False
    elif grad_on and (getattr(a, "requires_grad", False) or getattr(b, "requires_grad", False)):
        return False
    return True


cdef inline bint _npu_same_shape(object a, object b):
    """True if a and b have identical shapes (no broadcasting on backward)."""
    if isinstance(a, TensorImpl) and isinstance(b, TensorImpl):
        return (<TensorImpl>a)._shape_tuple == (<TensorImpl>b)._shape_tuple
    return getattr(a, "shape", None) == getattr(b, "shape", None)


cdef inline int _npu_binary_hot_state(object a, object b):
    """Return 1 for inference, 2 for same-shape autograd, 0 for fallback."""
    cdef bint grad_on
    cdef bint requires_grad
    if not _npu_pair_ready(a, b):
        return 0
    grad_on = _grad_enabled_fast()
    if isinstance(a, TensorImpl) and isinstance(b, TensorImpl):
        requires_grad = (<TensorImpl>a).requires_grad or (<TensorImpl>b).requires_grad
    else:
        requires_grad = getattr(a, "requires_grad", False) or getattr(b, "requires_grad", False)
    if not grad_on or not requires_grad:
        return 1
    if not _npu_same_shape(a, b):
        return 0
    if _profiler_active() or _npu_autocast_active_flag:
        return 0
    return 2


cdef inline bint _npu_autograd_binary_fast_ok(object a, object b):
    """True if the NPU binary autograd fast path may attach grad metadata itself.

    Requires identical operand shapes: the fast backward nodes return the
    incoming grad (add) or grad*other (mul) without sum-to-shape reduction,
    so any broadcasting must fall back to the dispatcher's reducing backward.
    """
    return _npu_binary_hot_state(a, b) == 2


cdef inline bint _npu_unary_fast_ok(object a):
    """True if a single NPU tensor can use a direct inference fast path."""
    cdef bint grad_on
    if not _npu_unary_ready(a):
        return False
    grad_on = _grad_enabled_fast()
    if isinstance(a, TensorImpl):
        if grad_on and (<TensorImpl>a).requires_grad:
            return False
    elif grad_on and getattr(a, "requires_grad", False):
        return False
    return True


cdef inline int _npu_unary_hot_state(object a):
    """Return 1 for inference, 2 for autograd, 0 for fallback."""
    cdef bint grad_on
    cdef bint requires_grad
    if not _npu_unary_ready(a):
        return 0
    grad_on = _grad_enabled_fast()
    if isinstance(a, TensorImpl):
        requires_grad = (<TensorImpl>a).requires_grad
    else:
        requires_grad = getattr(a, "requires_grad", False)
    if not grad_on or not requires_grad:
        return 1
    if _profiler_active() or _npu_autocast_active_flag:
        return 0
    return 2


cdef inline bint _npu_autograd_unary_fast_ok(object a):
    """True if the NPU unary autograd fast path may attach grad metadata itself."""
    return _npu_unary_hot_state(a) == 2


cdef inline object _npu_accumulate_grad_node(TensorImpl t):
    cdef object acc = t._accumulate_grad_node
    if acc is None:
        acc = AccumulateGrad(t)
        t._accumulate_grad_node = acc
    return acc


cdef inline tuple _npu_edge_for(TensorImpl t):
    cdef object fn = t.grad_fn
    if fn is not None:
        return (fn, t._output_nr)
    # Leaf AccumulateGrad nodes are public graph-introspection metadata; the
    # engine accumulates leaf grads directly from node.inputs. Defer creating the
    # AccumulateGrad object until next_functions is actually inspected/backward
    # builds dependencies, while still freezing non-leaf grad_fn edges eagerly.
    return (None, 0 if t.requires_grad else t._output_nr)


cdef inline tuple _npu_materialize_leaf_edges(tuple inputs, tuple cached):
    cdef Py_ssize_t i
    cdef object inp
    cdef object fn
    cdef object output_nr
    cdef list out = None
    for i in range(len(cached)):
        inp = inputs[i]
        fn, output_nr = cached[i]
        if fn is None and isinstance(inp, TensorImpl) and (<TensorImpl>inp).requires_grad:
            if out is None:
                out = list(cached)
            out[i] = (_npu_accumulate_grad_node(<TensorImpl>inp), 0)
    if out is None:
        return cached
    return tuple(out)


cdef inline tuple _npu_binary_edges(object a, object b):
    return (_npu_edge_for(<TensorImpl>a), _npu_edge_for(<TensorImpl>b))


cdef inline object _npu_binary_edges_if_needed(object a, object b):
    if (<TensorImpl>a).grad_fn is None and (<TensorImpl>b).grad_fn is None:
        return None
    return _npu_binary_edges(a, b)


cdef inline tuple _npu_unary_edges(object a):
    return (_npu_edge_for(<TensorImpl>a),)


cdef inline object _npu_unary_edges_if_needed(object a):
    if (<TensorImpl>a).grad_fn is None:
        return None
    return _npu_unary_edges(a)


cdef inline tuple _npu_get_binary_edges(object a, object b, object cached):
    if cached is None:
        cached = _npu_binary_edges(a, b)
    cached = _npu_materialize_leaf_edges((a, b), <tuple>cached)
    return <tuple>cached


cdef inline tuple _npu_get_unary_edges(object a, object cached):
    if cached is None:
        cached = _npu_unary_edges(a)
    cached = _npu_materialize_leaf_edges((a,), <tuple>cached)
    return <tuple>cached


cdef class _NpuSavedTensor:
    cdef object _tensor_ref
    cdef int64_t _saved_version
    cdef bint _released
    cdef object _hooks
    cdef object _packed

    cdef void _init_fast(self, object tensor):
        self._tensor_ref = tensor
        self._saved_version = (<TensorImpl>tensor)._version_value
        self._released = False
        self._hooks = None
        self._packed = None

    def __init__(self, object tensor):
        self._init_fast(tensor)

    def register_hooks(self, *args):
        cdef object pack
        cdef object unpack
        cdef int64_t before_version
        cdef int64_t after_version
        cdef object packed
        if len(args) != 2:
            raise TypeError("incompatible function arguments")
        pack, unpack = args
        if not callable(pack) or not callable(unpack):
            raise TypeError("incompatible function arguments")
        if self._released:
            raise RuntimeError(
                "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). "
                "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). "
                "Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."
            )
        if self._hooks is not None:
            raise RuntimeError("SavedTensor hooks have already been set")
        before_version = (<TensorImpl>self._tensor_ref)._version_value
        packed = pack(self._tensor_ref)
        after_version = (<TensorImpl>self._tensor_ref)._version_value
        if before_version != after_version:
            raise RuntimeError("A saved tensor pack hook is modifying its input in place.")
        self._hooks = (pack, unpack)
        self._packed = packed

    def release(self):
        self._released = True

    def materialize(self):
        cdef object shape
        cdef object unpack
        cdef object result
        cdef int64_t current_version
        if self._released:
            raise RuntimeError(
                "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). "
                "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). "
                "Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."
            )
        current_version = (<TensorImpl>self._tensor_ref)._version_value
        if current_version != self._saved_version:
            shape = "x".join(str(d) for d in (<TensorImpl>self._tensor_ref)._shape_tuple)
            raise RuntimeError(
                "one of the variables needed for gradient computation has been modified by an inplace operation: "
                f"[torch.Tensor [{shape}]], which is output 0 of AsStridedBackward0, is at version {current_version}; "
                f"expected version {self._saved_version} instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, "
                "with torch.autograd.set_detect_anomaly(True)."
            )
        if self._hooks is None:
            return self._tensor_ref
        _, unpack = self._hooks
        result = unpack(self._packed)
        _ensure_base()
        if not isinstance(result, _BaseTensor):
            raise TypeError("Output of saved tensor unpack_hook expected to be a Tensor")
        return result


cdef inline object _npu_saved_tensor_no_hooks(object tensor):
    cdef _NpuSavedTensor saved = _NpuSavedTensor.__new__(_NpuSavedTensor)
    saved._init_fast(tensor)
    return saved


cdef inline object _npu_saved_tensor(object tensor):
    if _npu_has_saved_hooks():
        return SavedTensor(tensor)
    return _npu_saved_tensor_no_hooks(tensor)


cdef inline object _npu_materialize_tensor_version(object tensor, int64_t saved_version, bint released):
    cdef object shape
    cdef int64_t current_version
    if released:
        raise RuntimeError(
            "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). "
            "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). "
            "Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."
        )
    current_version = (<TensorImpl>tensor)._version_value
    if current_version != saved_version:
        shape = "x".join(str(d) for d in (<TensorImpl>tensor)._shape_tuple)
        raise RuntimeError(
            "one of the variables needed for gradient computation has been modified by an inplace operation: "
            f"[torch.Tensor [{shape}]], which is output 0 of AsStridedBackward0, is at version {current_version}; "
            f"expected version {saved_version} instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, "
            "with torch.autograd.set_detect_anomaly(True)."
        )
    return tensor


cdef class _NpuSavedTensorProxy:
    cdef object _owner
    cdef int _slot
    cdef object _hooks
    cdef object _packed

    cdef void _init_fast(self, object owner, int slot):
        self._owner = owner
        self._slot = slot
        self._hooks = None
        self._packed = None

    def __init__(self, object owner, int slot):
        self._init_fast(owner, slot)

    def register_hooks(self, *args):
        cdef object pack
        cdef object unpack
        cdef object tensor
        cdef object packed
        cdef int64_t before_version
        cdef int64_t after_version
        if len(args) != 2:
            raise TypeError("incompatible function arguments")
        pack, unpack = args
        if not callable(pack) or not callable(unpack):
            raise TypeError("incompatible function arguments")
        if self._owner._saved_proxy_released_py(self._slot):
            raise RuntimeError(
                "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). "
                "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). "
                "Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."
            )
        if self._hooks is not None:
            raise RuntimeError("SavedTensor hooks have already been set")
        tensor = self._owner._saved_proxy_tensor_py(self._slot)
        before_version = (<TensorImpl>tensor)._version_value
        packed = pack(tensor)
        after_version = (<TensorImpl>tensor)._version_value
        if before_version != after_version:
            raise RuntimeError("A saved tensor pack hook is modifying its input in place.")
        self._hooks = (pack, unpack)
        self._packed = packed

    def release(self):
        self._owner._saved_proxy_release_py(self._slot)

    def materialize(self):
        cdef object unpack
        cdef object result
        cdef object tensor = self._owner._saved_proxy_materialize_py(self._slot)
        if self._hooks is None:
            return tensor
        _, unpack = self._hooks
        result = unpack(self._packed)
        _ensure_base()
        if not isinstance(result, _BaseTensor):
            raise TypeError("Output of saved tensor unpack_hook expected to be a Tensor")
        return result


cdef inline object _npu_saved_tensor_proxy(object owner, int slot):
    cdef _NpuSavedTensorProxy saved = _NpuSavedTensorProxy.__new__(_NpuSavedTensorProxy)
    saved._init_fast(owner, slot)
    return saved


cdef class _NpuAddBackward(_CyAutogradNode):
    cdef public object _self
    cdef public object _other
    cdef public object _alpha

    cdef void _init_fast(self, object self_, object other):
        global _npu_add_alpha_saved_value, _npu_add_saved_fields
        # Manual field initialization trims the NPU eager attach path while keeping
        # the same public Node attributes the engine/introspection APIs consume.
        self.inputs = (self_, other)
        self._saved_tensors_list = _npu_empty_saved_tensors
        if _npu_add_alpha_saved_value is None:
            _npu_add_alpha_saved_value = _SavedValue(1)
            _npu_add_saved_fields = {"alpha": _npu_add_alpha_saved_value}
        self._saved_fields = _npu_add_saved_fields
        self._hooks = None
        self._prehooks = None
        self._metadata = None
        self._name = "AddBackward0"
        self._anomaly_trace = None
        self._anomaly_parent = None
        self._next_functions_cache = _npu_binary_edges_if_needed(self_, other)
        self._self = self_
        self._other = other
        self._alpha = 1

    def __init__(self, self_, other):
        self._init_fast(self_, other)

    @property
    def next_functions(self):
        cached = _npu_get_binary_edges(self._self, self._other, self._next_functions_cache)
        self._next_functions_cache = cached
        return cached

    def backward(self, grad):
        grad_self = grad if (<TensorImpl>self._self).requires_grad else None
        grad_other = grad if (<TensorImpl>self._other).requires_grad else None
        return grad_self, grad_other



cdef inline object _mark_npu_owned_backward_grad(object grad):
    """Mark a fresh NPU backward output that leaf accumulation may steal."""
    if grad is not None:
        try:
            grad._candle_npu_owned_backward_grad = True
        except Exception:
            pass
    return grad


cdef class _NpuMulBackward(_CyAutogradNode):
    cdef public object _self
    cdef public object _other
    cdef int64_t _saved_self_version
    cdef int64_t _saved_other_version
    cdef bint _saved_self_released
    cdef bint _saved_other_released
    cdef object _saved_self_obj
    cdef object _saved_other_obj

    cdef void _init_fast(self, object self_, object other):
        # Manual field init + lazy hooks. No-hooks common case stores saved
        # metadata inline; raw SavedTensor proxy objects are created only for
        # graph introspection or explicit saved tensor hooks.
        cdef object saved_self
        cdef object saved_other
        self.inputs = (self_, other)
        self._hooks = None
        self._prehooks = None
        self._metadata = None
        self._name = "MulTensorBackward0"
        self._anomaly_trace = None
        self._anomaly_parent = None
        self._next_functions_cache = _npu_binary_edges_if_needed(self_, other)
        self._self = self_
        self._other = other
        self._saved_self_released = False
        self._saved_other_released = False
        if _npu_has_saved_hooks():
            saved_self = SavedTensor(self_)
            if self_ is other:
                saved_other = saved_self
            else:
                saved_other = SavedTensor(other)
            self._saved_self_obj = saved_self
            self._saved_other_obj = saved_other
            self._saved_self_version = 0
            self._saved_other_version = 0
        else:
            self._saved_self_obj = None
            self._saved_other_obj = None
            self._saved_self_version = (<TensorImpl>self_)._version_value
            self._saved_other_version = self._saved_self_version if self_ is other else (<TensorImpl>other)._version_value
        self._saved_tensors_list = _npu_empty_saved_tensors
        self._saved_fields = _npu_empty_saved_fields

    def __init__(self, self_, other):
        self._init_fast(self_, other)

    @property
    def next_functions(self):
        cached = _npu_get_binary_edges(self._self, self._other, self._next_functions_cache)
        self._next_functions_cache = cached
        return cached

    cpdef object _saved_proxy_tensor_py(self, int slot):
        if slot == 0:
            return self._self
        return self._other

    cpdef bint _saved_proxy_released_py(self, int slot):
        if slot == 0:
            return self._saved_self_released
        return self._saved_other_released

    cpdef void _saved_proxy_release_py(self, int slot):
        if slot == 0:
            self._saved_self_released = True
        else:
            self._saved_other_released = True

    cpdef object _saved_proxy_materialize_py(self, int slot):
        if slot == 0:
            return _npu_materialize_tensor_version(self._self, self._saved_self_version, self._saved_self_released)
        return _npu_materialize_tensor_version(self._other, self._saved_other_version, self._saved_other_released)

    cdef object _raw_saved_self_proxy(self):
        if self._saved_self_obj is None:
            self._saved_self_obj = _npu_saved_tensor_proxy(self, 0)
        return self._saved_self_obj

    cdef object _raw_saved_other_proxy(self):
        if self._saved_other_obj is None:
            if self._self is self._other:
                self._saved_other_obj = self._raw_saved_self_proxy()
            else:
                self._saved_other_obj = _npu_saved_tensor_proxy(self, 1)
        return self._saved_other_obj

    def saved_tensors(self):
        cdef object self_saved
        cdef object other_saved
        if self._saved_self_obj is not None:
            self_saved = self._saved_self_obj.materialize()
        else:
            self_saved = self._saved_proxy_materialize_py(0)
        if self._saved_other_obj is not None:
            other_saved = self._saved_other_obj.materialize()
        elif self._self is self._other and self._saved_self_obj is not None:
            other_saved = self._saved_self_obj.materialize()
        else:
            other_saved = self._saved_proxy_materialize_py(1)
        return (self_saved, other_saved)

    def release_saved_tensors(self):
        if self._saved_self_obj is not None:
            self._saved_self_obj.release()
        else:
            self._saved_self_released = True
        if self._self is self._other or self._saved_other_obj is self._saved_self_obj:
            self._saved_other_released = True
            return
        if self._saved_other_obj is not None:
            self._saved_other_obj.release()
        else:
            self._saved_other_released = True

    def __getattr__(self, name):
        if name == "_raw_saved_self":
            return self._raw_saved_self_proxy()
        if name == "_raw_saved_other":
            return self._raw_saved_other_proxy()
        if name == "_saved_self":
            if self._saved_self_obj is not None:
                return self._saved_self_obj.materialize()
            return self._saved_proxy_materialize_py(0)
        if name == "_saved_other":
            if self._saved_other_obj is not None:
                return self._saved_other_obj.materialize()
            if self._self is self._other and self._saved_self_obj is not None:
                return self._saved_self_obj.materialize()
            return self._saved_proxy_materialize_py(1)
        if name == "_raw_saved_tensors":
            return (self._raw_saved_self_proxy(), self._raw_saved_other_proxy())
        if name == "_saved_tensors":
            return self.saved_tensors()
        return _CyAutogradNode.__getattr__(self, name)

    def backward(self, grad):
        grad_self = None
        grad_other = None
        if self._saved_self_obj is not None:
            self_ = self._saved_self_obj.materialize()
        else:
            self_ = self._saved_proxy_materialize_py(0)
        if self._saved_other_obj is not None:
            other = self._saved_other_obj.materialize()
        elif self._self is self._other and self._saved_self_obj is not None:
            other = self._saved_self_obj.materialize()
        else:
            other = self._saved_proxy_materialize_py(1)
        _ensure_npu_refs()
        _ensure_npu_autograd_refs()
        # Under create_graph the gradient itself must be differentiable, so route
        # through the dispatched (graph-building) mul instead of the raw kernel.
        if _npu_create_graph_fn():
            if getattr(self_, "requires_grad", False):
                grad_self = _npu_dispatch_mul_fn("mul", None, grad, other)
            if getattr(other, "requires_grad", False):
                grad_other = _npu_dispatch_mul_fn("mul", None, grad, self_)
            return grad_self, grad_other
        if getattr(self_, "requires_grad", False):
            grad_self = _mark_npu_owned_backward_grad(_cy_fast_npu_mul(grad, other))
        if getattr(other, "requires_grad", False):
            grad_other = _mark_npu_owned_backward_grad(_cy_fast_npu_mul(grad, self_))
        return grad_self, grad_other


cdef class _NpuAddmmBackward(_CyAutogradNode):
    cdef public object _self
    cdef public object _mat1
    cdef public object _mat2
    cdef public object _beta
    cdef public object _alpha
    cdef public object _self_shape
    cdef int64_t _saved_mat1_version
    cdef int64_t _saved_mat2_version
    cdef bint _saved_mat1_released
    cdef bint _saved_mat2_released
    cdef object _saved_mat1_obj
    cdef object _saved_mat2_obj

    cdef void _init_fast(self, object self_, object mat1, object mat2):
        cdef object saved_mat1
        cdef object saved_mat2
        self.inputs = (self_, mat1, mat2)
        self._hooks = None
        self._prehooks = None
        self._metadata = None
        self._name = "AddmmBackward0"
        self._anomaly_trace = None
        self._anomaly_parent = None
        self._next_functions_cache = (_npu_edge_for(<TensorImpl>self_), _npu_edge_for(<TensorImpl>mat1), _npu_edge_for(<TensorImpl>mat2))
        self._self = self_
        self._mat1 = mat1
        self._mat2 = mat2
        self._beta = 1
        self._alpha = 1
        self._self_shape = (<TensorImpl>self_)._shape_tuple
        self._saved_mat1_released = False
        self._saved_mat2_released = False
        if _npu_has_saved_hooks():
            saved_mat1 = SavedTensor(mat1)
            if mat1 is mat2:
                saved_mat2 = saved_mat1
            else:
                saved_mat2 = SavedTensor(mat2)
            self._saved_mat1_obj = saved_mat1
            self._saved_mat2_obj = saved_mat2
            self._saved_mat1_version = 0
            self._saved_mat2_version = 0
        else:
            self._saved_mat1_obj = None
            self._saved_mat2_obj = None
            self._saved_mat1_version = (<TensorImpl>mat1)._version_value
            self._saved_mat2_version = self._saved_mat1_version if mat1 is mat2 else (<TensorImpl>mat2)._version_value
        self._saved_tensors_list = _npu_empty_saved_tensors
        self._saved_fields = _npu_empty_saved_fields

    def __init__(self, self_, mat1, mat2):
        self._init_fast(self_, mat1, mat2)

    cpdef tuple _engine_next_functions(self):
        return self._next_functions_cache

    @property
    def next_functions(self):
        cached = _npu_materialize_leaf_edges(self.inputs, self._next_functions_cache)
        self._next_functions_cache = cached
        return cached

    cpdef object _saved_proxy_tensor_py(self, int slot):
        if slot == 0:
            return self._mat1
        return self._mat2

    cpdef bint _saved_proxy_released_py(self, int slot):
        if slot == 0:
            return self._saved_mat1_released
        return self._saved_mat2_released

    cpdef void _saved_proxy_release_py(self, int slot):
        if slot == 0:
            self._saved_mat1_released = True
        else:
            self._saved_mat2_released = True

    cpdef object _saved_proxy_materialize_py(self, int slot):
        if slot == 0:
            return _npu_materialize_tensor_version(self._mat1, self._saved_mat1_version, self._saved_mat1_released)
        return _npu_materialize_tensor_version(self._mat2, self._saved_mat2_version, self._saved_mat2_released)

    cdef object _raw_saved_mat1_proxy(self):
        if self._saved_mat1_obj is None:
            self._saved_mat1_obj = _npu_saved_tensor_proxy(self, 0)
        return self._saved_mat1_obj

    cdef object _raw_saved_mat2_proxy(self):
        if self._saved_mat2_obj is None:
            if self._mat1 is self._mat2:
                self._saved_mat2_obj = self._raw_saved_mat1_proxy()
            else:
                self._saved_mat2_obj = _npu_saved_tensor_proxy(self, 1)
        return self._saved_mat2_obj

    def saved_tensors(self):
        cdef object mat1_saved
        cdef object mat2_saved
        if self._saved_mat1_obj is not None:
            mat1_saved = self._saved_mat1_obj.materialize()
        else:
            mat1_saved = self._saved_proxy_materialize_py(0)
        if self._saved_mat2_obj is not None:
            mat2_saved = self._saved_mat2_obj.materialize()
        elif self._mat1 is self._mat2 and self._saved_mat1_obj is not None:
            mat2_saved = self._saved_mat1_obj.materialize()
        else:
            mat2_saved = self._saved_proxy_materialize_py(1)
        return (mat1_saved, mat2_saved)

    def release_saved_tensors(self):
        if self._saved_mat1_obj is not None:
            self._saved_mat1_obj.release()
        else:
            self._saved_mat1_released = True
        if self._mat1 is self._mat2 or self._saved_mat2_obj is self._saved_mat1_obj:
            self._saved_mat2_released = True
            return
        if self._saved_mat2_obj is not None:
            self._saved_mat2_obj.release()
        else:
            self._saved_mat2_released = True

    def __getattr__(self, name):
        if name == "_raw_saved_mat1":
            return self._raw_saved_mat1_proxy()
        if name == "_raw_saved_mat2":
            return self._raw_saved_mat2_proxy()
        if name == "_saved_mat1":
            if self._saved_mat1_obj is not None:
                return self._saved_mat1_obj.materialize()
            return self._saved_proxy_materialize_py(0)
        if name == "_saved_mat2":
            if self._saved_mat2_obj is not None:
                return self._saved_mat2_obj.materialize()
            if self._mat1 is self._mat2 and self._saved_mat1_obj is not None:
                return self._saved_mat1_obj.materialize()
            return self._saved_proxy_materialize_py(1)
        if name == "_raw_saved_tensors":
            return (self._raw_saved_mat1_proxy(), self._raw_saved_mat2_proxy())
        if name == "_saved_tensors":
            return self.saved_tensors()
        return _CyAutogradNode.__getattr__(self, name)

    def backward(self, grad):
        cdef object mat1
        cdef object mat2
        cdef object grad_self = None
        cdef object grad_mat1 = None
        cdef object grad_mat2 = None
        if self._saved_mat1_obj is not None:
            mat1 = self._saved_mat1_obj.materialize()
        else:
            mat1 = self._saved_proxy_materialize_py(0)
        if self._saved_mat2_obj is not None:
            mat2 = self._saved_mat2_obj.materialize()
        elif self._mat1 is self._mat2 and self._saved_mat1_obj is not None:
            mat2 = self._saved_mat1_obj.materialize()
        else:
            mat2 = self._saved_proxy_materialize_py(1)
        _ensure_npu_refs()
        _ensure_npu_autograd_refs()
        if _npu_create_graph_fn():
            _ensure_dispatch()
            if getattr(self._self, "requires_grad", False):
                grad_self = _dispatch_fn("sum", None, grad, dim=0, keepdim=False)
            if getattr(mat1, "requires_grad", False):
                grad_mat1 = _dispatch_fn("mm_mat1_backward", None, grad, mat2, mat1.shape, None, None, 1)
            if getattr(mat2, "requires_grad", False):
                grad_mat2 = _dispatch_fn("mm_mat2_backward", None, grad, mat1, mat2.shape, None, None, 1)
            return grad_self, grad_mat1, grad_mat2
        if getattr(self._self, "requires_grad", False):
            grad_self = _mark_npu_owned_backward_grad(_cy_fast_npu_sum(grad, 0, False))
        if getattr(mat1, "requires_grad", False):
            grad_mat1 = _mark_npu_owned_backward_grad(_cy_fast_npu_mm_mat1_backward(grad, mat2, 1))
        if getattr(mat2, "requires_grad", False):
            grad_mat2 = _mark_npu_owned_backward_grad(_cy_fast_npu_mm_mat2_backward(grad, mat1, 1))
        return grad_self, grad_mat1, grad_mat2


cdef class _NpuLinearBackward(_CyAutogradNode):
    cdef public object _input
    cdef public object _weight
    cdef public object _bias
    cdef public object _input_shape
    cdef int64_t _saved_input_version
    cdef int64_t _saved_weight_version
    cdef bint _saved_input_released
    cdef bint _saved_weight_released
    cdef object _saved_input_obj
    cdef object _saved_weight_obj

    cdef void _init_fast(self, object input_, object weight, object bias):
        cdef object saved_input
        cdef object saved_weight
        self.inputs = (input_, weight, bias)
        self._hooks = None
        self._prehooks = None
        self._metadata = None
        self._name = "LinearBackward0"
        self._anomaly_trace = None
        self._anomaly_parent = None
        self._next_functions_cache = (_npu_edge_for(<TensorImpl>input_), _npu_edge_for(<TensorImpl>weight), _npu_edge_for(<TensorImpl>bias))
        self._input = input_
        self._weight = weight
        self._bias = bias
        self._input_shape = (<TensorImpl>input_)._shape_tuple
        self._saved_input_released = False
        self._saved_weight_released = False
        if _npu_has_saved_hooks():
            saved_input = SavedTensor(input_)
            saved_weight = SavedTensor(weight)
            self._saved_input_obj = saved_input
            self._saved_weight_obj = saved_weight
            self._saved_input_version = 0
            self._saved_weight_version = 0
        else:
            self._saved_input_obj = None
            self._saved_weight_obj = None
            self._saved_input_version = (<TensorImpl>input_)._version_value
            self._saved_weight_version = (<TensorImpl>weight)._version_value
        self._saved_tensors_list = _npu_empty_saved_tensors
        self._saved_fields = _npu_empty_saved_fields

    def __init__(self, input_, weight, bias):
        self._init_fast(input_, weight, bias)

    cpdef tuple _engine_next_functions(self):
        return self._next_functions_cache

    @property
    def next_functions(self):
        cached = _npu_materialize_leaf_edges(self.inputs, self._next_functions_cache)
        self._next_functions_cache = cached
        return cached

    cpdef object _saved_proxy_tensor_py(self, int slot):
        if slot == 0:
            return self._input
        return self._weight

    cpdef bint _saved_proxy_released_py(self, int slot):
        if slot == 0:
            return self._saved_input_released
        return self._saved_weight_released

    cpdef void _saved_proxy_release_py(self, int slot):
        if slot == 0:
            self._saved_input_released = True
        else:
            self._saved_weight_released = True

    cpdef object _saved_proxy_materialize_py(self, int slot):
        if slot == 0:
            return _npu_materialize_tensor_version(self._input, self._saved_input_version, self._saved_input_released)
        return _npu_materialize_tensor_version(self._weight, self._saved_weight_version, self._saved_weight_released)

    cdef object _raw_saved_input_proxy(self):
        if self._saved_input_obj is None:
            self._saved_input_obj = _npu_saved_tensor_proxy(self, 0)
        return self._saved_input_obj

    cdef object _raw_saved_weight_proxy(self):
        if self._saved_weight_obj is None:
            self._saved_weight_obj = _npu_saved_tensor_proxy(self, 1)
        return self._saved_weight_obj

    def saved_tensors(self):
        cdef object input_saved
        cdef object weight_saved
        if self._saved_input_obj is not None:
            input_saved = self._saved_input_obj.materialize()
        else:
            input_saved = self._saved_proxy_materialize_py(0)
        if self._saved_weight_obj is not None:
            weight_saved = self._saved_weight_obj.materialize()
        else:
            weight_saved = self._saved_proxy_materialize_py(1)
        return (input_saved, weight_saved)

    def release_saved_tensors(self):
        if self._saved_input_obj is not None:
            self._saved_input_obj.release()
        else:
            self._saved_input_released = True
        if self._saved_weight_obj is not None:
            self._saved_weight_obj.release()
        else:
            self._saved_weight_released = True

    def __getattr__(self, name):
        if name == "_raw_saved_input":
            return self._raw_saved_input_proxy()
        if name == "_raw_saved_weight":
            return self._raw_saved_weight_proxy()
        if name == "_saved_input":
            if self._saved_input_obj is not None:
                return self._saved_input_obj.materialize()
            return self._saved_proxy_materialize_py(0)
        if name == "_saved_weight":
            if self._saved_weight_obj is not None:
                return self._saved_weight_obj.materialize()
            return self._saved_proxy_materialize_py(1)
        if name == "_raw_saved_tensors":
            return (self._raw_saved_input_proxy(), self._raw_saved_weight_proxy())
        if name == "_saved_tensors":
            return self.saved_tensors()
        return _CyAutogradNode.__getattr__(self, name)

    def backward(self, grad):
        cdef object input_
        cdef object weight
        cdef object grad_input = None
        cdef object grad_weight = None
        cdef object grad_bias = None
        cdef object grad_2d
        cdef object input_2d
        cdef object flat_shape
        cdef object grad_input_shape
        if self._saved_input_obj is not None:
            input_ = self._saved_input_obj.materialize()
        else:
            input_ = self._saved_proxy_materialize_py(0)
        if self._saved_weight_obj is not None:
            weight = self._saved_weight_obj.materialize()
        else:
            weight = self._saved_proxy_materialize_py(1)
        _ensure_npu_refs()
        _ensure_npu_autograd_refs()
        if _npu_create_graph_fn():
            _ensure_dispatch()
            if getattr(input_, "requires_grad", False):
                grad_input = _dispatch_fn("matmul", None, grad, weight)
            if (<TensorImpl>grad)._ndim == 2:
                grad_2d = grad
                input_2d = input_
            else:
                flat_shape = ((<TensorImpl>grad)._c_numel // (<TensorImpl>grad)._c_shape[(<TensorImpl>grad)._ndim - 1], (<TensorImpl>grad)._c_shape[(<TensorImpl>grad)._ndim - 1])
                grad_2d = _dispatch_fn("reshape", None, grad, flat_shape)
                input_2d = _dispatch_fn("reshape", None, input_, ((<TensorImpl>input_)._c_numel // (<TensorImpl>input_)._c_shape[(<TensorImpl>input_)._ndim - 1], (<TensorImpl>input_)._c_shape[(<TensorImpl>input_)._ndim - 1]))
            if getattr(weight, "requires_grad", False):
                grad_t = _dispatch_fn("transpose", None, grad_2d, 0, 1)
                grad_weight = _dispatch_fn("matmul", None, grad_t, input_2d)
            if getattr(self._bias, "requires_grad", False):
                grad_bias = _dispatch_fn("sum", None, grad_2d, dim=0, keepdim=False)
            return grad_input, grad_weight, grad_bias
        if (<TensorImpl>grad)._ndim == 2:
            grad_2d = grad
            input_2d = input_
            grad_input_shape = self._input_shape
        else:
            flat_shape = ((<TensorImpl>grad)._c_numel // (<TensorImpl>grad)._c_shape[(<TensorImpl>grad)._ndim - 1], (<TensorImpl>grad)._c_shape[(<TensorImpl>grad)._ndim - 1])
            if grad.is_contiguous():
                grad_2d = (<TensorImpl>grad).cy_view(flat_shape)
            elif _tensor_has_all_zero_stride(<TensorImpl>grad):
                # Full-sum backward supplies a scalar-expanded grad with all
                # strides zero.  Reshaping that view is metadata-only, but the
                # generic Python reshape path is expensive in LinearBackward.
                grad_2d = (<TensorImpl>grad).cy_as_strided(flat_shape, (0, 0), (<TensorImpl>grad)._c_offset)
            else:
                _ensure_dispatch()
                grad_2d = _dispatch_fn("reshape", None, grad, flat_shape)
            input_2d = (<TensorImpl>input_).cy_view(((<TensorImpl>input_)._c_numel // (<TensorImpl>input_)._c_shape[(<TensorImpl>input_)._ndim - 1], (<TensorImpl>input_)._c_shape[(<TensorImpl>input_)._ndim - 1]))
            grad_input_shape = self._input_shape
        if getattr(input_, "requires_grad", False):
            grad_input = _cy_fast_npu_matmul(grad_2d, weight)
            if (<TensorImpl>grad_input)._shape_tuple != grad_input_shape:
                grad_input = (<TensorImpl>grad_input).cy_view(grad_input_shape)
            grad_input = _mark_npu_owned_backward_grad(grad_input)
        if getattr(weight, "requires_grad", False):
            grad_weight = _mark_npu_owned_backward_grad(_cy_fast_npu_mm_mat2_backward(input_2d, grad_2d, 1))
        if getattr(self._bias, "requires_grad", False):
            grad_bias = _mark_npu_owned_backward_grad(_cy_fast_npu_sum(grad_2d, 0, False))
        return grad_input, grad_weight, grad_bias


cdef class _NpuLayerNormBackward(_CyAutogradNode):
    cdef public object _input
    cdef public object _weight
    cdef public object _bias
    cdef public object _normalized_shape
    cdef public object _eps
    cdef public object _backward_data
    cdef int64_t _saved_input_version
    cdef bint _saved_input_released
    cdef object _saved_input_obj

    cdef void _init_fast(self, object input_, object weight, object bias,
                         object normalized_shape, object eps, object backward_data):
        cdef object saved_input
        self.inputs = (input_, weight, bias)
        self._hooks = None
        self._prehooks = None
        self._metadata = None
        self._name = "LayerNormBackward0"
        self._anomaly_trace = None
        self._anomaly_parent = None
        self._next_functions_cache = (_npu_edge_for(<TensorImpl>input_), _npu_edge_for(<TensorImpl>weight), _npu_edge_for(<TensorImpl>bias))
        self._input = input_
        self._weight = weight
        self._bias = bias
        self._normalized_shape = tuple(normalized_shape)
        self._eps = eps
        self._backward_data = backward_data
        self._saved_input_released = False
        if _npu_has_saved_hooks():
            saved_input = SavedTensor(input_)
            self._saved_input_obj = saved_input
            self._saved_input_version = 0
        else:
            self._saved_input_obj = None
            self._saved_input_version = (<TensorImpl>input_)._version_value
        self._saved_tensors_list = _npu_empty_saved_tensors
        self._saved_fields = _npu_empty_saved_fields

    def __init__(self, input_, weight, bias, normalized_shape, eps, backward_data):
        self._init_fast(input_, weight, bias, normalized_shape, eps, backward_data)

    cpdef tuple _engine_next_functions(self):
        return self._next_functions_cache

    @property
    def next_functions(self):
        cached = _npu_materialize_leaf_edges(self.inputs, self._next_functions_cache)
        self._next_functions_cache = cached
        return cached

    cpdef object _saved_proxy_tensor_py(self, int slot):
        return self._input

    cpdef bint _saved_proxy_released_py(self, int slot):
        return self._saved_input_released

    cpdef void _saved_proxy_release_py(self, int slot):
        self._saved_input_released = True

    cpdef object _saved_proxy_materialize_py(self, int slot):
        return _npu_materialize_tensor_version(self._input, self._saved_input_version, self._saved_input_released)

    cdef object _raw_saved_input_proxy(self):
        if self._saved_input_obj is None:
            self._saved_input_obj = _npu_saved_tensor_proxy(self, 0)
        return self._saved_input_obj

    def saved_tensors(self):
        if self._saved_input_obj is not None:
            return (self._saved_input_obj.materialize(),)
        return (self._saved_proxy_materialize_py(0),)

    def release_saved_tensors(self):
        if self._saved_input_obj is not None:
            self._saved_input_obj.release()
        else:
            self._saved_input_released = True

    def __getattr__(self, name):
        if name == "_raw_saved_input":
            return self._raw_saved_input_proxy()
        if name == "_saved_input":
            if self._saved_input_obj is not None:
                return self._saved_input_obj.materialize()
            return self._saved_proxy_materialize_py(0)
        if name == "_raw_saved_tensors":
            return (self._raw_saved_input_proxy(),)
        if name == "_saved_tensors":
            return self.saved_tensors()
        return _CyAutogradNode.__getattr__(self, name)

    def backward(self, grad):
        cdef object input_
        cdef object grad_input
        cdef object grad_weight
        cdef object grad_bias
        if self._saved_input_obj is not None:
            input_ = self._saved_input_obj.materialize()
        else:
            input_ = self._saved_proxy_materialize_py(0)
        _ensure_npu_refs()
        _ensure_npu_autograd_refs()
        if _npu_create_graph_fn():
            _ensure_dispatch()
            from candle._backends.autograd import _layer_norm_backward as _generic_layer_norm_backward
            return _generic_layer_norm_backward(
                grad,
                self._input,
                input_,
                _npu_fast_backward_keyset,
                (self._normalized_shape, self._weight, self._bias, self._eps),
                {},
                self._backward_data,
            )
        grad_input, grad_weight, grad_bias = _cy_fast_npu_layer_norm_backward(
            grad,
            input_,
            self._backward_data,
            self._normalized_shape,
            self._weight,
            self._bias,
        )
        return (
            _mark_npu_owned_backward_grad(grad_input),
            _mark_npu_owned_backward_grad(grad_weight),
            _mark_npu_owned_backward_grad(grad_bias),
        )


cdef class _NpuGeluBackward(_CyAutogradNode):
    cdef public object _self
    cdef int64_t _saved_self_version
    cdef bint _saved_self_released
    cdef object _saved_self_obj

    cdef void _init_fast(self, object self_):
        cdef object saved_self
        self.inputs = (self_,)
        self._hooks = None
        self._prehooks = None
        self._metadata = None
        self._name = "GeluBackward0"
        self._anomaly_trace = None
        self._anomaly_parent = None
        self._next_functions_cache = _npu_unary_edges_if_needed(self_)
        self._self = self_
        self._saved_self_released = False
        if _npu_has_saved_hooks():
            saved_self = SavedTensor(self_)
            self._saved_self_obj = saved_self
            self._saved_self_version = 0
        else:
            self._saved_self_obj = None
            self._saved_self_version = (<TensorImpl>self_)._version_value
        self._saved_tensors_list = _npu_empty_saved_tensors
        self._saved_fields = _npu_empty_saved_fields

    def __init__(self, self_):
        self._init_fast(self_)

    @property
    def next_functions(self):
        cached = _npu_get_unary_edges(self._self, self._next_functions_cache)
        self._next_functions_cache = cached
        return cached

    cpdef object _saved_proxy_tensor_py(self, int slot):
        return self._self

    cpdef bint _saved_proxy_released_py(self, int slot):
        return self._saved_self_released

    cpdef void _saved_proxy_release_py(self, int slot):
        self._saved_self_released = True

    cpdef object _saved_proxy_materialize_py(self, int slot):
        return _npu_materialize_tensor_version(self._self, self._saved_self_version, self._saved_self_released)

    cdef object _raw_saved_self_proxy(self):
        if self._saved_self_obj is None:
            self._saved_self_obj = _npu_saved_tensor_proxy(self, 0)
        return self._saved_self_obj

    def saved_tensors(self):
        if self._saved_self_obj is not None:
            return (self._saved_self_obj.materialize(),)
        return (self._saved_proxy_materialize_py(0),)

    def release_saved_tensors(self):
        if self._saved_self_obj is not None:
            self._saved_self_obj.release()
        else:
            self._saved_self_released = True

    def __getattr__(self, name):
        if name == "_raw_saved_self":
            return self._raw_saved_self_proxy()
        if name == "_saved_self":
            if self._saved_self_obj is not None:
                return self._saved_self_obj.materialize()
            return self._saved_proxy_materialize_py(0)
        if name == "_raw_saved_tensors":
            return (self._raw_saved_self_proxy(),)
        if name == "_saved_tensors":
            if self._saved_self_obj is not None:
                return (self._saved_self_obj.materialize(),)
            return (self._saved_proxy_materialize_py(0),)
        return _CyAutogradNode.__getattr__(self, name)

    def backward(self, grad):
        if self._saved_self_obj is not None:
            self_ = self._saved_self_obj.materialize()
        else:
            self_ = self._saved_proxy_materialize_py(0)
        _ensure_npu_autograd_refs()
        if not getattr(self_, "requires_grad", False):
            return (None,)
        return (_mark_npu_owned_backward_grad(_cy_fast_npu_gelu_backward(grad, self_)),)


cdef class _NpuSiluBackward(_CyAutogradNode):
    cdef public object _self
    cdef int64_t _saved_self_version
    cdef bint _saved_self_released
    cdef object _saved_self_obj

    cdef void _init_fast(self, object self_):
        cdef object saved_self
        self.inputs = (self_,)
        self._hooks = None
        self._prehooks = None
        self._metadata = None
        self._name = "SiluBackward0"
        self._anomaly_trace = None
        self._anomaly_parent = None
        self._next_functions_cache = _npu_unary_edges_if_needed(self_)
        self._self = self_
        self._saved_self_released = False
        if _npu_has_saved_hooks():
            saved_self = SavedTensor(self_)
            self._saved_self_obj = saved_self
            self._saved_self_version = 0
        else:
            self._saved_self_obj = None
            self._saved_self_version = (<TensorImpl>self_)._version_value
        self._saved_tensors_list = _npu_empty_saved_tensors
        self._saved_fields = _npu_empty_saved_fields

    def __init__(self, self_):
        self._init_fast(self_)

    @property
    def next_functions(self):
        cached = _npu_get_unary_edges(self._self, self._next_functions_cache)
        self._next_functions_cache = cached
        return cached

    cpdef object _saved_proxy_tensor_py(self, int slot):
        return self._self

    cpdef bint _saved_proxy_released_py(self, int slot):
        return self._saved_self_released

    cpdef void _saved_proxy_release_py(self, int slot):
        self._saved_self_released = True

    cpdef object _saved_proxy_materialize_py(self, int slot):
        return _npu_materialize_tensor_version(self._self, self._saved_self_version, self._saved_self_released)

    cdef object _raw_saved_self_proxy(self):
        if self._saved_self_obj is None:
            self._saved_self_obj = _npu_saved_tensor_proxy(self, 0)
        return self._saved_self_obj

    def saved_tensors(self):
        if self._saved_self_obj is not None:
            return (self._saved_self_obj.materialize(),)
        return (self._saved_proxy_materialize_py(0),)

    def release_saved_tensors(self):
        if self._saved_self_obj is not None:
            self._saved_self_obj.release()
        else:
            self._saved_self_released = True

    def __getattr__(self, name):
        if name == "_raw_saved_self":
            return self._raw_saved_self_proxy()
        if name == "_saved_self":
            if self._saved_self_obj is not None:
                return self._saved_self_obj.materialize()
            return self._saved_proxy_materialize_py(0)
        if name == "_raw_saved_tensors":
            return (self._raw_saved_self_proxy(),)
        if name == "_saved_tensors":
            if self._saved_self_obj is not None:
                return (self._saved_self_obj.materialize(),)
            return (self._saved_proxy_materialize_py(0),)
        return _CyAutogradNode.__getattr__(self, name)

    def backward(self, grad):
        if self._saved_self_obj is not None:
            self_ = self._saved_self_obj.materialize()
        else:
            self_ = self._saved_proxy_materialize_py(0)
        _ensure_npu_autograd_refs()
        if not getattr(self_, "requires_grad", False):
            return (None,)
        return (_mark_npu_owned_backward_grad(_cy_fast_npu_silu_backward(grad, self_)),)


cdef class _NpuFlashSdpaBackward(_CyAutogradNode):
    cdef public object _query
    cdef public object _key
    cdef public object _value
    cdef public object _output
    cdef public object _softmax_max
    cdef public object _softmax_sum
    cdef public double _scale_factor
    cdef int64_t _saved_query_version
    cdef int64_t _saved_key_version
    cdef int64_t _saved_value_version
    cdef int64_t _saved_output_version
    cdef int64_t _saved_softmax_max_version
    cdef int64_t _saved_softmax_sum_version
    cdef bint _saved_query_released
    cdef bint _saved_key_released
    cdef bint _saved_value_released
    cdef bint _saved_output_released
    cdef bint _saved_softmax_max_released
    cdef bint _saved_softmax_sum_released
    cdef object _saved_query_obj
    cdef object _saved_key_obj
    cdef object _saved_value_obj
    cdef object _saved_output_obj
    cdef object _saved_softmax_max_obj
    cdef object _saved_softmax_sum_obj

    cdef void _init_fast(self, object query, object key, object value,
                         object output, object softmax_max, object softmax_sum,
                         double scale_factor):
        cdef object saved_query
        cdef object saved_key
        cdef object saved_value
        cdef object saved_output
        cdef object saved_softmax_max
        cdef object saved_softmax_sum
        self.inputs = (query, key, value)
        self._saved_tensors_list = _npu_empty_saved_tensors
        self._saved_fields = _npu_empty_saved_fields
        self._hooks = None
        self._prehooks = None
        self._metadata = None
        self._name = "_NpuFlashSdpaBackward"
        self._anomaly_trace = None
        self._anomaly_parent = None
        self._query = query
        self._key = key
        self._value = value
        self._output = output
        self._softmax_max = softmax_max
        self._softmax_sum = softmax_sum
        self._scale_factor = scale_factor
        self._next_functions_cache = (
            _npu_edge_for(<TensorImpl>query),
            _npu_edge_for(<TensorImpl>key),
            _npu_edge_for(<TensorImpl>value),
        )
        self._saved_query_released = False
        self._saved_key_released = False
        self._saved_value_released = False
        self._saved_output_released = False
        self._saved_softmax_max_released = False
        self._saved_softmax_sum_released = False
        if _npu_has_saved_hooks():
            saved_query = SavedTensor(query)
            saved_key = SavedTensor(key)
            saved_value = SavedTensor(value)
            saved_output = SavedTensor(output)
            saved_softmax_max = SavedTensor(softmax_max)
            saved_softmax_sum = SavedTensor(softmax_sum)
            self._saved_query_obj = saved_query
            self._saved_key_obj = saved_key
            self._saved_value_obj = saved_value
            self._saved_output_obj = saved_output
            self._saved_softmax_max_obj = saved_softmax_max
            self._saved_softmax_sum_obj = saved_softmax_sum
            self._saved_query_version = 0
            self._saved_key_version = 0
            self._saved_value_version = 0
            self._saved_output_version = 0
            self._saved_softmax_max_version = 0
            self._saved_softmax_sum_version = 0
        else:
            self._saved_query_obj = None
            self._saved_key_obj = None
            self._saved_value_obj = None
            self._saved_output_obj = None
            self._saved_softmax_max_obj = None
            self._saved_softmax_sum_obj = None
            self._saved_query_version = (<TensorImpl>query)._version_value
            self._saved_key_version = (<TensorImpl>key)._version_value
            self._saved_value_version = (<TensorImpl>value)._version_value
            self._saved_output_version = (<TensorImpl>output)._version_value
            self._saved_softmax_max_version = (<TensorImpl>softmax_max)._version_value
            self._saved_softmax_sum_version = (<TensorImpl>softmax_sum)._version_value

    def __init__(self, query, key, value, output, softmax_max, softmax_sum, scale_factor):
        self._init_fast(query, key, value, output, softmax_max, softmax_sum, scale_factor)

    cpdef tuple _engine_next_functions(self):
        return self._next_functions_cache

    @property
    def next_functions(self):
        cached = _npu_materialize_leaf_edges(self.inputs, self._next_functions_cache)
        self._next_functions_cache = cached
        return cached

    cdef object _saved_at(self, int slot):
        if slot == 0:
            if self._saved_query_obj is not None:
                return self._saved_query_obj.materialize()
            return _npu_materialize_tensor_version(self._query, self._saved_query_version, self._saved_query_released)
        if slot == 1:
            if self._saved_key_obj is not None:
                return self._saved_key_obj.materialize()
            return _npu_materialize_tensor_version(self._key, self._saved_key_version, self._saved_key_released)
        if slot == 2:
            if self._saved_value_obj is not None:
                return self._saved_value_obj.materialize()
            return _npu_materialize_tensor_version(self._value, self._saved_value_version, self._saved_value_released)
        if slot == 3:
            if self._saved_output_obj is not None:
                return self._saved_output_obj.materialize()
            return _npu_materialize_tensor_version(self._output, self._saved_output_version, self._saved_output_released)
        if slot == 4:
            if self._saved_softmax_max_obj is not None:
                return self._saved_softmax_max_obj.materialize()
            return _npu_materialize_tensor_version(self._softmax_max, self._saved_softmax_max_version, self._saved_softmax_max_released)
        if self._saved_softmax_sum_obj is not None:
            return self._saved_softmax_sum_obj.materialize()
        return _npu_materialize_tensor_version(self._softmax_sum, self._saved_softmax_sum_version, self._saved_softmax_sum_released)

    def saved_tensors(self):
        return (self._saved_at(0), self._saved_at(1), self._saved_at(2),
                self._saved_at(3), self._saved_at(4), self._saved_at(5))

    def release_saved_tensors(self):
        if self._saved_query_obj is not None:
            self._saved_query_obj.release()
            self._saved_key_obj.release()
            self._saved_value_obj.release()
            self._saved_output_obj.release()
            self._saved_softmax_max_obj.release()
            self._saved_softmax_sum_obj.release()
        else:
            self._saved_query_released = True
            self._saved_key_released = True
            self._saved_value_released = True
            self._saved_output_released = True
            self._saved_softmax_max_released = True
            self._saved_softmax_sum_released = True

    def backward(self, grad_out):
        cdef object q = self._saved_at(0)
        cdef object k = self._saved_at(1)
        cdef object v = self._saved_at(2)
        cdef object out = self._saved_at(3)
        cdef object softmax_max = self._saved_at(4)
        cdef object softmax_sum = self._saved_at(5)
        cdef object grad_q
        cdef object grad_k
        cdef object grad_v
        cdef object packed_qkv_grad = None
        cdef object q_meta
        cdef object grad_shape
        cdef object grad_stride
        if not grad_out.is_contiguous():
            grad_shape = tuple(grad_out.shape)
            grad_stride = tuple(grad_out.stride())
            if (
                grad_shape != tuple(q.shape)
                or len(grad_stride) != 4
                or (grad_stride[len(grad_stride) - 1] != 1 and any(stride != 0 for stride in grad_stride))
            ):
                grad_out = grad_out.contiguous()
        q_meta = getattr(q, "_candle_packed_qkv_projection_meta", None)
        if (
            q_meta is not None
            and getattr(k, "_candle_packed_qkv_projection_meta", None) == q_meta[:4] + (1,)
            and getattr(v, "_candle_packed_qkv_projection_meta", None) == q_meta[:4] + (2,)
            and q_meta[4] == 0
        ):
            try:
                grad_q, grad_k, grad_v, packed_qkv_grad = _cy_fast_sdpa_flash_attention_backward_packed_qkv(
                    grad_out, q, k, v, out, softmax_max, softmax_sum,
                    self._scale_factor, q_meta[0], q_meta[1], q_meta[2], q_meta[3]
                )
            except (AttributeError, TypeError, ValueError):
                packed_qkv_grad = None
        if packed_qkv_grad is None:
            grad_q, grad_k, grad_v = _cy_fast_sdpa_flash_attention_backward(
                grad_out, q, k, v, out, softmax_max, softmax_sum, self._scale_factor
            )
        for grad in (grad_q, grad_k, grad_v):
            try:
                grad._candle_npu_owned_backward_grad = True
                if packed_qkv_grad is not None:
                    grad._candle_packed_qkv_backward_grad = packed_qkv_grad
            except (AttributeError, TypeError):
                pass
        return (
            _mark_npu_owned_backward_grad(grad_q),
            _mark_npu_owned_backward_grad(grad_k),
            _mark_npu_owned_backward_grad(grad_v),
        )


cdef inline object _attach_npu_flash_sdpa_grad(object out, object q, object k, object v,
                                               object softmax_max, object softmax_sum,
                                               double scale_factor):
    cdef _NpuFlashSdpaBackward grad_fn = _NpuFlashSdpaBackward.__new__(_NpuFlashSdpaBackward)
    grad_fn._init_fast(q, k, v, out, softmax_max, softmax_sum, scale_factor)
    (<TensorImpl>out).grad_fn = grad_fn
    (<TensorImpl>out).requires_grad = True
    return out


def npu_flash_sdpa(query, key, value, scale_factor):
    """Cython NPU FlashAttention SDPA forward with Cython autograd node."""
    cdef object out
    cdef object softmax_max
    cdef object softmax_sum
    if not isinstance(query, TensorImpl) or not isinstance(key, TensorImpl) or not isinstance(value, TensorImpl):
        return None
    if (<TensorImpl>query)._device_type != 1 or (<TensorImpl>key)._device_type != 1 or (<TensorImpl>value)._device_type != 1:
        return None
    try:
        out, softmax_max, softmax_sum = _cy_fast_sdpa_flash_attention(query, key, value, float(scale_factor))
    except (TypeError, ValueError):
        return None
    if _grad_enabled_fast() and (
        (<TensorImpl>query).requires_grad or (<TensorImpl>key).requires_grad or (<TensorImpl>value).requires_grad
    ):
        return _attach_npu_flash_sdpa_grad(out, query, key, value, softmax_max, softmax_sum, float(scale_factor))
    return out


cdef class _NpuSplitPackedQkvProjectionBackward(_CyAutogradNode):
    cdef public object _packed
    cdef public object _input_shape
    cdef public int64_t _embed
    cdef public int64_t _heads
    cdef public bint _batch_first

    cdef void _init_fast(self, object packed, object input_shape, int64_t embed, int64_t heads, bint batch_first):
        self.inputs = (packed,)
        self._saved_tensors_list = _npu_empty_saved_tensors
        self._saved_fields = _npu_empty_saved_fields
        self._hooks = None
        self._prehooks = None
        self._metadata = None
        self._name = "SplitPackedQkvProjectionBackward"
        self._anomaly_trace = None
        self._anomaly_parent = None
        self._packed = packed
        self._input_shape = input_shape
        self._embed = embed
        self._heads = heads
        self._batch_first = batch_first
        self._next_functions_cache = (_npu_edge_for(<TensorImpl>packed),)
        self._candle_multi_output_backward_count = 3

    def __init__(self, packed, input_shape, embed, heads, batch_first):
        self._init_fast(packed, input_shape, embed, heads, batch_first)

    cpdef tuple _engine_next_functions(self):
        return self._next_functions_cache

    @property
    def next_functions(self):
        cached = _npu_materialize_leaf_edges(self.inputs, self._next_functions_cache)
        self._next_functions_cache = cached
        return cached

    def backward(self, grads):
        cdef object grad_q
        cdef object grad_k
        cdef object grad_v
        cdef object packed_grad
        cdef object ref
        cdef object raw_shape
        cdef object stack
        cdef object zeros
        cdef int64_t head_dim
        cdef int64_t bsz
        cdef int64_t tgt_len
        if not isinstance(grads, (tuple, list)):
            return (None,)
        grad_q = grads[0] if len(grads) > 0 else None
        grad_k = grads[1] if len(grads) > 1 else None
        grad_v = grads[2] if len(grads) > 2 else None
        if grad_q is not None and grad_k is not None and grad_v is not None:
            packed_grad = getattr(grad_q, "_candle_packed_qkv_backward_grad", None)
            if (
                packed_grad is not None
                and getattr(grad_k, "_candle_packed_qkv_backward_grad", None) is packed_grad
                and getattr(grad_v, "_candle_packed_qkv_backward_grad", None) is packed_grad
            ):
                return (_mark_npu_owned_backward_grad(packed_grad),)
            packed_grad = _cy_fast_packed_qkv_projection_backward(
                grad_q, grad_k, grad_v, self._input_shape, self._embed, self._heads, self._batch_first)
            return (_mark_npu_owned_backward_grad(packed_grad),)

        ref = grad_q if grad_q is not None else grad_k if grad_k is not None else grad_v
        if ref is None:
            return (None,)
        from candle._functional import stack, zeros
        head_dim = self._embed // self._heads
        if self._batch_first:
            bsz = self._input_shape[0]
            tgt_len = self._input_shape[1]
            raw_shape = (bsz, tgt_len, self._embed)
            if grad_q is None:
                grad_q = zeros(raw_shape, dtype=ref.dtype, device=ref.device)
            else:
                grad_q = grad_q.transpose(1, 2).reshape(bsz, tgt_len, self._embed)
            if grad_k is None:
                grad_k = zeros(raw_shape, dtype=ref.dtype, device=ref.device)
            else:
                grad_k = grad_k.transpose(1, 2).reshape(bsz, tgt_len, self._embed)
            if grad_v is None:
                grad_v = zeros(raw_shape, dtype=ref.dtype, device=ref.device)
            else:
                grad_v = grad_v.transpose(1, 2).reshape(bsz, tgt_len, self._embed)
        else:
            tgt_len = self._input_shape[0]
            bsz = self._input_shape[1]
            raw_shape = (tgt_len, bsz, self._embed)
            if grad_q is None:
                grad_q = zeros(raw_shape, dtype=ref.dtype, device=ref.device)
            else:
                grad_q = grad_q.transpose(1, 2).transpose(0, 1).reshape(tgt_len, bsz, self._embed)
            if grad_k is None:
                grad_k = zeros(raw_shape, dtype=ref.dtype, device=ref.device)
            else:
                grad_k = grad_k.transpose(1, 2).transpose(0, 1).reshape(tgt_len, bsz, self._embed)
            if grad_v is None:
                grad_v = zeros(raw_shape, dtype=ref.dtype, device=ref.device)
            else:
                grad_v = grad_v.transpose(1, 2).transpose(0, 1).reshape(tgt_len, bsz, self._embed)
        packed_grad = stack((grad_q, grad_k, grad_v), dim=-2).reshape(self._input_shape)
        return (_mark_npu_owned_backward_grad(packed_grad),)


cdef inline object _attach_npu_packed_qkv_projection_grad(object outputs, object packed,
                                                          object input_shape, int64_t embed,
                                                          int64_t heads, bint batch_first):
    cdef _NpuSplitPackedQkvProjectionBackward grad_fn = _NpuSplitPackedQkvProjectionBackward.__new__(_NpuSplitPackedQkvProjectionBackward)
    cdef object q = outputs[0]
    cdef object k = outputs[1]
    cdef object v = outputs[2]
    cdef object meta = (input_shape, embed, heads, batch_first)
    grad_fn._init_fast(packed, input_shape, embed, heads, batch_first)
    (<TensorImpl>q).grad_fn = grad_fn
    (<TensorImpl>k).grad_fn = grad_fn
    (<TensorImpl>v).grad_fn = grad_fn
    (<TensorImpl>q).requires_grad = True
    (<TensorImpl>k).requires_grad = True
    (<TensorImpl>v).requires_grad = True
    (<TensorImpl>q)._output_nr = 0
    (<TensorImpl>k)._output_nr = 1
    (<TensorImpl>v)._output_nr = 2
    try:
        q._candle_packed_qkv_projection_meta = meta + (0,)
        k._candle_packed_qkv_projection_meta = meta + (1,)
        v._candle_packed_qkv_projection_meta = meta + (2,)
    except (AttributeError, TypeError):
        pass
    return outputs


def split_packed_qkv_projection(packed, embed_dim, num_heads, batch_first=False):
    """Cython NPU fast path for packed self-attention Q/K/V projection split."""
    cdef object outputs
    cdef object input_shape
    cdef int64_t embed = int(embed_dim)
    cdef int64_t heads = int(num_heads)
    cdef bint is_batch_first = bool(batch_first)
    if not isinstance(packed, TensorImpl) or (<TensorImpl>packed)._device_type != 1:
        return None
    try:
        outputs = _cy_fast_packed_qkv_projection_forward(packed, embed, heads, is_batch_first)
    except (RuntimeError, TypeError, ValueError):
        return None
    if outputs is None:
        return None
    input_shape = (<TensorImpl>packed)._shape_tuple
    if _grad_enabled_fast() and (<TensorImpl>packed).requires_grad:
        return _attach_npu_packed_qkv_projection_grad(outputs, packed, input_shape, embed, heads, is_batch_first)
    return outputs


cdef inline object _attach_npu_addmm_grad(object result, object input, object mat1, object mat2, object beta, object alpha):
    cdef TensorImpl out = <TensorImpl>result
    cdef _NpuAddmmBackward grad_fn = _NpuAddmmBackward.__new__(_NpuAddmmBackward)
    grad_fn._init_fast(input, mat1, mat2)
    out.grad_fn = grad_fn
    out.requires_grad = True
    return result


cdef inline object _attach_npu_linear_grad(object result, object input, object weight, object bias):
    cdef TensorImpl out = <TensorImpl>result
    cdef _NpuLinearBackward grad_fn = _NpuLinearBackward.__new__(_NpuLinearBackward)
    grad_fn._init_fast(input, weight, bias)
    out.grad_fn = grad_fn
    out.requires_grad = True
    return result


cdef inline object _attach_npu_add_grad(object result, object a, object b):
    cdef TensorImpl out = <TensorImpl>result
    cdef _NpuAddBackward grad_fn = _NpuAddBackward.__new__(_NpuAddBackward)
    grad_fn._init_fast(a, b)
    out.grad_fn = grad_fn
    out.requires_grad = True
    return result


cdef inline object _attach_npu_mul_grad(object result, object a, object b):
    cdef TensorImpl out = <TensorImpl>result
    cdef _NpuMulBackward grad_fn = _NpuMulBackward.__new__(_NpuMulBackward)
    grad_fn._init_fast(a, b)
    out.grad_fn = grad_fn
    out.requires_grad = True
    return result


cdef inline object _attach_npu_layer_norm_grad(object result, object input, object weight, object bias, object normalized_shape, object eps):
    cdef TensorImpl out = <TensorImpl>result
    cdef _NpuLayerNormBackward grad_fn = _NpuLayerNormBackward.__new__(_NpuLayerNormBackward)
    grad_fn._init_fast(input, weight, bias, normalized_shape, eps, getattr(result, "_backward_data", None))
    out.grad_fn = grad_fn
    out.requires_grad = True
    return result


cdef inline object _attach_npu_gelu_grad(object result, object a):
    cdef TensorImpl out = <TensorImpl>result
    cdef _NpuGeluBackward grad_fn = _NpuGeluBackward.__new__(_NpuGeluBackward)
    grad_fn._init_fast(a)
    out.grad_fn = grad_fn
    out.requires_grad = True
    return result


cdef inline object _attach_npu_silu_grad(object result, object a):
    cdef TensorImpl out = <TensorImpl>result
    cdef _NpuSiluBackward grad_fn = _NpuSiluBackward.__new__(_NpuSiluBackward)
    grad_fn._init_fast(a)
    out.grad_fn = grad_fn
    out.requires_grad = True
    return result


def _has_torch_function(args, kwargs):
    """Fast check: do any tensor args have __torch_function__ overrides?"""
    cdef object val

    _ensure_base()

    for val in args:
        if _check_value(val):
            return True

    if kwargs:
        for val in kwargs.values():
            if _check_value(val):
                return True

    return False


def _handle_torch_function(func, args, kwargs):
    """Dispatch to __torch_function__ if any arg is an overriding tensor subclass."""
    cdef object types
    cdef object val
    cdef object sorted_types
    cdef object cls
    cdef object result

    if not _has_torch_function(args, kwargs):
        return NotImplemented

    types = set()

    for val in args:
        _collect_types(val, types)

    if kwargs:
        for val in kwargs.values():
            _collect_types(val, types)

    sorted_types = sorted(types, key=lambda c: len(c.__mro__), reverse=True)
    for cls in sorted_types:
        result = cls.__torch_function__(func, types, args, kwargs or {})
        if result is not NotImplemented:
            return result

    return NotImplemented


def add(a=None, b=None, *, alpha=1, out=None):
    """Fast add: skip __torch_function__ when both args are base Tensor."""
    cdef object r
    cdef int npu_state

    # Front-load the exact-base NPU hot path before loading Python fallback
    # callables. This is the common L0 eager path and avoids several Python
    # guard/import checks before reaching _npu_add_fn.
    if a is not None and b is not None and alpha == 1 and out is None:
        if _exact_base_npu_pair(a, b):
            npu_state = _exact_npu_binary_hot_state(a, b)
            if npu_state == 1 and not _npu_profiler_active_flag:
                return _cy_fast_npu_add_exact(<TensorImpl>a, <TensorImpl>b)
            if npu_state == 2:
                return _attach_npu_add_grad(_cy_fast_npu_add_exact(<TensorImpl>a, <TensorImpl>b), a, b)

    _ensure_originals()

    if a is None or b is None:
        # Delegate to original for proper fallback behavior.
        return _py_add_fn(a, b, alpha=alpha, out=out) if a is not None else _py_add_fn()

    if _is_base_tensor(a) and (_is_base_tensor(b) or not hasattr(b, "__torch_function__")):
        if alpha != 1:
            b = _dispatch_fn("mul", None, b, alpha)
        elif _is_npu_tensor_pair(a, b):
            npu_state = _npu_binary_hot_state(a, b)
            if npu_state == 1 and not _profiler_active():
                return _cy_fast_npu_add(a, b)
            if out is None and npu_state == 2:
                return _attach_npu_add_grad(_cy_fast_npu_add(a, b), a, b)
        return _dispatch_fn("add", None, a, b)

    r = _handle_torch_function(_py_add_fn, (a, b), {"alpha": alpha, "out": out})
    if r is not NotImplemented:
        return r

    if alpha != 1:
        b = _dispatch_fn("mul", None, b, alpha)
    return _dispatch_fn("add", None, a, b)


def mul(a, b, *, out=None):
    """Fast mul: skip __torch_function__ when both args are base Tensor."""
    cdef object r
    cdef object result
    cdef object kwargs
    cdef int npu_state

    # Front-load the exact-base NPU hot path before loading Python fallback
    # callables. The out= path keeps the existing copy_ handling below.
    if out is None:
        if _exact_base_npu_pair(a, b):
            npu_state = _exact_npu_binary_hot_state(a, b)
            if npu_state == 1 and not _npu_profiler_active_flag:
                return _cy_fast_npu_mul_exact(<TensorImpl>a, <TensorImpl>b)
            if npu_state == 2:
                return _attach_npu_mul_grad(_cy_fast_npu_mul_exact(<TensorImpl>a, <TensorImpl>b), a, b)

    _ensure_originals()

    if _is_base_tensor(a) and (_is_base_tensor(b) or not hasattr(b, "__torch_function__")):
        if _is_npu_tensor_pair(a, b):
            npu_state = _npu_binary_hot_state(a, b)
            if npu_state == 1 and not _profiler_active():
                result = _cy_fast_npu_mul(a, b)
                if out is not None:
                    out.copy_(result)
                    return out
                return result
            if out is None and npu_state == 2:
                return _attach_npu_mul_grad(_cy_fast_npu_mul(a, b), a, b)
        result = _dispatch_fn("mul", None, a, b)
        if out is not None:
            out.copy_(result)
            return out
        return result

    kwargs = {}
    if out is not None:
        kwargs["out"] = out
    r = _handle_torch_function(_py_mul_fn, (a, b), kwargs)
    if r is not NotImplemented:
        return r

    result = _dispatch_fn("mul", None, a, b)
    if out is not None:
        out.copy_(result)
        return out
    return result


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Fast NPU layer_norm wrapper with Python-dispatch fallback."""
    cdef int state
    cdef object result
    state = _exact_npu_layer_norm_hot_state(input, weight, bias, normalized_shape)
    if state != 0:
        try:
            result = _cy_fast_npu_layer_norm(input, normalized_shape, weight, bias, eps)
        except ValueError:
            result = None
        if result is not None:
            if state == 2:
                return _attach_npu_layer_norm_grad(result, input, weight, bias, normalized_shape, eps)
            return result

    from candle.nn import functional as _nn_functional
    return _nn_functional._py_layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=eps)


def gelu(a, approximate='none'):
    """Fast nn.functional.gelu NPU eager route for the native GELU variant."""
    cdef int npu_state
    if approximate == 'none':
        if _exact_base_npu_unary(a):
            npu_state = _exact_npu_unary_hot_state(a)
            if npu_state == 1 and not _npu_profiler_active_flag and not _npu_autocast_active_flag:
                return _cy_fast_npu_gelu_exact(<TensorImpl>a)
            if npu_state == 2:
                return _attach_npu_gelu_grad(_cy_fast_npu_gelu_exact(<TensorImpl>a), a)
    from candle.nn import functional as _nn_functional
    return _nn_functional._py_gelu(a, approximate=approximate)


def silu(a, inplace=False):
    """Fast nn.functional.silu NPU inference route."""
    cdef int npu_state
    if not inplace:
        if _exact_base_npu_unary(a):
            npu_state = _exact_npu_unary_hot_state(a)
            if npu_state == 1 and not _npu_profiler_active_flag and not _npu_autocast_active_flag:
                return _cy_fast_npu_silu_exact(<TensorImpl>a)
            if npu_state == 2:
                return _attach_npu_silu_grad(_cy_fast_npu_silu_exact(<TensorImpl>a), a)
    from candle.nn import functional as _nn_functional
    return _nn_functional._py_silu(a, inplace=inplace)


def sub(a, b, *, alpha=1):
    """Fast sub: skip __torch_function__ when both args are base Tensor."""
    cdef object r

    _ensure_originals()

    if _is_base_tensor(a) and (_is_base_tensor(b) or not hasattr(b, "__torch_function__")):
        if alpha != 1:
            b = _dispatch_fn("mul", None, b, alpha)
        elif _is_npu_tensor_pair(a, b):
            _ensure_npu_refs()
            if _npu_fast_ok(a, b):
                return _npu_sub_fn(a, b)
        return _dispatch_fn("sub", None, a, b)

    r = _handle_torch_function(_py_sub_fn, (a, b), {"alpha": alpha})
    if r is not NotImplemented:
        return r

    if alpha != 1:
        b = _dispatch_fn("mul", None, b, alpha)
    return _dispatch_fn("sub", None, a, b)


def div(a, b, *, rounding_mode=None):
    """Fast div: skip __torch_function__ when both args are base Tensor."""
    cdef object r

    _ensure_originals()

    if _is_base_tensor(a) and (_is_base_tensor(b) or not hasattr(b, "__torch_function__")):
        if rounding_mode == "trunc":
            return _dispatch_fn("trunc_divide", None, a, b)
        if rounding_mode == "floor":
            return _dispatch_fn("floor_divide", None, a, b)
        if _is_npu_tensor_pair(a, b):
            _ensure_npu_refs()
            if _npu_fast_ok(a, b):
                return _npu_div_fn(a, b)
        return _dispatch_fn("true_divide", None, a, b)

    r = _handle_torch_function(_py_div_fn, (a, b), {"rounding_mode": rounding_mode})
    if r is not NotImplemented:
        return r

    if rounding_mode == "trunc":
        return _dispatch_fn("trunc_divide", None, a, b)
    if rounding_mode == "floor":
        return _dispatch_fn("floor_divide", None, a, b)
    return _dispatch_fn("true_divide", None, a, b)


def matmul(a, b, *, out=None):
    """Fast matmul: skip __torch_function__ when both args are base Tensor."""
    cdef object r
    cdef object result
    cdef object kwargs
    cdef bint requires_grad

    # Front-load safe 2D exact NPU inference matmul before loading Python
    # fallback callables.  Training/autograd cases stay on dispatch until a
    # dedicated matmul autograd attach path is implemented and tested.
    if _exact_base_npu_pair(a, b):
        requires_grad = (<TensorImpl>a).requires_grad or (<TensorImpl>b).requires_grad
        if (
            out is None
            and not _functionalize_active_flag
            and not _pipeline_active_flag
            and not _npu_profiler_active_flag
            and not _npu_autocast_active_flag
            and (not _grad_enabled_fast() or not requires_grad)
            and (<TensorImpl>a)._ndim == 2
            and (<TensorImpl>b)._ndim == 2
            and (<TensorImpl>a)._dtype_code == (<TensorImpl>b)._dtype_code
            and (<TensorImpl>a)._device_index == (<TensorImpl>b)._device_index
            and (<TensorImpl>a)._c_shape[1] == (<TensorImpl>b)._c_shape[0]
        ):
            return _cy_fast_npu_matmul_exact(<TensorImpl>a, <TensorImpl>b)

    _ensure_originals()

    if _is_base_tensor(a) and _is_base_tensor(b):
        result = _dispatch_fn("matmul", None, a, b)
        if out is not None:
            out.copy_(result)
            return out
        return result

    kwargs = {}
    if out is not None:
        kwargs["out"] = out
    r = _handle_torch_function(_py_matmul_fn, (a, b), kwargs)
    if r is not NotImplemented:
        return r

    result = _dispatch_fn("matmul", None, a, b)
    if out is not None:
        out.copy_(result)
        return out
    return result


def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    """Fast addmm: direct NPU kernel plus autograd attach for safe base tensors."""
    cdef int state
    cdef object result

    state = _exact_npu_addmm_hot_state(input, mat1, mat2, beta, alpha)
    if state != 0:
        try:
            result = _cy_fast_npu_addmm(input, mat1, mat2, beta, alpha)
        except ValueError:
            result = None
        if result is not None:
            if state == 2:
                return _attach_npu_addmm_grad(result, input, mat1, mat2, beta, alpha)
            return result

    _ensure_dispatch()
    return _dispatch_fn("addmm", getattr(getattr(input, "device", None), "type", None), input, mat1, mat2, beta=beta, alpha=alpha)


def linear(input, weight, bias=None):
    """Fast linear: fuse safe NPU weight.t() + addmm glue in Cython."""
    cdef int state
    cdef object weight_t
    cdef object mat1
    cdef object result
    cdef object out_shape
    cdef TensorImpl inp
    cdef TensorImpl out

    state = _exact_npu_linear_hot_state(input, weight, bias)
    if state != 0:
        inp = <TensorImpl>input
        weight_t = (<TensorImpl>weight).cy_transpose(0, 1)
        if inp._ndim == 2:
            mat1 = input
            out_shape = None
        else:
            mat1 = inp.cy_view((inp._c_numel // inp._c_shape[inp._ndim - 1], inp._c_shape[inp._ndim - 1]))
            out_shape = inp._shape_tuple[:inp._ndim - 1] + ((<TensorImpl>weight)._c_shape[0],)
        try:
            result = _cy_fast_npu_addmm(bias, mat1, weight_t, 1, 1)
        except ValueError:
            result = None
        if result is not None:
            if out_shape is not None:
                result = (<TensorImpl>result).cy_view(out_shape)
            if state == 2:
                return _attach_npu_linear_grad(result, input, weight, bias)
            return result

    from candle.nn import functional as _nn_functional
    return _nn_functional._py_linear(input, weight, bias)


def relu(a):
    """Fast relu: skip __torch_function__ for base Tensor."""
    cdef object r

    _ensure_originals()

    if _is_base_tensor(a):
        return _dispatch_fn("relu", None, a)

    r = _handle_torch_function(_py_relu_fn, (a,), {})
    if r is not NotImplemented:
        return r

    return _dispatch_fn("relu", None, a)


def transpose(*args, **kwargs):
    cdef object a
    cdef object dim0
    cdef object dim1
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not args:
        return _dispatch_fn("transpose", None, *args, **kwargs)

    a = args[0]
    if len(args) >= 3:
        dim0 = args[1]
        dim1 = args[2]
    else:
        dim0 = kwargs.get("dim0")
        dim1 = kwargs.get("dim1")

    if not _is_base_tensor(a):
        return _dispatch_fn("transpose", None, *args, **kwargs)
    if type(dim0) is not int or type(dim1) is not int:
        return _dispatch_fn("transpose", None, *args, **kwargs)

    v = a.cy_transpose(dim0, dim1)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "transpose",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def reshape(*args, **kwargs):
    cdef object a
    cdef object shape
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v
    cdef object shape_list
    cdef object infer_idx
    cdef object known_size
    cdef object size
    cdef object new_size
    cdef object idx
    cdef object dim

    _ensure_base()
    _ensure_dispatch()

    if not args:
        return _dispatch_fn("reshape", None, *args, **kwargs)

    a = args[0]
    if len(args) >= 2:
        shape = args[1]
    else:
        shape = kwargs.get("shape")

    if not _is_base_tensor(a):
        return _dispatch_fn("reshape", None, *args, **kwargs)

    if isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, (tuple, list)):
        shape = tuple(shape)
    else:
        return _dispatch_fn("reshape", None, *args, **kwargs)

    if not a.is_contiguous():
        return _dispatch_fn("reshape", None, *args, **kwargs)

    size = 1
    for dim in a.shape:
        size *= dim

    infer_idx = None
    known_size = 1
    shape_list = list(shape)
    for idx, dim in enumerate(shape_list):
        if dim == -1:
            if infer_idx is not None:
                raise RuntimeError("only one dimension can be inferred")
            infer_idx = idx
            continue
        known_size *= dim

    if infer_idx is not None:
        if known_size == 0 or size % known_size != 0:
            raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {size}")
        shape_list[infer_idx] = size // known_size

    shape = tuple(shape_list)
    new_size = 1
    for dim in shape:
        new_size *= dim
    if size != new_size:
        raise ValueError("reshape size mismatch")

    v = a.cy_view(shape)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "reshape",
        "shape": tuple(shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    from candle.autograd import forward_ad
    level = forward_ad._current_level()
    if level >= 0:
        tangent = forward_ad.get_tangent(a, level)
        if tangent is not None:
            v._fw_set(level, tangent.reshape(shape))
    return v


def flatten(a, start_dim=0, end_dim=-1):
    cdef object ndim
    cdef object start
    cdef object end
    cdef object flattened
    cdef object d
    cdef object new_shape
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("flatten", a.device.type, a, start_dim, end_dim)

    ndim = len(a.shape)
    if ndim == 0:
        return reshape(a, (1,))

    start = start_dim if start_dim >= 0 else start_dim + ndim
    end = end_dim if end_dim >= 0 else end_dim + ndim
    if start < 0 or start >= ndim:
        raise IndexError("Dimension out of range")
    if end < 0 or end >= ndim:
        raise IndexError("Dimension out of range")
    if start > end:
        raise RuntimeError("flatten() has invalid args: start_dim cannot come after end_dim")

    flattened = 1
    for d in a.shape[start:end + 1]:
        flattened *= d
    new_shape = a.shape[:start] + (flattened,) + a.shape[end + 1:]

    v = reshape(a, new_shape)
    source_view_meta = getattr(a, "_view_meta", None) or {}
    creation_mode = (getattr(v, "_view_meta", None) or {}).get("creation_mode", source_view_meta.get("creation_mode"))
    creation_kind = (getattr(v, "_view_meta", None) or {}).get("creation_kind", source_view_meta.get("creation_kind"))
    v._view_meta = {
        "op": "flatten",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def unflatten(a, dim, sizes):
    cdef object ndim
    cdef object d
    cdef object shape
    cdef object size_idx
    cdef object infer_idx
    cdef object known_size
    cdef object old_size
    cdef object new_size
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("unflatten", a.device.type, a, dim, sizes)

    if not isinstance(dim, int):
        return _dispatch_fn("unflatten", a.device.type, a, dim, sizes)

    if isinstance(sizes, list):
        sizes = tuple(sizes)
    elif not isinstance(sizes, tuple):
        return _dispatch_fn("unflatten", a.device.type, a, dim, sizes)

    if not a.is_contiguous():
        return _dispatch_fn("unflatten", a.device.type, a, dim, sizes)

    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    if d < 0 or d >= ndim:
        raise IndexError("Dimension out of range")

    old_size = a.shape[d]
    infer_idx = None
    known_size = 1
    for size_idx, new_size in enumerate(sizes):
        if new_size == -1:
            if infer_idx is not None:
                raise RuntimeError("only one dimension can be inferred")
            infer_idx = size_idx
            continue
        known_size *= new_size

    sizes = list(sizes)
    if infer_idx is not None:
        if known_size == 0 or old_size % known_size != 0:
            raise RuntimeError(f"unflatten: Provided sizes {tuple(sizes)} don't multiply up to the size of dim {d} ({old_size}) in the input tensor")
        sizes[infer_idx] = old_size // known_size

    new_size = 1
    for size_idx in sizes:
        new_size *= size_idx
    if new_size != old_size:
        raise RuntimeError(f"unflatten: Provided sizes {tuple(sizes)} don't multiply up to the size of dim {d} ({old_size}) in the input tensor")

    shape = a.shape[:d] + tuple(sizes) + a.shape[d + 1:]
    v = a.cy_view(shape)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "unflatten",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def squeeze(a, dim=None):
    cdef object shape
    cdef object stride
    cdef object pairs
    cdef object ndim
    cdef object targets
    cdef object item
    cdef object d
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v
    cdef object input_shape

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("squeeze", a.device.type, a, dim)

    # 1B-B: requires_grad path also takes the fast view path. After 1B-B's
    # dispatch-registration deletion, there is no autograd kernel to look up
    # for squeeze; the engine view-rebase block owns gradient flow via the
    # _view_func / _rev_view_func attached below.
    input_shape = tuple(a.shape)
    shape = list(a.shape)
    stride = list(a.stride)
    if dim is not None:
        if isinstance(dim, (list, tuple)):
            if dim:
                ndim = len(shape)
                targets = set()
                for item in dim:
                    d = item if item >= 0 else item + ndim
                    targets.add(d)
                pairs = [
                    (s, st)
                    for idx, (s, st) in enumerate(zip(shape, stride))
                    if idx not in targets or s != 1
                ]
                shape = [p[0] for p in pairs]
                stride = [p[1] for p in pairs]
        else:
            d = dim if dim >= 0 else dim + len(shape)
            if 0 <= d < len(shape) and shape[d] == 1:
                del shape[d]
                del stride[d]
    else:
        pairs = [(s, st) for s, st in zip(shape, stride) if s != 1]
        shape = [p[0] for p in pairs]
        stride = [p[1] for p in pairs]

    v = a.cy_as_strided(tuple(shape), tuple(stride), a.offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "squeeze",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    _attach_squeeze_view_funcs(v, dim, input_shape)
    return v


def _attach_squeeze_view_funcs(result, dim, input_shape):
    """Attach view_func/rev_view_func for squeeze so engine rebase owns grad."""
    def _squeeze_view_func(new_base, _dim=dim):
        if _dim is None:
            return new_base.squeeze()
        return new_base.squeeze(_dim)

    def _squeeze_rev_view_func(grad_view, _shape=input_shape):
        return grad_view.reshape(_shape)

    result._view_func = _squeeze_view_func
    result._rev_view_func = _squeeze_rev_view_func


def unsqueeze(a, dim):
    cdef object ndim
    cdef object d
    cdef object shape
    cdef object stride
    cdef object new_stride
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("unsqueeze", a.device.type, a, dim)
    if getattr(a, "requires_grad", False):
        return _dispatch_fn("unsqueeze", a.device.type, a, dim)

    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim + 1
    shape = list(a.shape)
    stride = list(a.stride)
    new_stride = stride[d] * shape[d] if d < ndim else 1
    shape.insert(d, 1)
    stride.insert(d, new_stride)

    v = a.cy_as_strided(tuple(shape), tuple(stride), a.offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "unsqueeze",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def permute(a, dims):
    cdef object ndim
    cdef object normalized
    cdef object d
    cdef object shape
    cdef object stride
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("permute", a.device.type, a, dims)
    if getattr(a, "requires_grad", False):
        return _dispatch_fn("permute", a.device.type, a, dims)

    ndim = len(a.shape)
    if isinstance(dims, list):
        dims = tuple(dims)
    elif not isinstance(dims, tuple):
        return _dispatch_fn("permute", a.device.type, a, dims)

    normalized = []
    for d in dims:
        d = d if d >= 0 else d + ndim
        normalized.append(d)

    shape = [a.shape[d] for d in normalized]
    stride = [a.stride[d] for d in normalized]
    v = a.cy_as_strided(tuple(shape), tuple(stride), a.offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "permute",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def slice(input, dim, start=0, end=9223372036854775807, step=1):
    cdef object ndim
    cdef object d
    cdef object dim_size
    cdef object start_idx
    cdef object end_idx
    cdef object step_idx
    cdef object length
    cdef object new_shape
    cdef object new_stride
    cdef object new_offset
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(input):
        return _dispatch_fn("slice", input.device.type, input, dim, start, end, step)

    ndim = len(input.shape)
    d = dim if dim >= 0 else dim + ndim
    if d < 0 or d >= ndim:
        raise IndexError("Dimension out of range")

    dim_size = input.shape[d]
    step_idx = int(step)
    if step_idx == 0:
        raise ValueError("slice step cannot be zero")

    start_idx, end_idx, step_idx = _builtins.slice(start, end, step_idx).indices(dim_size)
    length = len(range(start_idx, end_idx, step_idx))
    new_shape = list(input.shape)
    new_shape[d] = length
    new_stride = list(input.stride)
    new_stride[d] = new_stride[d] * step_idx
    new_offset = input.offset + start_idx * input.stride[d]
    v = input.cy_as_strided(tuple(new_shape), tuple(new_stride), new_offset)

    source_view_meta = getattr(input, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if input._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "slice",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def narrow(a, dim, start, length):
    cdef object ndim
    cdef object d
    cdef object new_shape
    cdef object new_offset
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("narrow", a.device.type, a, dim, start, length)
    if getattr(a, "requires_grad", False):
        return _dispatch_fn("narrow", a.device.type, a, dim, start, length)

    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    new_shape = list(a.shape)
    new_shape[d] = int(length)
    new_offset = a.offset + int(start) * a.stride[d]
    v = a.cy_as_strided(tuple(new_shape), tuple(a.stride), new_offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "narrow",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def select(a, dim, index):
    cdef object ndim
    cdef object d
    cdef object idx
    cdef object new_shape
    cdef object new_stride
    cdef object new_offset
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("select", a.device.type, a, dim, index)
    if getattr(a, "requires_grad", False):
        return _dispatch_fn("select", a.device.type, a, dim, index)

    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    idx = int(index)
    if idx < 0:
        idx += a.shape[d]
    new_shape = list(a.shape)
    del new_shape[d]
    new_stride = list(a.stride)
    new_offset = a.offset + idx * a.stride[d]
    del new_stride[d]
    v = a.cy_as_strided(tuple(new_shape), tuple(new_stride), new_offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "select",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def expand(a, sizes):
    cdef object sizes_tuple
    cdef object ndiff
    cdef object src_shape
    cdef object src_stride
    cdef object out_shape
    cdef object out_stride
    cdef object i
    cdef object sz
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("expand", a.device.type, a, sizes)
    if getattr(a, "requires_grad", False):
        return _dispatch_fn("expand", a.device.type, a, sizes)

    if isinstance(sizes, list):
        sizes_tuple = tuple(sizes)
    elif isinstance(sizes, tuple):
        sizes_tuple = sizes
    else:
        return _dispatch_fn("expand", a.device.type, a, sizes)

    ndiff = len(sizes_tuple) - len(a.shape)
    if ndiff < 0:
        raise RuntimeError("expand: number of sizes must be >= tensor dim")

    src_shape = (1,) * ndiff + a.shape
    src_stride = (0,) * ndiff + a.stride
    out_shape = []
    out_stride = []
    for i, sz in enumerate(sizes_tuple):
        if sz == -1:
            out_shape.append(src_shape[i])
            out_stride.append(src_stride[i])
        elif src_shape[i] == 1:
            out_shape.append(sz)
            out_stride.append(0)
        elif src_shape[i] == sz:
            out_shape.append(sz)
            out_stride.append(src_stride[i])
        else:
            raise RuntimeError(
                f"expand: size {sz} not compatible with dim size {src_shape[i]}"
            )

    v = a.cy_as_strided(tuple(out_shape), tuple(out_stride), a.offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "expand",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def movedim(a, source, destination):
    cdef object ndim
    cdef object source_tuple
    cdef object destination_tuple
    cdef object order
    cdef object dst_order
    cdef object dst_idx
    cdef object shape
    cdef object stride
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("movedim", a.device.type, a, source, destination)

    ndim = len(a.shape)
    if isinstance(source, int):
        source_tuple = (source,)
    elif isinstance(source, list):
        source_tuple = tuple(source)
    else:
        source_tuple = source

    if isinstance(destination, int):
        destination_tuple = (destination,)
    elif isinstance(destination, list):
        destination_tuple = tuple(destination)
    else:
        destination_tuple = destination

    if not isinstance(source_tuple, tuple) or not isinstance(destination_tuple, tuple):
        return _dispatch_fn("movedim", a.device.type, a, source, destination)

    source_tuple = tuple(s % ndim for s in source_tuple)
    destination_tuple = tuple(d % ndim for d in destination_tuple)

    order = [n for n in range(ndim) if n not in source_tuple]
    dst_order = sorted(range(len(destination_tuple)), key=lambda i: destination_tuple[i])
    for dst_idx in dst_order:
        order.insert(destination_tuple[dst_idx], source_tuple[dst_idx])

    shape = [a.shape[d] for d in order]
    stride = [a.stride[d] for d in order]
    v = a.cy_as_strided(tuple(shape), tuple(stride), a.offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "movedim",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def diagonal(a, offset=0, dim1=0, dim2=1):
    cdef object ndim
    cdef object d1
    cdef object d2
    cdef object shape
    cdef object stride
    cdef object size1
    cdef object size2
    cdef object diag_len
    cdef object base_offset
    cdef object out_shape
    cdef object out_stride
    cdef object i
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("diagonal", a.device.type, a, offset, dim1, dim2)

    ndim = len(a.shape)
    d1 = dim1 if dim1 >= 0 else dim1 + ndim
    d2 = dim2 if dim2 >= 0 else dim2 + ndim

    shape = list(a.shape)
    stride = list(a.stride)
    size1 = shape[d1]
    size2 = shape[d2]

    if offset >= 0:
        diag_len = max(0, min(size1, size2 - offset))
        base_offset = a.offset + offset * stride[d2]
    else:
        diag_len = max(0, min(size1 + offset, size2))
        base_offset = a.offset + (-offset) * stride[d1]

    out_shape = [shape[i] for i in range(ndim) if i not in (d1, d2)]
    out_stride = [stride[i] for i in range(ndim) if i not in (d1, d2)]
    out_shape.append(diag_len)
    out_stride.append(stride[d1] + stride[d2])

    v = a.cy_as_strided(tuple(out_shape), tuple(out_stride), base_offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "diagonal",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def view_as_real(a):
    cdef object out_dtype
    cdef object shape
    cdef object stride
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("view_as_real", a.device.type, a)

    if not a.is_complex():
        raise RuntimeError("view_as_real expects a complex tensor")
    if a.dtype.itemsize % 2 != 0:
        raise RuntimeError("view_as_real expects complex dtype with even itemsize")
    if a.dtype.name == "complex64":
        from candle import float32 as out_dtype
    elif a.dtype.name == "complex128":
        from candle import float64 as out_dtype
    elif a.dtype.name == "complex32":
        from candle import float16 as out_dtype
    else:
        raise RuntimeError("view_as_real expects a supported complex dtype")

    shape = tuple(a.shape) + (2,)
    stride = tuple(s * 2 for s in a.stride) + (1,)
    v = a.cy_as_strided(shape, stride, a.offset * 2)
    v._storage = a.storage()._reinterpret(out_dtype)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "view_as_real",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def view_as_complex(a):
    cdef object out_dtype
    cdef object ndim
    cdef object last_dim
    cdef object shape
    cdef object stride
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("view_as_complex", a.device.type, a)

    ndim = len(a.shape)
    if ndim == 0:
        raise RuntimeError("view_as_complex expects last dimension of size 2")
    if a.is_complex():
        raise RuntimeError("view_as_complex expects a non-complex tensor")
    last_dim = a.shape[ndim - 1]
    if last_dim != 2:
        raise RuntimeError("view_as_complex expects last dimension of size 2")
    if a.dtype.name == "float16":
        from candle import complex32 as out_dtype
    elif a.dtype.name == "float32":
        from candle import complex64 as out_dtype
    elif a.dtype.name == "float64":
        from candle import complex128 as out_dtype
    else:
        raise RuntimeError("view_as_complex is only supported for half, float and double tensors")

    shape = tuple(a.shape[i] for i in range(ndim - 1))
    stride = tuple(a.stride[i] // 2 for i in range(ndim - 1))
    v = a.cy_as_strided(shape, stride, a.offset // 2)
    v._storage = a.storage()._reinterpret(out_dtype)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "view_as_complex",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def unfold(a, dimension, size, step):
    cdef object ndim
    cdef object d
    cdef object dim_size
    cdef object n_windows
    cdef object shape
    cdef object stride
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("unfold", a.device.type, a, dimension, size, step)

    ndim = len(a.shape)
    d = dimension if dimension >= 0 else dimension + ndim
    dim_size = a.shape[d]
    n_windows = max(0, (dim_size - size) // step + 1)

    shape = list(a.shape)
    stride = list(a.stride)
    shape[d] = n_windows
    shape.append(size)
    stride[d] = stride[d] * step
    stride.append(a.stride[d])

    v = a.cy_as_strided(tuple(shape), tuple(stride), a.offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "unfold",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def unbind(a, dim=0):
    cdef object ndim
    cdef object d
    cdef object dim_size
    cdef object i
    cdef object idx
    cdef object new_shape
    cdef object new_stride
    cdef object new_offset
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object outputs
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("unbind", a.device.type, a, dim)
    if getattr(a, "requires_grad", False):
        return _dispatch_fn("unbind", a.device.type, a, dim)

    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    dim_size = a.shape[d]
    outputs = []

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    else:
        creation_kind = "multi_view"

    new_shape = list(a.shape)
    del new_shape[d]
    new_stride = list(a.stride)
    del new_stride[d]

    for i in range(dim_size):
        idx = int(i)
        new_offset = a.offset + idx * a.stride[d]
        v = a.cy_as_strided(tuple(new_shape), tuple(new_stride), new_offset)
        v._view_meta = {
            "op": "select",
            "shape": tuple(v.shape),
            "stride": tuple(v.stride),
            "offset": int(v.offset),
            "creation_mode": creation_mode,
            "creation_kind": creation_kind,
        }
        outputs.append(v)
    return tuple(outputs)


def split(a, split_size_or_sections, dim=0):
    cdef object ndim
    cdef object d
    cdef object dim_size
    cdef object outputs
    cdef object step
    cdef object start
    cdef object end
    cdef object size
    cdef object new_shape
    cdef object new_offset
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("split", a.device.type, a, split_size_or_sections, dim)
    if getattr(a, "requires_grad", False):
        if not _grad_enabled_fast() and getattr(getattr(a, "device", None), "type", None) == "npu":
            from candle._backends.npu.ops.shape import split as _npu_split
            return _npu_split(a, split_size_or_sections, dim)
        return _dispatch_fn("split", a.device.type, a, split_size_or_sections, dim)

    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    dim_size = a.shape[d]
    outputs = []

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    else:
        creation_kind = "multi_view"

    if isinstance(split_size_or_sections, int):
        if split_size_or_sections <= 0:
            raise ValueError("split_size must be > 0")
        step = split_size_or_sections
        for start in range(0, dim_size, step):
            end = start + step
            if end > dim_size:
                end = dim_size
            new_shape = list(a.shape)
            new_shape[d] = end - start
            new_offset = a.offset + int(start) * a.stride[d]
            v = a.cy_as_strided(tuple(new_shape), tuple(a.stride), new_offset)
            v._view_meta = {
                "op": "narrow",
                "shape": tuple(v.shape),
                "stride": tuple(v.stride),
                "offset": int(v.offset),
                "creation_mode": creation_mode,
                "creation_kind": creation_kind,
            }
            outputs.append(v)
    else:
        if sum(split_size_or_sections) != dim_size:
            raise ValueError("split sections must sum to dim size")
        start = 0
        for size in split_size_or_sections:
            new_shape = list(a.shape)
            new_shape[d] = size
            new_offset = a.offset + int(start) * a.stride[d]
            v = a.cy_as_strided(tuple(new_shape), tuple(a.stride), new_offset)
            v._view_meta = {
                "op": "narrow",
                "shape": tuple(v.shape),
                "stride": tuple(v.stride),
                "offset": int(v.offset),
                "creation_mode": creation_mode,
                "creation_kind": creation_kind,
            }
            outputs.append(v)
            start += size
    return tuple(outputs)


def chunk(a, chunks, dim=0):
    cdef object ndim
    cdef object d
    cdef object dim_size
    cdef object actual_chunks
    cdef object chunk_size
    cdef object size

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("chunk", a.device.type, a, chunks, dim)

    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    dim_size = a.shape[d]
    if chunks <= 0:
        raise ValueError("chunks must be > 0")
    actual_chunks = chunks if dim_size == 0 else min(chunks, dim_size)
    if actual_chunks == 0:
        return tuple()
    chunk_size = (dim_size + actual_chunks - 1) // actual_chunks
    return split(a, chunk_size, d)


def _split_sizes_from_indices(dim_size, indices):
    cdef list sizes = []
    cdef object raw_index
    cdef object index
    cdef object start = 0

    for raw_index in indices:
        index = int(raw_index)
        if index < 0:
            index += dim_size
        if index < 0:
            index = 0
        elif index > dim_size:
            index = dim_size
        sizes.append(index - start if index >= start else 0)
        start = index
    sizes.append(dim_size - start if dim_size >= start else 0)
    return sizes


def vsplit(a, split_size_or_sections):
    cdef object sizes
    cdef object sections
    cdef object dim_size
    cdef object size
    cdef object extra

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("vsplit", a.device.type, a, split_size_or_sections)

    if a.dim() < 2:
        raise RuntimeError(
            f"torch.vsplit requires a tensor with at least 2 dimension, but got a tensor with {a.dim()} dimensions!"
        )
    if isinstance(split_size_or_sections, int):
        sections = split_size_or_sections
        if sections <= 0:
            raise RuntimeError("torch.vsplit sections must be > 0")
        dim_size = a.shape[0]
        size, extra = divmod(dim_size, sections)
        if extra != 0:
            raise RuntimeError(
                f"torch.vsplit attempted to split along dimension 0, "
                f"but the size of the dimension {dim_size} is not divisible by the split_size {sections}!"
            )
        sizes = [size + 1] * extra + [size] * (sections - extra)
        return split(a, tuple(sizes), 0)
    sizes = _split_sizes_from_indices(a.shape[0], split_size_or_sections)
    return split(a, tuple(sizes), 0)


def hsplit(a, split_size_or_sections):
    cdef object dim
    cdef object sizes
    cdef object sections
    cdef object dim_size
    cdef object size
    cdef object extra

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("hsplit", a.device.type, a, split_size_or_sections)

    if a.dim() < 1:
        raise RuntimeError(
            f"torch.hsplit requires a tensor with at least 1 dimension, but got a tensor with {a.dim()} dimensions!"
        )
    dim = 0 if a.dim() == 1 else 1
    if isinstance(split_size_or_sections, int):
        sections = split_size_or_sections
        if sections <= 0:
            raise RuntimeError("torch.hsplit sections must be > 0")
        dim_size = a.shape[dim]
        size, extra = divmod(dim_size, sections)
        if extra != 0:
            raise RuntimeError(
                f"torch.hsplit attempted to split along dimension {dim}, "
                f"but the size of the dimension {dim_size} is not divisible by the split_size {sections}!"
            )
        sizes = [size + 1] * extra + [size] * (sections - extra)
        return split(a, tuple(sizes), dim)
    sizes = _split_sizes_from_indices(a.shape[dim], split_size_or_sections)
    return split(a, tuple(sizes), dim)


def dsplit(a, split_size_or_sections):
    cdef object sizes
    cdef object sections
    cdef object dim_size
    cdef object size
    cdef object extra

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("dsplit", a.device.type, a, split_size_or_sections)

    if a.dim() < 3:
        raise RuntimeError(
            f"torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with {a.dim()} dimensions!"
        )

    if isinstance(split_size_or_sections, int):
        sections = split_size_or_sections
        if sections <= 0:
            raise RuntimeError("torch.dsplit sections must be > 0")
        dim_size = a.shape[2]
        size, extra = divmod(dim_size, sections)
        if extra != 0:
            raise RuntimeError(
                f"torch.dsplit attempted to split along dimension 2, "
                f"but the size of the dimension {dim_size} is not divisible by the split_size {sections}!"
            )
        sizes = [size + 1] * extra + [size] * (sections - extra)
        return split(a, tuple(sizes), 2)
    sizes = _split_sizes_from_indices(a.shape[2], split_size_or_sections)
    return split(a, tuple(sizes), 2)


def view(*args, **kwargs):
    cdef object a
    cdef object shape
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v
    cdef object shape_list
    cdef object infer_idx
    cdef object known_size
    cdef object size
    cdef object new_size
    cdef object idx
    cdef object dim

    _ensure_base()
    _ensure_dispatch()

    if not args:
        return _dispatch_fn("view", None, *args, **kwargs)

    a = args[0]
    if len(args) >= 2:
        shape = args[1]
    else:
        shape = kwargs.get("shape")

    if not _is_base_tensor(a):
        return _dispatch_fn("view", None, *args, **kwargs)

    if isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, (tuple, list)):
        shape = tuple(shape)
    else:
        return _dispatch_fn("view", None, *args, **kwargs)

    if not a.is_contiguous():
        raise RuntimeError(
            "view size is not compatible with input tensor's size and stride "
            "(at least one dimension spans across two contiguous subspaces). "
            "Use .reshape(...) instead."
        )

    size = 1
    for dim in a.shape:
        size *= dim

    infer_idx = None
    known_size = 1
    shape_list = list(shape)
    for idx, dim in enumerate(shape_list):
        if dim == -1:
            if infer_idx is not None:
                raise RuntimeError("only one dimension can be inferred")
            infer_idx = idx
            continue
        known_size *= dim

    if infer_idx is not None:
        if known_size == 0 or size % known_size != 0:
            raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {size}")
        shape_list[infer_idx] = size // known_size

    shape = tuple(shape_list)
    new_size = 1
    for dim in shape:
        new_size *= dim
    if size != new_size:
        raise ValueError("view size mismatch")

    v = a.cy_view(shape)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "view",
        "shape": tuple(shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v



def neg(a):
    """Fast neg: skip __torch_function__ for base Tensor."""
    cdef object r

    _ensure_originals()

    if _is_base_tensor(a):
        return _dispatch_fn("neg", a.device.type, a)

    r = _handle_torch_function(_py_neg_fn, (a,), {})
    if r is not NotImplemented:
        return r

    return _dispatch_fn("neg", a.device.type, a)
