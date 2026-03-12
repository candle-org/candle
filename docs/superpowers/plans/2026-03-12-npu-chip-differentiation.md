# NPU Chip Differentiation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the monolithic `npu/ops.py` (11K+ lines) into functional domain modules and extend `ops_soc.py` to differentiate 910A/910B/310B/310P chips across operator support, dtype support, and distributed op capabilities.

**Architecture:** Convert `ops.py` to an `ops/` package by renaming it to `ops/__init__.py`, then incrementally extract domain modules. Each extraction leaves external imports unchanged. Extend `ops_soc.py` with three capability dimensions.

**Tech Stack:** Python, ctypes (ACLNN bindings), existing candle dispatch system.

**Spec:** `docs/superpowers/specs/2026-03-12-npu-chip-differentiation-design.md`

---

## Chunk 1: Foundation — Package Conversion + Capability Table

### Task 1: Convert `ops.py` to `ops/` package

This is the critical foundation step. By converting the single file into a package `__init__.py`, all existing imports (`from .ops import xxx`) continue to work unchanged.

**Files:**
- Rename: `src/candle/_backends/npu/ops.py` → `src/candle/_backends/npu/ops/__init__.py`

- [ ] **Step 1: Create the ops/ directory and move the file**

```bash
cd src/candle/_backends/npu
mkdir ops
git mv ops.py ops/__init__.py
```

- [ ] **Step 2: Fix relative imports inside ops/__init__.py**

The file is now one level deeper. All relative imports like `from ..._dtype import ...` need an extra dot, and `from . import aclnn` becomes `from .. import aclnn`.

Find and update these imports at the top of `ops/__init__.py`:

```python
# OLD (lines 1-12):
from ..._dtype import bool as bool_dtype
from ..._dtype import int32 as int32_dtype
from ..._dtype import int64 as int64_dtype
from ..._dtype import float32 as float_dtype
from ..._storage import npu_typed_storage_from_ptr
from ..common import view as view_backend
reshape = view_backend.reshape
from . import aclnn
from . import runtime as npu_runtime
from . import state as npu_state
from . import ops_soc

# NEW:
from ...._dtype import bool as bool_dtype
from ...._dtype import int32 as int32_dtype
from ...._dtype import int64 as int64_dtype
from ...._dtype import float32 as float_dtype
from ...._storage import npu_typed_storage_from_ptr
from ...common import view as view_backend
reshape = view_backend.reshape
from .. import aclnn
from .. import runtime as npu_runtime
from .. import state as npu_state
from .. import ops_soc
```

Also search for any other relative imports deeper in the file (e.g., `from ..._dtype`, `from ..._storage`, `from ..common`) and add one extra dot to each.

- [ ] **Step 3: Verify imports work**

Run: `source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -c "from candle._backends.npu.ops import add, matmul, relu; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Run NPU tests to verify nothing broke**

Run: `source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/npu/ -x -q --tb=short 2>&1 | tail -20`
Expected: All tests pass (or same pass/fail as before)

- [ ] **Step 5: Commit**

```bash
git add -A src/candle/_backends/npu/ops/
git commit -m "refactor(npu): convert ops.py to ops/ package

Move ops.py to ops/__init__.py and fix relative imports.
No functional changes — all external imports remain unchanged."
```

---

### Task 2: Extend `ops_soc.py` with dtype and distributed capability tables

**Files:**
- Modify: `src/candle/_backends/npu/ops_soc.py`
- Delete: `src/candle/_backends/npu/ops_910a.py`
- Delete: `src/candle/_backends/npu/ops_910b.py`
- Delete: `src/candle/_backends/npu/ops_310b.py`
- Delete: `src/candle/_backends/npu/ops_310p.py`

- [ ] **Step 1: Rewrite ops_soc.py with expanded capability tables**

Replace the full contents of `src/candle/_backends/npu/ops_soc.py` with:

```python
"""SoC-aware capability lookup for NPU ops.

Centralises all chip-specific differences:
  1. Operator fallback  — ops needing composite workaround
  2. Dtype support       — global and per-op dtype restrictions
  3. Distributed ops     — collective op availability
  4. Chip flags          — fine-grained feature switches
"""

from . import runtime as npu_runtime

# ── 1. Operator fallback ─────────────────────────────────────────────
# Ops that must use on-device composite workaround on a given chip.

_FALLBACK_OPS = {
    "910a": frozenset(),
    "910b": frozenset(),
    "310b": frozenset({
        "atan2", "where", "flip", "argsort", "sort", "topk",
        "diag", "lerp", "remainder", "isclose", "softplus",
        "uniform_", "normal_", "layer_norm", "mish",
        "batch_norm", "dropout", "take_along_dim", "gather",
    }),
    "310p": frozenset(),
}

# ── 2. Dtype support ─────────────────────────────────────────────────
# 2a. Dtypes globally unsupported on a chip (all ops).
_UNSUPPORTED_DTYPES_GLOBAL = {
    "910a": frozenset(),
    "910b": frozenset(),
    "310b": frozenset({"bfloat16"}),
    "310p": frozenset(),
}

# 2b. Per-op dtype restrictions (override / extend global).
_UNSUPPORTED_DTYPES_PER_OP = {
    "910a": {},
    "910b": {},
    "310b": {},
    "310p": {},
}

# ── 3. Distributed op support ────────────────────────────────────────
_SUPPORTED_DISTRIBUTED_OPS = {
    "910a": frozenset({"all_reduce", "broadcast", "all_gather"}),
    "910b": frozenset({
        "all_reduce", "broadcast", "all_gather",
        "alltoall_v", "reduce_scatter",
    }),
    "310b": frozenset(),
    "310p": frozenset(),
}

# ── 4. Chip flags ────────────────────────────────────────────────────
_CHIP_FLAGS = {
    "910a": {
        "use_smallop_arange_1d": False,
        "use_smallop_linspace": False,
    },
    "910b": {
        "use_smallop_arange_1d": False,
        "use_smallop_linspace": False,
    },
    "310b": {
        "use_smallop_arange_1d": True,
        "use_smallop_linspace": True,
    },
    "310p": {
        "use_smallop_arange_1d": False,
        "use_smallop_linspace": False,
    },
}

# ── Internal helpers ─────────────────────────────────────────────────

_cached_profile = None


def current_profile():
    """Return cached SoC profile string (e.g. '910b')."""
    global _cached_profile
    if _cached_profile is None:
        _cached_profile = npu_runtime.soc_profile()
    return _cached_profile


def _resolve_profile(profile):
    return current_profile() if profile is None else str(profile).lower()


# ── Public API ───────────────────────────────────────────────────────

def use_fallback(op_name, profile=None):
    """Return True if *op_name* should use composite fallback."""
    p = _resolve_profile(profile)
    ops = _FALLBACK_OPS.get(p, frozenset())
    return op_name in ops


def fallback_ops(profile=None):
    """Return the full set of fallback ops for *profile*."""
    return _FALLBACK_OPS.get(_resolve_profile(profile), frozenset())


def check_dtype_support(op_name, dtype, profile=None):
    """Return True if *dtype* is supported for *op_name* on *profile*.

    Checks global restrictions first, then per-op restrictions.
    """
    p = _resolve_profile(profile)
    dtype_name = dtype if isinstance(dtype, str) else getattr(dtype, "name", str(dtype))
    # Global restriction
    if dtype_name in _UNSUPPORTED_DTYPES_GLOBAL.get(p, frozenset()):
        return False
    # Per-op restriction
    per_op = _UNSUPPORTED_DTYPES_PER_OP.get(p, {})
    if op_name in per_op and dtype_name in per_op[op_name]:
        return False
    return True


def is_distributed_op_supported(op_name, profile=None):
    """Return True if distributed *op_name* is available on *profile*."""
    p = _resolve_profile(profile)
    supported = _SUPPORTED_DISTRIBUTED_OPS.get(p, frozenset())
    return op_name in supported


def chip_flag(name, profile=None, default=False):
    """Query a chip-specific feature flag."""
    p = _resolve_profile(profile)
    flags = _CHIP_FLAGS.get(p)
    if flags is None:
        return bool(default)
    return bool(flags.get(name, default))


# ── Convenience aliases (backward-compatible) ────────────────────────

def capability(name, profile=None, default=False):
    """Alias for chip_flag (backward compatibility)."""
    return chip_flag(name, profile=profile, default=default)


def use_smallop_arange_1d(profile=None):
    return chip_flag("use_smallop_arange_1d", profile=profile, default=False)


def use_smallop_linspace(profile=None):
    return chip_flag("use_smallop_linspace", profile=profile, default=False)
```

- [ ] **Step 2: Delete the empty per-chip policy modules**

```bash
git rm src/candle/_backends/npu/ops_910a.py
git rm src/candle/_backends/npu/ops_910b.py
git rm src/candle/_backends/npu/ops_310b.py
git rm src/candle/_backends/npu/ops_310p.py
```

- [ ] **Step 3: Verify ops_soc still works**

Run: `source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -c "from candle._backends.npu import ops_soc; print(ops_soc.use_fallback('atan2', profile='310b')); print(ops_soc.check_dtype_support('matmul', 'bfloat16', profile='310b')); print(ops_soc.is_distributed_op_supported('alltoall_v', profile='910b'))"`
Expected: `True`, `False`, `True`

- [ ] **Step 4: Run NPU tests**

Run: `source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/npu/ -x -q --tb=short 2>&1 | tail -20`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(npu): extend ops_soc with dtype and distributed capability tables

Add three capability dimensions: operator fallback, dtype support,
and distributed op support. Remove empty per-chip policy modules
(ops_910a/910b/310b/310p) — all routing now centralized in ops_soc."
```

---

## Chunk 2: Extract Helper Utilities + First Domain Modules

### Task 3: Extract `ops/_helpers.py` — shared utilities

All domain modules will depend on these helpers. Extract them first.

**Files:**
- Create: `src/candle/_backends/npu/ops/_helpers.py`
- Modify: `src/candle/_backends/npu/ops/__init__.py`

- [ ] **Step 1: Create `_helpers.py` with shared utilities**

Read `ops/__init__.py` and extract these functions (approximately lines 1-320 based on exploration):
- `_unwrap_storage()`
- `_wrap_tensor()`
- `_dtype_itemsize()`
- `_cast_tensor_dtype()`
- `_broadcast_shape()`
- `_broadcast_shape_checked()`
- `_npu_broadcast_to()`
- `_npu_arange_1d()`
- `_use_soc_fallback()`
- `_npu_add_scalar_()`
- `_npu_linear_index()`
- `npu_index_put_impl()`
- `_matmul_out_shape()`
- `_normalize_tensor_sequence_args()`
- `_iter_indices()`
- `_broadcast_index()`
- `_batch_offset()`
- `_unary_op()`
- `_binary_op()`
- `_numel()`
- `_normalize_reduction_dims()`
- `_reduce_out_shape()`
- `_reduce_dim_sizes()`
- `_broadcast_dims_to_out()`
- `_scalar_to_npu_tensor()`
- `_scalar_to_npu_tensor_no_add()`
- `_nan_like()`

The file should start with:

```python
"""Shared helper utilities for NPU ops modules."""

from ...._dtype import bool as bool_dtype
from ...._dtype import int32 as int32_dtype
from ...._dtype import int64 as int64_dtype
from ...._dtype import float32 as float_dtype
from ...._storage import npu_typed_storage_from_ptr
from ...common import view as view_backend
reshape = view_backend.reshape
from .. import aclnn
from .. import runtime as npu_runtime
from .. import state as npu_state
from .. import ops_soc
```

Move each helper function from `ops/__init__.py` into `_helpers.py`. In `ops/__init__.py`, replace each removed function with an import:

```python
from ._helpers import (
    _unwrap_storage, _wrap_tensor, _numel, _dtype_itemsize,
    _cast_tensor_dtype, _broadcast_shape, _broadcast_shape_checked,
    _npu_broadcast_to, _npu_arange_1d, _use_soc_fallback,
    _npu_add_scalar_, _npu_linear_index, npu_index_put_impl,
    _matmul_out_shape, _normalize_tensor_sequence_args,
    _iter_indices, _broadcast_index, _batch_offset,
    _unary_op, _binary_op,
    _normalize_reduction_dims, _reduce_out_shape,
    _reduce_dim_sizes, _broadcast_dims_to_out,
    _scalar_to_npu_tensor, _scalar_to_npu_tensor_no_add, _nan_like,
)
```

Remove the duplicate import block at the top of `ops/__init__.py` (the `from ...._dtype` etc. lines) since `_helpers.py` now owns them. But keep `from .. import ops_soc` etc. if any remaining functions in `__init__.py` use them directly.

**Important:** Some helpers reference each other (e.g., `_npu_arange_1d` calls `_npu_add_scalar_`). Make sure they are all in `_helpers.py` together.

- [ ] **Step 2: Verify imports**

Run: `source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -c "from candle._backends.npu.ops._helpers import _unwrap_storage, _unary_op; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run NPU tests**

Run: `source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/npu/ -x -q --tb=short 2>&1 | tail -20`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(npu): extract shared helpers into ops/_helpers.py

Move ~40 shared utility functions used across all op domains into
a dedicated _helpers module. No functional changes."
```

---

### Task 4: Extract `ops/math.py` — arithmetic + unary math

**Files:**
- Create: `src/candle/_backends/npu/ops/math.py`
- Modify: `src/candle/_backends/npu/ops/__init__.py`

- [ ] **Step 1: Create `math.py`**

Move these functions from `ops/__init__.py` into `ops/math.py`:

**Arithmetic (binary):** `add`, `sub`, `mul`, `div`
**In-place arithmetic:** `add_`, `sub_`, `mul_`, `div_`
**Unary math:** `abs`, `neg`, `sign`, `signbit`, `square`, `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tan`, `tanh`, `sigmoid`, `sinh`, `cosh`, `erf`, `erfc`, `floor`, `ceil`, `round`, `trunc`, `frac`, `log2`, `log10`, `exp2`, `expm1`, `log1p`, `asin`, `acos`, `atan`, `asinh`, `acosh`, `atanh`
**Binary math:** `atan2`, `pow`, `floor_divide`
**Float classification:** `isfinite`, `isinf`, `isnan`, `isposinf`, `isneginf`

The file header:

```python
"""Arithmetic and unary math operations for NPU."""

from ._helpers import (
    _unwrap_storage, _wrap_tensor, _unary_op, _binary_op,
    _cast_tensor_dtype, _broadcast_shape, _npu_broadcast_to,
    _scalar_to_npu_tensor, _scalar_to_npu_tensor_no_add,
    # ... add whatever helpers these functions actually use
)
from .. import aclnn
from .. import runtime as npu_runtime
from .. import state as npu_state
from .. import ops_soc
```

- [ ] **Step 2: Update `ops/__init__.py`**

Replace the moved function definitions with:

```python
from .math import (
    abs, add, sub, mul, div, neg, sign, signbit, square,
    exp, log, sqrt, rsqrt, sin, cos, tan, tanh, sigmoid,
    sinh, cosh, erf, erfc, floor, ceil, round, trunc, frac,
    log2, log10, exp2, expm1, log1p, asin, acos, atan,
    asinh, acosh, atanh, atan2, pow, floor_divide,
    isfinite, isinf, isnan, isposinf, isneginf,
    add_, sub_, mul_, div_,
)
```

- [ ] **Step 3: Verify + run tests**

Run: `source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -c "from candle._backends.npu.ops import add, exp, isnan; print('OK')"` → `OK`

Run: `source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/npu/ -x -q --tb=short 2>&1 | tail -20` → All pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(npu): extract ops/math.py — arithmetic + unary math"
```

---

### Task 5: Extract `ops/comparison.py` — comparison, logical, bitwise

**Files:**
- Create: `src/candle/_backends/npu/ops/comparison.py`
- Modify: `src/candle/_backends/npu/ops/__init__.py`

- [ ] **Step 1: Create `comparison.py`**

Move: `eq`, `ne`, `le`, `lt`, `gt`, `ge`, `logical_and`, `logical_or`, `logical_not`, `logical_xor`, `bitwise_not`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `equal`, `allclose`, `isclose`

- [ ] **Step 2: Update `ops/__init__.py`** — add `from .comparison import (...)`

- [ ] **Step 3: Verify + run tests** → All pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(npu): extract ops/comparison.py — comparison, logical, bitwise"
```

---

### Task 6: Extract `ops/reduce.py` — reductions + cumulative

**Files:**
- Create: `src/candle/_backends/npu/ops/reduce.py`
- Modify: `src/candle/_backends/npu/ops/__init__.py`

- [ ] **Step 1: Create `reduce.py`**

Move: `argmax`, `argmin`, `amax`, `amin`, `median`, `kthvalue`, `sum_`, `mean`, `all_`, `any_`, `count_nonzero`, `min_`, `max_`, `maximum`, `minimum`, `fmin`, `fmax`, `searchsorted`, `unique`, `topk`, `argsort`, `sort`, `cumsum`, `cumprod`, `cummax`, `cummin_op`, `var_`, `std_`, `norm_`, `prod_`, `nansum`, `logsumexp_op`, `nanmean_op`, `nanmedian_op`, `nanquantile_op`, `quantile_op`, `aminmax_op`, `aminmax_aclnn`, `renorm_op`, `argwhere_op`

Also move helper functions: `_normalize_reduction_dims`, `_reduce_out_shape`, `_reduce_dim_sizes`, `_broadcast_dims_to_out` — if not already in `_helpers.py`, move them there first and import from `_helpers`.

- [ ] **Step 2: Update `ops/__init__.py`** — add `from .reduce import (...)`

- [ ] **Step 3: Verify + run tests** → All pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(npu): extract ops/reduce.py — reductions + cumulative"
```

---

## Chunk 3: Shape, Activation, Normalization

### Task 7: Extract `ops/shape.py` — shape, view, indexing (~2500 lines)

This is the largest module. Contains reshape-related ops, flip/roll/rot90, cat/stack/split, indexing (getitem/setitem), scatter/gather, and related helpers.

**Files:**
- Create: `src/candle/_backends/npu/ops/shape.py`
- Modify: `src/candle/_backends/npu/ops/__init__.py`

- [ ] **Step 1: Create `shape.py`**

Move all shape/view/index functions:

**Shape manipulation:** `flatten_op`, `flip`, `roll`, `rot90`, `tile`, `repeat`, `repeat_interleave`, `tril`, `triu`, `tril_indices`, `triu_indices`, `diag`, `cartesian_prod`, `block_diag`, `contiguous`, `unfold`, `unflatten_op`, `broadcast_to_op`, `movedim_op`, `moveaxis_op`, `diagonal_op`, `one_hot`

**Stacking/splitting:** `cat`, `concatenate`, `stack`, `chunk`, `split`, `vsplit`, `hsplit`, `dsplit`, `unbind`, `hstack`, `vstack`, `row_stack`, `dstack`, `column_stack`

**Indexing:** `getitem`, `setitem`, `gather`, `index_select`, `take`, `take_along_dim`, `masked_select`, `scatter`, `scatter_`, `scatter_add_`, `masked_scatter_`, `masked_fill`, `masked_fill_`, `index_put_`, `index_put`, `index_copy_`, `index_fill_`, `index_add_`, `nonzero`, `narrow`, `select`, `expand`

**Also move private helpers** that are only used by shape ops: `_npu_basic_getitem_view`, `_npu_basic_getitem_with_strided_slices`, `_npu_aclnn_slice`, `_npu_advanced_getitem`, `_npu_expand`, `_npu_advanced_setitem`, etc.

- [ ] **Step 2: Update `ops/__init__.py`** — add `from .shape import (...)`

- [ ] **Step 3: Verify + run tests** → All pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(npu): extract ops/shape.py — shape, view, indexing"
```

---

### Task 8: Extract `ops/activation.py` — activation functions

**Files:**
- Create: `src/candle/_backends/npu/ops/activation.py`
- Modify: `src/candle/_backends/npu/ops/__init__.py`

- [ ] **Step 1: Create `activation.py`**

Move: `relu`, `relu_`, `relu6`, `sigmoid`, `softplus`, `hardtanh`, `silu`, `gelu`, `leaky_relu`, `elu`, `mish`, `prelu`, `selu_op`, `celu_op`, `threshold_op`, `hardshrink_op`, `softshrink_op`, `hardswish_op`, `hardsigmoid_op`, `softsign_op`, `rrelu_op`, `softmax`, `log_softmax`, `embedding`

- [ ] **Step 2: Update `ops/__init__.py`** — add `from .activation import (...)`

- [ ] **Step 3: Verify + run tests** → All pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(npu): extract ops/activation.py — activation functions"
```

---

### Task 9: Extract `ops/norm.py` — normalization ops

**Files:**
- Create: `src/candle/_backends/npu/ops/norm.py`
- Modify: `src/candle/_backends/npu/ops/__init__.py`

- [ ] **Step 1: Create `norm.py`**

Move: `layer_norm`, `_layer_norm_310b_fallback`, `batch_norm`, `_batch_norm_310b_fallback`, `group_norm`, `instance_norm`, `rms_norm`, `normalize_op`

- [ ] **Step 2: Update `ops/__init__.py`** — add `from .norm import (...)`

- [ ] **Step 3: Verify + run tests** → All pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(npu): extract ops/norm.py — normalization ops"
```

---

## Chunk 4: Linear Algebra, Convolution, Remaining Domains

### Task 10: Extract `ops/linalg.py` — linear algebra (~2000 lines)

**Files:**
- Create: `src/candle/_backends/npu/ops/linalg.py`
- Modify: `src/candle/_backends/npu/ops/__init__.py`

- [ ] **Step 1: Create `linalg.py`**

Move: `matmul`, `dot`, `mv`, `outer`, `mm_op`, `bmm_op`, `addmm`, `baddbmm`, `einsum_`, `linalg_qr`, `linalg_inv`, `linalg_vector_norm_op`, `linalg_norm_op`, `linalg_matrix_norm_op`, `linalg_multi_dot_op`, `linalg_matrix_power_op`, `linalg_vander_op`, `linalg_cholesky_op`, `linalg_cond_op`, `linalg_det_op`, `linalg_slogdet_op`, `linalg_eig_op`, `linalg_eigh_op`, `linalg_eigvals_op`, `linalg_eigvalsh_op`, `linalg_householder_product_op`, `linalg_lstsq_op`, `linalg_lu_op`, `linalg_lu_factor_op`, `linalg_lu_solve_op`, `linalg_matrix_exp_op`, `linalg_matrix_rank_op`, `linalg_pinv_op`, `linalg_solve_op`, `linalg_solve_triangular_op`, `linalg_svd_op`, `linalg_svdvals_op`, `linalg_tensorinv_op`, `linalg_tensorsolve_op`, `matrix_power_op`, `det_op`, `inner_op`, `tensordot_op`, `trace_op`, `cross_op`, `dist_op`, `cdist_op`

Also move: `_matmul_out_shape`, `_batch_offset` — if not already in `_helpers.py`.

- [ ] **Step 2: Update `ops/__init__.py`** — add `from .linalg import (...)`

- [ ] **Step 3: Verify + run tests** → All pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(npu): extract ops/linalg.py — linear algebra"
```

---

### Task 11: Extract `ops/conv.py` — convolution + pooling + upsampling

**Files:**
- Create: `src/candle/_backends/npu/ops/conv.py`
- Modify: `src/candle/_backends/npu/ops/__init__.py`

- [ ] **Step 1: Create `conv.py`**

Move: `conv1d`, `conv2d`, `conv3d_op`, `conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d_op`, `max_pool1d_op`, `max_pool2d`, `max_pool3d`, `avg_pool1d_op`, `avg_pool2d`, `avg_pool3d_op`, `adaptive_avg_pool1d_op`, `adaptive_avg_pool2d`, `adaptive_avg_pool3d_op`, `adaptive_max_pool1d_op`, `adaptive_max_pool2d`, `upsample_nearest1d_op`, `upsample_nearest2d`, `upsample_bilinear2d`, `upsample_bicubic2d_op`, `upsample_linear1d_op`, `im2col_op`, `col2im_op`, `grid_sample_op`, `affine_grid_op`, `pad`, `pad_sequence`, `ctc_loss_op`

- [ ] **Step 2: Update `ops/__init__.py`** — add `from .conv import (...)`

- [ ] **Step 3: Verify + run tests** → All pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(npu): extract ops/conv.py — conv, pooling, upsampling"
```

---

### Task 12: Extract `ops/random.py` — random + initialization

**Files:**
- Create: `src/candle/_backends/npu/ops/random.py`
- Modify: `src/candle/_backends/npu/ops/__init__.py`

- [ ] **Step 1: Create `random.py`**

Move: `uniform_`, `normal_`, `bernoulli_`, `exponential_`, `log_normal_`, `cauchy_`, `geometric_`, `randint_`, `random_`, `randperm`, `fill_`, `zero_`, `copy_`, `erfinv_`, `clamp_`, `reciprocal_`, `relu_`, `dropout`, `_dropout_310b_mask`, `uniform_op`

Also move private helpers for fallbacks (e.g., `_uniform_310b_fallback`, `_normal_310b_fallback` if they exist).

- [ ] **Step 2: Update `ops/__init__.py`** — add `from .random import (...)`

- [ ] **Step 3: Verify + run tests** → All pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(npu): extract ops/random.py — random, init, in-place"
```

---

### Task 13: Extract `ops/elementwise.py` — misc element-wise

**Files:**
- Create: `src/candle/_backends/npu/ops/elementwise.py`
- Modify: `src/candle/_backends/npu/ops/__init__.py`

- [ ] **Step 1: Create `elementwise.py`**

Move: `where`, `lerp`, `addcmul`, `addcdiv`, `logaddexp`, `logaddexp2`, `hypot`, `remainder`, `fmod`, `clamp`, `clamp_min`, `clamp_max`, `heaviside_op`, `diff_op`, `isreal_op`, `isin_op`, `bincount_op`, `bincount_aclnn`, `bucketize_op`, `histc_op`, `histogram_op`

Also move any private fallback helpers (e.g., `_remainder_310b_fallback`).

- [ ] **Step 2: Update `ops/__init__.py`** — add `from .elementwise import (...)`

- [ ] **Step 3: Verify + run tests** → All pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(npu): extract ops/elementwise.py — misc element-wise"
```

---

### Task 14: Extract `ops/special.py` — special functions + FFT

**Files:**
- Create: `src/candle/_backends/npu/ops/special.py`
- Modify: `src/candle/_backends/npu/ops/__init__.py`

- [ ] **Step 1: Create `special.py`**

Move all `special_*` functions and all `fft_*` functions.

- [ ] **Step 2: Update `ops/__init__.py`** — add `from .special import (...)`

- [ ] **Step 3: Verify + run tests** → All pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(npu): extract ops/special.py — special functions + FFT"
```

---

### Task 15: Extract `ops/optim.py` — optimizer step ops

**Files:**
- Create: `src/candle/_backends/npu/ops/optim.py`
- Modify: `src/candle/_backends/npu/ops/__init__.py`

- [ ] **Step 1: Create `optim.py`**

Move: `_adam_step_op`, `_adamw_step_op`, `_sgd_step_op`, `_adagrad_step_op`, `_rmsprop_step_op`, `_adadelta_step_op`, `_adamax_step_op`, `_asgd_step_op`, `_nadam_step_op`, `_radam_step_op`, `_rprop_step_op`, `_sparse_adam_step_op`

- [ ] **Step 2: Update `ops/__init__.py`** — add `from .optim import (...)`

- [ ] **Step 3: Verify + run tests** → All pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(npu): extract ops/optim.py — optimizer step ops"
```

---

## Chunk 5: Distributed + Cleanup

### Task 16: Create `distributed/` directory

**Files:**
- Create: `src/candle/_backends/npu/distributed/__init__.py`
- Create: `src/candle/_backends/npu/distributed/collective.py`

- [ ] **Step 1: Create `distributed/__init__.py`**

```python
"""Distributed collective operations for NPU backend."""

from .collective import *
```

- [ ] **Step 2: Create `collective.py` skeleton**

```python
"""Distributed collective op implementations with chip-aware guards."""

from .. import ops_soc


def _check_distributed_support(op_name):
    """Raise RuntimeError if op is not supported on current chip."""
    if not ops_soc.is_distributed_op_supported(op_name):
        raise RuntimeError(
            f"Distributed op '{op_name}' is not supported on "
            f"{ops_soc.current_profile()}"
        )
```

This is a skeleton — actual distributed op implementations will be added as separate work items when the distributed ops are implemented. The infrastructure is in place.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat(npu): add distributed/ skeleton with chip-aware guards"
```

---

### Task 17: Clean up `ops/__init__.py` — convert to pure re-export

After all domain modules are extracted, `ops/__init__.py` should contain only re-exports.

**Files:**
- Modify: `src/candle/_backends/npu/ops/__init__.py`

- [ ] **Step 1: Replace `ops/__init__.py` with pure re-exports**

At this point, all functions have been moved out. Replace the entire file with:

```python
"""NPU ops — re-exports from domain modules.

External import paths are unchanged:
    from candle._backends.npu.ops import add, matmul, relu
"""

from .math import *          # noqa: F401,F403
from .comparison import *    # noqa: F401,F403
from .reduce import *        # noqa: F401,F403
from .shape import *         # noqa: F401,F403
from .activation import *    # noqa: F401,F403
from .norm import *          # noqa: F401,F403
from .linalg import *        # noqa: F401,F403
from .conv import *          # noqa: F401,F403
from .random import *        # noqa: F401,F403
from .elementwise import *   # noqa: F401,F403
from .special import *       # noqa: F401,F403
from .optim import *         # noqa: F401,F403
```

- [ ] **Step 2: Verify all named imports still resolve**

Run: `source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -c "from candle._backends.npu.ops import add, matmul, relu, layer_norm, conv2d, uniform_, linalg_qr, fft_fft_op, _adam_step_op, where, eq, argmax; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Run full NPU test suite**

Run: `source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/npu/ -v --tb=short 2>&1 | tail -40`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(npu): finalize ops/__init__.py as pure re-export hub"
```

---

### Task 18: Final verification and lint

- [ ] **Step 1: Run pylint**

Run: `source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pylint src/candle/_backends/npu/ops/ src/candle/_backends/npu/ops_soc.py src/candle/_backends/npu/distributed/ --disable=all --enable=E 2>&1 | tail -20`
Expected: No errors (warnings are OK)

- [ ] **Step 2: Verify creation.py lazy imports still work**

Run: `source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -c "from candle._backends.npu.creation import rand_create; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Verify test_npu_streams.py import still works**

Run: `source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -c "import candle._backends.npu.ops as npu_ops; print(hasattr(npu_ops, 'add'))"`
Expected: `True`

- [ ] **Step 4: Run contract and CPU tests to verify no side effects**

Run: `source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/ tests/contract/ -x -q --tb=short 2>&1 | tail -20`
Expected: All pass

- [ ] **Step 5: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix(npu): address lint/import issues from ops split"
```

---

## Cross-Cutting Concerns

### Import dependency graph

```
ops/__init__.py  (re-export hub)
  ├── ops/_helpers.py  (shared utilities, imports aclnn/runtime/state/ops_soc)
  ├── ops/math.py      (imports _helpers, aclnn, ops_soc)
  ├── ops/comparison.py
  ├── ops/reduce.py
  ├── ops/shape.py     (may import from math.py for sub/add used in indexing)
  ├── ops/activation.py
  ├── ops/norm.py      (may import from math.py, reduce.py for composite fallbacks)
  ├── ops/linalg.py
  ├── ops/conv.py
  ├── ops/random.py
  ├── ops/elementwise.py
  ├── ops/special.py
  └── ops/optim.py
```

### Handling cross-module function calls

Some domain modules call functions from other domains (e.g., `norm.py`'s `layer_norm` composite fallback uses `mean` and `sqrt` from `math.py`). These should be imported from sibling modules:

```python
# norm.py
from .math import sub, mul, sqrt
from .reduce import mean
```

This creates cross-module imports within the `ops/` package, which is fine — they are all part of the same package and Python handles circular imports at the module level as long as the import happens at function call time or the modules don't have circular top-level imports. If circular import issues arise, use lazy imports inside functions.

### Task execution order

Tasks MUST be executed in order (1 → 18). Each task depends on the previous one being complete because it modifies `ops/__init__.py` incrementally.

### Rollback strategy

If any task fails tests, `git stash` or `git reset --soft HEAD~1` to undo the last commit and debug. The incremental approach means each step is independently revertible.
