# _C/ Cython Directory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `_cython/` + `_C.py` with `_C/` Cython directory that mirrors torch's C++ source structure.

**Architecture:** Move all `.pyx`/`.pxd` files from `_cython/` into new `_C/` directory. Merge `_C.py` TensorBase methods + `_tensor_api.pyx` standalone functions into new `_C/_TensorBase.pyx` as class methods. Merge `_C.py` StorageBase + backend storage classes into new `_C/_Storage.pyx`. `_C/__init__.py` becomes thin stub layer (torch._C equivalent).

**Tech Stack:** Cython 3.x, Python 3.11, numpy

---

## File Structure

```
Create:
  _C/__init__.py              (stub layer, ~150 lines)
  _C/_TensorBase.pyx          (TensorBase class, ~2500 lines)
  _C/_TensorBase.pxd          (Cython header)
  _C/_Storage.pyx             (StorageBase + backend, ~500 lines)
  _C/_Storage.pxd             (Cython header)

Modify:
  candle/__init__.py           (update import paths)
  _functional.py               (update _cython import paths)
  All 47 files importing from _cython (batch sed)

Delete:
  _C.py                       (split into _C/__init__.py + _C/_TensorBase.pyx + _C/_Storage.pyx)
  _cython/                    (moved into _C/)
  _cython/_tensor_api.pyx     (absorbed into _C/_TensorBase.pyx)
  _install_tensor_api()       (methods directly on class)

Move:
  _cython/*.pyx _cython/*.pxd → _C/*.pyx _C/*.pxd
```

---

### Task 0: Pre-flight — verify Cython compilation works

- [ ] **Step 1: Check Cython compilation tooling**

```bash
# Check if Cython is available
conda run -n candle311 python -c "import Cython; print(Cython.__version__)"

# Check if pyximport works
conda run -n candle311 python -c "import pyximport; pyximport.install(); print('pyximport OK')"

# Check if the project has a setup.py/build system
ls setup.py pyproject.toml setup.cfg 2>/dev/null
```

If Cython compilation is not available, skip the `.pyx` changes in Tasks 2-3 and only do the directory restructure (Task 1) + import path updates (Task 4).

---

### Task 1: Scaffold _C/ directory (Phase A)

**Files:**
- Move: `src/candle/_cython/` → `src/candle/_C/`
- Create: `src/candle/_C/__init__.py`
- Modify: `src/candle/__init__.py`

- [ ] **Step 1: Move _cython/ to _C/**

```bash
cd src/candle
mv _cython _C
```

- [ ] **Step 2: Write _C/__init__.py (thin stub layer)**

Take the stub portion of `_C.py` (lines 1-637: all stubs, StorageBase, NPU helpers, storage factories, legacy aliases, __getattr__, _compute_strides, _bf16_to_f32, _f32_to_bf16) and move them into `_C/__init__.py`. Also add the `_HAS_*` flag imports from old `_cython/__init__.py`.

The old `_cython/__init__.py` content (HAS flags + imports) gets merged into the top of `_C/__init__.py`.

```python
# _C/__init__.py — torch._C equivalent
# pylint: disable=import-error,no-name-in-module,possibly-unused-variable

# ---------------------------------------------------------------------------
# Cython feature flags
# ---------------------------------------------------------------------------
_HAS_CYTHON_DISPATCH = False
_HAS_CYTHON_ALLOCATOR = False
_HAS_CYTHON_STORAGE = False
# ... (all HAS flags from old _cython/__init__.py)
_HAS_CYTHON_TENSOR_API = False

try:
    from ._dispatch import cy_dispatch, cy_dispatch_with_keyset  # noqa: F401
    _HAS_CYTHON_DISPATCH = True
except ImportError:
    pass
# ... (all try/except import blocks from old _cython/__init__.py)

# ---------------------------------------------------------------------------
# Re-exports from Cython
# ---------------------------------------------------------------------------
from ._TensorBase import TensorBase, _TensorBase
from ._tensor_impl import TensorImpl, _StrideTuple
from ._Storage import StorageBase

# ---------------------------------------------------------------------------
# Torch-compat stubs
# ---------------------------------------------------------------------------
import abc

def _add_docstr(obj, docstr):
    obj.__doc__ = docstr
    return obj

class _disabled_torch_dispatch_impl:
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass

_torch_function_enabled = True

class DisableTorchFunctionSubclass:
    def __init__(self): pass
    def __enter__(self):
        global _torch_function_enabled
        self._prev = _torch_function_enabled
        _torch_function_enabled = False
        return self
    def __exit__(self, *args):
        global _torch_function_enabled
        _torch_function_enabled = self._prev

def _has_storage(tensor):
    return hasattr(tensor, '_storage') and tensor._storage is not None

def _get_tracing_state():
    return None

def _get_privateuse():
    return "npu"

class _dlpack_exchange_api:
    @staticmethod
    def to_dlpack(tensor): raise NotImplementedError("DLPack not supported")
    @staticmethod
    def from_dlpack(dlpack): raise NotImplementedError("DLPack not supported")

def _to_dlpack(tensor):
    raise NotImplementedError("DLPack not supported")

def _to_dlpack_versioned(tensor, version):
    raise NotImplementedError("DLPack not supported")

class _VariableFunctions:
    @staticmethod
    def rsub(tensor, other):
        import numpy as np
        if isinstance(other, (int, float, bool, complex, np.integer, np.floating)):
            from .._functional import mul as _mul
            result = _mul(tensor, -1)
            return result + other
        from .._functional import sub as _sub
        return _sub(other, tensor)

def _get_PyTorchFileReader():
    from .._stream import PyTorchFileReader
    return PyTorchFileReader

def _get_PyTorchFileWriter():
    from .._stream import PyTorchFileWriter
    return PyTorchFileWriter

def _get_privateuse1_backend_name():
    return "npu"

# ---------------------------------------------------------------------------
# NPU helpers
# ---------------------------------------------------------------------------
def _npu_probe_model_dirs():
    from .._backends.npu import runtime as _npu_runtime
    return _npu_runtime._probe_model_dirs()

def _npu_model_dir():
    from .._backends.npu import runtime as _npu_runtime
    return _npu_runtime._model_dir()

def _npu_aclnn_available():
    from .._backends.npu import aclnn as _aclnn
    return _aclnn.is_available()

def _npu_aclnn_symbols_ok():
    from .._backends.npu import aclnn as _aclnn
    return _aclnn.symbols_ok()

def _npu_aclnn_ones_zero_ok():
    from .._backends.npu import aclnn as _aclnn
    return _aclnn.ones_zero_symbols_ok()

def _npu_device_count():
    from .._backends.npu import runtime as _npu_runtime
    return _npu_runtime.device_count()

# ---------------------------------------------------------------------------
# Storage factory functions (will move to _C/_Storage.pyx in Phase C)
# ---------------------------------------------------------------------------
# Keep from _C.py lines 248-637 (storage factories, _install_typed_storage_compat, legacy aliases, __getattr__)
```

- [ ] **Step 3: Update candle/__init__.py import path**

`from . import _C` should now pick up `_C/__init__.py` (Python package) instead of `_C.py` (module). Delete `_C.py`.

```python
# candle/__init__.py line ~78 — keep as-is
from . import _C  # now imports _C/__init__.py
```

- [ ] **Step 4: Update all _cython → _C import paths (47 files)**

```bash
# Batch replace import paths in Python files
find src/candle/ -name "*.py" -exec sed -i 's/from \._cython/from ._C/g; s/from candle\._cython/from candle._C/g' {} +
find src/candle/ -name "*.pyx" -exec sed -i 's/from \._cython/from ._C/g; s/from candle\._cython/from candle._C/g' {} +
find src/candle/ -name "*.pxd" -exec sed -i 's/from \._cython/from ._C/g; s/from candle\._cython/from candle._C/g' {} +

# Update setup.py extension paths
sed -i 's/candle\._cython\./candle._C./g' setup.py
sed -i 's|src/candle/_cython/|src/candle/_C/|g' setup.py

# Update tests
find tests/ -name "*.py" -exec sed -i 's/from candle\._cython/from candle._C/g' {} +
```

- [ ] **Step 5: Update _functional.py lazy import paths**

```bash
sed -i 's/from \._cython\._functional_ops/from ._C._functional_ops/g' src/candle/_functional.py
sed -i 's/from \._cython\._fast_ops/from ._C._fast_ops/g' src/candle/_functional.py
```

- [ ] **Step 6: Verify no _cython references remain**

```bash
grep -rn "_cython" --include="*.py" --include="*.pyx" src/candle/ | grep -v __pycache__ | grep -v "/_C/"
```
Expected: no output (or only comments/docs referencing "Cython" generically)

- [ ] **Step 7: Compile Cython and test basic import**

```bash
cd src/candle/_C && python -c "import pyximport; pyximport.install()"
# Or use the project's build system
conda run -n candle311 python -c "import candle; print(candle.Tensor)"
```
Expected: `import candle` succeeds

- [ ] **Step 8: Run full test suite**

```bash
conda run -n candle311 python -m pytest tests/cpu/ tests/contract/ --tb=line
```
Expected: same 44 failures as before (bypasses tests)

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: move _cython/ → _C/, replace _C.py with _C/__init__.py"
```

---

### Task 2: Create _C/_TensorBase.pyx (Phase B)

**Files:**
- Create: `src/candle/_C/_TensorBase.pyx`
- Create: `src/candle/_C/_TensorBase.pxd`
- Modify: `src/candle/_C/__init__.py`
- Delete: `src/candle/_C/_tensor_api.pyx`

- [ ] **Step 1: Write _TensorBase.pxd header**

```cython
# _C/_TensorBase.pxd
from ._tensor_impl cimport TensorImpl

cdef class TensorBase(TensorImpl):
    pass
```

- [ ] **Step 2: Write _TensorBase.pyx class skeleton**

```cython
# cython: language_level=3, boundscheck=False, wraparound=False
# _C/_TensorBase.pyx
from ._tensor_impl cimport TensorImpl
from ._tensor_impl import TensorImpl as _TensorImpl_py, _StrideTuple, cy_init_tensor_fields

cdef class TensorBase(TensorImpl):
    """torch._C.TensorBase equivalent — all tensor methods live here."""

    _DEVICE_MAP = {"cpu": 0, "npu": 1, "cuda": 2, "mps": 3, "meta": 4}

    def __init__(self, storage, shape, stride, offset=0, requires_grad=False):
        cy_init_tensor_fields(
            self, storage, tuple(shape), _StrideTuple(stride),
            int(offset), bool(requires_grad),
            None, None, None, None, False, False, None, 0, None,
        )

    # === Core properties ===
    # ... (all methods from _C.py TensorBase class, lines 669-1190)
    # ... (all methods from _tensor_api.pyx converted to class methods)
```

- [ ] **Step 3: Migrate _C.py TensorBase methods into _TensorBase.pyx**

Take ALL methods from `_C.py` class `TensorBase(TensorImpl):` (lines 669-1190, everything between the class definition and `_TensorBase = TensorBase`). Convert module-level references (`_compute_strides`, `_bf16_to_f32`, `_f32_to_bf16`) to either inline or lazy imports. Replace `from .storage import` with `from candle.storage import`. Replace `from .utils.hooks import` with `from candle.utils.hooks import`.

Key pattern for all methods — use lazy imports:
```cython
cdef class TensorBase(TensorImpl):
    def is_contiguous(self, memory_format=None):
        from candle._C import _compute_strides
        expected = _compute_strides(self.shape)
        return self.stride == expected
```

- [ ] **Step 4: Migrate _tensor_api.pyx standalone functions into _TensorBase.pyx class methods**

For each `def tensor_xxx(self, ...)` in `_tensor_api.pyx`:
- Rename to `def xxx(self, ...)`
- Move into `cdef class TensorBase(TensorImpl):` body
- Keep all internal helper calls (`_ensure_functional_refs()`, `_ensure_npu_refs()`, etc.)

```cython
# Before (_tensor_api.pyx):
def tensor_add(self, other):
    _ensure_functional_refs()
    return _add_fn(self, other)

# After (_TensorBase.pyx, inside TensorBase class):
def __add__(self, other):
    _ensure_functional_refs()
    return _add_fn(self, other)
```

Note: `_tensor_api.pyx` has 314 functions. The mapping from function name to method name follows a pattern:
- `tensor_add` → `__add__`
- `tensor_sub` → `__sub__`
- `tensor_mul` → `__mul__`
- `tensor_neg` → `__neg__` (also `neg`)
- `tensor_clone` → `clone`
- `tensor_detach` → `detach`
- `tensor_to` → `to`
- `tensor_backward` → `backward`
- `tensor_add_` → `add_`
- `tensor_mul_` → `mul_`
- `tensor_add_method` → `add`
- `tensor_sub_method` → `sub`
- etc.

The full mapping is determined by the `_install_tensor_api()` assignments in `_C.py` lines 1196-1493.

- [ ] **Step 5: Delete _tensor_api.pyx and remove _install_tensor_api()**

```bash
rm src/candle/_C/_tensor_api.pyx
```

Remove `_install_tensor_api()` function and its call from `_C/__init__.py` and `candle/__init__.py`.

- [ ] **Step 6: Update _C/__init__.py import**

```python
# _C/__init__.py — update to import from _TensorBase
from ._TensorBase import TensorBase, _TensorBase
```

- [ ] **Step 7: Compile and test**

```bash
conda run -n candle311 python -m pytest tests/cpu/ tests/contract/ --tb=line
```
Expected: same result as before

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: create _C/_TensorBase.pyx, absorb _tensor_api.pyx methods"
```

---

### Task 3: Create _C/_Storage.pyx (Phase C)

**Files:**
- Create: `src/candle/_C/_Storage.pyx`
- Modify: `src/candle/_C/__init__.py`

- [ ] **Step 1: Write _Storage.pyx**

Move StorageBase + all backend storage classes + factory functions from `_C/__init__.py` into `_C/_Storage.pyx`. This includes:

- `StorageBase(StorageImpl)` — inherits from `_C._storage_impl.StorageImpl`
- `_NPUUntypedStorage`
- `_MPSUntypedStorage`
- `_MetaUntypedStorage`
- `PendingStorage`
- All factory functions: `typed_storage_from_numpy`, `empty_cpu_typed_storage`, `meta_typed_storage_from_shape`, `npu_typed_storage_from_ptr`, `mps_typed_storage_from_numpy`, `cuda_typed_storage_from_numpy`, `pinned_cpu_typed_storage_from_numpy`, etc.
- `_install_typed_storage_compat()`
- Legacy storage aliases (`FloatStorage`, etc.)
- `__getattr__` for lazy storage access

```cython
# _C/_Storage.pyx
from ._storage_impl cimport StorageImpl

cdef class StorageBase(StorageImpl):
    """torch._C.StorageBase equivalent."""
    # ... all storage methods
```

- [ ] **Step 2: Update _C/__init__.py**

```python
from ._Storage import StorageBase, _install_typed_storage_compat
```

- [ ] **Step 3: Compile and test**

```bash
conda run -n candle311 python -m pytest tests/cpu/ tests/contract/ --tb=line
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: create _C/_Storage.pyx, move storage classes from _C/__init__.py"
```

---

### Task 4: Cleanup and final verification (Phase D)

**Files:**
- Modify: `src/candle/_C/__init__.py`
- Delete: old files no longer needed

- [ ] **Step 1: Remove _install_tensor_api() call from candle/__init__.py**

```bash
sed -i '/_install_tensor_api/d' src/candle/__init__.py
```

- [ ] **Step 2: Final grep for _cython references**

```bash
grep -rn "_cython" --include="*.py" --include="*.pyx" --include="*.pxd" src/candle/ tests/ | grep -v __pycache__ | grep -v "/_C/" | grep -v "\.pyc"
```
Expected: no output (or only generic comments)

- [ ] **Step 3: Run pylint**

```bash
conda run -n candle311 pylint src/candle/ --rcfile=.github/pylint.conf
```

- [ ] **Step 4: Run full test suite**

```bash
conda run -n candle311 python -m pytest tests/cpu/ tests/contract/ --tb=line
```
Expected: 2850+ passed, 44 failed (bypasses tests)

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: cleanup, remove _install_tensor_api, final _C/ structure"
```
