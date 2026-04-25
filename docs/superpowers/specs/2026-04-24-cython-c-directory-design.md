# Design: Cython `_C/` Directory — torch C++ Layer Alignment

## 1. Objective

Replace `_cython/` + `_C.py` with `_C/` directory that directly mirrors torch's C++ source structure. All Cython `.pyx` files live under `_C/`, and `_C/__init__.py` serves as the `torch._C` equivalent (thin stub layer).

## 2. Target Structure

```
src/candle/_C/                        torch csrc equivalent
├── __init__.py                       torch._C stubs + re-exports
├── _tensor_impl.pxd/pxx             TensorImpl (internal metadata)
├── _TensorBase.pxd/pxx              TensorBase(TensorImpl) ← NEW, absorbs _tensor_api.pyx
├── _storage_impl.pxd/pxx            StorageImpl
├── _Storage.pyx                     StorageBase + backend storage ← NEW, absorbs _C.py storage
├── _device.pxd/pxx                  Device
├── _dtype.pxd/pxx                   Dtype
├── _autograd_node.pyx               AutogradNode
├── _autograd_engine.pyx             AutogradEngine
├── _autograd_function.pyx           AutogradFunction
├── _autograd_graph.pyx              AutogradGraph
├── _autograd_ops.pyx                AutogradOps
├── _dispatcher_core.pyx             DispatchCore
├── _dispatch.pyx                    Dispatch
├── _functional_ops.pyx              Fast functional ops
├── _fast_ops.pyx                    Fast ops
├── _stream.pyx                      Stream
├── _storage.pyx                     CyCPUUntypedStorage etc.
├── _cpu_kernels.pyx                 CPU kernels
├── _future.pyx                      Future
├── _allocator.pyx                   Allocator
├── _dataloader_ops.pyx              DataLoader ops
├── _mps_compute.pyx                 MPS compute shaders
├── _mps_helpers.pyx                 MPS helpers
├── _mps_ops.pyx                     MPS ops
├── _npu_ops.pyx                     NPU ops
├── _npu_storage.pyx                 NPU storage
├── _aclgraph.pyx                    Ascend ACL graph
├── _aclnn_ffi.pyx                   Ascend ACLNN FFI
└── _aclrt_ffi.pxd/pxx              Ascend ACL runtime FFI
```

## 3. File Changes

### 3.1 NEW: `_C/_TensorBase.pyx`

Merges two sources into one `cdef class TensorBase(TensorImpl)`:

**Source A — `_C.py` TensorBase methods (~168 methods):**
- `__init__`, `data`, `__delattr__`, `untyped_storage`, `storage`, `data_ptr`
- `ndim`, `is_floating_point`, `is_complex`, `is_contiguous`, `contiguous`
- `_numpy_view`, `reshape`, `view`, `flatten`, `transpose`, `t`, `T`
- `set_`, `as_strided`, `_ones_like`, `numpy`, `backward`
- `pin_memory`, `is_pinned`, `retain_grad`, `requires_grad_`
- `detach`, `detach_`, `register_hook`, `_is_view`, `_check_inplace`
- All `_` (in-place) methods: `add_`, `mul_`, `relu_`, `zero_`, `fill_`, `copy_`, `abs_`, `neg_`, ...
- All dunder methods: `__add__`, `__sub__`, `__mul__`, `__matmul__`, `__neg__`, `__rsub__`, ...
- All dispatch methods: `add`, `sub`, `mul`, `div`, `matmul`, `pow`, `sum`, `mean`, ...
- New tensor factories: `new_empty`, `new_zeros`, `new_ones`, `new_full`, `new_tensor`
- Device movement: `cpu`, `npu`, `mps`, `cuda`, `to`, `_to_dtype`
- `__repr__`, `__str__`, `__len__`, `__iter__`, `__hash__`

**Source B — `_cython/_tensor_api.pyx` functions (314 functions):**
- Each `def tensor_xxx(self, ...)` becomes `def xxx(self, ...)` on TensorBase
- Module-level `_ensure_functional_refs()` calls stay as-is inside methods
- `_tensor_api.pyx` deleted after migration

**Pattern:**
```cython
# Before (tensor_api.pyx standalone + monkey-patch)
def tensor_add(self, other):
    _ensure_functional_refs()
    return _add_fn(self, other)
# Then in _install_tensor_api(): TensorBase.__add__ = tensor_add

# After (TensorBase method)
cdef class TensorBase(TensorImpl):
    def __add__(self, other):
        _ensure_functional_refs()
        return _add_fn(self, other)
```

**Imports:** Methods use lazy `from candle._functional import ...` inside method bodies (same pattern as current `__richcmp__` in `_tensor_impl.pyx`).

### 3.2 NEW: `_C/_Storage.pyx`

Absorbs from `_C.py`:
- `StorageBase(StorageImpl)` — inherits from Cython StorageImpl
- `_NPUUntypedStorage`, `_MPSUntypedStorage`, `_MetaUntypedStorage`
- `PendingStorage`
- All factory functions: `typed_storage_from_numpy`, `empty_cpu_typed_storage`, etc.
- `_install_typed_storage_compat()` (or fold directly into class)

### 3.3 `_C/__init__.py` (replaces `_C.py`)

Thin stub layer — only what torch._C exposes as Python stubs:
```python
# Re-exports from Cython
from ._TensorBase import TensorBase, _TensorBase
from ._tensor_impl import TensorImpl
from ._Storage import StorageBase
from ._stream import PyTorchFileReader, PyTorchFileWriter

# Torch-compat stubs (not in C++ but needed by _tensor.py)
def _add_docstr(obj, docstr): ...
class _disabled_torch_dispatch_impl: ...
class DisableTorchFunctionSubclass: ...
def _has_storage(tensor): ...
def _get_tracing_state(): ...
class _VariableFunctions: ...
class _dlpack_exchange_api: ...
def _get_privateuse(): ...
# Legacy storage aliases (FloatStorage, etc.)

# NPU helpers
def _npu_probe_model_dirs(): ...
def _npu_device_count(): ...
```

### 3.4 DELETED

- `_cython/` entire directory (moved into `_C/`)
- `_C.py` (split into `_C/__init__.py` + `_C/_TensorBase.pyx` + `_C/_Storage.pyx`)
- `_cython/_tensor_api.pyx` (absorbed into `_C/_TensorBase.pyx`)
- `_install_tensor_api()` (methods directly on class)

### 3.5 UPDATED imports

47 files import from `_cython`. Pattern change:
```python
# Before
from ._cython._tensor_impl import cy_make_tensor_from_storage
from candle._cython._tensor_api import tensor_add

# After
from ._C._tensor_impl import cy_make_tensor_from_storage
# tensor_add no longer needed — it's TensorBase.__add__
```

`_cython/__init__.py` → `_C/__init__.py` (the `_HAS_CYTHON_*` flags and imports).

## 4. Implementation Phases

### Phase A: Scaffold `_C/` directory
1. `mv _cython _C` — move entire directory
2. Rename `_C.py` → `_C/__init__old.py` (keep for reference)
3. Write new `_C/__init__.py` with stub layer
4. Update `__init__.py`: `from . import _C` still works (now imports `_C/__init__.py`)

### Phase B: TensorBase consolidation
1. Create `_C/_TensorBase.pyx` with `cdef class TensorBase(TensorImpl)`
2. Migrate methods from `_C.py` TensorBase → `_TensorBase.pyx`
3. Migrate functions from `_tensor_api.pyx` → `_TensorBase.pyx` methods
4. Delete `_tensor_api.pyx`
5. Remove `_install_tensor_api()` call from `__init__.py`

### Phase C: Storage consolidation
1. Create `_C/_Storage.pyx` with StorageBase + backend classes
2. Migrate storage code from `_C.py` → `_Storage.pyx`
3. `_install_typed_storage_compat()` → direct class methods

### Phase D: Cleanup
1. Remove old `_C.py` / `_C/__init__old.py`
2. Update all import paths across codebase (47 files)
3. Compile and test

## 5. Class Hierarchy (Final State)

```
Cython (compiled):
  TensorImpl               (_C/_tensor_impl.pyx)
    └─ TensorBase          (_C/_TensorBase.pyx) ← all methods here
  StorageImpl              (_C/_storage_impl.pyx)
    └─ StorageBase         (_C/_Storage.pyx)
  Node, Edge, ...          (_C/_autograd_node.pyx)

Python:
  _C.TensorBase(TensorBase)  (_C/__init__.py) ← thin stub overrides only
    └─ Tensor               (_tensor.py, torch copy)
  _C.StorageBase(StorageBase)
    └─ UntypedStorage, TypedStorage  (storage.py, torch copy)
```

## 6. Risk Mitigation

- **Phase A is no-op for behavior** — pure file move, verify tests unchanged
- **Phase B is additive** — old `_tensor_api.pyx` kept until new methods verified
- **47 import paths to update** — scripted `sed` replacement, verified by grep
- **Cython compilation** — each `.pyx` change requires recompilation; work in batches
