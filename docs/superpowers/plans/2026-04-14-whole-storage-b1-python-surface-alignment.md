# Whole Storage Batch B1 Python Surface Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align Candle’s Python storage surface to `torch/storage.py` while keeping the whole-storage-mechanism architecture in view, removing private CPU storage implementation classes and bookkeeping helpers from `candle._storage` without breaking serialization, multiprocessing, or built-in CPU/CUDA/NPU storage entry points.

**Architecture:** This is the first implementation batch derived from the whole-storage-mechanism v2 master spec. It focuses on the Python storage boundary only: `candle._storage` should look more like a public shell, while private CPU implementation classes and bookkeeping helpers move behind the runtime/helper boundary in `_cython/_storage.pyx`. The batch also preserves the built-in device-storage story by keeping CPU/CUDA/NPU public factories available and by keeping multiprocessing/serialization behavior working through internal imports rather than public-shell leakage.

**Tech Stack:** Python, Cython, pytest, Candle Storage runtime

---

## File Structure / Responsibilities

- `src/candle/_storage.py` — public Python storage shell; after B1 it should expose storage API and factories, not private CPU implementation classes or bookkeeping helpers
- `src/candle/_cython/_storage.pyx` — private storage helper/runtime layer; receives the moved CPU untyped storage classes and bookkeeping helpers
- `src/candle/serialization.py` — internal storage rebuild consumer; must stop importing private CPU storage classes from `candle._storage`
- `src/candle/multiprocessing/reductions.py` — internal reduction/rebuild consumer; must stop importing private CPU storage classes from `candle._storage`
- `src/candle/multiprocessing/__init__.py` — public multiprocessing surface; may continue to expose `cleanup_shared_files()` / `shared_files_count()` while importing them from the private helper layer
- `tests/contract/test_storage_contract.py` — primary storage public-surface regression rail
- `tests/contract/test_tensor_storage_owner_contract.py` — owner-boundary regression rail for storage-related Tensor behavior
- `docs/superpowers/specs/2026-04-14-whole-storage-mechanism-realignment-master-design.md` — v2 master spec that defines the whole storage target

---

### Task 1: Add whole-storage B1 public-surface rails

**Files:**
- Modify: `tests/contract/test_storage_contract.py`
- Modify: `tests/contract/test_tensor_storage_owner_contract.py`
- Reference: `src/candle/_storage.py:1-220`
- Reference: `src/candle/multiprocessing/__init__.py:1-55`

- [ ] **Step 1: Add non-exposure and public-entry rails to the storage contract file**

Append these tests to `tests/contract/test_storage_contract.py`:

```python
import candle._storage as candle_storage
```

```python
def test_storage_module_does_not_expose_shared_file_bookkeeping_helpers():
    helper_names = {
        "_register_shared_file",
        "_unregister_shared_file",
        "_cleanup_shared_file",
        "_close_fd_and_cleanup",
        "cleanup_shared_files",
        "shared_files_count",
    }

    exported = {name for name in helper_names if hasattr(candle_storage, name)}

    assert exported == set()
```

```python
def test_storage_module_still_exposes_public_storage_entry_points():
    public_names = {
        "TypedStorage",
        "UntypedStorage",
        "typed_storage_from_numpy",
        "empty_cpu_typed_storage",
        "meta_typed_storage_from_shape",
        "npu_typed_storage_from_ptr",
        "pinned_cpu_typed_storage_from_numpy",
    }

    exported = {name for name in public_names if hasattr(candle_storage, name)}

    assert exported == public_names
```

```python
def test_storage_module_does_not_expose_private_untyped_storage_classes():
    assert not hasattr(candle_storage, "_PinnedCPUUntypedStorage")
    assert not hasattr(candle_storage, "_CPUUntypedStorage")
```

```python
def test_multiprocessing_storage_bookkeeping_surface_still_exists():
    import candle.multiprocessing as candle_mp

    assert callable(candle_mp.cleanup_shared_files)
    assert callable(candle_mp.shared_files_count)
```

- [ ] **Step 2: Add the narrow pinned-factory availability rail**

Append this test to `tests/contract/test_tensor_storage_owner_contract.py`:

```python
def test_public_pinned_storage_factory_still_exists():
    import candle._storage as candle_storage

    assert hasattr(candle_storage, "pinned_cpu_typed_storage_from_numpy")
```

- [ ] **Step 3: Run the new B1 rails to verify the baseline**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && PYTHONPATH=/home/dndx/lvyufeng/candle/.worktrees/runtime-rebuild-a3/src \
  conda run --no-capture-output -n candle python -m pytest \
  tests/contract/test_storage_contract.py::test_storage_module_does_not_expose_shared_file_bookkeeping_helpers \
  tests/contract/test_storage_contract.py::test_storage_module_still_exposes_public_storage_entry_points \
  tests/contract/test_storage_contract.py::test_storage_module_does_not_expose_private_untyped_storage_classes \
  tests/contract/test_storage_contract.py::test_multiprocessing_storage_bookkeeping_surface_still_exists \
  tests/contract/test_tensor_storage_owner_contract.py::test_public_pinned_storage_factory_still_exists \
  -v --tb=short
```

Expected:
- the public-entry and multiprocessing-surface rails pass
- the helper/class non-exposure rails fail before implementation

- [ ] **Step 4: Commit the B1 failing rails**

```bash
git add tests/contract/test_storage_contract.py tests/contract/test_tensor_storage_owner_contract.py
git commit -m "test(storage): add B1 whole-storage surface rails"
```

---

### Task 2: Move CPU untyped implementation classes behind the helper layer

**Files:**
- Modify: `src/candle/_storage.py`
- Modify: `src/candle/_cython/_storage.pyx`
- Modify: `src/candle/serialization.py`
- Modify: `src/candle/multiprocessing/reductions.py`
- Test: `tests/contract/test_storage_contract.py`
- Test: `tests/contract/test_tensor_storage_owner_contract.py`

- [ ] **Step 1: Add the CPU untyped storage helper classes to `_cython/_storage.pyx`**

At the top of `src/candle/_cython/_storage.pyx`, add the imports needed for the moved CPU helper classes:

```python
import ctypes
import mmap
import os
import tempfile
import threading
import weakref

import numpy as np
```

Then add these helper classes below the shared-file bookkeeping helpers section:

```python
class CyPinnedCPUUntypedStorage:
    def __init__(self, array, ptr, filename=None, shared=False, device=None):
        from candle._device import device as Device
        from candle._backends.npu import runtime as npu_runtime

        self.device = device or Device("cpu")
        self._array = array
        self._shared = shared
        self._filename = filename
        self._ptr = int(ptr)
        self._finalizer = weakref.finalize(self, npu_runtime.free_host, self._ptr)

    def nbytes(self):
        return int(self._array.nbytes)

    def data_ptr(self):
        return int(self._array.ctypes.data)

    def buffer(self):
        return self._array

    def resize_(self, new_nbytes):
        raise NotImplementedError("pinned storage cannot resize")

    def share_memory_(self):
        self._shared = True
        return self

    def is_shared(self):
        return self._shared

    def is_pinned(self):
        return True

    @classmethod
    def from_file(cls, filename, shared=False):
        data = np.memmap(filename, mode="r+", dtype=np.uint8)
        return cls(data, int(data.ctypes.data), filename=filename, shared=shared)

    def filename(self):
        return self._filename
```

```python
class CyCPUUntypedStorage:
    def __init__(
        self,
        array,
        filename=None,
        shared=False,
        device=None,
        mmap_obj=None,
        fd=None,
        tmp_file=None,
        sharing_mechanism=None,
        cleanup_finalizer=None,
    ):
        from candle._device import device as Device

        self.device = device or Device("cpu")
        self._array = array
        self._shared = shared
        self._filename = filename
        self._mmap = mmap_obj
        self._fd = fd
        self._tmp_file = tmp_file
        self._sharing_mechanism = sharing_mechanism
        self._cleanup_finalizer = cleanup_finalizer

    def nbytes(self):
        return int(self._array.nbytes)

    def data_ptr(self):
        return int(self._array.ctypes.data)

    def buffer(self):
        return self._array

    def resize_(self, new_nbytes):
        if self._filename is not None or self._shared:
            raise RuntimeError("Trying to resize storage that is not resizable")
        new_array = np.empty(int(new_nbytes), dtype=np.uint8)
        old_bytes = self._array.view(np.uint8)
        copy_bytes = min(old_bytes.size, new_array.size)
        new_array[:copy_bytes] = old_bytes[:copy_bytes]
        self._array = new_array
        return self

    def share_memory_(self, strategy="file_descriptor"):
        if self._shared:
            return self

        nbytes = int(self._array.nbytes)
        if nbytes == 0:
            self._shared = True
            self._sharing_mechanism = strategy
            return self

        if strategy == "file_descriptor":
            fd, filename = tempfile.mkstemp(prefix="candle_fd_", suffix=".bin")
            try:
                os.ftruncate(fd, nbytes)
                mm = mmap.mmap(fd, nbytes)
            except Exception:
                os.close(fd)
                raise

            dst = np.frombuffer(mm, dtype=np.uint8, count=nbytes)
            dst[:] = self._array.view(np.uint8).reshape(-1)

            self._array = dst
            self._mmap = mm
            self._fd = fd
            self._tmp_file = filename
            self._filename = filename
            self._shared = True
            self._sharing_mechanism = "file_descriptor"

            if self._cleanup_finalizer is not None:
                self._cleanup_finalizer.detach()
            self._cleanup_finalizer = weakref.finalize(
                self,
                cy_cleanup_shared_resource,
                int(fd),
                filename,
            )
            return self

        if strategy == "file_system":
            fd, filename = tempfile.mkstemp(prefix="candle_shm_", suffix=".bin")
            try:
                with open(fd, "wb", closefd=False) as f:
                    f.truncate(nbytes)
            finally:
                os.close(fd)

            mmap_arr = np.memmap(filename, mode="r+", dtype=np.uint8, shape=(nbytes,))
            mmap_arr[:] = self._array.view(np.uint8).reshape(-1)

            self._array = mmap_arr
            self._filename = filename
            self._shared = True
            self._sharing_mechanism = "file_system"
            cy_register_shared_file(filename)

            if self._cleanup_finalizer is not None:
                self._cleanup_finalizer.detach()
            self._cleanup_finalizer = None
            return self

        raise ValueError(f"unsupported sharing strategy: {strategy}")

    def is_shared(self):
        return self._shared

    def shared_memory_meta(self):
        if self._sharing_mechanism == "file_descriptor":
            return {
                "mechanism": "file_descriptor",
                "fd": int(self._fd),
                "filename": self._filename,
                "nbytes": int(self._array.nbytes),
            }
        if self._sharing_mechanism == "file_system":
            return {
                "mechanism": "file_system",
                "filename": self._filename,
                "nbytes": int(self._array.nbytes),
            }
        return None

    def typed_view(self, dtype, size):
        from candle._dtype import to_numpy_dtype
        return np.frombuffer(self._array, dtype=to_numpy_dtype(dtype), count=int(size))

    @classmethod
    def from_shared_memory(cls, filename, nbytes):
        data = np.memmap(filename, mode="r+", dtype=np.uint8, shape=(int(nbytes),))
        cy_register_shared_file(filename)
        return cls(data, filename=filename, shared=True, sharing_mechanism="file_system")

    @classmethod
    def from_shared_fd(cls, fd, nbytes, filename=None):
        fd = int(fd)
        mm = mmap.mmap(fd, int(nbytes))
        arr = np.frombuffer(mm, dtype=np.uint8, count=int(nbytes))
        storage = cls(
            arr,
            filename=filename,
            shared=True,
            mmap_obj=mm,
            fd=fd,
            tmp_file=filename,
            sharing_mechanism="file_descriptor",
        )
        storage._cleanup_finalizer = weakref.finalize(
            storage,
            cy_cleanup_shared_resource,
            fd,
            filename,
        )
        return storage

    @classmethod
    def from_file(cls, filename, shared=False):
        data = np.memmap(filename, mode="r+", dtype=np.uint8)
        return cls(data, filename=filename, shared=shared)

    def filename(self):
        return self._filename
```

- [ ] **Step 2: Verify the new non-exposure rails fail before `_storage.py` is updated**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && PYTHONPATH=/home/dndx/lvyufeng/candle/.worktrees/runtime-rebuild-a3/src \
  conda run --no-capture-output -n candle python -m pytest \
  tests/contract/test_storage_contract.py::test_storage_module_does_not_expose_private_untyped_storage_classes \
  tests/contract/test_tensor_storage_owner_contract.py::test_public_pinned_storage_factory_still_exists \
  -v --tb=short
```

Expected:
- private-class non-exposure rail still FAILS before the `_storage.py` public-shell cleanup
- pinned factory rail PASSES

- [ ] **Step 3: Replace the private CPU class definitions in `_storage.py` with private Cython imports**

In `src/candle/_storage.py`, add these imports near the top:

```python
from ._cython._storage import (  # pylint: disable=import-error,no-name-in-module
    CyCPUUntypedStorage,
    CyPinnedCPUUntypedStorage,
    cy_cleanup_shared_files as _cy_cleanup_shared_files,
    cy_cleanup_shared_resource as _cy_cleanup_shared_resource,
    cy_register_shared_file as _cy_register_shared_file,
)
```

Then remove the entire definitions of:

```python
class _PinnedCPUUntypedStorage(UntypedStorage):
    ...

class _CPUUntypedStorage(UntypedStorage):
    ...
```

Replace all remaining private uses in `_storage.py` as follows:

```python
return CyCPUUntypedStorage(self.buffer()[index], filename=None, shared=self._shared, device=self.device)
```

```python
return CyCPUUntypedStorage.from_file(filename, shared=shared)
```

```python
untyped = CyPinnedCPUUntypedStorage(raw, host_ptr, device=device)
```

```python
untyped = CyCPUUntypedStorage(arr.view(np.uint8), device=device)
```

- [ ] **Step 4: Update internal consumers to import the private CPU helper class from `_cython._storage`**

In `src/candle/serialization.py`, replace:

```python
from ._storage import TypedStorage, _CPUUntypedStorage, typed_storage_from_numpy
```

with:

```python
from ._cython._storage import CyCPUUntypedStorage  # pylint: disable=import-error,no-name-in-module
from ._storage import TypedStorage, typed_storage_from_numpy
```

And replace the call sites:

```python
_CPUUntypedStorage.from_shared_memory(...)
_CPUUntypedStorage.from_shared_fd(...)
```

with:

```python
CyCPUUntypedStorage.from_shared_memory(...)
CyCPUUntypedStorage.from_shared_fd(...)
```

In `src/candle/multiprocessing/reductions.py`, replace:

```python
from .._storage import _CPUUntypedStorage, TypedStorage
```

with:

```python
from .._cython._storage import CyCPUUntypedStorage  # pylint: disable=import-error,no-name-in-module
from .._storage import TypedStorage
```

And replace these uses:

```python
return _CPUUntypedStorage.from_shared_memory(...)
return _CPUUntypedStorage.from_shared_fd(...)
if isinstance(untyped, _CPUUntypedStorage):
reduction.register(_CPUUntypedStorage, _reduce_cpu_storage)
```

with:

```python
return CyCPUUntypedStorage.from_shared_memory(...)
return CyCPUUntypedStorage.from_shared_fd(...)
if isinstance(untyped, CyCPUUntypedStorage):
reduction.register(CyCPUUntypedStorage, _reduce_cpu_storage)
```

- [ ] **Step 5: Rebuild extensions and run the focused B1 rails**

Run:
```bash
cd /home/dndx/lvyufeng/candle/.worktrees/runtime-rebuild-a3 && \
source /opt/miniconda3/etc/profile.d/conda.sh && \
PYTHONPATH=/home/dndx/lvyufeng/candle/.worktrees/runtime-rebuild-a3/src \
conda run --no-capture-output -n candle python setup.py build_ext --inplace
```

Then run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && PYTHONPATH=/home/dndx/lvyufeng/candle/.worktrees/runtime-rebuild-a3/src \
  conda run --no-capture-output -n candle python -m pytest \
  tests/contract/test_storage_contract.py::test_storage_module_does_not_expose_private_untyped_storage_classes \
  tests/contract/test_tensor_storage_owner_contract.py::test_public_pinned_storage_factory_still_exists \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 6: Run one internal-consumer smoke check**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && PYTHONPATH=/home/dndx/lvyufeng/candle/.worktrees/runtime-rebuild-a3/src \
  conda run --no-capture-output -n candle python - <<'PY'
import candle.serialization
import candle.multiprocessing.reductions
print("ok")
PY
```

Expected: prints `ok`.

- [ ] **Step 7: Commit the CPU private-class downshift**

```bash
git add src/candle/_storage.py src/candle/_cython/_storage.pyx src/candle/serialization.py src/candle/multiprocessing/reductions.py tests/contract/test_storage_contract.py tests/contract/test_tensor_storage_owner_contract.py
git commit -m "refactor(storage): move pinned host storage internals out of shell"
```

---

### Task 3: Preserve multiprocessing bookkeeping API while keeping `_storage.py` clean

**Files:**
- Modify: `src/candle/multiprocessing/__init__.py`
- Modify: `src/candle/_storage.py`
- Modify: `src/candle/_cython/_storage.pyx`
- Test: `tests/contract/test_storage_contract.py`

- [ ] **Step 1: Add a focused multiprocessing API rail if missing**

If not already present from Task 1, ensure `tests/contract/test_storage_contract.py` contains:

```python
def test_multiprocessing_storage_bookkeeping_surface_still_exists():
    import candle.multiprocessing as candle_mp

    assert callable(candle_mp.cleanup_shared_files)
    assert callable(candle_mp.shared_files_count)
```

- [ ] **Step 2: Keep multiprocessing importing bookkeeping from the private helper layer**

In `src/candle/multiprocessing/__init__.py`, ensure the import is:

```python
from .._cython._storage import (  # pylint: disable=import-error,no-name-in-module
    cy_cleanup_shared_files as _cleanup_shared_files,
    cy_shared_files_count as _shared_files_count,
)
```

and **not** from `.._storage`.

- [ ] **Step 3: Re-run the bookkeeping surface tests**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && PYTHONPATH=/home/dndx/lvyufeng/candle/.worktrees/runtime-rebuild-a3/src \
  conda run --no-capture-output -n candle python -m pytest \
  tests/contract/test_storage_contract.py::test_storage_module_does_not_expose_shared_file_bookkeeping_helpers \
  tests/contract/test_storage_contract.py::test_multiprocessing_storage_bookkeeping_surface_still_exists \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 4: Commit the multiprocessing surface preservation if it required any additional change**

```bash
git add src/candle/multiprocessing/__init__.py src/candle/_storage.py src/candle/_cython/_storage.pyx tests/contract/test_storage_contract.py
git commit -m "refactor(storage): preserve multiprocessing bookkeeping api"
```

If no code change was required in this task because Task 2 already landed the exact final form, skip this commit step.

---

### Task 4: Validate whole-storage B1 and preserve built-in storage surface

**Files:**
- Validate: `tests/contract/test_storage_contract.py`
- Validate: `tests/contract/test_tensor_storage_owner_contract.py`
- Validate: `src/candle/serialization.py`
- Validate: `src/candle/multiprocessing/reductions.py`
- Validate: `src/candle/multiprocessing/__init__.py`

- [ ] **Step 1: Run the full whole-storage B1 validation rails**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && PYTHONPATH=/home/dndx/lvyufeng/candle/.worktrees/runtime-rebuild-a3/src \
  conda run --no-capture-output -n candle python -m pytest \
  tests/contract/test_storage_contract.py \
  tests/contract/test_tensor_storage_owner_contract.py \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 2: Run focused import smoke checks for the moved private CPU helpers**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && PYTHONPATH=/home/dndx/lvyufeng/candle/.worktrees/runtime-rebuild-a3/src \
  conda run --no-capture-output -n candle python - <<'PY'
import candle.serialization
import candle.multiprocessing
import candle.multiprocessing.reductions
print("storage-b1-ok")
PY
```

Expected: prints `storage-b1-ok`.

- [ ] **Step 3: Review the final diff against the whole-storage B1 goal**

Manually verify the diff satisfies all of these checks:

```text
- src/candle/_storage.py is closer in role to torch/storage.py than before
- candle._storage no longer exposes shared-file bookkeeping helpers
- candle._storage no longer exposes _PinnedCPUUntypedStorage or _CPUUntypedStorage
- public storage factories and entry points remain available, including pinned_cpu_typed_storage_from_numpy and npu_typed_storage_from_ptr
- serialization and multiprocessing reductions no longer depend on importing private CPU storage classes from candle._storage
- candle.multiprocessing still exposes cleanup_shared_files() and shared_files_count()
- no dispatcher / autograd / backend scope creep was introduced
- NPU remains part of the built-in storage surface story, not a plugin-only afterthought
```

- [ ] **Step 4: Commit the validated whole-storage B1 batch**

```bash
git add src/candle/_storage.py src/candle/_cython/_storage.pyx src/candle/serialization.py src/candle/multiprocessing/reductions.py src/candle/multiprocessing/__init__.py tests/contract/test_storage_contract.py tests/contract/test_tensor_storage_owner_contract.py
git commit -m "refactor(storage): align python storage surface with torch"
```
