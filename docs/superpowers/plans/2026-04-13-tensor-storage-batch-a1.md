# Tensor / Storage Batch A1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define and lightly enforce Tensor/Storage ownership boundaries so Python shell code stops expanding as a runtime owner before deeper semantic migration begins.

**Architecture:** Keep this batch intentionally narrow. Do not redesign behavior across dispatcher, autograd, or NPU. Instead, make ownership boundaries explicit in Tensor/Storage code, add small guardrail tests that lock those boundaries in, and reduce a few obvious Python-owned runtime responsibilities only where the move is local and low-risk. The steady-state direction is Python shell + Cython runtime core.

**Tech Stack:** Python, Cython, pytest, Candle Tensor/Storage runtime

---

## File Structure / Responsibilities

- `src/candle/_tensor.py` — public Tensor shell; should expose methods and compatibility glue, but should not keep growing as a runtime owner
- `src/candle/_cython/_tensor_impl.pxd` — Cython TensorImpl field declarations; this is the authoritative list of Tensor runtime-owned state
- `src/candle/_cython/_tensor_impl.pyx` — Cython TensorImpl runtime behavior and cached metadata ownership
- `src/candle/_storage.py` — Python storage shell and public storage API surface
- `src/candle/_cython/_storage.pyx` — Cython storage fast/runtime helpers
- `tests/contract/test_tensor_storage_owner_contract.py` — new focused contract tests for Tensor/Storage ownership boundaries in this batch
- `tests/contract/test_tensor_alias_version_contract.py` — existing alias/version guardrails to keep from regressing while tightening boundaries
- `docs/superpowers/specs/2026-04-13-tensor-storage-batch-a1-design.md` — approved scope boundary for this batch

---

### Task 1: Add contract tests that lock the A1 ownership boundary

**Files:**
- Create: `tests/contract/test_tensor_storage_owner_contract.py`
- Test: `tests/contract/test_tensor_alias_version_contract.py`
- Reference: `src/candle/_tensor.py`
- Reference: `src/candle/_cython/_tensor_impl.pyx`

- [ ] **Step 1: Write the failing ownership-boundary test file**

```python
import candle as torch


def test_tensor_impl_declares_runtime_owned_fields():
    x = torch.tensor([1.0, 2.0])
    impl = x
    for name in (
        "_storage",
        "_device_obj",
        "_dtype_obj",
        "_version_value",
        "_base",
        "_view_meta",
        "_dispatch_keys",
    ):
        assert hasattr(impl, name), name


def test_tensor_python_shell_still_exposes_public_storage_api():
    x = torch.tensor([1.0, 2.0])
    storage = x.storage()
    untyped = x.untyped_storage()
    assert storage is not None
    assert untyped is not None


def test_tensor_data_setter_still_routes_through_runtime_storage_swap():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    before = x._version_counter.value
    x.data = y
    assert x.tolist() == [3.0, 4.0]
    assert x._version_counter.value == before + 1


def test_storage_public_surface_still_exposes_data_ptr_and_untyped_storage():
    x = torch.tensor([1.0, 2.0])
    storage = x.storage()
    untyped = storage.untyped_storage()
    assert isinstance(storage.data_ptr(), int)
    assert isinstance(untyped.data_ptr(), int)
```

- [ ] **Step 2: Run the new contract file to establish the current baseline**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/contract/test_tensor_storage_owner_contract.py -v --tb=short
```

Expected: PASS or very small targeted failures. If it fails, the failures describe current boundary ambiguity that this batch will fix.

- [ ] **Step 3: Re-run the existing alias/version contract as a no-regression baseline**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/contract/test_tensor_alias_version_contract.py -v --tb=short
```

Expected: PASS. If this fails before any edits, stop and record the pre-existing failure rather than broadening A1.

- [ ] **Step 4: Commit the test-only boundary lock-in**

```bash
git add tests/contract/test_tensor_storage_owner_contract.py tests/contract/test_tensor_alias_version_contract.py
git commit -m "test(contract): lock tensor storage ownership boundary"
```

---

### Task 2: Make TensorImpl field ownership explicit in Cython declarations

**Files:**
- Modify: `src/candle/_cython/_tensor_impl.pxd`
- Modify: `src/candle/_cython/_tensor_impl.pyx`
- Test: `tests/contract/test_tensor_storage_owner_contract.py`

- [ ] **Step 1: Tighten the TensorImpl field comments in the `.pxd` to mark runtime-owned state explicitly**

Update the block comments in `src/candle/_cython/_tensor_impl.pxd` so the fields are grouped as runtime-owned categories, e.g.:

```cython
cdef class TensorImpl:
    # -- runtime-owned tensor metadata --
    cdef int64_t _c_shape[MAX_NDIM]
    cdef int64_t _c_stride[MAX_NDIM]
    cdef public int _ndim
    cdef public int64_t _c_numel
    cdef public int64_t _c_offset

    # -- runtime-owned device/dtype caches --
    cdef public int _device_type
    cdef public int _device_index
    cdef public object _device_obj
    cdef public int _dtype_code
    cdef public int _itemsize
    cdef public object _dtype_obj

    # -- runtime-owned storage and autograd attachment --
    cdef public object _storage
    cdef public int64_t _version_value
    cdef public object _base
    cdef public object _view_meta
    cdef public unsigned int _dispatch_keys
```

- [ ] **Step 2: Mirror that ownership grouping in `_tensor_impl.pyx` top-level comments**

Adjust the module/class comments near the top of `src/candle/_cython/_tensor_impl.pyx` so they explicitly say TensorImpl is the owner of tensor runtime metadata and cached runtime state, while Python `Tensor` is the public shell.

Use wording like:

```cython
"""Cython TensorImpl — runtime owner for Tensor metadata.

This module is the steady-state owner for tensor runtime fields such as
shape/stride/offset, cached device/dtype state, dispatch-key-relevant state,
version attachment, and base/view metadata attachment points. Public Python
Tensor methods should forward into this runtime rather than grow new owner
state in `_tensor.py`.
"""
```

- [ ] **Step 3: Run the ownership contract test again**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/contract/test_tensor_storage_owner_contract.py -v --tb=short
```

Expected: PASS.

- [ ] **Step 4: Commit the explicit TensorImpl owner declaration**

```bash
git add src/candle/_cython/_tensor_impl.pxd src/candle/_cython/_tensor_impl.pyx tests/contract/test_tensor_storage_owner_contract.py
git commit -m "refactor(tensor): declare TensorImpl runtime ownership boundary"
```

---

### Task 3: Reduce duplicated Python Tensor owner helpers to thin adapters

**Files:**
- Modify: `src/candle/_tensor.py:180-231`
- Modify: `src/candle/_cython/_tensor_impl.pyx:80-159`
- Test: `tests/contract/test_tensor_storage_owner_contract.py`

- [ ] **Step 1: Write a narrow adapter test for Tensor shell/device-dtype helpers**

Append this test to `tests/contract/test_tensor_storage_owner_contract.py`:

```python
def test_tensor_shell_device_dtype_helpers_preserve_runtime_cache_values():
    x = torch.tensor([1.0, 2.0])
    before_device = x.device
    before_dtype = x.dtype
    x._set_device_from_storage(before_device)
    x._set_dtype_from_storage(before_dtype)
    assert x.device == before_device
    assert x.dtype == before_dtype
```

- [ ] **Step 2: Run the single new test before changing code**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/contract/test_tensor_storage_owner_contract.py::test_tensor_shell_device_dtype_helpers_preserve_runtime_cache_values -v --tb=short
```

Expected: PASS. This protects behavior while we narrow the Python helpers.

- [ ] **Step 3: Change `_tensor.py` helper methods into explicit thin adapters**

In `src/candle/_tensor.py`, replace the current bodies of `_set_device_from_storage()` and `_set_dtype_from_storage()` with direct delegation to the Cython owner helpers:

```python
    def _set_device_from_storage(self, dev):
        self._set_device_from_obj(dev)

    def _set_dtype_from_storage(self, dtype):
        self._set_dtype_from_obj(dtype)
```

Do not leave duplicated Python-side device/dtype recomputation logic in place.

- [ ] **Step 4: Run the focused ownership test and the alias/version no-regression subset**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_storage_owner_contract.py::test_tensor_shell_device_dtype_helpers_preserve_runtime_cache_values \
  tests/contract/test_tensor_alias_version_contract.py::test_detach_shares_version_counter_with_source \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit the Python-shell thinning change**

```bash
git add src/candle/_tensor.py src/candle/_cython/_tensor_impl.pyx tests/contract/test_tensor_storage_owner_contract.py
git commit -m "refactor(tensor): narrow Python shell device dtype helpers"
```

---

### Task 4: Make storage shell/runtime ownership explicit without broad behavior change

**Files:**
- Modify: `src/candle/_storage.py`
- Modify: `src/candle/_cython/_storage.pyx`
- Test: `tests/contract/test_storage_contract.py`
- Test: `tests/contract/test_tensor_storage_owner_contract.py`

- [ ] **Step 1: Add a focused storage boundary test**

Append this test to `tests/contract/test_tensor_storage_owner_contract.py`:

```python
def test_typed_storage_public_api_routes_through_untyped_runtime_owner():
    x = torch.tensor([1.0, 2.0])
    storage = x.storage()
    untyped = storage.untyped_storage()
    assert storage.data_ptr() == untyped.data_ptr()
    assert storage.device == untyped.device
```

- [ ] **Step 2: Run the focused storage boundary test**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/contract/test_tensor_storage_owner_contract.py::test_typed_storage_public_api_routes_through_untyped_runtime_owner -v --tb=short
```

Expected: PASS.

- [ ] **Step 3: Clarify storage shell/runtime split in code comments, not behavior**

In `src/candle/_storage.py`, add or update comments/docstrings so that:

- `UntypedStorage` is explicitly described as the public storage shell
- typed/untyped storage pointer ownership facts are described as runtime truths
- Python storage API methods are described as wrappers around runtime-owned backing data

In `src/candle/_cython/_storage.pyx`, update the module docstring to clarify it is the storage runtime/helper layer, not merely an optimization layer.

Use wording like:

```cython
"""Cython storage helpers for runtime-owned storage operations.

This module exists to keep storage-critical runtime behavior out of the Python
API shell and to centralize low-level storage helpers that later phases can
extend without growing new Python owners.
"""
```

- [ ] **Step 4: Run the focused storage tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_storage_contract.py \
  tests/contract/test_tensor_storage_owner_contract.py::test_typed_storage_public_api_routes_through_untyped_runtime_owner \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit the storage ownership clarification**

```bash
git add src/candle/_storage.py src/candle/_cython/_storage.pyx tests/contract/test_storage_contract.py tests/contract/test_tensor_storage_owner_contract.py
git commit -m "refactor(storage): document shell versus runtime ownership"
```

---

### Task 5: Capture the A2 migration inventory directly in code-adjacent comments

**Files:**
- Modify: `src/candle/_tensor.py`
- Modify: `src/candle/_storage.py`
- Modify: `docs/superpowers/specs/2026-04-13-tensor-storage-batch-a1-design.md`

- [ ] **Step 1: Add narrow A2 migration comments in `_tensor.py`**

Near the remaining Python-owned runtime-sensitive methods in `src/candle/_tensor.py`, add concise comments like:

```python
    # A1 boundary note: this remains in the Python Tensor shell temporarily.
    # A2 should migrate version/alias/view-sensitive runtime behavior into
    # _cython/_tensor_impl.pyx rather than growing new shell ownership here.
```

Only add comments where they directly mark still-temporary ownership; do not annotate unrelated methods.

- [ ] **Step 2: Add narrow A2 migration comments in `_storage.py`**

For the storage shell sections that still expose low-level ownership-sensitive behavior, add comments like:

```python
# A1 boundary note: this file remains the public storage API shell.
# Future runtime-sensitive storage ownership should move into Cython helpers
# instead of expanding Python-side owner state here.
```

- [ ] **Step 3: Update the A1 spec file with the concrete A2 migration inventory**

Append this subsection to `docs/superpowers/specs/2026-04-13-tensor-storage-batch-a1-design.md`:

```markdown
## 13. Concrete A2 Migration Inventory

A2 should target the following runtime-sensitive responsibilities that remain temporary after A1:

- version-sensitive Python Tensor shell behavior that still reaches into runtime-owned state
- remaining alias/view boundary helpers that still live in `_tensor.py`
- any storage-shell logic that still behaves like a low-level owner rather than a public API wrapper
```

- [ ] **Step 4: Run the smallest ownership boundary suite one final time**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_storage_owner_contract.py \
  tests/contract/test_storage_contract.py \
  tests/contract/test_tensor_alias_version_contract.py::test_detach_shares_version_counter_with_source \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit the A2 handoff inventory**

```bash
git add src/candle/_tensor.py src/candle/_storage.py docs/superpowers/specs/2026-04-13-tensor-storage-batch-a1-design.md tests/contract/test_tensor_storage_owner_contract.py tests/contract/test_storage_contract.py
git commit -m "docs(runtime): record tensor storage A2 migration inventory"
```

---

## Self-review

### Spec coverage

- A1 requires explicit Tensor/Storage ownership boundaries: covered by Tasks 1-4
- A1 requires no cross-subsystem edits: enforced by file scope in every task
- A1 requires a migration inventory for A2: covered by Task 5
- A1 requires narrow validation only: each task uses focused contract tests only

### Placeholder scan

- No TBD/TODO placeholders remain in tasks
- Each task contains exact file paths and exact commands
- Code-bearing steps include concrete code blocks

### Type consistency

- Tensor helper names match current code (`_set_device_from_storage`, `_set_dtype_from_storage`, `_set_device_from_obj`, `_set_dtype_from_obj`)
- Test file name is introduced once and used consistently: `tests/contract/test_tensor_storage_owner_contract.py`
- A2 references are consistent with the parent spec and batch design

---

Plan complete and saved to `docs/superpowers/plans/2026-04-13-tensor-storage-batch-a1.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
