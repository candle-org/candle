# Tensor / Storage Batch A2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move version-sensitive and view-sensitive runtime truth out of the Python `Tensor` shell and into the Cython `TensorImpl`, while staying strictly inside the Tensor / Storage subsystem.

**Architecture:** Implement A2 in two bounded phases. A2a converges version/detach/set_-related runtime truth into `src/candle/_cython/_tensor_impl.pyx`, reducing Python shell ownership for version-sensitive behavior. A2b then converges common view base/version attachment truth into the same Cython runtime center, while intentionally leaving some user-facing `_view_meta` payload assembly in the Python shell for now. The batch still does not touch dispatcher, autograd runtime, or backend code.

**Tech Stack:** Python, Cython, pytest, Candle Tensor/Storage runtime

---

## File Structure / Responsibilities

- `src/candle/_tensor.py` — public Tensor shell; should continue shrinking as a runtime owner for version/view-sensitive semantics
- `src/candle/_cython/_tensor_impl.pxd` — authoritative declaration of Tensor runtime-owned fields and callable Cython interfaces
- `src/candle/_cython/_tensor_impl.pyx` — runtime truth center for version, base/view linkage, and view-derived tensor creation
- `tests/contract/test_tensor_alias_version_contract.py` — primary correctness rail for alias/version/view semantics in A2
- `tests/contract/test_inplace_view_rules.py` — guardrail for view/inplace legality after view-truth migration
- `tests/contract/test_tensor_storage_owner_contract.py` — ownership boundary guardrail added in A1, should remain green throughout A2
- `tests/contract/test_storage_contract.py` — storage shell/runtime boundary regression guardrail
- `docs/superpowers/specs/2026-04-14-tensor-storage-batch-a2-design.md` — approved A2 design and scope boundary

---

### Task 1: Add A2a-focused failing tests for detach/set_ runtime truth

**Files:**
- Modify: `tests/contract/test_tensor_alias_version_contract.py`
- Reference: `src/candle/_tensor.py:562-572`
- Reference: `src/candle/_tensor.py:652-661`
- Reference: `src/candle/_cython/_tensor_impl.pyx:212-241,555-619`

- [ ] **Step 1: Add focused A2a tests to the existing alias/version contract file**

Append these tests to `tests/contract/test_tensor_alias_version_contract.py`:

```python
def test_detach_preserves_storage_shape_stride_offset_runtime_truth():
    base = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    view = base.view((4,))
    detached = view.detach()

    assert detached.storage().data_ptr() == view.storage().data_ptr()
    assert detached.shape == view.shape
    assert detached.stride == view.stride
    assert detached.offset == view.offset
    assert detached.requires_grad is False
    assert detached.grad_fn is None


def test_detach_of_view_preserves_base_version_sharing_truth():
    base = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    view = base.view((4,))
    detached = view.detach()

    before = detached._version_counter.value
    base._version_counter.bump()
    assert detached._version_counter.value == before + 1


def test_set_preserves_device_dtype_runtime_truth():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    out = x.set_(x.storage(), 1, (2,), (1,))

    assert out is x
    assert x.device.type == "cpu"
    assert x.dtype == torch.float32
    assert x.tolist() == [1.0, 2.0]
```

- [ ] **Step 2: Run the three new tests to confirm the current baseline**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_alias_version_contract.py::test_detach_preserves_storage_shape_stride_offset_runtime_truth \
  tests/contract/test_tensor_alias_version_contract.py::test_detach_of_view_preserves_base_version_sharing_truth \
  tests/contract/test_tensor_alias_version_contract.py::test_set_preserves_device_dtype_runtime_truth \
  -v --tb=short
```

Expected: PASS or narrow A2a-relevant failures only.

- [ ] **Step 3: Run the existing A2a primary rails as the no-regression baseline**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_alias_version_contract.py::test_detach_shares_version_counter_with_source \
  tests/contract/test_tensor_alias_version_contract.py::test_set_bumps_version_counter_once \
  tests/contract/test_tensor_alias_version_contract.py::test_set_on_view_bumps_shared_version_counter_once \
  tests/contract/test_tensor_alias_version_contract.py::test_setitem_bumps_version_counter \
  tests/contract/test_tensor_alias_version_contract.py::test_dispatch_setitem_bumps_version_counter_exactly_once \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 4: Commit the A2a-focused test additions**

```bash
git add tests/contract/test_tensor_alias_version_contract.py
git commit -m "test(contract): add A2a tensor runtime truth rails"
```

---

### Task 2: Move detach runtime truth into TensorImpl helper(s)

**Files:**
- Modify: `src/candle/_cython/_tensor_impl.pxd`
- Modify: `src/candle/_cython/_tensor_impl.pyx`
- Modify: `src/candle/_tensor.py:652-661`
- Test: `tests/contract/test_tensor_alias_version_contract.py`

- [ ] **Step 1: Add a dedicated Cython detach helper declaration**

In `src/candle/_cython/_tensor_impl.pxd`, add a Cython API for detach-derived tensor construction:

```cython
cdef class TensorImpl:
    cpdef object cy_detach(self)
```

- [ ] **Step 2: Run the current detach-focused tests before implementation**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_alias_version_contract.py::test_detach_shares_version_counter_with_source \
  tests/contract/test_tensor_alias_version_contract.py::test_detach_preserves_storage_shape_stride_offset_runtime_truth \
  tests/contract/test_tensor_alias_version_contract.py::test_detach_of_view_preserves_base_version_sharing_truth \
  -v --tb=short
```

Expected: PASS before the refactor.

- [ ] **Step 3: Implement `cy_detach()` as the Cython fact source**

In `src/candle/_cython/_tensor_impl.pyx`, add:

```cython
    cpdef object cy_detach(self):
        cdef object tensor_type = type(self)
        cdef TensorImpl out = <TensorImpl>tensor_type.__new__(tensor_type)
        cy_init_tensor_fields(
            out,
            self._storage,
            self._shape_tuple,
            self._stride_tuple,
            self._c_offset,
            False,
            None,
            None,
            None,
            None,
            self._pending,
            False,
            None,
            self._version_value,
            self._version_counter,
        )
        return out
```

Keep the helper narrow: same storage/shape/stride/offset/runtime cache truth, `requires_grad=False`, `grad=None`, `grad_fn=None`, shared version-counter truth via `_version_counter`.

- [ ] **Step 4: Make Python `detach()` a thin shell over `cy_detach()`**

In `src/candle/_tensor.py`, replace the current detach body with:

```python
    def detach(self):
        return self.cy_detach()
```

Do not keep Python-side reconstruction logic in place.

- [ ] **Step 5: Re-run the detach-focused tests**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_alias_version_contract.py::test_detach_shares_version_counter_with_source \
  tests/contract/test_tensor_alias_version_contract.py::test_detach_preserves_storage_shape_stride_offset_runtime_truth \
  tests/contract/test_tensor_alias_version_contract.py::test_detach_of_view_preserves_base_version_sharing_truth \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit the detach migration**

```bash
git add src/candle/_cython/_tensor_impl.pxd src/candle/_cython/_tensor_impl.pyx src/candle/_tensor.py tests/contract/test_tensor_alias_version_contract.py
git commit -m "refactor(tensor): move detach runtime truth into TensorImpl"
```

---

### Task 3: Move set_-related runtime truth into TensorImpl helper(s)

**Files:**
- Modify: `src/candle/_cython/_tensor_impl.pxd`
- Modify: `src/candle/_cython/_tensor_impl.pyx`
- Modify: `src/candle/_tensor.py:540-573`
- Test: `tests/contract/test_tensor_alias_version_contract.py`

- [ ] **Step 1: Add a dedicated Cython helper declaration for `set_` runtime mutation**

In `src/candle/_cython/_tensor_impl.pxd`, add:

```cython
cdef class TensorImpl:
    cpdef object cy_set_runtime_truth(
        self,
        object typed_storage,
        tuple size,
        object stride,
        int64_t storage_offset,
    )
```

- [ ] **Step 2: Run the set_-focused tests before implementation**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_alias_version_contract.py::test_set_bumps_version_counter_once \
  tests/contract/test_tensor_alias_version_contract.py::test_set_on_view_bumps_shared_version_counter_once \
  tests/contract/test_tensor_alias_version_contract.py::test_set_preserves_device_dtype_runtime_truth \
  -v --tb=short
```

Expected: PASS before the refactor.

- [ ] **Step 3: Implement `cy_set_runtime_truth()` in Cython**

In `src/candle/_cython/_tensor_impl.pyx`, add:

```cython
    cpdef object cy_set_runtime_truth(
        self,
        object typed_storage,
        tuple size,
        object stride,
        int64_t storage_offset,
    ):
        cdef object device = getattr(typed_storage, "device", None)
        cdef object dtype = getattr(typed_storage, "dtype", None)
        self._storage = typed_storage
        self._set_shape(size)
        self._set_stride(stride)
        self._c_offset = storage_offset
        if device is not None:
            self._set_device_from_obj(device)
        if dtype is not None:
            self._set_dtype_from_obj(dtype)
        self._bump_version()
        return self
```

Keep ownership-local logic here only. Do not move argument validation into Cython in this task.

- [ ] **Step 4: Make Python `set_()` a thin shell once validation is done**

In `src/candle/_tensor.py`, keep the current Python-side argument checks and bounds checks, but replace the post-validation runtime mutation block with:

```python
        self._check_inplace()
        return self.cy_set_runtime_truth(
            typed_storage,
            size,
            _StrideTuple(stride),
            int(storage_offset),
        )
```

Remove the old direct assignments to `_storage`, `shape`, `stride`, `offset`, `_set_device_from_storage`, `_set_dtype_from_storage`, and `_bump_version()` from the Python shell.

- [ ] **Step 5: Re-run the set_-focused tests**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_alias_version_contract.py::test_set_bumps_version_counter_once \
  tests/contract/test_tensor_alias_version_contract.py::test_set_on_view_bumps_shared_version_counter_once \
  tests/contract/test_tensor_alias_version_contract.py::test_set_preserves_device_dtype_runtime_truth \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit the set_ runtime-truth migration**

```bash
git add src/candle/_cython/_tensor_impl.pxd src/candle/_cython/_tensor_impl.pyx src/candle/_tensor.py tests/contract/test_tensor_alias_version_contract.py
git commit -m "refactor(tensor): move set_ runtime truth into TensorImpl"
```

---

### Task 4: Add A2b-focused view metadata contract tests

**Files:**
- Modify: `tests/contract/test_tensor_alias_version_contract.py`
- Modify: `tests/contract/test_inplace_view_rules.py`
- Reference: `src/candle/_tensor.py:311-388,414-451`
- Reference: `src/candle/_cython/_tensor_impl.pyx:468-549`

- [ ] **Step 1: Add focused `_base` / `_view_meta` tests**

Append these tests to `tests/contract/test_tensor_alias_version_contract.py`:

```python
def test_view_runtime_truth_sets_base_to_source_tensor():
    base = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    view = base.view((4,))
    assert view._base is base


def test_as_strided_runtime_truth_sets_base_to_source_tensor():
    base = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    view = base.as_strided((2, 2), (2, 1))
    assert view._base is base


def test_view_runtime_truth_records_view_meta():
    base = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    view = base.view((4,))
    assert view._view_meta is not None
    assert view._view_meta["op"] == "view"
```

- [ ] **Step 2: Run the new focused A2b tests**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_alias_version_contract.py::test_view_runtime_truth_sets_base_to_source_tensor \
  tests/contract/test_tensor_alias_version_contract.py::test_as_strided_runtime_truth_sets_base_to_source_tensor \
  tests/contract/test_tensor_alias_version_contract.py::test_view_runtime_truth_records_view_meta \
  -v --tb=short
```

Expected: PASS or narrow A2b-relevant failure only.

- [ ] **Step 3: Run the existing A2b primary rails as baseline**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_alias_version_contract.py::test_as_strided_view_shares_version_counter_with_source \
  tests/contract/test_tensor_alias_version_contract.py::test_diagonal_view_shares_version_counter_with_source \
  tests/contract/test_tensor_alias_version_contract.py::test_movedim_view_shares_version_counter_with_source \
  tests/contract/test_tensor_alias_version_contract.py::test_moveaxis_view_shares_version_counter_with_source \
  tests/contract/test_tensor_alias_version_contract.py::test_expand_view_shares_version_counter_with_source \
  tests/contract/test_tensor_alias_version_contract.py::test_broadcast_to_view_shares_version_counter_with_source \
  tests/contract/test_tensor_alias_version_contract.py::test_split_view_shares_version_counter_with_source \
  tests/contract/test_tensor_alias_version_contract.py::test_chunk_view_shares_version_counter_with_source \
  tests/contract/test_tensor_alias_version_contract.py::test_unary_inplace_preserves_view_aliasing \
  tests/contract/test_inplace_view_rules.py \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 4: Commit the A2b-focused test additions**

```bash
git add tests/contract/test_tensor_alias_version_contract.py tests/contract/test_inplace_view_rules.py
git commit -m "test(contract): add A2b view runtime truth rails"
```

---

### Task 5: Move common view truth into TensorImpl view constructors

**Files:**
- Modify: `src/candle/_cython/_tensor_impl.pyx`
- Modify: `src/candle/_tensor.py:311-388,414-451`
- Test: `tests/contract/test_tensor_alias_version_contract.py`
- Test: `tests/contract/test_inplace_view_rules.py`

- [ ] **Step 1: Add a Cython helper for common view attachment truth**

In `src/candle/_cython/_tensor_impl.pyx`, add a small internal helper near the view constructors:

```cython
cdef inline void _attach_view_runtime_truth(TensorImpl base, TensorImpl view):
    view._version_value = base._version_value
    view._base = base._base if base._base is not None else base
    view._vc_proxy = None
    view._view_meta = None
    view._pending = False
    view._retain_grad = False
    view._backward_hooks = None
    view._output_nr = 0
```
```

- [ ] **Step 2: Apply the helper inside `cy_view()` and `cy_as_strided()`**

Replace the repeated block in both functions with:

```cython
        _attach_view_runtime_truth(self, v)
```

Retain the existing storage/device/dtype/dispatch/grad field propagation.

- [ ] **Step 3: Keep Python view shells only for high-level metadata augmentation**

In `src/candle/_tensor.py`, leave user-facing view/transpose shells in place, but ensure they no longer try to establish base/version truth that is already set by `cy_view()` / `cy_as_strided()` / `cy_transpose()`.

The Python shell may still attach user-facing `_view_meta["op"]` detail where needed, but it should not overwrite the Cython-owned `_base` / version-sharing truth. Full `_view_meta` payload construction does not need to move into Cython in this task.

- [ ] **Step 4: Run the new A2b tests and the primary rails**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_alias_version_contract.py::test_view_runtime_truth_sets_base_to_source_tensor \
  tests/contract/test_tensor_alias_version_contract.py::test_as_strided_runtime_truth_sets_base_to_source_tensor \
  tests/contract/test_tensor_alias_version_contract.py::test_view_runtime_truth_records_view_meta \
  tests/contract/test_tensor_alias_version_contract.py::test_as_strided_view_shares_version_counter_with_source \
  tests/contract/test_tensor_alias_version_contract.py::test_unary_inplace_preserves_view_aliasing \
  tests/contract/test_inplace_view_rules.py \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit the view-truth migration**

```bash
git add src/candle/_cython/_tensor_impl.pyx src/candle/_tensor.py tests/contract/test_tensor_alias_version_contract.py tests/contract/test_inplace_view_rules.py
git commit -m "refactor(tensor): centralize view runtime truth in TensorImpl"
```

---

### Task 6: Run the full A2 boundary suite

**Files:**
- No code changes required unless failures are found.

- [ ] **Step 1: Run the A2 total rails**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_alias_version_contract.py \
  tests/contract/test_inplace_view_rules.py \
  tests/contract/test_tensor_storage_owner_contract.py \
  tests/contract/test_storage_contract.py \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 2: If failures are found, fix only Tensor/Storage-local regressions**

Allowed files for fixes:

```text
src/candle/_tensor.py
src/candle/_cython/_tensor_impl.pyx
src/candle/_cython/_tensor_impl.pxd
src/candle/_storage.py
src/candle/_cython/_storage.pyx
tests/contract/test_tensor_alias_version_contract.py
tests/contract/test_inplace_view_rules.py
tests/contract/test_tensor_storage_owner_contract.py
tests/contract/test_storage_contract.py
```

Do not fix A2 failures by changing dispatcher, autograd, or backend files.

- [ ] **Step 3: Re-run the full A2 boundary suite after any fix**

Run the same command again.
Expected: PASS.

- [ ] **Step 4: Commit the A2 verification gate if any follow-up fixes were needed**

```bash
git add src/candle/_tensor.py src/candle/_cython/_tensor_impl.pyx src/candle/_cython/_tensor_impl.pxd src/candle/_storage.py src/candle/_cython/_storage.pyx tests/contract/test_tensor_alias_version_contract.py tests/contract/test_inplace_view_rules.py tests/contract/test_tensor_storage_owner_contract.py tests/contract/test_storage_contract.py
git commit -m "test(contract): verify tensor storage A2 boundary suite"
```

If no fixes were needed, skip this commit.

---

## Self-review

### Spec coverage

- A2 stays strictly inside Tensor/Storage: all tasks are confined to `_tensor.py`, `_tensor_impl.pyx/.pxd`, and contract tests
- A2a covers version/detach/set_-related runtime truth: Tasks 1-3
- A2b covers `_base` / `_view_meta` / view-family truth: Tasks 4-5
- Existing contract rails remain the primary safety net: every task reuses `test_tensor_alias_version_contract.py` and `test_inplace_view_rules.py`
- A2 total validation is explicit: Task 6

### Placeholder scan

- No TBD/TODO placeholders remain in tasks
- Every test/implementation step contains exact code or exact commands
- No task says "similar to above" or relies on undefined helper names outside the plan

### Type consistency

- New Cython methods are named consistently: `cy_detach`, `cy_set_runtime_truth`
- New internal helper name is consistent: `_attach_view_runtime_truth`
- Tests consistently refer to `_base`, `_view_meta`, `_version_counter`, `detach`, and `set_`

---

Plan complete and saved to `docs/superpowers/plans/2026-04-14-tensor-storage-batch-a2.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
