# Tensor / Storage Batch A3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `Tensor.data` shallow metadata-copy runtime truth out of the Python `Tensor` shell and into the Cython `TensorImpl`, aligning Candle more closely with PyTorch's `TensorImpl::shallow_copy_from()` ownership model while staying strictly inside the Tensor / Storage subsystem.

**Architecture:** Keep `Tensor.data` as a Python property for public API validation and user-facing errors, but introduce one narrow Cython helper that becomes the single fact source for storage / shape / stride / offset / device / dtype refresh and version-sensitive mutation on the `data` path. The batch is PyTorch-led, references torch_npu only as a runtime-ownership direction check, and preserves Candle’s long-term direction that NPU storage is built-in framework-owned device storage like CUDA rather than a plugin-style layer.

**Tech Stack:** Python, Cython, pytest, Candle Tensor/Storage runtime

---

## File Structure / Responsibilities

- `src/candle/_tensor.py` — public Tensor shell; after A3 it should only validate `Tensor.data` arguments and forward the actual runtime mutation to Cython
- `src/candle/_cython/_tensor_impl.pxd` — authoritative declaration of Tensor runtime-owned fields and callable Cython interfaces; declare the new A3 helper here
- `src/candle/_cython/_tensor_impl.pyx` — runtime truth center for tensor metadata mutation; implement the new `Tensor.data` shallow-copy helper here
- `tests/contract/test_tensor_storage_owner_contract.py` — ownership-boundary rails for proving `Tensor.data` now routes through a runtime-owned helper rather than Python field writes
- `tests/contract/test_tensor_alias_version_contract.py` — alias/version rails for proving version behavior and metadata-copy behavior stay stable after the migration
- `tests/contract/test_inplace_view_rules.py` — regression guardrail for view / inplace legality after the ownership move
- `tests/contract/test_storage_contract.py` — storage-shell regression guardrail
- `tests/npu/test_npu_fast_storage.py` — optional targeted NPU confidence rail if the current contract changes touch device-storage refresh behavior on NPU; do not expand A3 scope beyond a narrow smoke verification
- `docs/superpowers/specs/2026-04-14-tensor-storage-batch-a3-design.md` — approved A3 design and scope boundary

---

### Task 1: Add A3-focused failing tests for `Tensor.data` runtime truth

**Files:**
- Modify: `tests/contract/test_tensor_storage_owner_contract.py`
- Modify: `tests/contract/test_tensor_alias_version_contract.py`
- Reference: `src/candle/_tensor.py:199-215`
- Reference: `src/candle/_cython/_tensor_impl.pyx:569-587`

- [ ] **Step 1: Add focused `Tensor.data` ownership tests to the storage-owner contract file**

Append these tests to `tests/contract/test_tensor_storage_owner_contract.py`:

```python
def test_tensor_data_setter_preserves_source_storage_stride_and_offset_truth():
    x = torch.tensor([1.0, 2.0])
    z = torch.tensor([9.0, 3.0, 4.0])
    y = z[1:]

    x.data = y

    assert x.storage().data_ptr() == y.storage().data_ptr()
    assert x.stride == y.stride
    assert x.offset == y.offset
    assert x.tolist() == [3.0, 4.0]


def test_tensor_data_setter_preserves_runtime_device_dtype_caches():
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([3.0, 4.0], dtype=torch.float32)

    x.data = y

    assert x.device.type == y.device.type
    assert x.dtype == y.dtype
```

- [ ] **Step 2: Add focused version rail for the `Tensor.data` path**

Append this test to `tests/contract/test_tensor_alias_version_contract.py` near the other version-sensitive mutation tests:

```python
def test_data_setter_bumps_version_counter_once():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    before = x._version_counter.value

    x.data = y

    assert x._version_counter.value == before + 1
    assert x.tolist() == [3.0, 4.0]
```

- [ ] **Step 3: Run the new `Tensor.data` tests to verify the current baseline**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_storage_owner_contract.py::test_tensor_data_setter_preserves_source_storage_stride_and_offset_truth \
  tests/contract/test_tensor_storage_owner_contract.py::test_tensor_data_setter_preserves_runtime_device_dtype_caches \
  tests/contract/test_tensor_alias_version_contract.py::test_data_setter_bumps_version_counter_once \
  -v --tb=short
```

Expected: PASS or a narrow A3-relevant failure only.

- [ ] **Step 4: Run existing `Tensor.data` ownership baseline tests**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_storage_owner_contract.py::test_tensor_data_setter_still_routes_through_runtime_storage_swap \
  tests/contract/test_tensor_alias_version_contract.py::test_set_bumps_version_counter_once \
  tests/contract/test_tensor_alias_version_contract.py::test_set_preserves_device_dtype_runtime_truth \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit the A3 test additions**

```bash
git add tests/contract/test_tensor_storage_owner_contract.py tests/contract/test_tensor_alias_version_contract.py
git commit -m "test(contract): add A3 tensor data runtime truth rails"
```

---

### Task 2: Add a dedicated Cython helper for `Tensor.data` shallow metadata-copy truth

**Files:**
- Modify: `src/candle/_cython/_tensor_impl.pxd`
- Modify: `src/candle/_cython/_tensor_impl.pyx`
- Test: `tests/contract/test_tensor_storage_owner_contract.py`
- Test: `tests/contract/test_tensor_alias_version_contract.py`

- [ ] **Step 1: Declare the A3 helper in `_tensor_impl.pxd`**

Add this declaration to `src/candle/_cython/_tensor_impl.pxd` near the existing `cy_detach` / `cy_set_runtime_truth` declarations:

```cython
    cpdef object cy_set_data_runtime_truth_from(self, object other)
```

- [ ] **Step 2: Run the focused `Tensor.data` tests before implementation**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_storage_owner_contract.py::test_tensor_data_setter_still_routes_through_runtime_storage_swap \
  tests/contract/test_tensor_storage_owner_contract.py::test_tensor_data_setter_preserves_source_storage_stride_and_offset_truth \
  tests/contract/test_tensor_storage_owner_contract.py::test_tensor_data_setter_preserves_runtime_device_dtype_caches \
  tests/contract/test_tensor_alias_version_contract.py::test_data_setter_bumps_version_counter_once \
  -v --tb=short
```

Expected: PASS before the refactor.

- [ ] **Step 3: Implement the A3 helper in `_tensor_impl.pyx`**

Add this method to `src/candle/_cython/_tensor_impl.pyx` next to `cy_set_runtime_truth`:

```cython
    cpdef object cy_set_data_runtime_truth_from(self, object other):
        cdef TensorImpl src = <TensorImpl>other
        self._storage = src._storage
        self._set_shape(src._shape_tuple)
        self._set_stride(src._stride_tuple)
        self._c_offset = src._c_offset
        self._device_type = src._device_type
        self._device_index = src._device_index
        self._device_obj = src._device_obj
        self._dtype_code = src._dtype_code
        self._itemsize = src._itemsize
        self._dtype_obj = src._dtype_obj
        self._recompute_dispatch_keys()
        self._bump_version()
        return self
```

This helper is intentionally narrow:
- it shallow-copies storage / shape / stride / offset truth from the source tensor
- it refreshes runtime device / dtype cache fields from the source runtime owner
- it keeps the version bump rule in one Cython-owned place
- it does **not** redesign autograd behavior or view metadata behavior

- [ ] **Step 4: Re-run the focused `Tensor.data` tests after adding the helper**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_storage_owner_contract.py::test_tensor_data_setter_still_routes_through_runtime_storage_swap \
  tests/contract/test_tensor_storage_owner_contract.py::test_tensor_data_setter_preserves_source_storage_stride_and_offset_truth \
  tests/contract/test_tensor_storage_owner_contract.py::test_tensor_data_setter_preserves_runtime_device_dtype_caches \
  tests/contract/test_tensor_alias_version_contract.py::test_data_setter_bumps_version_counter_once \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit the Cython helper addition**

```bash
git add src/candle/_cython/_tensor_impl.pxd src/candle/_cython/_tensor_impl.pyx tests/contract/test_tensor_storage_owner_contract.py tests/contract/test_tensor_alias_version_contract.py
git commit -m "refactor(tensor): add A3 data runtime truth helper"
```

---

### Task 3: Make Python `Tensor.data` a thin shell over the Cython helper

**Files:**
- Modify: `src/candle/_tensor.py:199-215`
- Modify: `tests/contract/test_tensor_storage_owner_contract.py`
- Modify: `tests/contract/test_tensor_alias_version_contract.py`

- [ ] **Step 1: Replace Python-side field mutation in the `data` setter**

In `src/candle/_tensor.py`, replace the body after validation with this forwarding call:

```python
    @data.setter
    def data(self, new_data):
        """Replace the tensor's data with new_data (in-place)."""
        if not isinstance(new_data, Tensor):
            raise TypeError(f"data must be a Tensor, got {type(new_data).__name__}")
        if new_data.shape != self.shape:
            raise RuntimeError(f"shape mismatch: expected {self.shape}, got {new_data.shape}")
        if new_data.dtype != self.dtype:
            raise RuntimeError(f"dtype mismatch: expected {self.dtype}, got {new_data.dtype}")
        self.cy_set_data_runtime_truth_from(new_data)
```

Do **not** keep direct Python-side assignments to `_storage`, `stride`, `offset`, or `_bump_version()`.

- [ ] **Step 2: Add one regression test that the public validation still stays in Python**

Append this test to `tests/contract/test_tensor_storage_owner_contract.py`:

```python
def test_tensor_data_setter_rejects_non_tensor_input_before_runtime_mutation():
    x = torch.tensor([1.0, 2.0])

    with pytest.raises(TypeError, match=r"data must be a Tensor"):
        x.data = [3.0, 4.0]
```

Also add the import at the top of that file if needed:

```python
import pytest
```

- [ ] **Step 3: Run the `Tensor.data` contract tests after the Python-shell slimming**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_storage_owner_contract.py \
  tests/contract/test_tensor_alias_version_contract.py -k "data_setter or set_preserves_device_dtype_runtime_truth" \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 4: Commit the Python-shell forwarding change**

```bash
git add src/candle/_tensor.py tests/contract/test_tensor_storage_owner_contract.py tests/contract/test_tensor_alias_version_contract.py
git commit -m "refactor(tensor): route data setter through TensorImpl"
```

---

### Task 4: Run A3 validation rails and a narrow NPU confidence check

**Files:**
- Validate: `tests/contract/test_tensor_storage_owner_contract.py`
- Validate: `tests/contract/test_tensor_alias_version_contract.py`
- Validate: `tests/contract/test_inplace_view_rules.py`
- Validate: `tests/contract/test_storage_contract.py`
- Optional validate: `tests/npu/test_npu_fast_storage.py`

- [ ] **Step 1: Run the full A3 contract rails**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/contract/test_tensor_storage_owner_contract.py \
  tests/contract/test_tensor_alias_version_contract.py \
  tests/contract/test_inplace_view_rules.py \
  tests/contract/test_storage_contract.py \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 2: Run a narrow NPU storage confidence check if NPU hardware is available**

Run:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
  tests/npu/test_npu_fast_storage.py::test_typed_storage_interface \
  tests/npu/test_npu_fast_storage.py::test_storage_data_ptr_nonzero \
  -v --tb=short
```

Expected:
- PASS on an NPU-capable machine
- SKIP only if NPU is unavailable

This step is only a confidence rail. Do **not** widen A3 into backend redesign work if these tests expose unrelated NPU issues.

- [ ] **Step 3: Review the final diff against the A3 spec before committing**

Manually verify the diff satisfies all of these checks:

```text
- src/candle/_tensor.py no longer directly assigns _storage / stride / offset or calls _bump_version() in the data setter
- src/candle/_cython/_tensor_impl.pyx owns the single data-path runtime mutation helper
- version bump behavior for Tensor.data is single-sourced in Cython
- no dispatcher / autograd / backend files were modified
- NPU direction remains framework-owned built-in storage by architecture, without introducing any torch_npu-style plugin layer
```

- [ ] **Step 4: Commit the validated A3 batch**

```bash
git add src/candle/_tensor.py src/candle/_cython/_tensor_impl.pxd src/candle/_cython/_tensor_impl.pyx tests/contract/test_tensor_storage_owner_contract.py tests/contract/test_tensor_alias_version_contract.py
git commit -m "refactor(tensor): move data runtime truth into TensorImpl"
```
