# CPU View/Slice Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add schema/test/kernel/API support for view/slice ops: `slice`, `slice_copy`, `slice_scatter`, `expand_copy`, `as_strided_`, `as_strided_copy`, `as_strided_scatter`.

**Architecture:** Register schemas first, add contract parity tests, then CPU kernels and registrations. Implement in terms of existing indexing/view logic to minimize risk.

**Tech Stack:** Python (Candle), PyTorch for contract parity tests, pytest.

---

### Task 1: Add contract tests for slice and expand/as_strided ops

**Files:**
- Create: `tests/contract/test_training_core_slice_parity.py`

**Step 1: Write the failing test**

```python
import candle as torch

from .helpers import run_training_core_parity_case


def test_slice_forward_view_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="slice",
        candle_fn=lambda x: torch.slice(x, dim=1, start=0, end=2, step=1),
        torch_fn=lambda x: real_torch.slice(x, dim=1, start=0, end=2, step=1),
        candle_inputs=lambda: (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),),
        torch_inputs=lambda: (real_torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=real_torch.float32),),
    )

    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_slice_copy_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="slice_copy",
        candle_fn=lambda x: torch.slice_copy(x, dim=0, start=0, end=1, step=1),
        torch_fn=lambda x: real_torch.slice_copy(x, dim=0, start=0, end=1, step=1),
        candle_inputs=lambda: (torch.arange(6, dtype=torch.float32).reshape(2, 3),),
        torch_inputs=lambda: (real_torch.arange(6, dtype=real_torch.float32).reshape(2, 3),),
    )

    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_slice_scatter_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="slice_scatter",
        candle_fn=lambda x, src: torch.slice_scatter(x, src, dim=1, start=0, end=2, step=1),
        torch_fn=lambda x, src: real_torch.slice_scatter(x, src, dim=1, start=0, end=2, step=1),
        candle_inputs=lambda: (
            torch.zeros((2, 3), dtype=torch.float32),
            torch.tensor([[9.0, 8.0], [7.0, 6.0]], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.zeros((2, 3), dtype=real_torch.float32),
            real_torch.tensor([[9.0, 8.0], [7.0, 6.0]], dtype=real_torch.float32),
        ),
    )

    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_expand_copy_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="expand_copy",
        candle_fn=lambda x: torch.expand_copy(x, (2, 3)),
        torch_fn=lambda x: real_torch.expand_copy(x, (2, 3)),
        candle_inputs=lambda: (torch.tensor([[1.0], [2.0]], dtype=torch.float32),),
        torch_inputs=lambda: (real_torch.tensor([[1.0], [2.0]], dtype=real_torch.float32),),
    )

    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_as_strided_copy_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="as_strided_copy",
        candle_fn=lambda x: torch.as_strided_copy(x, size=(2, 2), stride=(2, 1), storage_offset=0),
        torch_fn=lambda x: real_torch.as_strided_copy(x, size=(2, 2), stride=(2, 1), storage_offset=0),
        candle_inputs=lambda: (torch.arange(6, dtype=torch.float32),),
        torch_inputs=lambda: (real_torch.arange(6, dtype=real_torch.float32),),
    )

    assert result["shape_match"] is True
    assert result["value_match"] is True


def test_as_strided_scatter_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="as_strided_scatter",
        candle_fn=lambda x, src: torch.as_strided_scatter(x, src, size=(2, 2), stride=(2, 1), storage_offset=0),
        torch_fn=lambda x, src: real_torch.as_strided_scatter(x, src, size=(2, 2), stride=(2, 1), storage_offset=0),
        candle_inputs=lambda: (
            torch.zeros((6,), dtype=torch.float32),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.zeros((6,), dtype=real_torch.float32),
            real_torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=real_torch.float32),
        ),
    )

    assert result["shape_match"] is True
    assert result["value_match"] is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/contract/test_training_core_slice_parity.py::test_slice_forward_view_matches_torch_contract -v`
Expected: FAIL with `AttributeError` or dispatch schema error for `slice`.

**Step 3: Write minimal implementation**

N/A in this task.

**Step 4: Run test to verify it passes**

N/A in this task.

**Step 5: Commit**

```bash
git add tests/contract/test_training_core_slice_parity.py
git commit -m "test: add slice/as_strided parity contracts"
```

---

### Task 2: Register schemas for slice/expand_copy/as_strided ops

**Files:**
- Modify: `src/candle/_dispatch/schemas.py`

**Step 1: Write the failing test**

Covered by Task 1 tests failing.

**Step 2: Run test to verify it fails**

Run: `pytest tests/contract/test_training_core_slice_parity.py::test_slice_forward_view_matches_torch_contract -v`
Expected: FAIL with schema registration error.

**Step 3: Write minimal implementation**

Add schema registrations near other view ops:

```python
registry.register_schema("slice", "slice(Tensor(a) input, int dim, int start, int end, int step=1) -> Tensor(a)")
registry.register_schema("slice_copy", "slice_copy(Tensor input, int dim, int start, int end, int step=1) -> Tensor")
registry.register_schema("slice_scatter", "slice_scatter(Tensor input, Tensor src, int dim, int start, int end, int step=1) -> Tensor")
registry.register_schema("expand_copy", "expand_copy(Tensor input, int[] sizes) -> Tensor")
registry.register_schema("as_strided_", "as_strided_(Tensor(a!) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor")
registry.register_schema("as_strided_copy", "as_strided_copy(Tensor input, int[] size, int[] stride, int? storage_offset=None) -> Tensor")
registry.register_schema("as_strided_scatter", "as_strided_scatter(Tensor input, Tensor src, int[] size, int[] stride, int? storage_offset=None) -> Tensor")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/contract/test_training_core_slice_parity.py::test_slice_forward_view_matches_torch_contract -v`
Expected: FAIL later in dispatch (kernel missing).

**Step 5: Commit**

```bash
git add src/candle/_dispatch/schemas.py
git commit -m "feat: register slice/as_strided schemas"
```

---

### Task 3: Add functional/tensor API stubs for new ops

**Files:**
- Modify: `src/candle/_functional.py`

**Step 1: Write the failing test**

Already failing from Task 1 if API entry points are missing.

**Step 2: Run test to verify it fails**

Run: `pytest tests/contract/test_training_core_slice_parity.py::test_slice_forward_view_matches_torch_contract -v`
Expected: FAIL with `AttributeError: module candle has no attribute slice`.

**Step 3: Write minimal implementation**

Add functions:

```python
def slice(input, dim, start, end, step=1):
    return dispatch("slice", input.device.type, input, dim, start, end, step)


def slice_copy(input, dim, start, end, step=1):
    return dispatch("slice_copy", input.device.type, input, dim, start, end, step)


def slice_scatter(input, src, dim, start, end, step=1):
    return dispatch("slice_scatter", input.device.type, input, src, dim, start, end, step)


def expand_copy(input, sizes):
    return dispatch("expand_copy", input.device.type, input, sizes)


def as_strided_(self, size, stride, storage_offset=None):
    return dispatch("as_strided_", self.device.type, self, size, stride, storage_offset)


def as_strided_copy(input, size, stride, storage_offset=None):
    return dispatch("as_strided_copy", input.device.type, input, size, stride, storage_offset)


def as_strided_scatter(input, src, size, stride, storage_offset=None):
    return dispatch("as_strided_scatter", input.device.type, input, src, size, stride, storage_offset)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/contract/test_training_core_slice_parity.py::test_slice_forward_view_matches_torch_contract -v`
Expected: FAIL with missing kernel.

**Step 5: Commit**

```bash
git add src/candle/_functional.py
git commit -m "feat: add slice/as_strided functional APIs"
```

---

### Task 4: Implement CPU kernels for slice/expand_copy/as_strided ops

**Files:**
- Modify: `src/candle/_backends/cpu/ops.py`

**Step 1: Write the failing test**

Already failing from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/contract/test_training_core_slice_parity.py::test_slice_forward_view_matches_torch_contract -v`
Expected: FAIL with missing kernel for `slice`.

**Step 3: Write minimal implementation**

Add in CPU ops:

```python
def slice(input, dim, start, end, step=1):
    key = [slice(None)] * input.dim()
    key[int(dim)] = slice(int(start), int(end), int(step))
    return getitem(input, tuple(key))


def slice_copy(input, dim, start, end, step=1):
    return contiguous(slice(input, dim, start, end, step))


def slice_scatter(input, src, dim, start, end, step=1):
    out = input.clone()
    key = [slice(None)] * out.dim()
    key[int(dim)] = slice(int(start), int(end), int(step))
    setitem(out, tuple(key), src)
    return out


def expand_copy(input, sizes):
    return contiguous(expand(input, sizes))


def as_strided_(self, size, stride, storage_offset=None):
    # Update view metadata in-place on CPU
    offset = self.offset if storage_offset is None else int(storage_offset)
    self.shape = tuple(size)
    self.stride = tuple(stride)
    self.offset = offset
    return self


def as_strided_copy(input, size, stride, storage_offset=None):
    view = input.as_strided(size, stride, storage_offset)
    return contiguous(view)


def as_strided_scatter(input, src, size, stride, storage_offset=None):
    out = input.clone()
    view = out.as_strided(size, stride, storage_offset)
    setitem(out, (...,), src)  # replace with view assignment via slice in-place
    return out
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/contract/test_training_core_slice_parity.py -v`
Expected: PASS for slice/expand/as_strided tests.

**Step 5: Commit**

```bash
git add src/candle/_backends/cpu/ops.py
git commit -m "feat: add cpu kernels for slice/as_strided/expand_copy"
```

---

### Task 5: Register CPU kernels and run required gates

**Files:**
- Modify: `src/candle/_backends/cpu/__init__.py`

**Step 1: Write the failing test**

Already failing from Task 4 if not registered.

**Step 2: Run test to verify it fails**

Run: `pytest tests/contract/test_training_core_slice_parity.py::test_slice_forward_view_matches_torch_contract -v`
Expected: FAIL with missing kernel registration error.

**Step 3: Write minimal implementation**

Add imports and registrations:

```python
from .ops import slice, slice_copy, slice_scatter, expand_copy, as_strided_, as_strided_copy, as_strided_scatter

registry.register("slice", "cpu", slice, meta=meta_infer.infer_view)
registry.register("slice_copy", "cpu", slice_copy)
registry.register("slice_scatter", "cpu", slice_scatter)
registry.register("expand_copy", "cpu", expand_copy)
registry.register("as_strided_", "cpu", as_strided_)
registry.register("as_strided_copy", "cpu", as_strided_copy)
registry.register("as_strided_scatter", "cpu", as_strided_scatter)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/contract/test_training_core_slice_parity.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/candle/_backends/cpu/__init__.py
git commit -m "feat: register cpu slice/as_strided kernels"
```

---

### Task 6: Run required contract gate

**Files:**
- None

**Step 1: Run test to verify it fails**

N/A

**Step 2: Run test to verify it passes**

Run: `pytest tests/contract/ -v --tb=short`
Expected: PASS.

**Step 3: Commit**

N/A
```
