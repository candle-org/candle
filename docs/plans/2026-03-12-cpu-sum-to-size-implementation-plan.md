# sum_to_size Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `sum_to_size` with full PyTorch error-message parity on CPU, including schema validation, kernel, meta inference, autograd, and expanded contract tests.

**Architecture:** Implement size validation in schema layer, CPU kernel in ops, meta inference for shapes, and autograd backward that expands grad to input shape. Expand contract tests to diff against real torch errors.

**Tech Stack:** Python, pytest, candle dispatch/autograd/meta frameworks.

---

### Task 1: Expand contract tests for full error-message parity

**Files:**
- Modify: `tests/contract/test_training_core_sum_to_size_parity.py`

**Step 1: Write the failing test**

Add tests that compare candle vs torch for:
- top-level invalid sizes: `True`, `1.5`, `"1"`, `None`
- first element invalid: `("1", 3)`, `(1.5, 3)`, `(None, 3)`, `([1], 3)`
- later element invalid: `(1, "1")`, `(1, 1.5)`, `(1, None)`, `(1, [3])`, `(1, True, "a")`
- bool element first vs later (`(True, 3)` invalid, `(1, True)` valid)
- empty list accepted
- size mismatch errors for `[0, 3]`, `[-1, 3]`, `[1, 2, 3]`, `[2, 2]`
- view behavior when size equals input shape (storage sharing)

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_sum_to_size_parity.py -v`
Expected: FAIL with missing kernel / schema validation errors.

---

### Task 2: Add schema validation for sum_to_size size argument

**Files:**
- Modify: `src/candle/_dispatch/schema.py`

**Step 1: Write the failing test**

No new test; use expanded contract test.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_sum_to_size_parity.py::test_sum_to_size_type_error_matches_torch_contract -v`
Expected: FAIL with mismatched error messages.

**Step 3: Write minimal implementation**

Add `_validate_sum_to_size_size` inside `_validate_types`:
- If `size` is bool/float/str/None → `TypeError` with `sum_to_size(): argument 'size' (position 1) must be tuple of ints, not <type>`.
- If `size` is list/tuple:
  - empty sequence → accept
  - first element invalid (`not int` or `bool` or `None` or list/tuple/str/float) → `TypeError` "must be tuple of ints, but found element of type <type> at pos 0"
  - later invalid → `TypeError` "failed to unpack the object at pos <idx+1> with error \"type must be tuple of ints,but got <type>\"".
  - treat bool elements as ints except for position 0 (per PyTorch).
- If `size` is int → accept.

Wire into `_validate_types` when `op_short_name == "sum_to_size"` and param is `size`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_sum_to_size_parity.py::test_sum_to_size_type_error_matches_torch_contract -v`
Expected: PASS.

**Step 5: Commit**

`git add src/candle/_dispatch/schema.py tests/contract/test_training_core_sum_to_size_parity.py`
`git commit -m "test: expand sum_to_size contract parity"`

---

### Task 3: Implement CPU kernel for sum_to_size

**Files:**
- Modify: `src/candle/_backends/cpu/ops.py`

**Step 1: Write the failing test**

Use existing contract test `test_sum_to_size_forward_matches_torch_contract`.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_sum_to_size_parity.py::test_sum_to_size_forward_matches_torch_contract -v`
Expected: FAIL with missing kernel registration.

**Step 3: Write minimal implementation**

Add `sum_to_size(a, size)`:
- Normalize size to tuple of ints.
- If size matches `a.shape`, return `a`.
- Validate expandability against `a.shape` (right-aligned):
  - If `len(size) > a.dim()` or any target dim <= 0 or mismatch where target not 1 and not equal → raise `RuntimeError` with `size {[...]} is not expandable to size {[...]}.'
- Reduce `a` using numpy:
  - While extra leading dims exist, sum over axis 0.
  - For each dimension where target == 1 and current dim > 1, sum over that axis with keepdims.
- Return tensor from numpy with original dtype/device.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_sum_to_size_parity.py::test_sum_to_size_forward_matches_torch_contract -v`
Expected: PASS.

**Step 5: Commit**

`git add src/candle/_backends/cpu/ops.py`
`git commit -m "feat: add cpu sum_to_size"`

---

### Task 4: Add meta inference and CPU registration

**Files:**
- Modify: `src/candle/_backends/meta/infer.py`
- Modify: `src/candle/_backends/cpu/__init__.py`

**Step 1: Write the failing test**

Use existing contract tests.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_sum_to_size_parity.py::test_sum_to_size_scalar_shape_matches_torch_contract -v`
Expected: FAIL or mismatch.

**Step 3: Write minimal implementation**

Add `infer_sum_to_size(a, size)`:
- Determine target shape like kernel logic.
- Raise same expandability `RuntimeError` for invalid sizes.
- Return `TensorSpec(shape=target, stride=_contiguous_stride(target), dtype=a.dtype)`.

Register in `src/candle/_backends/cpu/__init__.py`:
- Add import for `sum_to_size` and register: `registry.register("sum_to_size", "cpu", sum_to_size, meta=meta_infer.infer_sum_to_size)`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_sum_to_size_parity.py::test_sum_to_size_scalar_shape_matches_torch_contract -v`
Expected: PASS.

**Step 5: Commit**

`git add src/candle/_backends/meta/infer.py src/candle/_backends/cpu/__init__.py`
`git commit -m "feat: register sum_to_size meta"`

---

### Task 5: Add autograd backward for sum_to_size

**Files:**
- Modify: `src/candle/_backends/autograd.py`

**Step 1: Write the failing test**

Use `test_sum_to_size_backward_matches_torch_contract`.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_sum_to_size_parity.py::test_sum_to_size_backward_matches_torch_contract -v`
Expected: FAIL with missing autograd kernel.

**Step 3: Write minimal implementation**

Add `_sum_to_size_backward(grad, a, _saved_a, keyset, args, kwargs)`:
- Extract target size from args.
- Use `redispatch("expand", keyset, grad, a.shape)` after ensuring grad is reshaped to target size.
- Return expanded grad for input; no grad for size.

Register in autograd table:
- Add `("sum_to_size", lambda: _autograd_unary_args("sum_to_size", _sum_to_size_backward, save_input=False))`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_sum_to_size_parity.py::test_sum_to_size_backward_matches_torch_contract -v`
Expected: PASS.

**Step 5: Commit**

`git add src/candle/_backends/autograd.py`
`git commit -m "feat: add sum_to_size autograd"`

---

### Task 6: Full contract gate

**Step 1: Run full contract tests**

Run: `PYTHONPATH=src pytest tests/contract/ -v --tb=short`
Expected: PASS.

**Step 2: Final commit (if needed)**

`git add tests/contract/test_training_core_sum_to_size_parity.py`
`git commit -m "test: expand sum_to_size parity coverage"`

