# CPU Torch Parity P0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the highest-risk CPU parity gaps against torch by fixing dtype promotion and reduction dtype semantics, then exposing missing high-frequency top-level in-place APIs.

**Architecture:** Keep the existing dispatch architecture unchanged and patch behavior at the CPU kernel + functional wrapper layer. Use torch-vs-candle parity tests as the contract, starting from failing tests to prevent regressions and to encode exact expected behavior. Scope is intentionally narrow (P0 only): arithmetic/reduction semantics and top-level API visibility; no long-tail operator expansion.

**Tech Stack:** Python, numpy-backed CPU kernels, candle dispatch registry, pytest

---

### Task 1: Add failing parity tests for arithmetic dtype promotion

**Files:**
- Modify: `tests/cpu/test_ops_cpu.py`
- Reference: `src/candle/_backends/cpu/ops.py`

**Step 1: Write failing tests (torch parity contract)**

Add tests that compare `candle` behavior with `torch` for promotion-sensitive cases:

```python
def test_add_dtype_promotion_matches_torch_cpu():
    import torch as real_torch
    a = torch.tensor([1, 2, 3], dtype=torch.int64)
    b = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    out = torch.add(a, b)

    ta = real_torch.tensor([1, 2, 3], dtype=real_torch.int64)
    tb = real_torch.tensor([0.5, 0.5, 0.5], dtype=real_torch.float32)
    tout = real_torch.add(ta, tb)

    assert out.dtype == tout.dtype
    np.testing.assert_allclose(out.numpy(), tout.numpy())
```

Also add `mul/div/true_divide` coverage and `bool + int64` case.

**Step 2: Run target tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/cpu/test_ops_cpu.py -k "dtype_promotion" -v`
Expected: FAIL due to current lhs-cast behavior.

**Step 3: Minimal implementation in CPU kernels**

In `src/candle/_backends/cpu/ops.py`, stop forcing output dtype to `a.dtype` for mixed-type arithmetic. Compute result dtype using numpy result type of the actual operation, then map to candle dtype and construct tensor with that dtype.

Implementation constraints:
- Keep tensor device behavior unchanged.
- Preserve shape/stride handling.
- Do not broaden scope to unrelated ops yet.

**Step 4: Re-run targeted tests**

Run: `PYTHONPATH=src pytest tests/cpu/test_ops_cpu.py -k "dtype_promotion" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/cpu/test_ops_cpu.py src/candle/_backends/cpu/ops.py
git commit -m "fix(cpu): align arithmetic dtype promotion with torch on cpu"
```

---

### Task 2: Add failing parity tests for reduction dtype semantics (`mean`, `sum`)

**Files:**
- Modify: `tests/cpu/test_ops_cpu.py`
- Modify: `tests/cpu/test_top_level_ops.py`
- Reference: `src/candle/_functional.py`
- Reference: `src/candle/_backends/cpu/ops.py`

**Step 1: Write failing tests for `mean` on integer input**

Add tests that enforce torch parity:

```python
def test_mean_int_without_dtype_matches_torch_error():
    import torch as real_torch
    x = torch.tensor([1, 2, 3], dtype=torch.int64)
    with pytest.raises(RuntimeError):
        torch.mean(x)

    tx = real_torch.tensor([1, 2, 3], dtype=real_torch.int64)
    with pytest.raises(RuntimeError):
        real_torch.mean(tx)
```

Also test that `mean(..., dtype=torch.float32)` works and matches torch.

**Step 2: Write failing tests for `sum(dtype=...)` accumulation behavior**

Add overflow-sensitive contract:

```python
def test_sum_dtype_accumulates_in_target_dtype():
    import torch as real_torch
    x = torch.tensor([120, 120], dtype=torch.int8)
    out = torch.sum(x, dtype=torch.int64)

    tx = real_torch.tensor([120, 120], dtype=real_torch.int8)
    tout = real_torch.sum(tx, dtype=real_torch.int64)

    assert out.item() == tout.item() == 240
```

**Step 3: Run target tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/cpu/test_ops_cpu.py -k "mean_int|sum_dtype" -v`
Expected: FAIL.

**Step 4: Minimal implementation changes**

- In `src/candle/_backends/cpu/ops.py::sum_`, support `dtype` by performing reduction in target dtype when provided.
- In `src/candle/_backends/cpu/ops.py::mean_` or wrapper layer, reject integer/bool input without explicit `dtype`, matching torch error behavior.
- In `src/candle/_functional.py::mean`, keep `dtype` kwarg behavior but ensure error parity when `dtype` is absent.

**Step 5: Re-run target tests**

Run: `PYTHONPATH=src pytest tests/cpu/test_ops_cpu.py -k "mean_int|sum_dtype" -v`
Expected: PASS.

**Step 6: Commit**

```bash
git add tests/cpu/test_ops_cpu.py tests/cpu/test_top_level_ops.py src/candle/_backends/cpu/ops.py src/candle/_functional.py
git commit -m "fix(cpu): align mean/sum dtype semantics with torch"
```

---

### Task 3: Add missing top-level in-place API wrappers for high-frequency ops

**Files:**
- Modify: `src/candle/_functional.py`
- Modify: `src/candle/__init__.py`
- Modify: `tests/cpu/test_top_level_ops.py`

**Step 1: Write failing top-level API tests**

Add tests for top-level callable presence and behavior:

```python
def test_top_level_relu_inplace_exists_and_mutates():
    x = torch.tensor([-1.0, 2.0])
    out = torch.relu_(x)
    assert out is x
    np.testing.assert_allclose(x.numpy(), np.array([0.0, 2.0]))
```

Add at least: `relu_`, `add_`, `mul_`, `clamp_`, `zero_`, `copy_`.

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/cpu/test_top_level_ops.py -k "top_level_.*inplace|relu_" -v`
Expected: FAIL with missing attribute/function.

**Step 3: Implement minimal wrappers and exports**

- Add wrappers in `src/candle/_functional.py` that dispatch to corresponding inplace ops.
- Export in `src/candle/__init__.py` imports and `__all__`.
- Keep behavior delegated to existing tensor/dispatch semantics.

**Step 4: Re-run targeted tests**

Run: `PYTHONPATH=src pytest tests/cpu/test_top_level_ops.py -k "top_level_.*inplace|relu_" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/candle/_functional.py src/candle/__init__.py tests/cpu/test_top_level_ops.py
git commit -m "feat(api): expose high-frequency top-level inplace ops"
```

---

### Task 4: Optional P0.5 safety: explicit `candle.nn` package export in root module

**Files:**
- Modify: `src/candle/__init__.py`
- Modify: `tests/cpu/test_import.py`

**Step 1: Write failing import contract test**

```python
def test_candle_nn_module_is_exposed():
    import candle
    assert hasattr(candle, "nn")
    assert hasattr(candle.nn, "functional")
```

**Step 2: Run test and verify failure (if currently failing in clean env)**

Run: `PYTHONPATH=src pytest tests/cpu/test_import.py -k "candle_nn_module_is_exposed" -v`
Expected: FAIL or flaky behavior depending on import order.

**Step 3: Minimal implementation**

In `src/candle/__init__.py`, explicitly add:

```python
from . import nn
```

and include `"nn"` in `__all__`.

**Step 4: Re-run target test**

Run: `PYTHONPATH=src pytest tests/cpu/test_import.py -k "candle_nn_module_is_exposed" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/candle/__init__.py tests/cpu/test_import.py
git commit -m "fix(api): explicitly expose candle.nn in root package"
```

---

### Task 5: Regression and focused parity verification

**Files:**
- Reuse modified files only

**Step 1: Run focused CPU suite for touched areas**

Run:

```bash
PYTHONPATH=src pytest tests/cpu/test_ops_cpu.py tests/cpu/test_top_level_ops.py tests/cpu/test_import.py -v
```

Expected: PASS for new + surrounding tests.

**Step 2: Run smoke checks for unaffected hot paths**

Run:

```bash
PYTHONPATH=src pytest tests/cpu/test_nn_functional.py tests/cpu/test_autograd_ops.py -q
```

Expected: PASS.

**Step 3: Summarize parity outcomes in commit message or PR description**

Include exact behavior changes:
- arithmetic dtype promotion now torch-consistent for covered cases
- `mean` integer behavior now torch-consistent
- `sum(dtype=...)` accumulation behavior now torch-consistent
- top-level inplace API coverage improved

**Step 4: Final commit (if needed for any leftovers)**

```bash
git add -A
git commit -m "test(cpu): add parity regression coverage for dtype semantics and top-level inplace APIs"
```

---

### Notes / Non-goals for this plan

- Do not implement long-tail torch API additions in this batch.
- Do not alter non-CPU backends unless shared code path changes are strictly required.
- Do not refactor dispatch architecture.
- Keep patches minimal and behavior-driven by tests.
