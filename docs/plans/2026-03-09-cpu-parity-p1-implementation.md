# CPU Parity P1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close high-impact nn.functional API parity gaps for CPU by adding missing torch-compatible wrappers and behavior for pooling-with-indices, adaptive max pooling variants, max unpooling, and inplace activation wrappers.

**Architecture:** Keep backend kernels as source of truth and extend `candle.nn.functional` wrappers first, only adding minimal kernel logic where missing (for max_unpool ops). Drive all behavior through torch-comparison tests in CPU suite and preserve existing dispatch architecture. Scope is intentionally constrained to high-frequency training/inference entry points.

**Tech Stack:** Python, numpy-backed CPU kernels, candle dispatch, pytest

---

### Task 1: Add failing nn.functional parity tests for P1 target APIs

**Files:**
- Modify: `tests/cpu/test_nn_functional.py`
- Reference: `src/candle/nn/functional.py`

**Step 1: Write failing tests for wrappers that should exist**

Add tests for:
- `F.relu_`, `F.elu_`, `F.hardtanh_`, `F.rrelu_`, `F.threshold_`, `F.selu_`, `F.celu_`
- `F.max_pool1d_with_indices`, `F.max_pool2d_with_indices`
- `F.adaptive_max_pool1d`, `F.adaptive_max_pool2d(..., return_indices=True)`

Example test style:

```python
def test_relu_inplace_wrapper_matches_torch_shape_and_mutation():
    x = torch.tensor([-1.0, 2.0])
    out = F.relu_(x)
    assert out is x
    assert x.tolist() == [0.0, 2.0]
```

**Step 2: Run target tests and verify failure**

Run: `PYTHONPATH=src pytest tests/cpu/test_nn_functional.py -k "inplace_wrapper or with_indices or adaptive_max_pool" -v`
Expected: FAIL with missing attributes/wrappers.

**Step 3: Commit test-only red phase**

```bash
git add tests/cpu/test_nn_functional.py
git commit -m "test(cpu): add failing P1 nn.functional parity coverage"
```

---

### Task 2: Implement missing nn.functional wrappers for existing CPU kernels

**Files:**
- Modify: `src/candle/nn/functional.py`

**Step 1: Add minimal wrappers for inplace activations**

Implement wrappers:
- `relu_`, `elu_`, `hardtanh_`, `rrelu_`, `threshold_`, `selu_`, `celu_`

Implementation should delegate to existing non-inplace functions or tensor in-place methods with minimal semantics.

**Step 2: Add wrappers for pooling with indices**

Implement wrappers:
- `max_pool1d_with_indices`
- `max_pool2d_with_indices`

Return tuple `(output, indices)` with torch-compatible argument signatures.

**Step 3: Add/adjust adaptive max pool wrappers**

Implement wrappers:
- `adaptive_max_pool1d`
- `adaptive_max_pool2d` path for `return_indices=True`

**Step 4: Run target tests and verify green**

Run: `PYTHONPATH=src pytest tests/cpu/test_nn_functional.py -k "inplace_wrapper or with_indices or adaptive_max_pool" -v`
Expected: PASS.

**Step 5: Commit wrapper implementation**

```bash
git add src/candle/nn/functional.py
git commit -m "feat(nn.functional): add P1 wrappers for inplace activations and pooling with indices"
```

---

### Task 3: Add max_unpool CPU support (minimal, torch-aligned behavior)

**Files:**
- Modify: `src/candle/nn/functional.py`
- Modify: `src/candle/_backends/cpu/ops.py`
- Modify: `src/candle/_backends/cpu/__init__.py`
- Modify: `src/candle/_dispatch/schemas.py`
- Test: `tests/cpu/test_nn_functional.py`

**Step 1: Add failing tests for max unpool**

Add tests for:
- `F.max_unpool1d`
- `F.max_unpool2d`

Minimum behavior:
- correct output shape
- values scattered back by indices
- invalid index raises

**Step 2: Run tests and confirm failure**

Run: `PYTHONPATH=src pytest tests/cpu/test_nn_functional.py -k "max_unpool" -v`
Expected: FAIL.

**Step 3: Implement minimal CPU kernels and registration**

- Add CPU ops for `max_unpool1d` and `max_unpool2d` in `ops.py`.
- Register kernels in `_backends/cpu/__init__.py`.
- Add schemas in `_dispatch/schemas.py`.
- Add `nn.functional` wrappers.

**Step 4: Re-run max_unpool tests**

Run: `PYTHONPATH=src pytest tests/cpu/test_nn_functional.py -k "max_unpool" -v`
Expected: PASS.

**Step 5: Commit max_unpool batch**

```bash
git add src/candle/_backends/cpu/ops.py src/candle/_backends/cpu/__init__.py src/candle/_dispatch/schemas.py src/candle/nn/functional.py tests/cpu/test_nn_functional.py
git commit -m "feat(cpu): add max_unpool1d/2d parity support"
```

---

### Task 4: Regression verification and parity smoke

**Files:**
- Reuse touched files only

**Step 1: Run focused suites**

```bash
PYTHONPATH=src pytest tests/cpu/test_nn_functional.py tests/cpu/test_top_level_ops.py -v
```

**Step 2: Run broader safety smoke**

```bash
PYTHONPATH=src pytest tests/cpu/test_ops_cpu.py tests/cpu/test_autograd_ops.py -q
```

**Step 3: Commit any test stabilization changes**

```bash
git add -A
git commit -m "test(cpu): stabilize P1 parity and regression coverage"
```

---

### Non-goals

- No long-tail torch nn.functional parity (fractional pools, grouped_mm, etc.)
- No new linalg/fft/special expansion in this batch
- No cross-backend behavior changes unless shared wrapper paths require it
