# CPU Inplace Activations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align inplace activation behavior (relu_, elu_, hardtanh_, rrelu_, threshold_, selu_, celu_, leaky_relu_) with PyTorch on CPU, focusing on autograd leaf checks and non-leaf allowlist.

**Architecture:** Keep existing functional wrappers and dispatch. Add tests to cover leaf vs non-leaf inplace semantics. Ensure Tensor._check_inplace matches PyTorch rules. Avoid changing backend kernels unless test demands it.

**Tech Stack:** Python, candle dispatch/autograd, pytest

---

### Task 1: Add failing test for non-leaf requires_grad inplace behavior

**Files:**
- Modify: `tests/cpu/test_nn_functional.py`

**Step 1: Write failing test (non-leaf requires_grad should allow inplace)**

```python
def test_inplace_activation_non_leaf_requires_grad_allowed():
    base = torch.tensor([-1.0, 0.0, 2.0], device='cpu')
    x = base * 1.0  # non-leaf
    out = F.relu_(x)
    assert out is x
    assert torch.allclose(x, torch.tensor([0.0, 0.0, 2.0], device='cpu'), atol=1e-6)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/cpu/test_nn_functional.py -k "inplace_activation_non_leaf_requires_grad_allowed" -v`
Expected: FAIL with leaf-inplace RuntimeError (current behavior).

**Step 3: Commit red test**

```bash
git add tests/cpu/test_nn_functional.py
git commit -m "test(cpu): add non-leaf inplace activation behavior"
```

---

### Task 2: Fix inplace leaf check to allow non-leaf requires_grad tensors

**Files:**
- Modify: `src/candle/_tensor.py`

**Step 1: Implement minimal fix**

Adjust `_check_inplace` to raise only for leaf tensors requiring grad, not all requires_grad tensors. Confirm view-of-leaf rule still enforced.

**Step 2: Re-run test**

Run: `PYTHONPATH=src pytest tests/cpu/test_nn_functional.py -k "inplace_activation_non_leaf_requires_grad_allowed" -v`
Expected: PASS.

**Step 3: Run leaf error test**

Run: `PYTHONPATH=src pytest tests/cpu/test_nn_functional.py -k "inplace_activation_leaf_requires_grad_errors" -v`
Expected: PASS.

**Step 4: Commit green implementation**

```bash
git add src/candle/_tensor.py
git commit -m "fix(autograd): allow inplace on non-leaf requires_grad tensors"
```

---

### Task 3: Regression

**Step 1: Run focused suite**

Run: `PYTHONPATH=src pytest tests/cpu/test_nn_functional.py -k "inplace_activation" -v`
Expected: PASS.

**Step 2: Commit any stabilization changes (if needed)**

```bash
git add -A
git commit -m "test(cpu): stabilize inplace activation tests"
```
