# CPU Pool/Unpool (1D/2D) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align CPU max pooling with indices and max unpooling (1D/2D) semantics with PyTorch, using targeted tests and minimal kernel fixes.

**Architecture:** Keep existing dispatch/schema wiring. Drive behavior through nn.functional wrappers and CPU kernels in `src/candle/_backends/cpu/ops.py`. Focus on return_indices correctness, output_size inference, and index validation. Scope is limited to CPU and 1D/2D only.

**Tech Stack:** Python, numpy-backed CPU kernels, candle dispatch, pytest

---

### Task 1: Add/adjust failing CPU tests for pool/unpool 1D/2D edge cases

**Files:**
- Modify: `tests/cpu/test_nn_functional.py`

**Step 1: Write the failing test (indices correctness + output_size inference)**

Add focused tests (or sharpen existing ones):

```python
def test_max_pool2d_with_indices_flattening_matches_torch():
    import torch as torch_ref

    x = torch.tensor([[[[1.0, 3.0], [2.0, 4.0]]]], device='cpu')
    pooled, indices = F.max_pool2d_with_indices(x, kernel_size=2, stride=2)
    ref_pooled, ref_indices = torch_ref.nn.functional.max_pool2d(
        x,
        kernel_size=2,
        stride=2,
        return_indices=True,
    )
    assert torch.allclose(pooled, ref_pooled, atol=1e-6)
    assert torch.equal(indices, ref_indices)


def test_max_unpool1d_infers_output_size_when_none_matches_torch():
    import torch as torch_ref

    x = torch.tensor([[[1.0, 3.0, 2.0, 4.0]]], device='cpu')
    pooled, indices = F.max_pool1d_with_indices(x, kernel_size=2, stride=2)
    out = F.max_unpool1d(pooled, indices, kernel_size=2, stride=2)
    ref_pooled, ref_indices = torch_ref.nn.functional.max_pool1d(
        x,
        kernel_size=2,
        stride=2,
        return_indices=True,
    )
    ref_out = torch_ref.nn.functional.max_unpool1d(
        ref_pooled,
        ref_indices,
        kernel_size=2,
        stride=2,
    )
    assert out.shape == ref_out.shape
    assert torch.allclose(out, ref_out, atol=1e-6)



def test_max_pool1d_padding_too_large_matches_torch():
    import torch as torch_ref

    x = torch.tensor([[[1.0, 2.0, 3.0]]], device='cpu')
    with pytest.raises(RuntimeError):
        torch_ref.nn.functional.max_pool1d(x, kernel_size=3, stride=1, padding=2)
    with pytest.raises(RuntimeError):
        F.max_pool1d(x, kernel_size=3, stride=1, padding=2)


def test_max_pool2d_padding_too_large_matches_torch():
    import torch as torch_ref

    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device='cpu')
    with pytest.raises(RuntimeError):
        torch_ref.nn.functional.max_pool2d(x, kernel_size=2, stride=1, padding=2)
    with pytest.raises(RuntimeError):
        F.max_pool2d(x, kernel_size=2, stride=1, padding=2)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/cpu/test_nn_functional.py -k "max_pool2d_with_indices_flattening_matches_torch or max_unpool1d_infers_output_size_when_none_matches_torch or max_pool1d_padding_too_large_matches_torch or max_pool2d_padding_too_large_matches_torch" -v`

Expected: FAIL (incorrect indices or output_size inference)

**Step 3: Commit red tests**

```bash
git add tests/cpu/test_nn_functional.py
git commit -m "test(cpu): cover pool/unpool 1d/2d indices and output_size"
```

---

### Task 2: Fix CPU max_pool{1,2}d indices + max_unpool{1,2}d output_size inference

**Files:**
- Modify: `src/candle/_backends/cpu/ops.py`

**Step 1: Implement minimal fixes**

- Ensure `max_pool2d(..., return_indices=True)` returns flat indices in H*W order consistent with PyTorch.
- Ensure `max_pool1d(..., return_indices=True)` returns flat indices for 1D (already likely, confirm behavior with padding/stride).
- Add padding validation to `max_pool1d`/`max_pool2d` to match PyTorch (padding must be <= floor(kernel_size/2) per dim).
- In `max_unpool1d/2d`, infer `output_size` when `None`, using PyTorch’s formula based on input, kernel_size, stride, padding.
- Validate indices are within output range; raise `ValueError` or `IndexError`.

**Step 2: Re-run targeted tests**

Run: `PYTHONPATH=src pytest tests/cpu/test_nn_functional.py -k "max_pool2d_with_indices_flattening_matches_torch or max_unpool1d_infers_output_size_when_none_matches_torch or max_pool1d_padding_too_large_matches_torch or max_pool2d_padding_too_large_matches_torch" -v`

Expected: PASS

**Step 3: Commit green implementation**

```bash
git add src/candle/_backends/cpu/ops.py
git commit -m "fix(cpu): align pool/unpool 1d/2d indices and output_size"
```

---

### Task 3: Regression verification

**Files:**
- Reuse touched files only

**Step 1: Run focused CPU tests**

Run: `PYTHONPATH=src pytest tests/cpu/test_nn_functional.py -v`
Expected: PASS

**Step 2: Run adjacent regression**

Run: `PYTHONPATH=src pytest tests/cpu/test_top_level_ops.py -v`
Expected: PASS

**Step 3: Commit any stabilization changes (if needed)**

```bash
git add -A
git commit -m "test(cpu): stabilize pool/unpool 1d/2d regression"
```
