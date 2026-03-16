# Autograd Step 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `view_as_real`, `view_as_complex`, `var_mean`, and `scatter_reduce` with full reduce modes, and add contract tests.

**Architecture:**
- `view_as_real`/`view_as_complex` are view-like ops. Register schemas with alias sets and implement via backend view helpers.
- `var_mean` returns `(var, mean)` and uses existing ops.
- `scatter_reduce` supports reduce modes (`sum`, `prod`, `mean`, `amax`, `amin`) and `include_self` on CPU; other backends raise NotImplemented.

**Tech Stack:** Python, Candle dispatch/registry, pytest.

---

### Task 1: Contract Tests (Red)

**Files:**
- Create: `tests/contract/test_view_as_real_complex.py`
- Create: `tests/contract/test_var_mean_contract.py`
- Create: `tests/contract/test_scatter_reduce_contract.py`

**Step 1: Write failing tests**

```python
# tests/contract/test_view_as_real_complex.py
import candle as torch
import pytest

def test_view_as_real_complex_shape_dtype():
    x = torch.randn(2, 3, dtype=torch.complex64)
    y = torch.view_as_real(x)
    assert y.shape == (2, 3, 2)
    assert str(y.dtype).startswith('float')

def test_view_as_complex_roundtrip():
    x = torch.randn(2, 3, dtype=torch.complex64)
    y = torch.view_as_real(x)
    z = torch.view_as_complex(y)
    assert z.shape == x.shape
    assert str(z.dtype) == str(x.dtype)

def test_view_as_complex_invalid_last_dim():
    x = torch.randn(2, 3, 4)
    with pytest.raises(RuntimeError):
        torch.view_as_complex(x)
```

```python
# tests/contract/test_var_mean_contract.py
import candle as torch

def test_var_mean_returns_tuple():
    x = torch.randn(2, 3)
    v, m = torch.var_mean(x)
    assert v.shape == m.shape


def test_var_mean_dim_keepdim():
    x = torch.randn(2, 3)
    v, m = torch.var_mean(x, dim=1, keepdim=True)
    assert v.shape == (2, 1)
    assert m.shape == (2, 1)
```

```python
# tests/contract/test_scatter_reduce_contract.py
import candle as torch


def _make_inputs():
    base = torch.zeros(3, 5)
    index = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 1, 2], [2, 0, 1, 2, 0]])
    src = torch.arange(15).reshape(3, 5)
    return base, index, src


def test_scatter_reduce_sum_include_self():
    base, index, src = _make_inputs()
    out = torch.scatter_reduce(base, 0, index, src, reduce="sum", include_self=True)
    assert out.shape == base.shape


def test_scatter_reduce_prod_exclude_self():
    base, index, src = _make_inputs()
    out = torch.scatter_reduce(base, 0, index, src, reduce="prod", include_self=False)
    assert out.shape == base.shape


def test_scatter_reduce_mean():
    base, index, src = _make_inputs()
    out = torch.scatter_reduce(base, 0, index, src, reduce="mean", include_self=True)
    assert out.shape == base.shape


def test_scatter_reduce_amax_amin():
    base, index, src = _make_inputs()
    out_max = torch.scatter_reduce(base, 0, index, src, reduce="amax", include_self=True)
    out_min = torch.scatter_reduce(base, 0, index, src, reduce="amin", include_self=True)
    assert out_max.shape == base.shape
    assert out_min.shape == base.shape
```

**Step 2: Run tests to verify they fail**

Run:
- `pytest tests/contract/test_view_as_real_complex.py::test_view_as_real_complex_shape_dtype -v --tb=short`
- `pytest tests/contract/test_var_mean_contract.py::test_var_mean_returns_tuple -v --tb=short`
- `pytest tests/contract/test_scatter_reduce_contract.py::test_scatter_reduce_sum_include_self -v --tb=short`

Expected: failures due to missing APIs.

---

### Task 2: Schema + Functional API

**Files:**
- Modify: `src/candle/_dispatch/schemas.py`
- Modify: `src/candle/_functional.py`
- Modify: `src/candle/_tensor.py`
- Modify: `src/candle/__init__.py`

**Step 1: Register schemas**
- `view_as_real(Tensor input) -> Tensor`
- `view_as_complex(Tensor input) -> Tensor`
- `var_mean(Tensor input, int[]? dim=None, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)`
- `scatter_reduce(Tensor input, int dim, Tensor index, Tensor src, str reduce, bool include_self=True) -> Tensor`

**Step 2: Functional API**
- Add `view_as_real`, `view_as_complex`, `var_mean`, `scatter_reduce` in `src/candle/_functional.py` dispatching by op name.
- Add `Tensor.var_mean` in `src/candle/_tensor.py` calling functional.
- Export top-level in `src/candle/__init__.py`.

---

### Task 3: Backend Implementations

**Files:**
- Modify: `src/candle/_backends/common/view.py`
- Modify: `src/candle/_backends/cpu/ops.py`
- Modify: `src/candle/_backends/cpu/__init__.py`
- Modify: `src/candle/_backends/meta/__init__.py`
- Modify: `src/candle/_backends/meta/ops.py`

**Step 1: view_as_real/complex**
- Implement view helpers in `common/view.py` using `as_strided` to reinterpret shape/stride.
- Register CPU/meta kernels.

**Step 2: var_mean**
- Implement via existing `var` + `mean` in CPU/meta where needed; or register composite in autograd/functional.

**Step 3: scatter_reduce**
- Implement CPU kernel supporting `sum/prod/mean/amax/amin` and `include_self`.
- Meta kernel infers shape.

---

### Task 4: Tests + Contract Suite Gate

**Step 1:** Run targeted tests again (expect pass).
**Step 2:** Run `pytest tests/contract/ -v --tb=short`.

---

### Task 5: Pylint Gate

Run: `pylint src/candle/ --rcfile=pyproject.toml`

---

### Task 6: Commit + PR

```bash
git add tests/contract/test_view_as_real_complex.py tests/contract/test_var_mean_contract.py tests/contract/test_scatter_reduce_contract.py \
  src/candle/_dispatch/schemas.py src/candle/_functional.py src/candle/_tensor.py src/candle/__init__.py \
  src/candle/_backends/common/view.py src/candle/_backends/cpu/ops.py src/candle/_backends/cpu/__init__.py \
  src/candle/_backends/meta/__init__.py src/candle/_backends/meta/ops.py

git commit -m "feat: add view_as_real/complex var_mean scatter_reduce"

git fetch upstream main
git rebase upstream/main

git push -u origin compat/autograd-pr3

gh pr create -R candle-org/candle --base main --head lvyufeng:compat/autograd-pr3 \
  --title "feat: add view_as_real/complex var_mean scatter_reduce" \
  --body-file /tmp/pr_body_pr3.md
```
