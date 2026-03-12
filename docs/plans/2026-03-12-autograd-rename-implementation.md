# Autograd Rename Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename `candle._autograd` to `candle.autograd`, update internal imports, and add minimal `autograd._functions.Resize` support for PyTorch autograd compatibility.

**Architecture:** Move the autograd package to `src/candle/autograd/`, replace all `candle._autograd` references with `candle.autograd`, and remove the autograd alias in `src/_candle_torch_compat.py` so `torch.autograd` maps to `candle.autograd`. Add `candle/autograd/_functions` with `Resize` implemented as a `Function` using Candle tensor APIs.

**Tech Stack:** Python, candle autograd, pytest

---

### Task 1: Update compatibility tests for new autograd package

**Files:**
- Modify: `tests/test_torch_compat.py`

**Step 1: Write the failing test**

Update the autograd assertions to expect `candle.autograd`:

```python
assert m._resolve_candle_name("torch.autograd") == "candle.autograd"
assert m._resolve_candle_name("torch.autograd.graph") == "candle.autograd.graph"

import candle.autograd
assert torch.autograd is candle.autograd
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_torch_compat.py::TestResolveCandleName::test_aliased_autograd -v`
Expected: FAIL with old module name `candle._autograd` in assertion.

**Step 3: Commit**

```bash
git add tests/test_torch_compat.py
git commit -m "test: expect candle.autograd mapping"
```

---

### Task 2: Rename autograd package and update internal imports

**Files:**
- Rename: `src/candle/_autograd/` -> `src/candle/autograd/`
- Modify: `src/candle/__init__.py`
- Modify: `src/candle/_tensor.py`
- Modify: `src/candle/library.py`
- Modify: `src/candle/_dispatch/dispatcher.py`
- Modify: `src/candle/_functional.py`
- Modify: `src/candle/_backends/autograd.py`
- Modify: `src/candle/utils/checkpoint.py`
- Modify: `src/candle/nn/init.py`
- Modify: `src/candle/nn/parallel/distributed.py`
- Modify: `tests/cpu/test_autograd_function.py`
- Modify: `tests/cpu/test_custom_op.py`
- Modify: `tests/contract/test_saved_tensors_hooks.py`
- Modify: `tests/contract/test_autograd_engine_topo.py`

**Step 1: Rename the package directory**

Run:
```bash
mv src/candle/_autograd src/candle/autograd
```

**Step 2: Update all imports**

Replace `candle._autograd` or relative `_autograd` imports with `candle.autograd` equivalents. Examples:

```python
from .autograd.engine import backward as _backward
from ..autograd.grad_mode import no_grad
from candle.autograd import Function
```

**Step 3: Run test to verify it fails before fixes**

Run: `pytest tests/contract/test_autograd_engine_topo.py::test_saved_tensors_hooks_unpacked_once_per_node -v`
Expected: FAIL with import error for `candle._autograd`.

**Step 4: Verify tests pass after import updates**

Run: `pytest tests/contract/test_autograd_engine_topo.py::test_saved_tensors_hooks_unpacked_once_per_node -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/candle tests/cpu tests/contract
git commit -m "refactor: rename autograd package"
```

---

### Task 3: Update torch compatibility mapping

**Files:**
- Modify: `src/_candle_torch_compat.py`

**Step 1: Write the failing test**

Ensure `TestResolveCandleName::test_aliased_autograd` expects `candle.autograd` and `TestImportHook::test_torch_autograd` imports `candle.autograd`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_torch_compat.py::TestImportHook::test_torch_autograd -v`
Expected: FAIL with `torch.autograd` still resolving to `_autograd`.

**Step 3: Update mapping**

Remove the autograd entry from `_ALIASES` so `torch.autograd` maps to `candle.autograd` by default.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_torch_compat.py::TestImportHook::test_torch_autograd -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/_candle_torch_compat.py tests/test_torch_compat.py
git commit -m "fix: map torch.autograd to candle.autograd"
```

---

### Task 4: Add minimal autograd._functions.Resize

**Files:**
- Create: `src/candle/autograd/_functions/__init__.py`
- Create: `src/candle/autograd/_functions/tensor.py`
- Modify: `src/candle/autograd/__init__.py`

**Step 1: Write the failing test**

Add a unit test for `Resize` in `tests/cpu/test_autograd_function.py`:

```python
def test_autograd_resize_apply_roundtrip_shape():
    import candle.autograd as autograd
    import candle

    x = candle.rand(2, requires_grad=True)
    y = autograd._functions.Resize.apply(x, (2,))
    assert y.shape == (2,)
    y.sum().backward()
    assert x.grad.shape == (2,)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/cpu/test_autograd_function.py::test_autograd_resize_apply_roundtrip_shape -v`
Expected: FAIL with `AttributeError: module 'candle.autograd' has no attribute '_functions'`.

**Step 3: Implement Resize**

`src/candle/autograd/_functions/tensor.py`:

```python
import operator
from functools import reduce

from ..function import Function


class Resize(Function):
    @staticmethod
    def forward(ctx, tensor, sizes):
        ctx.sizes = tuple(sizes)
        ctx.numel = reduce(operator.mul, ctx.sizes, 1)
        if tensor.numel() != ctx.numel:
            raise RuntimeError(
                (
                    "requested resize to {} ({} elements in total), "
                    "but the given tensor has a size of {} ({} elements). "
                    "autograd's resize can only change the shape of a given "
                    "tensor, while preserving the number of elements. "
                ).format(
                    "x".join(map(str, ctx.sizes)),
                    ctx.numel,
                    "x".join(map(str, tensor.size())),
                    tensor.numel(),
                )
            )
        ctx.input_sizes = tensor.size()
        if tensor.is_contiguous():
            return tensor.contiguous().view(*ctx.sizes)
        return tensor.contiguous().view(*ctx.sizes)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous().view(ctx.input_sizes), None
```

`src/candle/autograd/_functions/__init__.py`:

```python
from .tensor import Resize

__all__ = ["Resize"]
```

In `src/candle/autograd/__init__.py`, export `_functions` to allow `autograd._functions.Resize` access:

```python
from . import _functions
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/cpu/test_autograd_function.py::test_autograd_resize_apply_roundtrip_shape -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/candle/autograd tests/cpu/test_autograd_function.py
git commit -m "feat: add autograd._functions.Resize"
```

---

### Task 5: Run compatibility and contract test gates

**Files:**
- Modify if needed: `compat/pytorch/xfail.yaml`

**Step 1: Run PyTorch autograd compatibility file**

Run: `python compat/pytorch/run.py --file test_autograd.py`
Expected: No collection errors; record failing tests.

**Step 2: Update xfail if needed**

If failures remain that are known gaps, add patterns with reasons to `compat/pytorch/xfail.yaml`.

**Step 3: Run contract suite**

Run: `pytest tests/contract/ -v --tb=short`
Expected: PASS.

**Step 4: Commit any xfail updates**

```bash
git add compat/pytorch/xfail.yaml
git commit -m "test: xfail autograd gaps"
```
