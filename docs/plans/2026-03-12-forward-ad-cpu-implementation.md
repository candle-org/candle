# Forward-Mode AD (CPU) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement PyTorch-aligned forward-mode AD on CPU: dual levels, make/unpack dual tensors, JVP propagation, and autograd helpers needed by `test_autograd.py`.

**Architecture:** Store forward-mode tangents per level on `Tensor`. `candle.autograd.forward_ad` manages level stack and dual tensor API. Dispatcher applies JVP rules after primal kernel execution. Missing JVP rules raise `RuntimeError`. Autograd graph helpers and `_calculate_shape` are implemented for compatibility.

**Tech Stack:** Python, candle autograd/dispatch, pytest

---

### Task 1: Add Tensor forward‑mode storage + helpers

**Files:**
- Modify: `src/candle/_tensor.py`
- Test: `tests/cpu/test_forward_ad.py` (new)

**Step 1: Write the failing test**

Create `tests/cpu/test_forward_ad.py` with:

```python
import candle
from candle.autograd import forward_ad


def test_forward_ad_level_stack_and_make_unpack():
    x = candle.rand(2)
    with forward_ad.dual_level():
        tangent = candle.ones_like(x)
        dual = forward_ad.make_dual(x, tangent)
        primal, t = forward_ad.unpack_dual(dual)
        assert primal is x
        assert t is tangent


def test_forward_ad_nested_levels_isolated():
    x = candle.rand(2)
    with forward_ad.dual_level():
        t1 = candle.ones_like(x)
        d1 = forward_ad.make_dual(x, t1)
        with forward_ad.dual_level():
            t2 = candle.full_like(x, 2.0)
            d2 = forward_ad.make_dual(x, t2)
            _, t = forward_ad.unpack_dual(d2)
            assert t is t2
        _, t = forward_ad.unpack_dual(d1)
        assert t is t1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/cpu/test_forward_ad.py::test_forward_ad_level_stack_and_make_unpack -v`
Expected: FAIL because forward_ad is missing / no tangent storage.

**Step 3: Implement minimal Tensor storage**

In `src/candle/_tensor.py`, add:

```python
    def _fw_get(self, level):
        return getattr(self, "_fw_tangents", {}).get(level)

    def _fw_set(self, level, tangent):
        tangents = getattr(self, "_fw_tangents", None)
        if tangents is None:
            tangents = {}
            self._fw_tangents = tangents
        tangents[level] = tangent

    def _fw_clear(self, level):
        tangents = getattr(self, "_fw_tangents", None)
        if tangents is None:
            return
        tangents.pop(level, None)
        if not tangents:
            self._fw_tangents = {}

    def _fw_has(self, level):
        tangents = getattr(self, "_fw_tangents", None)
        return bool(tangents) and level in tangents
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/cpu/test_forward_ad.py::test_forward_ad_level_stack_and_make_unpack -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/candle/_tensor.py tests/cpu/test_forward_ad.py
git commit -m "feat: add forward AD tensor storage"
```

---

### Task 2: Implement candle.autograd.forward_ad API

**Files:**
- Modify: `src/candle/autograd/forward_ad.py`
- Modify: `src/candle/autograd/__init__.py`
- Test: `tests/cpu/test_forward_ad.py`

**Step 1: Write failing tests**

Extend `tests/cpu/test_forward_ad.py`:

```python
import pytest


def test_forward_ad_requires_active_level():
    x = candle.rand(2)
    t = candle.ones_like(x)
    with pytest.raises(RuntimeError):
        forward_ad.make_dual(x, t)


def test_forward_ad_exit_requires_lifo():
    with forward_ad.dual_level() as lvl1:
        with forward_ad.dual_level() as lvl2:
            pass
        with pytest.raises(RuntimeError):
            forward_ad.exit_dual_level(level=lvl1)
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/cpu/test_forward_ad.py::test_forward_ad_requires_active_level -v`
Expected: FAIL.

**Step 3: Implement forward_ad API**

In `src/candle/autograd/forward_ad.py`, implement:
- thread‑local `_current_level` and `_level_stack`
- `enter_dual_level`, `exit_dual_level`, `dual_level` context manager
- `make_dual(primal, tangent, *, level=None)` with dtype/shape validation
- `unpack_dual(tensor, *, level=None)` returning `(primal, tangent)`
- `UnpackedDualTensor` namedtuple

**Step 4: Export from autograd**

In `src/candle/autograd/__init__.py` add:

```python
from . import forward_ad
```

**Step 5: Run tests to verify pass**

Run: `pytest tests/cpu/test_forward_ad.py -v`
Expected: PASS.

**Step 6: Commit**

```bash
git add src/candle/autograd/forward_ad.py src/candle/autograd/__init__.py tests/cpu/test_forward_ad.py
git commit -m "feat: implement forward AD API"
```

---

### Task 3: Add JVP registry + dispatcher integration

**Files:**
- Modify: `src/candle/autograd/forward_ad.py`
- Modify: `src/candle/_dispatch/dispatcher.py`
- Test: `tests/cpu/test_forward_ad.py`

**Step 1: Write failing test**

Add to `tests/cpu/test_forward_ad.py`:

```python

def test_forward_ad_add_jvp():
    x = candle.rand(2)
    y = candle.rand(2)
    with forward_ad.dual_level():
        tx = candle.ones_like(x)
        ty = candle.full_like(y, 2.0)
        x = forward_ad.make_dual(x, tx)
        y = forward_ad.make_dual(y, ty)
        z = candle.add(x, y)
        _, tz = forward_ad.unpack_dual(z)
        assert tz is not None
        assert (tz == tx + ty).all()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/cpu/test_forward_ad.py::test_forward_ad_add_jvp -v`
Expected: FAIL with missing JVP error.

**Step 3: Implement JVP registry**

In `src/candle/autograd/forward_ad.py`, add:

```python
_JVP_RULES = {}

def register_jvp(op_name, fn):
    _JVP_RULES[op_name] = fn


def get_jvp(op_name):
    return _JVP_RULES.get(op_name)
```

**Step 4: Integrate into dispatcher**

In `src/candle/_dispatch/dispatcher.py`, after `_run_kernel()` result:
- Determine active forward‑AD level
- If any input has tangent → require JVP rule
- Compute tangent(s) and attach to output(s)
- Raise `RuntimeError` if rule missing

**Step 5: Add JVP rule for add**

In `forward_ad.py`, register:

```python
register_jvp("add", lambda x, y, *, _tangents: _tangents[0] + _tangents[1])
```

**Step 6: Run test to verify pass**

Run: `pytest tests/cpu/test_forward_ad.py::test_forward_ad_add_jvp -v`
Expected: PASS.

**Step 7: Commit**

```bash
git add src/candle/autograd/forward_ad.py src/candle/_dispatch/dispatcher.py tests/cpu/test_forward_ad.py
git commit -m "feat: add forward AD JVP dispatch"
```

---

### Task 4: Implement core autograd helpers (graph + _calculate_shape)

**Files:**
- Modify: `src/candle/autograd/graph.py`
- Modify: `src/candle/autograd/__init__.py`
- Test: `tests/cpu/test_forward_ad.py`

**Step 1: Write failing test**

Add to `tests/cpu/test_forward_ad.py`:

```python

def test_gradient_edge_roundtrip():
    x = candle.rand(2, requires_grad=True)
    out = x.clone()
    edge = candle.autograd.graph.get_gradient_edge(x)
    assert edge.node is x.grad_fn
    assert edge.output_nr == x.output_nr
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/cpu/test_forward_ad.py::test_gradient_edge_roundtrip -v`
Expected: FAIL with missing GradientEdge.

**Step 3: Implement GradientEdge and get_gradient_edge**

In `src/candle/autograd/graph.py`:
- Add `GradientEdge` dataclass (node, output_nr)
- Add `get_gradient_edge(tensor)` returning `GradientEdge(tensor.grad_fn, tensor.output_nr)`

**Step 4: Implement _calculate_shape**

In `src/candle/autograd/__init__.py`, add `_calculate_shape` modeled after PyTorch using `tensor.shape` (no nested tensor support yet). It should raise if `GradientEdge` + batched grads used.

**Step 5: Run test to verify pass**

Run: `pytest tests/cpu/test_forward_ad.py::test_gradient_edge_roundtrip -v`
Expected: PASS.

**Step 6: Commit**

```bash
git add src/candle/autograd/graph.py src/candle/autograd/__init__.py tests/cpu/test_forward_ad.py
git commit -m "feat: add autograd graph helpers"
```

---

### Task 5: Add core JVP rules for common ops

**Files:**
- Modify: `src/candle/autograd/forward_ad.py`
- Test: `tests/cpu/test_forward_ad.py`

**Step 1: Write failing tests**

Add tests for mul/div/sum/mean/reshape/view/transpose in `tests/cpu/test_forward_ad.py`.

**Step 2: Run tests to verify failure**

Run: `pytest tests/cpu/test_forward_ad.py -k "jvp" -v`
Expected: FAIL with missing JVP errors.

**Step 3: Implement JVP rules**

Implement rules for:
- `add`, `sub`, `mul`, `div`
- `matmul`, `mm`, `mv`
- `sum`, `mean`
- `view`, `reshape`, `transpose`, `permute`
- `neg`, `exp`, `log`, `tanh`, `relu`
- `getitem`/slice

**Step 4: Run tests to verify pass**

Run: `pytest tests/cpu/test_forward_ad.py -k "jvp" -v`
Expected: PASS for covered ops.

**Step 5: Commit**

```bash
git add src/candle/autograd/forward_ad.py tests/cpu/test_forward_ad.py
git commit -m "feat: add core forward AD JVP rules"
```

---

### Task 6: Run PyTorch autograd compatibility tests and iterate

**Files:**
- Modify: `compat/pytorch/xfail.yaml` (if needed)

**Step 1: Run `test_autograd.py`**

Run: `python compat/pytorch/run.py --file test_autograd.py`
Expected: Tests collect and run; remaining failures captured.

**Step 2: Address failures or xfail**

For edge cases not relevant to training/transformers, add explicit patterns in `compat/pytorch/xfail.yaml` with reasons.

**Step 3: Run contract suite**

Run: `pytest tests/contract/ -v --tb=short`
Expected: PASS.

**Step 4: Commit xfail updates**

```bash
git add compat/pytorch/xfail.yaml
git commit -m "test: xfail forward AD edge cases"
```
