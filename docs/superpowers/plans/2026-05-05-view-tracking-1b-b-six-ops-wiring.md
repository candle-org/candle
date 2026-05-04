# View-tracking 1B-B: Wire 5 view ops, drop hand-added Backward classes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `_view_func` / `_rev_view_func` callables onto `flatten`, `squeeze`, `narrow`, `movedim`, `unflatten` so the engine view-rebase block at `src/candle/_C/_autograd_engine.pyx:315-325` owns their gradient flow. In the same PR, delete the 7 hand-added `*Backward0` classes plus the matching `*_autograd` Python wrappers, the 6 Cython squeeze companions, and 10 dispatch-registration entries — none of which are needed once view rebase fires.

**Architecture:** Hand-written closures attached to view tensors, not generator-driven. Each closure binds the original op kwargs (e.g. `start_dim`, `dim`, `start`, `length`, `source`, `destination`) and the input shape so `view_func(new_base)` replays the forward op and `rev_view_func(grad)` undoes the shape transformation. After deletion of registration entries, `squeeze`/`flatten`/`unflatten`/`movedim`/`narrow` dispatch falls through to the no-autograd default kernel, which produces a plain view via the forward path; the engine's rebase block (live since PR #372) takes over from there. `contiguous` stays unchanged — see spec for why.

**Tech Stack:** Python 3.11, Cython 3.x, numpy, candle's `_C/` compiled core, candle's `_dispatch/` registry, candle's autograd engine in `_C/_autograd_engine.pyx`.

**Spec:** `docs/superpowers/specs/2026-05-05-view-tracking-1b-b-six-ops-wiring.md`

**Predecessor PRs:** #372 (view-tracking infrastructure 1A), #373 (1B-A conditional-view forward path).

---

## Task 0: Pre-flight verification

Make sure the worktree builds and the current baseline tests pass before adding wiring.

**Files:**
- Verify: `src/candle/_C/_autograd_engine.pyx` lines 315-325 (rebase block already live)
- Verify: `src/candle/_backends/common/view.py:10-35` (`_make_view` already accepts `view_func`/`rev_view_func` kwargs)
- Verify: `src/candle/_generated/view_funcs.py` (static inventory of 19 names)

- [ ] **Step 1: Confirm worktree state and branch**

Run:
```bash
cd /home/jenkins/lvyufeng/candle/.worktrees/view-tracking-1b-b
git status
git log --oneline -3
```
Expected: clean tree, branch `view-tracking-1b-b`, latest commit `3815893 docs(specs): add view-tracking 1B-B design`.

- [ ] **Step 2: Rebuild Cython extensions to start from a known-good state**

Run:
```bash
PYTHONPATH=$PWD/src conda run -n candle311 python setup.py build_ext --inplace
```
Expected: build completes with no errors. Warnings are okay.

- [ ] **Step 3: Run existing view-fastpath + view-tracking-infrastructure tests as a baseline**

Run:
```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest \
  tests/cpu/test_view_fastpath.py \
  tests/cpu/test_view_tracking_infrastructure.py \
  -v --tb=short
```
Expected: all pass (these were green at the end of 1B-A and PR #372).

If any fails: STOP. Investigate before wiring.

---

## Task 1: Add red boundary tests for the 1B-B rebase wiring

Pin the new contract in tests before wiring any code, so the work is genuinely TDD.

**Files:**
- Create: `tests/cpu/test_view_rebase_wiring.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/cpu/test_view_rebase_wiring.py` with:

```python
"""Pin engine view-rebase wiring for the 5 ops landed in 1B-B.

After 1B-B:
  - flatten/unflatten/squeeze/narrow/movedim views carry _view_func + _rev_view_func.
  - Backward through these ops flows via engine rebase, not via *Backward0 classes.
  - Chained views rebase recursively.
  - The 7 hand-added Backward0 classes are gone from _generated/functions.py.
"""
import numpy as np
import candle as torch


# --- Per-op rebase semantics ----------------------------------------------

def test_flatten_view_carries_view_func_and_rev_view_func():
    t = torch.randn(3, 4)
    u = t.flatten()
    assert callable(u._view_func)
    assert callable(u._rev_view_func)
    g = torch.ones(u.shape)
    assert u._rev_view_func(g).shape == t.shape


def test_squeeze_view_carries_view_func_and_rev_view_func():
    t = torch.randn(1, 3, 1, 4)
    u = t.squeeze()
    assert callable(u._view_func)
    assert callable(u._rev_view_func)
    assert u._rev_view_func(torch.ones(u.shape)).shape == t.shape


def test_narrow_rev_view_func_pads_with_zeros():
    t = torch.randn(5, 4)
    u = t.narrow(0, 1, 3)
    g = torch.ones(u.shape)
    g_base = u._rev_view_func(g)
    assert g_base.shape == t.shape
    np.testing.assert_array_equal(g_base.numpy()[0], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(g_base.numpy()[1], np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(g_base.numpy()[4], np.zeros(4, dtype=np.float32))


def test_movedim_rev_view_func_swaps_axes_back():
    t = torch.randn(2, 3, 4, requires_grad=True)
    u = t.movedim(0, 2)
    assert tuple(u.shape) == (3, 4, 2)
    g = torch.ones(u.shape)
    assert u._rev_view_func(g).shape == t.shape


def test_unflatten_rev_view_func_reshapes():
    t = torch.randn(2, 6)
    u = t.unflatten(1, (2, 3))
    g = torch.ones(u.shape)
    assert u._rev_view_func(g).shape == t.shape


# --- End-to-end backward semantics ----------------------------------------

def test_flatten_backward_via_rebase():
    t = torch.randn(3, 4, requires_grad=True)
    u = t.flatten()
    u.sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), np.ones((3, 4), dtype=np.float32))


def test_narrow_backward_via_rebase():
    t = torch.randn(5, 4, requires_grad=True)
    u = t.narrow(0, 1, 3)
    u.sum().backward()
    expected = np.zeros((5, 4), dtype=np.float32)
    expected[1:4] = 1.0
    np.testing.assert_array_equal(t.grad.numpy(), expected)


def test_squeeze_backward_via_rebase():
    t = torch.randn(1, 3, 1, 4, requires_grad=True)
    u = t.squeeze()
    u.sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), np.ones((1, 3, 1, 4), dtype=np.float32))


def test_movedim_backward_via_rebase():
    t = torch.randn(2, 3, 4, requires_grad=True)
    u = t.movedim(0, 2)
    u.sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), np.ones((2, 3, 4), dtype=np.float32))


def test_unflatten_backward_via_rebase():
    t = torch.randn(2, 6, requires_grad=True)
    u = t.unflatten(1, (2, 3))
    u.sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), np.ones((2, 6), dtype=np.float32))


# --- Chained views rebase recursively -------------------------------------

def test_chained_view_rebase_flatten_then_narrow():
    t = torch.randn(3, 4, requires_grad=True)
    u = t.flatten().narrow(0, 0, 6)  # 2 nested views
    u.sum().backward()
    expected = np.zeros((3, 4), dtype=np.float32).reshape(-1)
    expected[:6] = 1.0
    np.testing.assert_array_equal(t.grad.numpy().reshape(-1), expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest \
  tests/cpu/test_view_rebase_wiring.py -v --tb=short
```

Expected: most assertions fail because `_view_func` / `_rev_view_func` are still `None` for these ops. Pre-existing backward tests (`test_*_backward_via_rebase`) likely pass already because the hand-added `*Backward0` classes still produce the right gradient — but they pass via the OLD path, not the rebase path. That's fine; the per-op `assert callable(u._view_func)` checks make the failure crisp.

- [ ] **Step 3: Commit the red tests**

```bash
git add tests/cpu/test_view_rebase_wiring.py
git commit -m "test(cpu): pin 1B-B view-rebase wiring contract for 5 ops"
```

---

## Task 2: Wire `view_func` / `rev_view_func` for `squeeze` in `_backends/common/view.py`

`squeeze` constructs its view via `_make_view` directly, so wiring is just adding the two kwargs.

**Files:**
- Modify: `src/candle/_backends/common/view.py:134-162`

- [ ] **Step 1: Edit the `squeeze` function to bind closures and pass them to `_make_view`**

Replace:
```python
def squeeze(a, dim=None):
    shape = list(a.shape)
    stride = list(a.stride)
    if dim is not None:
        if isinstance(dim, (list, tuple)):
            if dim:
                ndim = len(shape)
                targets = set()
                for item in dim:
                    d = item if item >= 0 else item + ndim
                    targets.add(d)
                pairs = [
                    (s, st)
                    for idx, (s, st) in enumerate(zip(shape, stride))
                    if idx not in targets or s != 1
                ]
                shape = [p[0] for p in pairs]
                stride = [p[1] for p in pairs]
        else:
            d = dim if dim >= 0 else dim + len(shape)
            if 0 <= d < len(shape) and shape[d] == 1:
                del shape[d]
                del stride[d]
    else:
        pairs = [(s, st) for s, st in zip(shape, stride) if s != 1]
        shape = [p[0] for p in pairs]
        stride = [p[1] for p in pairs]
    base = _get_base(a)
    return _make_view(base, shape, stride, a.offset, "squeeze", source=a)
```

With:
```python
def squeeze(a, dim=None):
    shape = list(a.shape)
    stride = list(a.stride)
    if dim is not None:
        if isinstance(dim, (list, tuple)):
            if dim:
                ndim = len(shape)
                targets = set()
                for item in dim:
                    d = item if item >= 0 else item + ndim
                    targets.add(d)
                pairs = [
                    (s, st)
                    for idx, (s, st) in enumerate(zip(shape, stride))
                    if idx not in targets or s != 1
                ]
                shape = [p[0] for p in pairs]
                stride = [p[1] for p in pairs]
        else:
            d = dim if dim >= 0 else dim + len(shape)
            if 0 <= d < len(shape) and shape[d] == 1:
                del shape[d]
                del stride[d]
    else:
        pairs = [(s, st) for s, st in zip(shape, stride) if s != 1]
        shape = [p[0] for p in pairs]
        stride = [p[1] for p in pairs]
    base = _get_base(a)
    input_shape = a.shape

    def _squeeze_view_func(new_base, _dim=dim):
        if _dim is None:
            return new_base.squeeze()
        return new_base.squeeze(_dim)

    def _squeeze_rev_view_func(grad_view, _shape=input_shape):
        return grad_view.reshape(_shape)

    return _make_view(
        base, shape, stride, a.offset, "squeeze",
        source=a,
        view_func=_squeeze_view_func,
        rev_view_func=_squeeze_rev_view_func,
    )
```

- [ ] **Step 2: Run squeeze-specific assertions to verify they pass**

Run:
```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest \
  tests/cpu/test_view_rebase_wiring.py::test_squeeze_view_carries_view_func_and_rev_view_func \
  -v --tb=short
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/candle/_backends/common/view.py
git commit -m "feat(view): wire view_func/rev_view_func on squeeze view"
```

---

## Task 3: Wire `view_func` / `rev_view_func` for `narrow` in `_backends/common/view.py`

`narrow` constructs its view via `_make_view` directly, so wiring is the same pattern. The `_rev_view_func` is non-trivial: pad with zeros and copy the gradient into the narrow window.

**Files:**
- Modify: `src/candle/_backends/common/view.py:177-183`

- [ ] **Step 1: Edit the `narrow` function**

Replace:
```python
def narrow(a, dim, start, length, *, creation_kind=None):
    d = dim if dim >= 0 else dim + len(a.shape)
    new_shape = list(a.shape)
    new_shape[d] = int(length)
    new_offset = a.offset + int(start) * a.stride[d]
    base = _get_base(a)
    return _make_view(base, tuple(new_shape), a.stride, new_offset, "narrow", source=a, creation_kind=creation_kind)
```

With:
```python
def narrow(a, dim, start, length, *, creation_kind=None):
    d = dim if dim >= 0 else dim + len(a.shape)
    new_shape = list(a.shape)
    new_shape[d] = int(length)
    new_offset = a.offset + int(start) * a.stride[d]
    base = _get_base(a)
    input_shape = a.shape
    start_int = int(start)
    length_int = int(length)

    def _narrow_view_func(new_base, _dim=d, _start=start_int, _len=length_int):
        return new_base.narrow(_dim, _start, _len)

    def _narrow_rev_view_func(grad_view, _shape=input_shape, _dim=d, _start=start_int, _len=length_int):
        # Pad grad_view back to input_shape with zeros at non-narrow positions.
        # Mirrors NarrowBackward0 / _narrow_backward_helper.
        from candle import zeros as _zeros
        grad_input = _zeros(_shape, dtype=grad_view.dtype, device=grad_view.device)
        grad_input.narrow(_dim, _start, _len).copy_(grad_view)
        return grad_input

    return _make_view(
        base, tuple(new_shape), a.stride, new_offset, "narrow",
        source=a, creation_kind=creation_kind,
        view_func=_narrow_view_func,
        rev_view_func=_narrow_rev_view_func,
    )
```

- [ ] **Step 2: Run narrow-specific assertions**

Run:
```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest \
  tests/cpu/test_view_rebase_wiring.py::test_narrow_rev_view_func_pads_with_zeros \
  -v --tb=short
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/candle/_backends/common/view.py
git commit -m "feat(view): wire view_func/rev_view_func on narrow view"
```

---

## Task 4: Wire `view_func` / `rev_view_func` for `flatten` in `_C/_tensor_api.pyx`

`flatten` does not call `_make_view` directly — it routes through `self.reshape(new_shape)` and overrides `_view_meta["op"]` afterwards. We attach the callables in the same after-the-fact step.

**Files:**
- Modify: `src/candle/_C/_tensor_api.pyx:806-842` (`tensor_flatten`)

- [ ] **Step 1: Edit `tensor_flatten` to attach the closures after reshape**

Replace:
```cython
def tensor_flatten(self, start_dim=0, end_dim=-1):
    cdef Py_ssize_t ndim = len(self.shape)
    cdef Py_ssize_t i
    cdef Py_ssize_t flattened = 1
    cdef tuple new_shape
    cdef object result
    cdef object meta

    if ndim == 0:
        result = self.reshape((1,))
        meta = getattr(result, "_view_meta", None)
        if meta is not None:
            meta = dict(meta)
            meta["op"] = "flatten"
            result._view_meta = meta
        return result
    if start_dim < 0:
        start_dim += ndim
    if end_dim < 0:
        end_dim += ndim
    if start_dim < 0 or start_dim >= ndim:
        raise IndexError("Dimension out of range")
    if end_dim < 0 or end_dim >= ndim:
        raise IndexError("Dimension out of range")
    if start_dim > end_dim:
        raise RuntimeError("flatten() has invalid args: start_dim cannot come after end_dim")

    for i in range(start_dim, end_dim + 1):
        flattened *= self.shape[i]
    new_shape = self.shape[:start_dim] + (flattened,) + self.shape[end_dim + 1:]
    result = self.reshape(new_shape)
    meta = getattr(result, "_view_meta", None)
    if meta is not None:
        meta = dict(meta)
        meta["op"] = "flatten"
        result._view_meta = meta
    return result
```

With:
```cython
def tensor_flatten(self, start_dim=0, end_dim=-1):
    cdef Py_ssize_t ndim = len(self.shape)
    cdef Py_ssize_t i
    cdef Py_ssize_t flattened = 1
    cdef tuple new_shape
    cdef tuple input_shape = tuple(self.shape)
    cdef object result
    cdef object meta

    if ndim == 0:
        result = self.reshape((1,))
        meta = getattr(result, "_view_meta", None)
        if meta is not None:
            meta = dict(meta)
            meta["op"] = "flatten"
            result._view_meta = meta
        if result is not self:
            _attach_flatten_view_funcs(result, 0, 0, input_shape)
        return result
    if start_dim < 0:
        start_dim += ndim
    if end_dim < 0:
        end_dim += ndim
    if start_dim < 0 or start_dim >= ndim:
        raise IndexError("Dimension out of range")
    if end_dim < 0 or end_dim >= ndim:
        raise IndexError("Dimension out of range")
    if start_dim > end_dim:
        raise RuntimeError("flatten() has invalid args: start_dim cannot come after end_dim")

    for i in range(start_dim, end_dim + 1):
        flattened *= self.shape[i]
    new_shape = self.shape[:start_dim] + (flattened,) + self.shape[end_dim + 1:]
    result = self.reshape(new_shape)
    meta = getattr(result, "_view_meta", None)
    if meta is not None:
        meta = dict(meta)
        meta["op"] = "flatten"
        result._view_meta = meta
    if result is not self:
        _attach_flatten_view_funcs(result, int(start_dim), int(end_dim), input_shape)
    return result


def _attach_flatten_view_funcs(result, start_dim, end_dim, input_shape):
    """Attach view_func/rev_view_func for flatten so engine rebase owns grad."""
    def _flatten_view_func(new_base, _start=start_dim, _end=end_dim):
        return new_base.flatten(_start, _end)

    def _flatten_rev_view_func(grad_view, _shape=input_shape):
        return grad_view.reshape(_shape)

    result._view_func = _flatten_view_func
    result._rev_view_func = _flatten_rev_view_func
```

Note: the helper is defined as a module-level Python `def` (not `cdef`) so closures work and Cython accepts it. The `result is not self` guard avoids attaching when reshape returned the same object (e.g. for already-flat inputs that `reshape` short-circuits).

- [ ] **Step 2: Rebuild Cython**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python setup.py build_ext --inplace
```
Expected: build succeeds.

- [ ] **Step 3: Run flatten-specific assertions**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest \
  tests/cpu/test_view_rebase_wiring.py::test_flatten_view_carries_view_func_and_rev_view_func \
  -v --tb=short
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/candle/_C/_tensor_api.pyx
git commit -m "feat(_C): wire view_func/rev_view_func on flatten output"
```

---

## Task 5: Wire `view_func` / `rev_view_func` for `unflatten` in `_C/_tensor_api.pyx`

Same pattern as `flatten` — attach closures after the existing reshape+meta override.

**Files:**
- Modify: `src/candle/_C/_tensor_api.pyx:1565-1582` (`tensor_unflatten`)

- [ ] **Step 1: Edit `tensor_unflatten`**

Replace:
```cython
def tensor_unflatten(self, dim, sizes):
    cdef Py_ssize_t ndim = len(self.shape)
    cdef tuple new_shape
    cdef object result
    cdef object meta
    if dim < 0:
        dim += ndim
    new_shape = self.shape[:dim] + tuple(sizes) + self.shape[dim + 1:]
    # Use reshape (not view) so non-contiguous input falls back to a copy
    # instead of raising. PyTorch's unflatten has the same conditional-view
    # semantics.
    result = self.reshape(new_shape)
    meta = getattr(result, "_view_meta", None)
    if meta is not None:
        meta = dict(meta)
        meta["op"] = "unflatten"
        result._view_meta = meta
    return result
```

With:
```cython
def tensor_unflatten(self, dim, sizes):
    cdef Py_ssize_t ndim = len(self.shape)
    cdef tuple new_shape
    cdef tuple input_shape = tuple(self.shape)
    cdef tuple sizes_tuple = tuple(sizes)
    cdef object result
    cdef object meta
    if dim < 0:
        dim += ndim
    new_shape = self.shape[:dim] + sizes_tuple + self.shape[dim + 1:]
    # Use reshape (not view) so non-contiguous input falls back to a copy
    # instead of raising. PyTorch's unflatten has the same conditional-view
    # semantics.
    result = self.reshape(new_shape)
    meta = getattr(result, "_view_meta", None)
    if meta is not None:
        meta = dict(meta)
        meta["op"] = "unflatten"
        result._view_meta = meta
    if result is not self:
        _attach_unflatten_view_funcs(result, int(dim), sizes_tuple, input_shape)
    return result


def _attach_unflatten_view_funcs(result, dim, sizes_tuple, input_shape):
    """Attach view_func/rev_view_func for unflatten so engine rebase owns grad."""
    def _unflatten_view_func(new_base, _dim=dim, _sizes=sizes_tuple):
        return new_base.unflatten(_dim, _sizes)

    def _unflatten_rev_view_func(grad_view, _shape=input_shape):
        return grad_view.reshape(_shape)

    result._view_func = _unflatten_view_func
    result._rev_view_func = _unflatten_rev_view_func
```

- [ ] **Step 2: Rebuild Cython**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python setup.py build_ext --inplace
```

- [ ] **Step 3: Run unflatten assertions**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest \
  tests/cpu/test_view_rebase_wiring.py::test_unflatten_rev_view_func_reshapes \
  -v --tb=short
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/candle/_C/_tensor_api.pyx
git commit -m "feat(_C): wire view_func/rev_view_func on unflatten output"
```

---

## Task 6: Wire `view_func` / `rev_view_func` for `movedim` in `_C/_tensor_api.pyx`

`movedim` has two paths: a no-grad fast-path (line 1953-1973, doesn't matter for autograd) and a `requires_grad=True` dispatched path (line 1924-1925). We wire only the dispatched path because that's where gradient flows.

**Files:**
- Modify: `src/candle/_C/_tensor_api.pyx:1909-1974` (`tensor_movedim`)

- [ ] **Step 1: Edit `tensor_movedim` to attach closures on the dispatched-path return value**

Replace the `if self.requires_grad:` early return and surrounding code:

Before:
```cython
def tensor_movedim(self, source, destination):
    cdef object ndim
    ...
    cdef object v

    _ensure_dispatch_ref()
    if self.requires_grad:
        return _dispatch_fn("movedim", self.device.type, self, source, destination)

    ndim = len(self.shape)
    ...
```

After:
```cython
def tensor_movedim(self, source, destination):
    cdef object ndim
    ...
    cdef object v
    cdef object grad_path_result

    _ensure_dispatch_ref()
    if self.requires_grad:
        grad_path_result = _dispatch_fn("movedim", self.device.type, self, source, destination)
        if grad_path_result is not self:
            _attach_movedim_view_funcs(grad_path_result, source, destination)
        return grad_path_result

    ndim = len(self.shape)
    ...
```

And add the helper at module scope (keep it next to `tensor_movedim`):

```cython
def _attach_movedim_view_funcs(result, source, destination):
    """Attach view_func/rev_view_func for movedim so engine rebase owns grad.

    movedim is its own inverse with src↔dst swapped, so rev_view_func
    just calls movedim(destination, source) on the grad view.
    """
    def _movedim_view_func(new_base, _src=source, _dst=destination):
        return new_base.movedim(_src, _dst)

    def _movedim_rev_view_func(grad_view, _src=source, _dst=destination):
        return grad_view.movedim(_dst, _src)

    result._view_func = _movedim_view_func
    result._rev_view_func = _movedim_rev_view_func
```

Note: the spec discusses tuple-input movedim (`movedim((0,1), (2,3))`) as a possible edge case. The math holds: `movedim(dst, src)` is the inverse of `movedim(src, dst)` for both scalar and tuple inputs because the reordering is symmetric. The new `test_movedim_rev_view_func_swaps_axes_back` test exercises the scalar case; the chained-view test exercises composition.

- [ ] **Step 2: Rebuild Cython**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python setup.py build_ext --inplace
```

- [ ] **Step 3: Run movedim assertions**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest \
  tests/cpu/test_view_rebase_wiring.py::test_movedim_rev_view_func_swaps_axes_back \
  -v --tb=short
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/candle/_C/_tensor_api.pyx
git commit -m "feat(_C): wire view_func/rev_view_func on movedim grad path"
```

---

## Task 7: Flip the four `_view_func is None` assertions in `tests/cpu/test_view_fastpath.py`

After 1B-B, `flatten` and `unflatten` views carry callables. The 1B-A pin needs to be flipped.

**Files:**
- Modify: `tests/cpu/test_view_fastpath.py` lines 41-50, 70-79

- [ ] **Step 1: Edit `test_flatten_shares_storage_when_input_is_contiguous`**

Replace:
```python
def test_flatten_shares_storage_when_input_is_contiguous():
    t = torch.randn(3, 4)
    u = t.flatten()
    assert u.storage() is t.storage()
    assert u._base is not None
    assert u._view_meta is not None
    assert u._view_meta["op"] == "flatten"
    assert u._view_func is None
    assert u._rev_view_func is None
```

With:
```python
def test_flatten_shares_storage_when_input_is_contiguous():
    t = torch.randn(3, 4)
    u = t.flatten()
    assert u.storage() is t.storage()
    assert u._base is not None
    assert u._view_meta is not None
    assert u._view_meta["op"] == "flatten"
    # 1B-B wiring: view carries view_func/rev_view_func; engine rebase owns grad.
    assert callable(u._view_func)
    assert callable(u._rev_view_func)
    g = torch.ones(u.shape)
    assert u._rev_view_func(g).shape == t.shape
```

- [ ] **Step 2: Edit `test_unflatten_shares_storage_when_input_is_contiguous`**

Replace:
```python
def test_unflatten_shares_storage_when_input_is_contiguous():
    t = torch.randn(2, 6)
    u = t.unflatten(1, (2, 3))
    assert u.storage() is t.storage()
    assert u._base is not None
    assert u._view_meta is not None
    assert u._view_meta["op"] == "unflatten"
    assert u._view_func is None
    assert u._rev_view_func is None
```

With:
```python
def test_unflatten_shares_storage_when_input_is_contiguous():
    t = torch.randn(2, 6)
    u = t.unflatten(1, (2, 3))
    assert u.storage() is t.storage()
    assert u._base is not None
    assert u._view_meta is not None
    assert u._view_meta["op"] == "unflatten"
    # 1B-B wiring: view carries view_func/rev_view_func; engine rebase owns grad.
    assert callable(u._view_func)
    assert callable(u._rev_view_func)
    g = torch.ones(u.shape)
    assert u._rev_view_func(g).shape == t.shape
```

- [ ] **Step 3: Run the updated tests**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest \
  tests/cpu/test_view_fastpath.py -v --tb=short
```
Expected: all 11 tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/cpu/test_view_fastpath.py
git commit -m "test(cpu): flip flatten/unflatten _view_func pins for 1B-B"
```

---

## Task 8: Run the full rebase wiring test file end-to-end

Confirm every assertion in `test_view_rebase_wiring.py` passes. The end-to-end backward tests (`test_*_backward_via_rebase`) exercise the engine block with the deletion deferred to Task 9 — they should pass either way, but seeing all green now means the wiring side is solid before we delete.

- [ ] **Step 1: Run the full test file**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest \
  tests/cpu/test_view_rebase_wiring.py -v --tb=short
```

Expected: all 11 tests pass.

If `test_chained_view_rebase_flatten_then_narrow` fails: the engine's recursive rebase call may be skipping a layer. Read the engine block at `src/candle/_C/_autograd_engine.pyx:315-325` and verify it recurses correctly. The `apply_hooks=False` arg on the recursive call is intentional — only the top frame applies hooks.

If a per-op assert fails: check that the corresponding wiring task (2/3/4/5/6) actually committed. Run `git log --oneline -10` to confirm.

---

## Task 9: Delete 7 hand-added Backward classes from `_generated/functions.py`

Remove the entire class bodies. Helpers like `_narrow_backward_helper` / `_movedim_backward_helper` can stay (small, harmless when unreferenced).

**Files:**
- Modify: `src/candle/_generated/functions.py`

- [ ] **Step 1: Identify exact line ranges before deleting**

Run:
```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -c "
import re, sys
with open('src/candle/_generated/functions.py') as f:
    text = f.read()
classes = ['SqueezeBackward0', 'SqueezeDimBackward0', 'SqueezeDimsBackward0',
           'FlattenBackward0', 'MovedimBackward0', 'NarrowBackward0', 'UnflattenBackward0']
for c in classes:
    m = re.search(rf'^class {c}\(Node\):', text, re.M)
    if m:
        lines = text[:m.start()].count('\n') + 1
        # find end: next class or function def at column 0
        rest = text[m.start():]
        end_match = re.search(r'\n(class |def )', rest[10:])
        if end_match:
            end_line = lines + rest[:10 + end_match.start()].count('\n')
        else:
            end_line = lines + rest.count('\n')
        print(f'{c}: lines {lines}-{end_line}')
    else:
        print(f'{c}: NOT FOUND')
"
```

Expected output: 7 line-range tuples. Use these to guide the next step.

- [ ] **Step 2: Delete each class body**

Use `Edit` tool (one class at a time) to delete each `class XxxBackward0(Node): ...` block, including its trailing blank line up to the next class/def. Order: `SqueezeBackward0`, `SqueezeDimBackward0`, `SqueezeDimsBackward0`, `FlattenBackward0`, `MovedimBackward0`, `NarrowBackward0`, `UnflattenBackward0`.

After each deletion, do not run anything yet — keep going. After all 7 are gone:

- [ ] **Step 3: Confirm the classes are gone**

```bash
grep -n "^class \(Squeeze\|SqueezeDim\|SqueezeDims\|Flatten\|Movedim\|Narrow\|Unflatten\)Backward0" src/candle/_generated/functions.py
```
Expected: empty output. (Note: `NarrowCopyBackward0` is a different class for `narrow_copy` — leave it alone.)

- [ ] **Step 4: Verify file still parses**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -c "from candle._generated import functions"
```
Expected: no import errors. If `ImportError`/`SyntaxError`: the deletion damaged surrounding code. Use `git diff src/candle/_generated/functions.py` to inspect.

- [ ] **Step 5: Commit**

```bash
git add src/candle/_generated/functions.py
git commit -m "refactor(generated): drop 7 hand-added view-op Backward classes"
```

---

## Task 10: Delete `*_autograd` wrappers from `_generated/variable_type.py`

Remove the Python wrappers for `squeeze`/`squeeze_`/`squeeze.dim`/`squeeze_.dim`/`squeeze.dims`/`squeeze_.dims`/`flatten`/`unflatten`/`movedim`/`narrow` plus their `*_post` siblings.

**Files:**
- Modify: `src/candle/_generated/variable_type.py`

- [ ] **Step 1: Locate exact lines**

Confirmed inventory (from project exploration):
* `squeeze_autograd` line 4353
* `squeeze_dim_autograd` line 4366
* `squeeze_dims_autograd` line 4380
* `squeeze__autograd` line 4394
* `squeeze__dim_autograd` line 4407
* `squeeze__dims_autograd` line 4421
* `squeeze_autograd_post` line 12627 (note: signature `*_squeeze_args` due to multi-overload)
* `squeeze_dim_autograd_post` line 12637
* `squeeze_dims_autograd_post` line 12648
* `squeeze__autograd_post` line 12659
* `squeeze__dim_autograd_post` line 12669
* `squeeze__dims_autograd_post` line 12680
* `flatten_autograd` line 17786
* `flatten_autograd_post` line 17802
* `movedim_autograd` line 18663
* `movedim_autograd_post` line 18679
* `narrow_autograd` line 18752
* `narrow_autograd_post` line 18769
* `unflatten_autograd` line 19599
* `unflatten_autograd_post` line 19615

- [ ] **Step 2: Delete each function body**

Use `Edit` tool one at a time. For each function: delete from `def XXX_autograd(...):` through the trailing blank line just before the next `def`. Work in reverse line order so earlier line numbers remain valid: 19615 → 19599 → 18769 → 18752 → 18679 → 18663 → 17802 → 17786 → 12680 → 12669 → 12659 → 12648 → 12637 → 12627 → 4421 → 4407 → 4394 → 4380 → 4366 → 4353.

- [ ] **Step 3: Confirm deletions**

```bash
grep -n "^def \(squeeze\|squeeze_\|flatten\|unflatten\|movedim\|narrow\)_autograd" src/candle/_generated/variable_type.py
```
Expected: empty output. (Note: `unsqueeze_autograd`, `narrow_copy_autograd` are different ops — they should stay.)

- [ ] **Step 4: Verify file still imports**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -c "from candle._generated import variable_type"
```
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/candle/_generated/variable_type.py
git commit -m "refactor(generated): drop view-op autograd wrappers from variable_type.py"
```

---

## Task 11: Delete squeeze companions from `_generated/_variable_type_cy.pyx`

The Cython companion holds 12 squeeze entries (6 `_autograd` + 6 `_autograd_post`). The 5 ops `flatten`/`unflatten`/`movedim`/`narrow` are NOT in the Cython companion — Python-only.

**Files:**
- Modify: `src/candle/_generated/_variable_type_cy.pyx`

- [ ] **Step 1: Locate squeeze entries**

Confirmed inventory:
* `squeeze_autograd` line 4711
* `squeeze_dim_autograd` line 4725
* `squeeze_dims_autograd` line 4740
* `squeeze__autograd` line 4755
* `squeeze__dim_autograd` line 4769
* `squeeze__dims_autograd` line 4784
* `squeeze_autograd_post` line 13615
* `squeeze_dim_autograd_post` line 13626
* `squeeze_dims_autograd_post` line 13638
* `squeeze__autograd_post` line 13650
* `squeeze__dim_autograd_post` line 13661
* `squeeze__dims_autograd_post` line 13673

- [ ] **Step 2: Delete each function body**

Same pattern as Task 10 — one at a time, reverse line order: 13673 → 13661 → 13650 → 13638 → 13626 → 13615 → 4784 → 4769 → 4755 → 4740 → 4725 → 4711.

- [ ] **Step 3: Confirm deletions**

```bash
grep -n "^def squeeze.*autograd" src/candle/_generated/_variable_type_cy.pyx
```
Expected: empty output.

- [ ] **Step 4: Rebuild Cython companion**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python setup.py build_ext --inplace
```
Expected: build succeeds.

- [ ] **Step 5: Verify import**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -c "from candle._generated import _variable_type_cy"
```
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/candle/_generated/_variable_type_cy.pyx
git commit -m "refactor(generated): drop squeeze autograd wrappers from Cython companion"
```

---

## Task 12: Delete dispatch-registration entries from `_generated/registration.py`

Remove the 10 `register_autograd_*` lines for the 5 ops.

**Files:**
- Modify: `src/candle/_generated/registration.py`

- [ ] **Step 1: Identify exact lines**

Confirmed inventory:
* line 273: `register_autograd_kernels('squeeze', default=_VT_PY.squeeze_autograd, ...)`
* line 274: `register_autograd_post_kernels('squeeze', _VT_PY.squeeze_autograd_post)`
* line 482: `register_autograd_kernels('flatten', default=_VT_PY.flatten_autograd, ...)`
* line 483: `register_autograd_post_kernels('flatten', _VT_PY.flatten_autograd_post)`
* line 484: `register_autograd_kernels('unflatten', default=_VT_PY.unflatten_autograd, ...)`
* line 485: `register_autograd_post_kernels('unflatten', _VT_PY.unflatten_autograd_post)`
* line 486: `register_autograd_kernels('movedim', default=_VT_PY.movedim_autograd, ...)`
* line 487: `register_autograd_post_kernels('movedim', _VT_PY.movedim_autograd_post)`
* line 498: `register_autograd_kernels('narrow', default=_VT_PY.narrow_autograd, ...)`
* line 499: `register_autograd_post_kernels('narrow', _VT_PY.narrow_autograd_post)`

- [ ] **Step 2: Delete each line**

Use `Edit` tool with `replace_all=False`. For each pair, delete both the `register_autograd_kernels(...)` line and the matching `register_autograd_post_kernels(...)` line. Work in reverse line order to keep numbers stable: 498-499 → 486-487 → 484-485 → 482-483 → 273-274.

- [ ] **Step 3: Confirm deletions**

```bash
grep -n "register_autograd.*\(squeeze\|flatten\|unflatten\|movedim\|narrow\)" src/candle/_generated/registration.py | grep -v unsqueeze | grep -v narrow_copy
```
Expected: empty output.

- [ ] **Step 4: Verify import**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -c "from candle._generated import registration"
```
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/candle/_generated/registration.py
git commit -m "refactor(dispatch): drop view-op autograd registrations; rebase owns grad"
```

---

## Task 13: Add codegen-drift coordination

Two artifacts: a comment in `derivatives.yaml` so future regenerator runs know not to re-emit, and a contract test enforcing the deletion.

**Files:**
- Modify: `tools/autograd/derivatives.yaml`
- Create: `tests/contract/test_view_ops_no_backward_class.py`

- [ ] **Step 1: Add comment in `derivatives.yaml` at the top of squeeze entries**

Find line 1627 (`- name: squeeze(...)` first squeeze entry) and prepend a comment block. Use `Edit` to replace:

```yaml
- name: squeeze(Tensor(a) self) -> Tensor(a)
  self: unsqueeze_to(grad, self.sym_sizes())
```

With:

```yaml
# View-tracked ops (1B-B): squeeze / squeeze_ / squeeze.dim / squeeze.dims /
# squeeze_.dim / squeeze_.dims gradients flow via view rebase
# (_view_func/_rev_view_func wired in src/candle/_backends/common/view.py).
# The generator MUST NOT emit *Backward0 classes or *_autograd wrappers
# for these entries. flatten/unflatten/movedim/narrow are not in this
# yaml file but follow the same rule.
- name: squeeze(Tensor(a) self) -> Tensor(a)
  self: unsqueeze_to(grad, self.sym_sizes())
```

- [ ] **Step 2: Write contract regression test**

Create `tests/contract/test_view_ops_no_backward_class.py`:

```python
"""Contract: 5 view ops (flatten, unflatten, squeeze[.dim/.dims], movedim, narrow)
gradient flow goes through engine view-rebase, NOT through hand-added Backward
classes. Regression guard against accidental re-emission by the autograd
generator (tasks #223-#225).
"""
import importlib


REMOVED_BACKWARD_CLASSES = (
    "SqueezeBackward0",
    "SqueezeDimBackward0",
    "SqueezeDimsBackward0",
    "FlattenBackward0",
    "UnflattenBackward0",
    "NarrowBackward0",
    "MovedimBackward0",
)

REMOVED_AUTOGRAD_WRAPPERS = (
    "squeeze_autograd",
    "squeeze_dim_autograd",
    "squeeze_dims_autograd",
    "squeeze__autograd",
    "squeeze__dim_autograd",
    "squeeze__dims_autograd",
    "flatten_autograd",
    "unflatten_autograd",
    "movedim_autograd",
    "narrow_autograd",
)


def test_view_op_backward_classes_are_not_emitted():
    functions = importlib.import_module("candle._generated.functions")
    for name in REMOVED_BACKWARD_CLASSES:
        assert not hasattr(functions, name), (
            f"{name} was deleted in 1B-B because view rebase owns its gradient. "
            f"If the generator re-emitted it, teach the generator to skip "
            f"view-tracked ops."
        )


def test_view_op_autograd_wrappers_are_not_emitted():
    variable_type = importlib.import_module("candle._generated.variable_type")
    for name in REMOVED_AUTOGRAD_WRAPPERS:
        assert not hasattr(variable_type, name), (
            f"{name} was deleted in 1B-B. View rebase owns this op's gradient; "
            f"a wrapper is unnecessary."
        )


def test_view_op_dispatch_registrations_are_absent():
    """The dispatcher registry should have no autograd kernel for these 5 ops.
    Falling through to the default no-autograd kernel is what makes the engine
    rebase block fire on the leaf view.
    """
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey

    for op in ("squeeze", "flatten", "unflatten", "movedim", "narrow"):
        entry = registry._entries.get(op)
        if entry is None:
            continue  # op not in registry at all is fine
        for key in (DispatchKey.Autograd, DispatchKey.AutogradCPU):
            assert key not in entry.kernels, (
                f"{op} should have NO {key} kernel after 1B-B; view rebase "
                f"owns gradient. Found kernel: {entry.kernels.get(key)}"
            )
```

Note: The test reaches into `registry._entries` and `entry.kernels` — these are private attributes. If the contract test fails because of a different attribute layout, inspect `src/candle/_dispatch/registry.py` and adapt.

- [ ] **Step 3: Run the contract test**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest \
  tests/contract/test_view_ops_no_backward_class.py -v --tb=short
```
Expected: all 3 tests PASS (since Tasks 9-12 already deleted the artifacts).

If `test_view_op_dispatch_registrations_are_absent` fails because of attribute names, edit the test to match `registry`'s actual layout. Don't loosen the assertion — keep it strict.

- [ ] **Step 4: Commit**

```bash
git add tools/autograd/derivatives.yaml tests/contract/test_view_ops_no_backward_class.py
git commit -m "test(contract): pin view-op deletion against codegen drift regression"
```

---

## Task 14: Run the focused 1B-B test surface

Confirm all view-tracking + rebase-wiring tests pass after every wiring + deletion is in.

- [ ] **Step 1: Run focused tests**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest \
  tests/cpu/test_view_fastpath.py \
  tests/cpu/test_view_rebase_wiring.py \
  tests/cpu/test_view_tracking_infrastructure.py \
  tests/contract/test_view_ops_no_backward_class.py \
  -v --tb=short
```

Expected: all green. If anything fails, re-read which task most likely introduced the regression and look at the diff.

- [ ] **Step 2: Snapshot status**

```bash
git status
git log --oneline -15
```
Expected: clean tree, ~10 new commits since `3815893`.

---

## Task 15: Run the broader autograd test surface

A wide regression check across ops touching `flatten`/`squeeze`/`narrow`/`movedim`/`unflatten`.

- [ ] **Step 1: Run autograd-heavy modules**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest \
  tests/cpu/test_autograd_api.py \
  tests/cpu/test_autograd_function.py \
  tests/contract/test_autograd_create_graph.py \
  tests/contract/test_autograd_engine_topo.py \
  tests/contract/test_autograd_retain_graph.py \
  -v --tb=short
```

Expected: all green. If a test fails:
- Read the failure message.
- If it's a missing attribute (e.g. `SqueezeBackward0`), inspect: a test was probably reaching into a Backward class internals. The fix is to update the test to look for view-rebase semantics instead, or to mark the test as 1B-B-superseded. Do NOT restore the deleted class.
- If it's a gradient mismatch, investigate the rebase math. Likely `_rev_view_func` is wrong for an edge case.

---

## Task 16: Pylint gate

```bash
cd /home/jenkins/lvyufeng/candle/.worktrees/view-tracking-1b-b
pylint src/candle/ --rcfile=.github/pylint.conf
```

Expected: 10.00/10. If anything is below: fix it. Common patterns from past PRs:

- `import-outside-toplevel` for `from candle import zeros as _zeros` inside `_narrow_rev_view_func`. The closure-local import is intentional (avoids a top-level circular import). Add `# pylint: disable=import-outside-toplevel` on that line.
- `unused-argument` warnings on closure default-args (`_dim=d`). These are kwargs-trick captures; add `# pylint: disable=unused-argument` if needed, but prefer rewriting the closure to use `functools.partial` if the disable would be too noisy.
- `too-many-locals` in `tensor_movedim` after our edit. If it crosses the threshold, factor the helper out further.

If pylint complains about the deleted lines (e.g., `_VT_PY` becomes unused if too many entries vanish): inspect `_VT_PY` references — `unsqueeze`, `narrow_copy`, etc. still use it, so it should stay in scope.

- [ ] **Step 1: Address all pylint warnings until score is 10.00/10**
- [ ] **Step 2: Commit any pylint fixes**

```bash
git add -p  # selectively stage pylint-related fixes
git commit -m "style: fix pylint warnings in 1B-B wiring"
```

(Skip this step if pylint passed clean on first try.)

---

## Task 17: Full CPU + contract gate

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest \
  tests/cpu/ tests/contract/ -v --tb=short
```

Expected: green or only pre-existing failures (which would have failed on `main` before 1B-B started). If a NEW failure appears:

1. Run `git diff main -- <suspected file>` to see what changed.
2. If it's a test that reached into a deleted class, fix the test (don't restore the class).
3. If it's a real autograd correctness regression, the wiring math is wrong; fix it.

Heavy autograd coverage exists for `flatten`/`squeeze`/`narrow`/`movedim`/`unflatten` across `tests/cpu/test_*.py` — these are the empirical proof that rebase math equals the old `*Backward0.apply` semantics.

- [ ] **Step 1: Run full gate**
- [ ] **Step 2: If any failures, triage and fix**
- [ ] **Step 3: Re-run gate until green**

---

## Task 18: NPU gate (if hardware available)

The 1B-B wiring is platform-agnostic Python/Cython, but NPU has its own dispatched paths for these ops.

- [ ] **Step 1: Check for NPU hardware**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -c "
import candle
try:
    is_avail = candle.npu.is_available()
except Exception as e:
    is_avail = False
print('NPU available:', is_avail)
"
```

- [ ] **Step 2: If NPU available, run NPU gate**

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest tests/npu/ -v --tb=short
```

Expected: same pass/fail profile as `main` before 1B-B. If a new NPU failure appears, treat it as Task 17 step 2 — the closures might be capturing CPU-specific objects.

- [ ] **Step 3: If NPU NOT available, skip and document**

In the PR description, note "NPU gate not run; no NPU hardware on this machine" so reviewers know the CI safety net catches NPU regressions.

---

## Task 19: Rebase onto upstream/main

```bash
git fetch upstream main
git rebase upstream/main
```

If conflicts: resolve in favor of the 1B-B intent (delete the Backward classes, keep the wiring). After resolving:

```bash
git rebase --continue
```

Re-run Tasks 14-17 to confirm the rebase didn't break anything.

- [ ] **Step 1: Rebase**
- [ ] **Step 2: Re-run focused tests + full gate**

---

## Task 20: Push branch and open PR

- [ ] **Step 1: Push to origin**

```bash
git push -u origin view-tracking-1b-b --force-with-lease
```

(`--force-with-lease` is needed because the rebase rewrites history.)

- [ ] **Step 2: Create the PR to upstream**

```bash
gh pr create --repo candle-org/candle --head lvyufeng:view-tracking-1b-b --base main \
  --title "feat(autograd): wire 5 view ops onto rebase, drop hand-added Backward classes (1B-B)" \
  --body "$(cat <<'EOF'
## Summary

Wires `_view_func` / `_rev_view_func` callables onto `flatten`, `squeeze`, `narrow`, `movedim`, `unflatten` view tensors, so the engine view-rebase block at `_C/_autograd_engine.pyx:315-325` (live since #372 but unreachable for these ops) takes ownership of their gradients. Deletes the 7 hand-added `*Backward0` classes and matching `*_autograd` wrappers + 10 dispatch-registration entries that are no longer needed.

`contiguous` stays as it was after #373 — semantics are already correct in autograd terms; see spec.

This is sub-batch 1B-B in the view-tracking initiative. Predecessor: #372 (1A) and #373 (1B-A).

## What changes

- `src/candle/_backends/common/view.py`: `squeeze` / `narrow` pass `view_func` + `rev_view_func` closures to `_make_view`.
- `src/candle/_C/_tensor_api.pyx`: `tensor_flatten` / `tensor_unflatten` / `tensor_movedim` attach closures to the result tensor.
- `src/candle/_generated/functions.py`: 7 `*Backward0` classes deleted (~200 lines).
- `src/candle/_generated/variable_type.py`: 20 `*_autograd` / `*_autograd_post` wrappers deleted (~150 lines).
- `src/candle/_generated/_variable_type_cy.pyx`: 12 squeeze Cython companions deleted.
- `src/candle/_generated/registration.py`: 10 `register_autograd_*` lines deleted.
- `tools/autograd/derivatives.yaml`: comment block at squeeze entries marking them as view-tracked.
- `tests/cpu/test_view_fastpath.py`: 1B-A `_view_func is None` pins flipped to `is callable`.
- `tests/cpu/test_view_rebase_wiring.py`: new file pinning rebase semantics.
- `tests/contract/test_view_ops_no_backward_class.py`: regression guard against codegen drift re-emission.

## Why

Wiring + deletion in one PR avoids the ambiguous window where both gradient paths exist (engine rebase runs first → deleted-but-not-yet Backward classes become dead code) and the broken window where neither exists.

After this lands, the 5 ops match PyTorch's pattern: `VIEW_FUNCTIONS` ops have no autograd-kernel registration; the engine rebases gradient onto the view's `_base` using the wired `_rev_view_func`.

## Test plan

- [x] `tests/cpu/test_view_fastpath.py` — 1B-A pins flipped, 11 tests pass
- [x] `tests/cpu/test_view_rebase_wiring.py` — new file, 11 tests covering per-op rebase, end-to-end backward, chained-view recursion
- [x] `tests/cpu/test_view_tracking_infrastructure.py` — 7 tests still pass
- [x] `tests/contract/test_view_ops_no_backward_class.py` — 3 tests asserting deletion stays
- [x] `pylint src/candle/ --rcfile=.github/pylint.conf` — 10.00/10
- [x] `pytest tests/cpu/ tests/contract/` — green
- [ ] `pytest tests/npu/` — only if NPU hardware available
- [ ] CI: pylint + test-cpu + test-mps

## Net diff

~150 lines wiring + ~250 lines tests + ~50 lines doc/comments minus ~600 lines of deleted hand-added Backward + autograd + registration code. Net negative.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Capture and report PR URL**

The `gh pr create` command prints a URL. Save it for the user.

---

## Self-review (do this BEFORE running any task)

After writing the plan, walk through it once with fresh eyes:

1. **Spec coverage:** Every section of the spec maps to a task.
   - Per-op wiring (5 ops) → Tasks 2-6
   - Test pin flip → Task 7
   - Backward class deletion → Task 9
   - Autograd wrapper deletion → Tasks 10-11
   - Registration deletion → Task 12
   - Codegen-drift coordination → Task 13
   - Contract regression test → Task 13
   - Full gate → Tasks 14-18
   - PR workflow → Tasks 19-20

2. **Placeholder scan:** No "TBD" or "implement later" — every step has the actual code or command.

3. **Type / signature consistency:** `_attach_flatten_view_funcs(result, start_dim, end_dim, input_shape)` and `_attach_unflatten_view_funcs(result, dim, sizes_tuple, input_shape)` have consistent positional arguments. The `_attach_movedim_view_funcs(result, source, destination)` matches its caller. The `_make_view(...)` signature already accepts `view_func` and `rev_view_func` (verified on line 10-11 of `_backends/common/view.py`).

4. **Dependency order:** Tests first (Task 1), then wiring (2-6), then test pin flip (7), then deletion (9-12), then drift coordination (13), then gates (14-18). The full gate cannot pass with deletion done but wiring missing — that's why all wiring tasks precede all deletion tasks.

5. **No application-specific hacks:** All wiring is generic PyTorch view-tracking; no per-application special cases.

6. **Worktree discipline:** All commits go on `view-tracking-1b-b` inside `.worktrees/`. No `main` modifications.

If issues: fix inline, no need to re-review.

---

## Out of scope

Not in this plan, deferred to future batches:

- `view`, `reshape`, `permute`, `transpose`, `unsqueeze` — already produce views via `_make_view` but without `view_func`/`rev_view_func` callables. They have generated `*Backward0` classes that work; wiring is convenient but not necessary for autograd correctness today.
- `expand`, `select`, `slice`, `as_strided`, `t`, `unfold` — same as above.
- `view_as_real`, `view_as_complex` — special: storage-dtype reinterpretation. Forward AD JVP rules also need attention.
- `contiguous` — semantically already correct.
- Generator-driven `view_funcs.py` body emission — defer until codegen-drift work (#223–#225) stabilizes the generator.
- Multi-output `Function.apply` topology refactor — separate batch.

---

## Verification

The full validation gate at the end of Task 20:

1. Focused tests: `tests/cpu/test_view_fastpath.py`, `tests/cpu/test_view_rebase_wiring.py`, `tests/cpu/test_view_tracking_infrastructure.py`, `tests/contract/test_view_ops_no_backward_class.py`
2. Autograd surface: `tests/cpu/test_autograd_*.py`, `tests/contract/test_autograd_*.py`
3. Full gate: `pytest tests/cpu/ tests/contract/`
4. NPU gate: `pytest tests/npu/` (if hardware available)
5. Pylint: `pylint src/candle/ --rcfile=.github/pylint.conf`
6. CI: pylint + test-cpu + test-mps green

If all pass and the PR is open with a clean diff, 1B-B is done.
