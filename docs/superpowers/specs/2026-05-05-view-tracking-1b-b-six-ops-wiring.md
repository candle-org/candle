# Sub-batch 1B-B Design: Wire 5 view ops onto view-tracking, remove hand-added Backward classes

**Date:** 2026-05-05
**Status:** Draft
**Predecessor:** PR #372 (view-tracking 1A — `_view_func`/`_rev_view_func` cdef attrs, `_make_view` plumbing, engine grad rebase, `view_funcs.py` static inventory) and PR #373 (1B-A — conditional-view forward path for `flatten`/`unflatten` / `_view_meta["op"]` fix).
**Successor:** none planned in 1B family. Future view ops (`expand`, `select`, `slice`, `as_strided`, `t`, `unfold`, `view_as_real`, `view_as_complex`) get their own batches.

## Goal

Wire `_view_func` / `_rev_view_func` callables for **5** of the `VIEW_FUNCTIONS` / `RETURNS_VIEWS_OF_INPUT` ops — `flatten`, `squeeze`, `narrow`, `movedim`, `unflatten` — so the engine's grad-rebase block in `_accumulate_tensor_grad` (live since PR #372 but unreached by these ops) takes ownership of their gradients.

In the same PR, delete the hand-added `*Backward0` classes from `src/candle/_generated/functions.py`, the `*_autograd` Python wrappers from `src/candle/_generated/variable_type.py`, and the corresponding `register_autograd_*` lines from `src/candle/_generated/registration.py` for these 5 ops. The result: gradient flow for these 5 ops routes exclusively through view rebase, not through hand-added Function nodes.

`contiguous` stays as it is — see [Why contiguous is excluded](#why-contiguous-is-excluded).

## Context

PR #372 added the runtime infrastructure: `_view_func` / `_rev_view_func` cdef attrs on `TensorImpl`, plumbed through `_make_view`, and the engine rebase block at `src/candle/_C/_autograd_engine.pyx` lines 315–325:

```python
# PyTorch-aligned view-rebase: if the leaf is a view that carries a
# _rev_view_func, redirect grad accumulation onto its base.
base = getattr(tensor, "_base", None)
rev_func = getattr(tensor, "_rev_view_func", None)
if base is not None and rev_func is not None:
    grad = rev_func(grad)
    return self._accumulate_tensor_grad(
        base, grad,
        mark_create_graph=mark_create_graph,
        apply_hooks=False,
    )
```

This block is live but currently unreachable for the 6 candidate ops (their views are constructed without `view_func`/`rev_view_func` set). 1B-B wires the 5 selected ops so this block fires.

PR #373 (1B-A) made `flatten`/`unflatten` produce real views (instead of always copying) and set `_view_meta["op"]` to the right op string, so a future routing-by-op-name strategy works. After 1B-A:

| Op | Forward produces | `_view_meta["op"]` | `_view_func` / `_rev_view_func` |
|---|---|---|---|
| `narrow` | view via `_make_view` | `"narrow"` | `None` (handled by hand-added `NarrowBackward0`) |
| `squeeze` | view via `_make_view` | `"squeeze"` | `None` (handled by `SqueezeBackward0` / `SqueezeDimBackward0` / `SqueezeDimsBackward0`) |
| `movedim` | view via direct `cy_as_strided`+`_view_meta` (no-grad fast-path) **or** dispatched op (grad path) | `"movedim"` | `None` (grad path uses `MovedimBackward0`) |
| `flatten` | view via `reshape` (contiguous input) or copy (1B-A behavior); `_view_meta["op"]` reset to `"flatten"` | `"flatten"` | `None` (handled by `FlattenBackward0`) |
| `unflatten` | same pattern as `flatten` | `"unflatten"` | `None` (handled by `UnflattenBackward0`) |
| `contiguous` | returns `self` when contiguous (1B-A); fresh-storage copy otherwise | n/a (no new view) | n/a |

1B-B fills in the `_view_func` / `_rev_view_func` columns for the first 5 ops and deletes their `*Backward0` classes.

## Why `contiguous` is excluded

`contiguous` has fundamentally different semantics from the other 5. After 1B-A:

* When input is already contiguous, `tensor_contiguous` returns `self` directly. The output is the same Python object as the input — there is no new view tensor to attach `_view_func` to. Gradient accumulates onto `self` automatically because `out is self`. Wiring is unnecessary.
* When input is non-contiguous, the dispatched `contiguous` allocates fresh storage and copies. The output has a new storage that does **not** alias the input. There is no view-tracking link to base; the genuine semantic is "identity grad through a copy." `ContiguousBackward0`'s `apply()` returns `(grad,)` — that's correct, and not something view-tracking can replace.

So `contiguous` is already correct in autograd terms after 1B-A. Putting it in 1B-B's scope would mean removing `ContiguousBackward0` and… replacing it with what? There is no view to attach a `view_func` to. The hand-added Backward class genuinely does the right thing for the copy case, and is genuinely unused (output is the same Python object) for the self-return case. Leave it alone.

## Why one PR, not two

The two halves of 1B-B (wire + delete) are tightly coupled:

* **If we wire `view_func`/`rev_view_func` without deleting the hand-added Backward classes**: both paths produce gradients for the 5 ops. The rebase block runs first and returns early (line 321), so the deleted-but-not-yet-deleted `*Backward0` classes become dead code. Functionally fine but confusing in code review and a regression-tracking liability.
* **If we delete the Backward classes without wiring**: the 5 ops have no autograd at all — every test using `requires_grad=True` with these ops breaks.

A single PR avoids the brief window where both paths exist *and* avoids the brief window where neither path exists.

## Scope of 1B-B

**In scope:**

1. Add `view_func` / `rev_view_func` callables to **5 ops**: `flatten`, `squeeze`, `narrow`, `movedim`, `unflatten`.
2. Delete **7 hand-added Backward classes** from `src/candle/_generated/functions.py`:
   * `SqueezeBackward0` (line 10677), `SqueezeDimBackward0` (line 10705), `SqueezeDimsBackward0` (line 10733)
   * `FlattenBackward0` (line 22246)
   * `MovedimBackward0` (line 23245)
   * `NarrowBackward0` (line 23340)
   * `UnflattenBackward0` (line 24268)
3. Delete the matching `squeeze_autograd` / `flatten_autograd` / `unflatten_autograd` / `movedim_autograd` / `narrow_autograd` (and their `*_post` siblings) Python wrappers in `src/candle/_generated/variable_type.py`.
4. Delete dispatch-registration entries in `src/candle/_generated/registration.py` for these 5 ops (the 10 `register_autograd_*` lines at registration.py lines 273-274, 482-485, 486-487, 498-499). After deletion, the dispatch system sees no autograd kernel registered for these ops and falls through to the default no-autograd kernel — which produces a plain view via the forward path. View rebase then handles gradient at engine level.
5. Update existing test `tests/cpu/test_view_fastpath.py`: flip the four `assert u._view_func is None` / `u._rev_view_func is None` assertions for `flatten` and `unflatten`.
6. New test file `tests/cpu/test_view_rebase_wiring.py` covering rebase semantics for all 5 ops.

**Out of scope:**

* `contiguous` (see above).
* Other view ops in `view_funcs.py::VIEW_OPS` (`view`, `reshape`, `permute`, `transpose`, `unsqueeze`, `expand`, `select`, `slice`, `as_strided`, `t`, `unfold`, `view_as_real`, `view_as_complex`).
* Generator changes — the `view_funcs.py` static inventory stays as the static inventory; per-op bodies are hand-written next to the ops they belong to. (Generator-driven wiring is a future option, but coupling to the drifting `gen_*.py` output is not worth the blast radius today.)
* Multi-output `Function.apply` topology refactor.
* Forward AD (`view_as_real` / `view_as_complex` JVP rules).

## Per-op wiring

The 5 ops divide by where the view is constructed.

### `flatten` and `unflatten`

After 1B-A, both go through `tensor_flatten` (`_C/_tensor_api.pyx:806`) and `tensor_unflatten` (`_C/_tensor_api.pyx:1565`), which call `self.reshape(new_shape)` and override `_view_meta["op"]`. We keep that, then attach `_view_func` and `_rev_view_func` on `result`.

```cython
# in tensor_flatten, after the existing meta override:
input_shape = self.shape  # captured before any reshape
def _flatten_view_func(new_base, _start=start_dim, _end=end_dim):
    return new_base.flatten(_start, _end)
def _flatten_rev_view_func(grad_view, _shape=input_shape):
    return grad_view.reshape(_shape)
result._view_func = _flatten_view_func
result._rev_view_func = _flatten_rev_view_func
```

The closure binds `start_dim`/`end_dim` (or `dim`/`sizes` for unflatten) and the input's pre-flatten shape so that:

* `_view_func(new_base)` replays the forward op when a base tensor is rebased.
* `_rev_view_func(grad_view)` reverses the forward shape transformation. For `flatten`/`unflatten` that's just a `reshape` back to the input shape — semantically identical to what `FlattenBackward0.backward` and `UnflattenBackward0.backward` already do (they call `redispatch("reshape", grad, input_.shape)`).

The `flatten`/`unflatten` view is created via `self.reshape(new_shape)` which goes through `_make_view` with `view_func=None`/`rev_view_func=None`. We can't change that path without breaking generic `reshape`. So we set the callables **after** the reshape returns. This is exactly the same pattern 1B-A uses to override `_view_meta["op"]`.

### `squeeze`

After 1B-A, `Tensor.squeeze` routes through `_C/_tensor_api.pyx::tensor_squeeze_method` (line 3150), which delegates to the Python `_functional_squeeze_fn` defined in `src/candle/_functional.py:1253`. That function either calls `_cy_squeeze` (if available) or `_squeeze_impl` from `_backends/common/view.py:134`.

The `_make_view` call site is `_backends/common/view.py:161-162`:

```python
return _make_view(base, shape, stride, a.offset, "squeeze", source=a)
```

We change this to:

```python
input_shape = a.shape
def _squeeze_view_func(new_base, _dim=dim):
    return new_base.squeeze(_dim) if _dim is not None else new_base.squeeze()
def _squeeze_rev_view_func(grad_view, _shape=input_shape):
    return grad_view.reshape(_shape)
return _make_view(
    base, shape, stride, a.offset, "squeeze",
    source=a,
    view_func=_squeeze_view_func,
    rev_view_func=_squeeze_rev_view_func,
)
```

Same mechanism for the dispatched path (`squeeze` with a specific dim that goes through the dispatched op path in `_functional.py:1267`). The dispatched path returns a tensor produced by the backend kernel — we wrap it in a thin layer that sets the callables before returning.

`SqueezeBackward0.backward`, `SqueezeDimBackward0.backward`, and `SqueezeDimsBackward0.backward` all do exactly `redispatch("reshape", grad, self_.shape)`. Our `_squeeze_rev_view_func` does the same.

### `narrow`

`Tensor.narrow` routes via `_C/_tensor_api.pyx::tensor_narrow_method` (line 2630). Two paths:

1. **`requires_grad=True`** (line 2641-2642): falls back to `_dispatch_fn("narrow", ...)`. Dispatch eventually calls the backend's narrow kernel, which on CPU calls `_backends/common/view.py::narrow` (line 177) — that's where `_make_view` is called.

2. **`requires_grad=False`**: no-grad fast-path uses `cy_as_strided` + manual `_view_meta` (lines 2649-2660). We don't wire here because no-grad means no gradient flows. (We *could* still set the callables for consistency, but it's wasted work — there's no rebase to trigger.)

We wire path 1 by editing `_backends/common/view.py::narrow`:

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
        # Mirrors `_narrow_backward_helper` from _generated/functions.py.
        from candle import zeros
        grad_input = zeros(_shape, dtype=grad_view.dtype, device=grad_view.device)
        grad_input.narrow(_dim, _start, _len).copy_(grad_view)
        return grad_input
    return _make_view(
        base, tuple(new_shape), a.stride, new_offset, "narrow",
        source=a, creation_kind=creation_kind,
        view_func=_narrow_view_func,
        rev_view_func=_narrow_rev_view_func,
    )
```

`NarrowBackward0.backward` calls `_narrow_backward_helper(grad, input_, dim, start, length, keyset)`. Our `_narrow_rev_view_func` does the same algorithm inline using `zeros` + `narrow().copy_()`. The helper can stay in `_generated/functions.py` for now — only the *class* gets deleted.

### `movedim`

`Tensor.movedim` routes via `_C/_tensor_api.pyx::tensor_movedim` (line 1909). Two paths:

1. **`requires_grad=True`** (line 1924-1925): falls back to `_dispatch_fn("movedim", ...)`. The dispatched movedim eventually constructs the view (currently producing a `MovedimBackward0` Function node).

2. **`requires_grad=False`**: no-grad fast-path uses `cy_as_strided` + manual `_view_meta` (lines 1953-1973). No gradient flows; no wiring needed.

We add wiring at the dispatched path. Approach: extend `tensor_movedim` to wire the result *after* dispatch returns, inside the `requires_grad` branch:

```cython
def tensor_movedim(self, source, destination):
    _ensure_dispatch_ref()
    if self.requires_grad:
        result = _dispatch_fn("movedim", self.device.type, self, source, destination)
        # Wire view_func / rev_view_func so the engine rebase block owns grad.
        input_shape = self.shape
        # Compute the inverse permutation to reverse movedim.
        # See PyTorch's movedim_backward for the math.
        result._view_func = lambda nb, _src=source, _dst=destination: nb.movedim(_src, _dst)
        result._rev_view_func = lambda g, _src=source, _dst=destination: g.movedim(_dst, _src)
        return result
    # ... existing no-grad fast path ...
```

`MovedimBackward0.backward` calls `_movedim_backward_helper(grad, input_, source, destination, keyset)`. The math: `movedim` is its own inverse with source↔destination swapped. So `_rev_view_func(g)` is just `g.movedim(destination, source)`. Same result, simpler statement.

### Summary table

| Op | Wire site | `_view_func` body | `_rev_view_func` body |
|---|---|---|---|
| `flatten` | `tensor_flatten` (after reshape) | `b.flatten(start, end)` | `g.reshape(input_shape)` |
| `unflatten` | `tensor_unflatten` (after reshape) | `b.unflatten(dim, sizes)` | `g.reshape(input_shape)` |
| `squeeze` | `_backends/common/view.squeeze` (in `_make_view` kwargs) | `b.squeeze(dim)` | `g.reshape(input_shape)` |
| `narrow` | `_backends/common/view.narrow` (in `_make_view` kwargs) | `b.narrow(dim, start, length)` | `zeros(shape); grad.narrow(dim,start,length).copy_(g)` |
| `movedim` | `tensor_movedim` (after dispatched call, requires_grad path) | `b.movedim(src, dst)` | `g.movedim(dst, src)` |

## Engine integration

After the wiring lands, the engine block at `_autograd_engine.pyx:315-325` fires whenever a downstream op's gradient lands on one of these 5 view tensors. The chain is:

1. Backward kicks off, walks the graph, eventually accumulates `grad_view` onto a tensor `v` produced by one of the 5 ops.
2. `_accumulate_tensor_grad(v, grad_view)` runs.
3. `v._base` is set (via `_make_view`) and `v._rev_view_func` is set (via 1B-B wiring) — engine takes the rebase branch.
4. Engine calls `grad_base = v._rev_view_func(grad_view)` and recurses with `_accumulate_tensor_grad(v._base, grad_base)`.
5. Recursion continues until grad lands on a leaf or another view with no `_rev_view_func`.

The recursion handles **chained views** correctly: `t.flatten().narrow(0, 0, 6)` produces a narrow view whose base is the flatten view whose base is `t`. Both have `_rev_view_func` after 1B-B; rebase runs twice, depositing grad on `t`.

## Removal: hand-added classes and dispatch wrappers

### Backward classes (delete from `_generated/functions.py`)

7 classes:

* `SqueezeBackward0` — line 10677, ~26 lines
* `SqueezeDimBackward0` — line 10705, ~28 lines
* `SqueezeDimsBackward0` — line 10733, ~28 lines
* `FlattenBackward0` — line 22246, ~30 lines
* `MovedimBackward0` — line 23245, ~30 lines
* `NarrowBackward0` — line 23340, ~32 lines
* `UnflattenBackward0` — line 24268, ~30 lines

The `_narrow_backward_helper` and `_movedim_backward_helper` module-level helpers in `functions.py` stay where they are (they're tiny and unreferenced after deletion is fine — they don't affect imports). A future cleanup PR can decide whether to delete them.

### `*_autograd` wrappers (delete from `_generated/variable_type.py`)

For each op there are typically two wrappers per dispatch key (`squeeze_autograd`, `squeeze_autograd_post`, etc.) and they appear in multiple places (variable_type.py has both pre-dispatch and post-dispatch entries). Identified entry points to delete:

* `squeeze` family: lines 4358, 4371, 4385, 4399, 4412, 4426, 12629, 12639, 12650, 12661, 12671, 12682
* `flatten`: lines 17791, 17804
* `movedim`: lines 18668, 18681
* `narrow`: lines 18757, 18771
* `unflatten`: lines 19604, 19617

Each is a small stanza like:

```python
def squeeze_autograd(...):
    ...
    grad_fn = _F.SqueezeBackward0((self_,), raw_keyset=raw_keyset, active_keyset=active_keyset)
    ...
```

Delete the function bodies; verify no other generated code references them.

### Dispatch registration (`_generated/registration.py`)

Delete all 10 `register_autograd_*` lines for the 5 ops:

```python
# Delete these lines:
register_autograd_kernels('squeeze', default=_VT_PY.squeeze_autograd, ...)
register_autograd_post_kernels('squeeze', _VT_PY.squeeze_autograd_post)
register_autograd_kernels('flatten', default=_VT_PY.flatten_autograd, ...)
register_autograd_post_kernels('flatten', _VT_PY.flatten_autograd_post)
register_autograd_kernels('unflatten', default=_VT_PY.unflatten_autograd, ...)
register_autograd_post_kernels('unflatten', _VT_PY.unflatten_autograd_post)
register_autograd_kernels('movedim', default=_VT_PY.movedim_autograd, ...)
register_autograd_post_kernels('movedim', _VT_PY.movedim_autograd_post)
register_autograd_kernels('narrow', default=_VT_PY.narrow_autograd, ...)
register_autograd_post_kernels('narrow', _VT_PY.narrow_autograd_post)
```

After deletion, no autograd kernel is registered for `squeeze`/`flatten`/`unflatten`/`movedim`/`narrow`. The dispatch system falls through to the default no-autograd kernel, which produces a plain view via the forward path (with `_view_func`/`_rev_view_func` set). View rebase at the engine level owns gradient routing. This matches PyTorch's pattern for `VIEW_FUNCTIONS` ops.

## Codegen drift coordination

`tools/autograd/gen_autograd_functions.py` (or wherever the generator lives) currently emits the 7 Backward classes and the `*_autograd` wrappers we are deleting. The codegen drift work tracked by tasks #223–#225 will eventually regenerate these files — at that point, the regenerator must NOT re-emit these 7 classes / 5 op wrappers.

Action items in 1B-B:

1. **Add a comment block at the top of `tools/autograd/derivatives.yaml`** (or wherever the entries live) for these 5 ops, marking them as "view-tracked; no Backward class generation." A future generator change must respect this.
2. **Add a regression test in `tests/contract/`**: assert that no `*Backward0` class for `squeeze`/`flatten`/`unflatten`/`narrow`/`movedim` exists in `_generated/functions.py`. This catches the regenerator if it accidentally re-emits.

## Test plan

### Update existing: `tests/cpu/test_view_fastpath.py`

Change line 48-49 from:
```python
assert u._view_func is None
assert u._rev_view_func is None
```
to:
```python
assert u._view_func is not None
assert u._rev_view_func is not None
# Functional check: rev rebases shape correctly
g = candle.ones(u.shape)
g_base = u._rev_view_func(g)
assert g_base.shape == t.shape
```

Same for line 77-78 (`unflatten` test).

### New: `tests/cpu/test_view_rebase_wiring.py`

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
    np.testing.assert_array_equal(g_base.numpy()[0], np.zeros(4))
    np.testing.assert_array_equal(g_base.numpy()[1], np.ones(4))
    np.testing.assert_array_equal(g_base.numpy()[4], np.zeros(4))


def test_movedim_rev_view_func_swaps_axes_back():
    t = torch.randn(2, 3, 4)
    u = t.movedim(0, 2)  # shape (3, 4, 2)
    assert u.shape == (3, 4, 2)
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


# --- Regression guard: hand-added Backward classes are gone ---------------

def test_hand_added_backward_classes_are_removed():
    from candle._generated import functions as _F
    for name in (
        "SqueezeBackward0", "SqueezeDimBackward0", "SqueezeDimsBackward0",
        "FlattenBackward0", "UnflattenBackward0",
        "NarrowBackward0", "MovedimBackward0",
    ):
        assert not hasattr(_F, name), (
            f"{name} should have been deleted in 1B-B; view rebase owns this op."
        )
```

### Targeted verification

```bash
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest \
  tests/cpu/test_view_fastpath.py \
  tests/cpu/test_view_rebase_wiring.py \
  tests/cpu/test_view_tracking_infrastructure.py \
  -v --tb=short
```

### Full gate

```bash
pylint src/candle/ --rcfile=.github/pylint.conf
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest tests/cpu/ tests/contract/ -v --tb=short
PYTHONPATH=$PWD/src conda run -n candle311 python -m pytest tests/npu/ -v --tb=short  # if NPU available
```

The CPU + contract gate has heavy autograd coverage for these 5 ops. The semantic equivalence claim ("rebase = old `*Backward0`'s `apply()`") is verified empirically by the gate passing with byte-identical gradients. If anything diverges, the gate flags it.

## Files touched

* `src/candle/_C/_tensor_api.pyx` — `tensor_flatten` / `tensor_unflatten` set `_view_func` + `_rev_view_func` on result; `tensor_movedim` sets them on dispatched-path output.
* `src/candle/_backends/common/view.py` — `squeeze` / `narrow` pass `view_func` + `rev_view_func` to `_make_view`.
* `src/candle/_generated/functions.py` — delete 7 `*Backward0` classes (~200 lines deleted).
* `src/candle/_generated/variable_type.py` — delete 5 sets of `*_autograd` / `*_autograd_post` wrappers (~150 lines deleted).
* `src/candle/_generated/registration.py` — delete 10 `register_autograd_*` lines for these 5 ops.
* `tools/autograd/derivatives.yaml` — add comment marking these ops as view-tracked, no Backward class generation.
* `tests/cpu/test_view_fastpath.py` — flip 4 None assertions, add functional rev_view_func checks.
* `tests/cpu/test_view_rebase_wiring.py` — new file (~150 lines).
* `tests/contract/test_no_hand_added_backward.py` (or fold into existing contract test) — assert removed classes stay removed (~10 lines).
* `docs/superpowers/specs/2026-05-05-view-tracking-1b-b-six-ops-wiring.md` — this spec.

Estimated diff: **net negative** — ~150 lines wiring + ~250 lines tests + ~50 lines doc/comments − ~600 lines of deleted hand-added Backward + autograd + registration code.

## Risks

1. **Engine rebase precondition**: the rebase block fires only when `tensor._base is not None`. After 1B-B, all 5 ops set `_base` via `_make_view` (or via `cy_as_strided` + manual meta in the no-grad case, which doesn't matter for autograd). ✓
2. **Test pin flip**: `tests/cpu/test_view_fastpath.py` lines 48-49, 77-78 explicitly assert `_view_func is None` for `flatten`/`unflatten` outputs. **1B-B flips these.** Document explicitly in PR description.
3. **`narrow` rebase implementation**: our `_narrow_rev_view_func` uses `zeros(shape) → narrow(...).copy_(grad)`. PyTorch's narrow_backward is the same pattern. Edge cases:
   * If `grad_view` is on a non-CPU device, `zeros(shape, device=grad_view.device)` must keep the device.
   * If `grad_view` requires grad (create_graph), the rebase must produce a grad-tracking output.
4. **`movedim` rebase identity**: `movedim(dst, src)` undoes `movedim(src, dst)` only when source and destination are bijective lists. For tuple inputs (`movedim((0,1), (2,3))`), this requires verifying the math. A failing test in `test_view_rebase_wiring.py` covers this case.
5. **Codegen drift regenerator**: when tasks #223–#225 regenerate `_generated/functions.py`, the 7 deleted classes will reappear unless the generator is taught to skip them. The contract test catches the regression but doesn't fix it. Coordinate with the codegen-drift work.
6. **Multi-output Function topology**: none of the 5 ops produce multiple outputs. Not a concern for 1B-B.

## Out of scope (future)

* `view`, `reshape`, `permute`, `transpose`, `unsqueeze` — already produce views via `_make_view` but without `view_func`/`rev_view_func`. Wiring follows the same pattern but isn't needed for autograd correctness because they have generated Backward classes that already work. Migrate when convenient, not in 1B-B.
* `expand`, `select`, `slice`, `as_strided`, `t`, `unfold` — same as above.
* `view_as_real`, `view_as_complex` — special because they reinterpret storage dtype. Forward AD JVP rules also need attention; defer.
* `contiguous` — already correct, see [Why contiguous is excluded](#why-contiguous-is-excluded).
* Generator-driven `view_funcs.py` body emission — defer until codegen-drift work (tasks #223–#225) stabilizes the generator.
