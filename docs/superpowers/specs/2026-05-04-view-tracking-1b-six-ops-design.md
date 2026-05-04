# Sub-batch 1B Design: Wire 6 shape/view ops onto view-tracking infrastructure

**Date:** 2026-05-04
**Status:** Draft (1B-A scope; 1B-B opens after 1B-A merges)
**Predecessor:** PR #372 (view-tracking infrastructure 1A — `_view_func`/`_rev_view_func` cdef attrs, `_make_view` parameters, engine grad rebase, codegen `view_funcs.py`).
**Successor scope:** 1B-B will wire all 6 ops onto `view_func`/`rev_view_func` and delete the hand-added Backward classes.

## Goal

Land six PyTorch `VIEW_FUNCTIONS` / `RETURNS_VIEWS_OF_INPUT` ops — `contiguous`, `flatten`, `squeeze`, `narrow`, `movedim`, `unflatten` — onto Candle's view-tracking infrastructure so backward flows through view rebase rather than hand-added identity Backward classes.

The work is split into two PRs to keep blast radius small:

* **1B-A (this spec, this PR):** make the **forward** path of the three currently-copying ops (`contiguous`, `flatten`, `unflatten`) match PyTorch's conditional semantics. `contiguous` returns `self` when already contiguous; `flatten`/`unflatten` return real views when view-compatible. No autograd or codegen changes.
* **1B-B (separate spec, separate PR):** wire all 6 ops into `view_func` / `rev_view_func`, drop their hand-added `*Backward0` classes from `_generated/`.

This document specifies **1B-A only**. 1B-B will get its own design doc when 1B-A is merged.

## Context

PR #372 added the infrastructure: a view tensor can carry an op-specific `_view_func` (replay forward) and `_rev_view_func` (rebase a gradient onto its base), and the autograd engine walks the rebase chain inside `_accumulate_tensor_grad`. That PR did not flip any ops to use it — the field defaults to `None`, the engine rebase block is dead code today, and the 6 view ops still rely on hand-added identity Backward classes (`ContiguousBackward0`, `FlattenBackward0`, `SqueezeBackward0`, `NarrowBackward0`, `MovedimBackward0`, `UnflattenBackward0` in `src/candle/_generated/functions.py`).

A direct audit of the 6 ops shows two distinct gaps from PyTorch:

| Op          | Currently in candle (forward)                   | PyTorch (forward)                                | Gap                              |
|-------------|--------------------------------------------------|---------------------------------------------------|----------------------------------|
| `narrow`    | view via `_make_view`                            | view (slice metadata only)                       | none (forward) — only autograd wiring left |
| `squeeze`   | view via `_make_view`                            | conditional view (in `VIEW_FUNCTIONS`)            | none (forward) — only autograd wiring left |
| `movedim`   | view via `_make_view`                            | view (permute-style stride remap)                | none (forward) — only autograd wiring left |
| `contiguous`| **always copies**                                | **conditional view** (returns self if already contiguous in requested format) | forward path differs |
| `flatten`   | **always copies**                                | **conditional view** (view if stride-collapsible over `[start_dim, end_dim]`, else copy) | forward path differs |
| `unflatten` | **always copies**                                | **conditional view** (delegates to `view`/`reshape` semantics; view if stride-compatible) | forward path differs |

So the 6 ops divide naturally:

* 3 ops (`narrow`, `squeeze`, `movedim`) already produce a real view in candle. Forward is correct. They are "forward-clean, autograd-pending."
* 3 ops (`contiguous`, `flatten`, `unflatten`) still always allocate a fresh storage and copy. Forward is wrong vs torch. They need a forward fix before any view-tracking wiring will work — you cannot rebase a gradient onto a view that isn't actually a view.

**1B-A fixes the forward path of the second group.** It does not touch autograd. After 1B-A:

* `contiguous(a)` returns `a` itself when `a` is already contiguous (matching torch's identity behavior); otherwise the existing copy path runs.
* `flatten(a, ...)` and `unflatten(a, ...)` return real views (via `_make_view`) when `a` is contiguous; otherwise the existing copy path runs.
* All 6 ops' Backward classes still run unchanged — autograd routing is 1B-B's job.

## Why split 1B into 1B-A + 1B-B

The two stages have different blast radii and different validation needs:

* **1B-A** changes 6 backend kernel functions (`contiguous`/`flatten`/`unflatten` × cpu/mps/npu). It changes runtime behavior for any caller that mutates the output of `contiguous`/`flatten`/`unflatten` — these are observable semantics and need their own regression coverage.
* **1B-B** changes generator output (`_generated/functions.py`, `_generated/variable_type.py`, `_generated/registration.py`) and adds `view_func`/`rev_view_func` callables on view ops. It changes autograd behavior — gradients reach the base via rebase instead of via hand-added Backward.

Mixing both into one PR would mean a 6-op forward semantic change *and* an autograd-routing change land in the same diff. Separating them makes each PR's scope auditable on its own and lets a regression bisect onto whichever change introduced it.

## Scope of 1B-A

**In scope:**

* Backend forward path for `contiguous`, `flatten`, `unflatten` on CPU, MPS, and NPU.
* `contiguous`: when input is already contiguous in the requested format, return the input itself (`return a`).
* `flatten` / `unflatten`: when input is contiguous, return a real view via `_make_view`. Otherwise keep the copy path.
* New test file `tests/cpu/test_view_fastpath.py` covering the three behaviors above.

**Out of scope (deferred to 1B-B):**

* Setting `view_func` / `rev_view_func` on any op output. They stay `None` after 1B-A. The infrastructure exists (PR #372) but is not exercised yet.
* Removing the hand-added `ContiguousBackward0` / `FlattenBackward0` / `SqueezeBackward0` / `NarrowBackward0` / `MovedimBackward0` / `UnflattenBackward0` classes. They keep running.
* Autograd-routing changes for `narrow`, `squeeze`, `movedim` (their forward is already correct).
* CUDA backend — no separate `contiguous`/`flatten`/`unflatten` impl exists in `_backends/cuda/`.

## Conditional-view semantics

Each of the three ops gets a **view fast-path** under specific conditions; otherwise it falls back to the existing copy path.

### `contiguous(a, memory_format=None)`

* `memory_format` is `None` or `contiguous_format` (default):
  * If `a.is_contiguous()`: **return `a` directly** (matching torch's `t.contiguous()` returning `self` for already-contiguous inputs — same Python identity, no `_make_view` wrap).
* `memory_format` is `channels_last`:
  * Already handled correctly: existing path returns `a` when `_is_channels_last_stride(...)` matches. No change needed.
* Otherwise: keep the existing copy path (`np.ascontiguousarray` on CPU, `aclnnInplaceCopy`-equivalent on NPU, `_dispatch_unary_gpu(a, "identity")` on MPS).

The functional contract is "guaranteed contiguous output." The view fast-path satisfies that contract because the input is already contiguous.

Note that today the cpu `contiguous` path **does not** short-circuit to `return a` even when the input is already contiguous in default format — it always allocates via `np.ascontiguousarray`. That's the bug 1B-A fixes for cpu. The mps `contiguous` already returns `a` on the fast-path (line 23) and is correct as-is.

### `flatten(a, start_dim=0, end_dim=-1)`

PyTorch's flatten returns a view when the strides over `[start_dim, end_dim]` already collapse to a single contiguous run. Concretely, that holds when:

* `start_dim == end_dim` (no-op flatten): just return `a`.
* The input is contiguous (full-tensor stride pattern matches `_contiguous_stride(shape)`): the flattened range is trivially collapsible. Return a view with shape `a.shape[:start] + (prod(a.shape[start:end+1]),) + a.shape[end+1:]` and stride `_contiguous_stride(new_shape)`.
* Sub-range stride-collapsible (general case): there exists a single stride `s` such that `a.stride[start_dim] = s * prod(a.shape[start_dim+1:end_dim+1])`, `a.stride[start_dim+1] = s * prod(a.shape[start_dim+2:end_dim+1])`, …, `a.stride[end_dim] = s`. PyTorch handles this with its general view-feasibility check. **For 1B-A we conservatively support only the "input is contiguous" path** — that covers the dominant case (post-Linear / post-conv tensors are contiguous) and avoids reproducing PyTorch's stride-feasibility math now. The general case can land later.

When the input is not contiguous (and `start_dim != end_dim`): keep the existing copy path.

### `unflatten(a, dim, sizes)`

PyTorch's unflatten delegates to `view` / `reshape` semantics. Conditional view holds when:

* The split dim's stride is compatible with the new shape — i.e., `a.is_contiguous()` is sufficient (most common case), or more narrowly when `a.stride[dim]` factors cleanly across `sizes`.
* For 1B-A we conservatively support only the "input is contiguous" path. Output shape is `a.shape[:d] + tuple(sizes) + a.shape[d+1:]`. Output stride is `_contiguous_stride(new_shape)`.

When the input is not contiguous: keep the existing copy path.

### Common pattern across the 3 ops

For `flatten` and `unflatten`:

```python
if a.is_contiguous():
    base = _get_base(a)
    return _make_view(base, new_shape, _contiguous_stride(new_shape), a.offset, op_name, source=a)
# else: existing copy path stays unchanged
```

with `op_name` set to `"flatten"` or `"unflatten"`. `view_func` and `rev_view_func` are not passed (default `None`) — 1B-A leaves the autograd path unchanged.

For `contiguous` the fast-path is simpler — `return a` directly (no `_make_view` wrap). Output is the same Python object as the input, matching torch's `t.contiguous() is t` identity when already contiguous.

## Per-backend changes

**CPU** — `src/candle/_backends/cpu/ops.py`:

* `contiguous` (line 1351): for the default-memory-format branch, after the existing `if getattr(memory_format, "_name", None) == "channels_last":` block, when `memory_format is None` or `contiguous_format`: if `a.is_contiguous()`, `return a`. Otherwise keep `np.ascontiguousarray(_to_numpy(a))` copy.
* `flatten` (line 3058): if `a.is_contiguous()`, build view via `_make_view` (import `view_backend` like `movedim` does at line 3082). Otherwise keep `np.ascontiguousarray` copy.
* `unflatten` (line 3068): same pattern as `flatten`. If `a.is_contiguous()`, view; else copy.

**MPS** — `src/candle/_backends/mps/ops/shape.py`:

* `contiguous` (line 23): the existing `if a.is_contiguous(): return a` already short-circuits. **Confirmed correct as-is.** No change needed for the contiguous fast-path. The change for MPS is only that this op now produces tensors that 1B-B can later wire to `view_func` — but no wiring happens here.
* `flatten` (line 1099): the existing GPU path constructs a fresh storage tensor via `_from_metal_buffer(_metal_buf(a), new_shape, ...)` — that's a same-storage tensor but it's *not* registered as a view (no `_make_view`, no `_base`). Replace the `_can_use_gpu(a) and a.is_contiguous()` branch to go through `view_backend._make_view(...)` instead. Non-contiguous GPU and CPU-fallback paths stay unchanged.
* `unflatten` (line 1123): same pattern as `flatten`.

**NPU** — `src/candle/_backends/npu/ops/shape.py`:

* `contiguous` (line 1428): currently always memcpy_d2d. Add a fast-path: if `a.is_contiguous()`, `return a`. Otherwise keep the existing memcpy path. (NPU has no memory_format parameter — only the default-format case applies.)
* `flatten_op` (line 1409): currently always returns `view_backend.reshape(a, new_shape)`. The current behavior already produces a view when the input is contiguous (because `view_backend.reshape` calls `_make_view` for contiguous inputs and `a.contiguous()`-then-view for non-contiguous ones). **Verify by reading `view_backend.reshape`** that this already meets the conditional-view contract; if so, no NPU change is needed for `flatten_op`. If not, add explicit conditional-view branching.
* `unflatten_op` (line 1974): same as `flatten_op` — already delegates to `view_backend.reshape`. Verify and adjust similarly.

**CUDA**: no `_backends/cuda/ops/shape.py` for these ops. No change.

## What changes in observable behavior

After 1B-A:

* `contiguous(a)` when `a` is already contiguous returns `a` itself (`out is a`). Was: a fresh copy with new storage.
* `flatten(a, ...)` / `unflatten(a, ...)` when `a` is contiguous return a view: `out.storage().data_ptr() == a.storage().data_ptr()`, `out._base is _get_base(a)`. Was: a fresh copy.
* For non-contiguous input, all three ops still allocate and copy as before.
* Mutating the fast-path output mutates the input's storage, and vice versa. PyTorch behaves the same way.

Code that called `contiguous()` / `flatten()` / `unflatten()` *to defensively detach storage* relied on undocumented behavior — torch never guaranteed that. Anything that wants a real copy should call `.clone()`. We expect very little candle code to depend on the old detach side-effect, but the regression test suite will surface it.

## Test plan

New file: `tests/cpu/test_view_fastpath.py`. Tests cover:

1. **`contiguous` returns self on the fast-path:**
   * `t = torch.randn(3, 4); u = t.contiguous(); assert u is t`
   * `t = torch.randn(3, 4).t(); u = t.contiguous(); assert u is not t and u.storage().data_ptr() != t.storage().data_ptr()` (non-contiguous still copies)

2. **`flatten` / `unflatten` storage sharing on contiguous fast-path:**
   * `t = torch.randn(3, 4); u = t.flatten(); assert u.storage().data_ptr() == t.storage().data_ptr()`
   * `t = torch.randn(2, 6); u = t.unflatten(1, (2, 3)); assert u.storage().data_ptr() == t.storage().data_ptr()`

3. **Mutation propagates through the fast-path view:**
   * `t = torch.randn(3, 4); u = t.flatten(); u[0] = 99.0; assert t.view(-1)[0].item() == 99.0`
   * Same for `unflatten`.

4. **Copy fallback still produces an independent allocation when input is non-contiguous:**
   * `t = torch.randn(4, 4).t()` (non-contiguous) `; u = t.flatten(); assert u.storage().data_ptr() != t.storage().data_ptr()`
   * Same for `unflatten`.

5. **`_base` and `_view_meta` are populated on the `flatten`/`unflatten` fast-path:**
   * `u._base is _get_base(t)` (resolves via `_make_view`'s `_get_base` call).
   * `u._view_meta["op"] in {"flatten", "unflatten"}`.
   * `u._view_func is None and u._rev_view_func is None` (1B-A doesn't wire those).
   * For `contiguous` fast-path: `u is t`, so `u._view_meta` reflects `t`'s pre-existing meta (no new view created) — there's no new view metadata to assert here.

6. **Backward through fast-path still works** (this is the safety net for "we didn't break the existing autograd path"):
   * `t = torch.randn(3, 4, requires_grad=True); u = t.flatten(); u.sum().backward(); assert torch.allclose(t.grad, torch.ones_like(t))`
   * Same for `unflatten`. For `contiguous` on already-contiguous input the existing tests already cover this since the function is now a true no-op.

7. **`memory_format=channels_last` for `contiguous` is unchanged:**
   * Existing tests in `tests/cpu/test_*memory_format*` already cover this. No new test needed; the change must not regress them.

## Files touched

* `src/candle/_backends/cpu/ops.py` — modify `contiguous`, `flatten`, `unflatten`.
* `src/candle/_backends/mps/ops/shape.py` — modify `flatten`, `unflatten` GPU paths. `contiguous` already correct.
* `src/candle/_backends/npu/ops/shape.py` — modify `contiguous`. Verify (and possibly noop) `flatten_op` and `unflatten_op` already delegate correctly through `view_backend.reshape`.
* `tests/cpu/test_view_fastpath.py` — new file.
* `docs/superpowers/specs/2026-05-04-view-tracking-1b-six-ops-design.md` — this spec.

Estimated diff: ~80–120 lines source change + ~150 lines new test.

## Verification

Per `CLAUDE.md` Rule 4, `src/candle/` changes require pylint + cpu/contract tests + local backend tests:

```bash
# Pylint
pylint src/candle/ --rcfile=.github/pylint.conf

# CPU + contract
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/ tests/contract/ -v --tb=short

# Local backend (this host has NPU; MPS host runs the MPS suite separately)
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/npu/ -v --tb=short
```

Targeted suites to surface 1B-A breakage early:

```bash
python -m pytest tests/cpu/test_view_fastpath.py -v --tb=short
python -m pytest tests/cpu/test_view_tracking_infrastructure.py -v --tb=short
python -m pytest tests/cpu/ -v -k "contiguous or flatten or unflatten" --tb=short
```

## Risks

* **Hidden mutation-aliasing:** existing candle code may have called `contiguous()` expecting the result to be storage-detached. Regression coverage above will catch the most common cases; rest gets caught by the cpu/contract gate.
* **MPS GPU path divergence:** `flatten`/`unflatten` on MPS GPU currently produce a same-storage tensor via `_from_metal_buffer` but without `_make_view` metadata. Switching to `_make_view` changes how downstream ops see this tensor (it now has `_base` and view meta). Risk surface: any MPS op that branches on `_base`. Audit `mps/ops/` for `_base is None` checks before merging.
* **NPU `flatten_op`/`unflatten_op` already-correct assumption:** spec says "verify and possibly no-op." If the verify step finds `view_backend.reshape` *doesn't* actually preserve view-share for the contiguous case, the NPU branch needs explicit conditional-view code, not a no-op.

## Out of scope (1B-B)

Captured here only so the next spec author knows what's pending:

* `_view_func` / `_rev_view_func` callables registered for: `contiguous`, `flatten`, `squeeze`, `narrow`, `movedim`, `unflatten`.
* Removal of `ContiguousBackward0`, `FlattenBackward0`, `SqueezeBackward0`, `NarrowBackward0`, `MovedimBackward0`, `UnflattenBackward0` from `src/candle/_generated/functions.py`.
* Removal of `*_autograd` wrappers in `src/candle/_generated/variable_type.py` for these 6 ops.
* `_VT_PY` → standard registration entries in `src/candle/_generated/registration.py` for these 6 ops.
* Engine rebase block in `_accumulate_tensor_grad` becomes live for these 6 ops (was dead code in PR #372).

1B-B will get its own design doc the day 1B-A is merged.
