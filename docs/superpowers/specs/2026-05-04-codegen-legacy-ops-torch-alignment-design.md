# Codegen legacy-ops torch alignment design

**Date:** 2026-05-04
**Status:** draft (awaiting user approval)
**Worktree:** `.worktrees/codegen-roundtrip`
**Tracking issue:** #371 (precursor) — to be split into multiple PRs

## Background

`tools/autograd/derivatives.yaml + tools/autograd/gen_*.py` is the declared single
source of truth for candle's autograd wrappers. In practice it is not: running

```bash
python -m tools.autograd.gen_autograd
```

against the checked-in `src/candle/_generated/{functions.py,variable_type.py,registration.py}`
strips ~275 KB of hand-added content — 110 fully registered ops, 193 `def`s, 102
backward classes, plus interspersed `_VT_PY` overrides, helper-function
ordering deltas, and hand-edited wrapper bodies — none of which derive from
`derivatives.yaml`.

A red contract test (`tests/contract/test_codegen_roundtrip.py`) pins this:
the generator must roundtrip the checked-in files byte-for-byte.

The user's directive: do not preserve the legacy versions; align candle with
PyTorch's actual autograd architecture instead.

## Goals

1. Make `tools/autograd/derivatives.yaml + tools/autograd/gen_*.py` candle's
   sole source of autograd wrapper code, byte-for-byte roundtripping the
   `_generated/` tree.
2. For every legacy op currently registered outside the generator, mirror
   PyTorch's actual handling: yaml entry, view-tracking, or composite forward.
3. Avoid keeping any wrapper, backward class, or registration line that does
   not match the corresponding PyTorch construct.

## Non-goals

- Adding ops PyTorch itself does not have.
- Generalising `formula_transpiler.py` beyond what these op families require.
- Performance optimisation of the affected paths during the migration. The
  result should be no slower than today; faster is bonus.

## PyTorch's actual autograd architecture for the 110 legacy ops

PyTorch handles its `derivatives.yaml`-absent ops through three distinct
codegen pipelines, all driven from the same `native_functions.yaml` schema:

### Route A — `gen_inplace_or_view_type.py` + `gen_view_funcs.py` (view ops)

For ops whose return type is annotated as a view (`Tensor(a) self -> Tensor(a)`)
or that appear in `VIEW_FUNCTIONS` / `RETURNS_VIEWS_OF_INPUT`, PyTorch:

- Emits an `ADInplaceOrView` kernel that calls `as_view(base, output, ...)`
  to record the parent base tensor and the forward/reverse view functions.
- Emits a reverse-view function (`gen_view_funcs.py`) whose backward is
  rebasing-aware and walks the `_base` chain instead of running a custom
  backward Node.
- Adds **no** entry to `derivatives.yaml`.

In this batch's scope, the following 6 candle legacy ops are torch-route-A:
`contiguous`, `flatten`, `squeeze`, `narrow`, `movedim`, `unflatten`.

### Route B — `derivatives.yaml` (yaml-derivable composites)

Ops with a closed-form gradient expressible in terms of already-supported
candle ops have a yaml entry whose backward formula `gen_variable_type.py`
transpiles into a Backward Node `backward()` body.

In this batch's scope: `broadcast_to`, `tile`, `moveaxis`, `take_along_dim`,
`repeat_interleave`. Each gets a `derivatives.yaml` entry.

### Route C — composite forward (no entry, no Backward Node)

Some ops decompose at the forward level into already-differentiable primitives.
Their forward implementation calls `redispatch("primitive_op", ...)` repeatedly,
and autograd records gradients through the decomposition automatically. No
yaml entry, no Backward Node.

In this batch's scope: `diff` (which today calls a hand-written
`_diff_backward_helper`). Its forward already decomposes through `narrow` +
`sub`; once `narrow` is on Route A, `diff` can drop its custom Backward Node
and rely on the decomposition.

## Scope: shape/view family (12 ops)

| op                   | torch route | candle today                       | this batch       |
|----------------------|-------------|-------------------------------------|------------------|
| `contiguous`         | A           | hand-added Backward = identity      | view-tracking    |
| `flatten`            | A           | hand-added Backward                 | view-tracking    |
| `squeeze`            | A           | hand-added Backward                 | view-tracking    |
| `narrow`             | A           | hand-added Backward                 | view-tracking    |
| `movedim`            | A           | hand-added Backward                 | view-tracking    |
| `unflatten`          | A           | hand-added Backward                 | view-tracking    |
| `broadcast_to`       | B           | hand-added Backward (`reduce_grad`) | yaml entry       |
| `tile`               | B           | hand-added Backward                 | yaml entry       |
| `moveaxis`           | B           | hand-added Backward                 | yaml entry       |
| `take_along_dim`     | B           | hand-added Backward                 | yaml entry       |
| `repeat_interleave`  | B           | hand-added Backward                 | yaml entry       |
| `diff`               | C           | hand-added Backward + helper        | composite fwd    |

## Architecture

The batch is large. We split it into four sub-batches that each ship a green
PR and leave `tests/contract/test_codegen_roundtrip.py` strictly less red than
before.

### Sub-batch 1A — view-tracking infrastructure (largest)

**Files added / modified:**

- `tools/autograd/gen_inplace_or_view_type.py` — new file mirroring PyTorch.
  Reads `native_functions.yaml`, emits per-view-op `ADInplaceOrView` kernels.
- `tools/autograd/gen_view_funcs.py` — new file. Emits reverse-view-func
  closures used by the autograd engine.
- `src/candle/_C/_TensorBase.pyx` — add `_view_func`, `_rev_view_func`,
  `_creation_meta` fields and accessors. Keep the existing `_base` chain.
- `src/candle/_C/_autograd_engine.pyx` — when a leaf tensor in the backward
  graph has `_base != None`, rebase its grad through `_view_func` /
  `_rev_view_func` instead of accumulating into the leaf directly.
- `src/candle/_C/_tensor_impl.pyx` — `as_view(base, output, view_func,
  rev_view_func)` constructor. Currently has a partial implementation; this
  fills in the missing fields.
- `tools/autograd/gen_autograd.py` — call `gen_inplace_or_view_type` and
  `gen_view_funcs` from `main()`.
- `src/candle/_generated/inplace_or_view_type.py` — new generated module.
- `src/candle/_generated/view_funcs.py` — new generated module.

**No** changes to the 12 ops in this sub-batch — only the infrastructure is
landed. Verification: existing autograd tests remain green; the new generated
files are byte-stable across regen.

### Sub-batch 1B — wire 6 view ops onto view-tracking

For `contiguous`, `flatten`, `squeeze`, `narrow`, `movedim`, `unflatten`:

- Add the schema annotation `Tensor(a) self -> Tensor(a)` (or its candle
  equivalent) to whatever already-existing schema declares these ops.
- Move them from candle's "ops with backward" set to candle's view-op set,
  so 1A's codegen picks them up.
- Delete their hand-added entries from `_generated/functions.py`,
  `_generated/variable_type.py`, `_generated/registration.py`.
- Verification: `tests/contract/test_codegen_roundtrip.py` shrinks by 6 ops.
  Existing autograd tests green.

### Sub-batch 1C — yaml entries for 5 composite ops

For `broadcast_to`, `tile`, `moveaxis`, `take_along_dim`, `repeat_interleave`:

- Write one `derivatives.yaml` entry each, with the backward formula matching
  the existing hand-written body. Delete the hand-added wrappers/Backward
  classes.
- If the formula uses any helper not in `formula_transpiler`, add the helper.
- Verification: `tests/contract/test_codegen_roundtrip.py` shrinks by 5 ops.

### Sub-batch 1D — composite forward for `diff`

- Reimplement `diff`'s forward as a sequence of `narrow` + `sub` calls.
  After 1B, `narrow` is view-tracked, so the decomposition is fully
  differentiable without a custom Backward Node.
- Delete `DiffBackward0`, `_diff_backward_helper`, `diff_autograd*`, the
  registration line, and the entry from `tests/contract/test_generated_registration_coverage.py::LEGACY_MANUAL_WRAPPERS`.
- Verification: roundtrip test shrinks to 110-12=98 remaining ops.

## Components

| Component                                | Responsibility                                                                                                | Owner |
|------------------------------------------|---------------------------------------------------------------------------------------------------------------|-------|
| `tools/autograd/gen_inplace_or_view_type.py` | Generate view-op kernels that call `as_view`.                                                              | new |
| `tools/autograd/gen_view_funcs.py`       | Generate reverse-view-func closures.                                                                          | new |
| `src/candle/_C/_TensorBase.pyx`          | Carry `_view_func`, `_rev_view_func`, `_creation_meta`.                                                       | extend |
| `src/candle/_C/_tensor_impl.pyx`         | `as_view` ctor populates the new fields.                                                                       | extend |
| `src/candle/_C/_autograd_engine.pyx`     | On grad arriving at a view, rebase via `_rev_view_func`.                                                      | extend |
| `tests/contract/test_codegen_roundtrip.py` | Asserts byte-for-byte roundtrip of `_generated/`.                                                            | new |
| `tests/cpu/test_view_tracking.py`        | Asserts `_base`/`_view_func` semantics on the 6 view ops match torch.                                          | new |

## Data flow

```
native_functions.yaml ──▶ gen_inplace_or_view_type ──▶ view kernel calls as_view(base, output, fwd, rev)
                                                              │
                                                              ▼
                                           Tensor with _base, _view_func, _rev_view_func set
                                                              │
                                                  backward()  │
                                                              ▼
                              autograd engine sees _base != None ──▶ rebase grad via _rev_view_func
                                                              │
                                                              ▼
                                           grad accumulates on the base tensor
```

## Error handling

- `gen_inplace_or_view_type.py` must error loudly if a schema declares a view
  output but the op has no candle backend kernel (today's "silent fall back to
  generated Backward Node" pattern is what causes hidden drift).
- `as_view` must error if `_base` is already set (no view-of-view-of-view
  cycles unless the engine explicitly walks them).
- The roundtrip contract test must error with the exact missing op name in
  its assertion message so future PRs see what's still legacy.

## Testing

- `tests/contract/test_codegen_roundtrip.py` (already added, currently red).
- `tests/cpu/test_view_tracking.py` (new, sub-batch 1A): for each of the 6
  view ops, assert `result._base is x`, `result._view_func is not None`,
  `x.grad` is correctly rebased after `result.sum().backward()`.
- Existing `tests/cpu/test_autograd_*.py` and `tests/contract/test_*.py`
  must remain green throughout.
- Pylint must remain at 10.00 / 10.

## Out of scope

- The other 98 legacy ops (`fft_*`, `linalg_*`, `conv*`, `pooling*`, `norm*`,
  `stack*`, `special_*`, etc.). Each becomes its own family-batch using the
  same routing decision tree. This spec only covers the shape/view family.
- Migrating the 14 hand-curated `_VT_PY=` registration overrides. Those will
  become trivially derivable once `gen_registration.py` knows which ops have
  Cython-side wrappers; that is a follow-up batch.
- Forward AD for the 12 ops. Their JVP rules can stay in `_C._forward_ad`'s
  default registry (already migrated).

## Verification

After sub-batch 1D lands:

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/contract/test_codegen_roundtrip.py -v
```

must show: roundtrip test passes for these 12 ops; assertion failure (if any)
lists only the remaining 98 legacy op names.

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/cpu/ tests/contract/ -q --tb=short
```

must remain green.

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  pylint src/candle/ --rcfile=.github/pylint.conf
```

must report 10.00 / 10.

## Risks

| Risk                                                                  | Mitigation                                                                                                  |
|-----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Sub-batch 1A is large and touches the autograd engine.                 | Land it as its own PR with no op changes; gate on existing autograd test suite.                              |
| View-tracking changes affect functionalize / NPU paths.                | `src/candle/_dispatch/functionalize.py` already manages `_base` propagation; the sub-batch only adds fields. |
| `repeat_interleave` and `take_along_dim` may need new transpiler ops. | Add them when their yaml entries land (sub-batch 1C); revert the sub-batch if cost balloons.                  |
| Removing `DiffBackward0` could regress `diff` numerically.             | Sub-batch 1D ships a numerical regression test against PyTorch's `torch.diff`.                               |
| `flatten` and `squeeze` schemas may already lack the view annotation. | Audit during sub-batch 1B; add the annotation as part of the same PR.                                       |

## Sub-batch deliverables

| Sub-batch | PR title (proposed) | Net roundtrip ops resolved |
|-----------|---------------------|----------------------------|
| 1A        | `feat(autograd): build PyTorch-aligned view-tracking infrastructure` | 0 |
| 1B        | `refactor(autograd): wire 6 shape/view ops to view-tracking, drop hand-added Backward classes` | 6 |
| 1C        | `feat(derivatives): add yaml entries for broadcast_to/tile/moveaxis/take_along_dim/repeat_interleave` | 5 |
| 1D        | `refactor(autograd): drop DiffBackward0 in favour of composite forward` | 1 |
| **Total** |                     | **12** of 110 |
