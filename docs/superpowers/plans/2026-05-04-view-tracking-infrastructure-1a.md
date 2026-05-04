# View-Tracking Infrastructure (Sub-Batch 1A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the PyTorch-aligned view-tracking infrastructure that lets candle replace per-op Backward Node classes for view ops with automatic gradient rebasing through the `_base` chain. This sub-batch lands the infrastructure ONLY — no op migrations yet — so existing autograd tests stay green and the contract roundtrip test stays unchanged.

**Architecture:** Replace today's opaque `_view_meta` dict on view tensors with two callables `_view_func` and `_rev_view_func` (matching torch semantics: `_view_func(new_base) -> Tensor` re-applies the forward view onto a new base; `_rev_view_func(grad) -> Tensor` walks a view-output gradient back to a base-shaped gradient). Add the autograd-engine logic that, when accumulating into a view tensor, calls `_rev_view_func` and accumulates onto the base instead. Add a small `tools/autograd/gen_view_funcs.py` codegen step that emits a stable inventory file mirroring torch's `VIEW_FUNCTIONS` table; the inventory file is data-only in 1A and consumed in 1B.

**Tech Stack:** Cython 3 (`.pyx` / `.pxd`), Python 3.11, candle's existing `_C` package, `_backends/common/view.py`, pytest, pylint 10/10 gate.

---

## File Structure

This sub-batch touches:

| File | Responsibility | Action |
|---|---|---|
| `src/candle/_C/_tensor_impl.pxd` | declare `_view_func`, `_rev_view_func` cdef public attributes on `TensorImpl` | modify |
| `src/candle/_C/_tensor_impl.pyx` | initialise the two new attributes; preserve existing `_view_meta` field for transition | modify |
| `src/candle/_backends/common/view.py` | extend `_make_view` to accept and store `view_func` / `rev_view_func`; default both to `None` for back-compat | modify |
| `src/candle/_C/_autograd_engine.pyx` | when `_accumulate_tensor_grad` lands grad on a tensor with `_base is not None` and `_rev_view_func is not None`, rebase grad and accumulate on `_base` | modify |
| `tools/autograd/gen_view_funcs.py` | new codegen step: emit `src/candle/_generated/view_funcs.py` listing the canonical view-op inventory (data only in 1A) | create |
| `tools/autograd/gen_autograd.py` | wire `gen_view_funcs` into `main()` | modify |
| `src/candle/_generated/view_funcs.py` | generated inventory file | create (auto) |
| `tests/cpu/test_view_tracking_infrastructure.py` | new tests asserting `_view_func` / `_rev_view_func` round-trip and rebase-on-grad correctness for an inline test op | create |
| `tests/contract/test_codegen_roundtrip.py` | extend to cover the new generated `view_funcs.py` | modify |

No op migrations: `contiguous` / `flatten` / `squeeze` / `narrow` / `movedim` / `unflatten` are NOT touched in this sub-batch. Their backward paths stay as-is until sub-batch 1B.

---

## Task 1: Pre-flight — confirm baseline build and test surface

**Files:**
- Verify only

- [ ] **Step 1: Confirm worktree is on the right branch and clean**

```bash
git rev-parse --abbrev-ref HEAD
git status
```

Expected: branch is `view-tracking-infrastructure` (or similar), tree clean except for any uncommitted plan / spec.

- [ ] **Step 2: Build Cython extensions in-place**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 python setup.py build_ext --inplace
```

Expected: build succeeds, no errors.

- [ ] **Step 3: Run existing autograd contract + cpu tests as baseline**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/cpu/test_autograd_api.py tests/cpu/test_view_dispatch.py \
                   tests/contract/test_generated_registration_coverage.py \
                   -q --tb=short
```

Expected: all green. If anything is red on baseline, stop and report — do not proceed.

---

## Task 2: Extend `TensorImpl` with `_view_func` and `_rev_view_func` cdef attributes

**Files:**
- Modify: `src/candle/_C/_tensor_impl.pxd`
- Modify: `src/candle/_C/_tensor_impl.pyx`
- Test: `tests/cpu/test_view_tracking_infrastructure.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cpu/test_view_tracking_infrastructure.py`:

```python
"""Pin the new view-tracking attributes on TensorImpl.

Sub-batch 1A only adds the storage; sub-batch 1B will populate it for the
real view ops.  Tests here only assert the attributes exist, default to
None, and accept assignment of a callable.
"""
import candle


def test_tensor_has_view_func_attribute_defaulting_to_none():
    t = candle.zeros((2, 3))
    assert hasattr(t, "_view_func")
    assert t._view_func is None


def test_tensor_has_rev_view_func_attribute_defaulting_to_none():
    t = candle.zeros((2, 3))
    assert hasattr(t, "_rev_view_func")
    assert t._rev_view_func is None


def test_view_func_attribute_accepts_callable_assignment():
    t = candle.zeros((2, 3))
    f = lambda base: base
    t._view_func = f
    assert t._view_func is f


def test_rev_view_func_attribute_accepts_callable_assignment():
    t = candle.zeros((2, 3))
    f = lambda grad: grad
    t._rev_view_func = f
    assert t._rev_view_func is f
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/cpu/test_view_tracking_infrastructure.py -v --tb=short
```

Expected: 4 failures with `AttributeError: 'Tensor' object has no attribute '_view_func'` (and `_rev_view_func`).

- [ ] **Step 3: Add the cdef public declarations**

Open `src/candle/_C/_tensor_impl.pxd`. Find the existing block declaring `_view_meta`:

```cython
    cdef public object _view_meta
```

Add two more cdef public attributes immediately below it:

```cython
    cdef public object _view_meta
    cdef public object _view_func
    cdef public object _rev_view_func
```

- [ ] **Step 4: Initialise them to `None` in `_tensor_impl.pyx`**

Open `src/candle/_C/_tensor_impl.pyx`. Find every place that initialises `_view_meta = None` (line 494 today, plus any post-construction helpers). Add equivalent initialisation lines:

```cython
        view._view_meta = None
        view._view_func = None
        view._rev_view_func = None
```

Search the file with `grep -n "_view_meta" src/candle/_C/_tensor_impl.pyx` to find every location. There must be initialisation everywhere `_view_meta` is currently set to a fresh value (including in any `as_view`-like helper).

- [ ] **Step 5: Rebuild Cython extensions**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 python setup.py build_ext --inplace
```

Expected: clean rebuild.

- [ ] **Step 6: Run the test**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/cpu/test_view_tracking_infrastructure.py -v --tb=short
```

Expected: 4 PASS.

- [ ] **Step 7: Run the autograd / view dispatch suite to confirm no regression**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/cpu/test_autograd_api.py tests/cpu/test_view_dispatch.py -q
```

Expected: same green baseline as Task 1 Step 3.

- [ ] **Step 8: Commit**

```bash
git add src/candle/_C/_tensor_impl.pxd src/candle/_C/_tensor_impl.pyx \
        tests/cpu/test_view_tracking_infrastructure.py
git commit -m "feat(_C): add _view_func / _rev_view_func attributes on TensorImpl"
```

---

## Task 3: Plumb `view_func` / `rev_view_func` through `_make_view`

**Files:**
- Modify: `src/candle/_backends/common/view.py:1-40`
- Test: `tests/cpu/test_view_tracking_infrastructure.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/cpu/test_view_tracking_infrastructure.py`:

```python
def test_make_view_propagates_view_func_and_rev_view_func():
    from candle._backends.common.view import _make_view
    base = candle.zeros((2, 3))
    base.requires_grad = True
    fwd = lambda new_base: new_base
    rev = lambda grad: grad
    view = _make_view(
        base,
        shape=(2, 3),
        stride=(3, 1),
        offset=0,
        op="identity",
        view_func=fwd,
        rev_view_func=rev,
    )
    assert view._view_func is fwd
    assert view._rev_view_func is rev
    assert view._base is base


def test_make_view_default_view_func_and_rev_view_func_is_none():
    from candle._backends.common.view import _make_view
    base = candle.zeros((2, 3))
    view = _make_view(base, shape=(2, 3), stride=(3, 1), offset=0, op="identity")
    assert view._view_func is None
    assert view._rev_view_func is None
```

- [ ] **Step 2: Run, expect FAIL**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/cpu/test_view_tracking_infrastructure.py -v --tb=short
```

Expected: 2 new failures with `TypeError: _make_view() got an unexpected keyword argument 'view_func'`.

- [ ] **Step 3: Extend `_make_view`**

Open `src/candle/_backends/common/view.py`. The current signature is:

```python
def _make_view(base, shape, stride, offset, op, source=None, *, creation_kind=None):
```

Replace with:

```python
def _make_view(base, shape, stride, offset, op, source=None, *, creation_kind=None,
               view_func=None, rev_view_func=None):
```

At the bottom of the function body (after `view._view_meta = {...}` block), add:

```python
    view._view_func = view_func
    view._rev_view_func = rev_view_func
    return view
```

Make sure the existing `return view` is removed (or replaced) so we don't double-return.

- [ ] **Step 4: Run the new tests**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/cpu/test_view_tracking_infrastructure.py -v --tb=short
```

Expected: 6 PASS.

- [ ] **Step 5: Run the existing view-dispatch suite**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/cpu/test_view_dispatch.py tests/cpu/test_tensor_view.py \
                   tests/contract/test_view_as_real_complex.py -q
```

Expected: same baseline; no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/candle/_backends/common/view.py \
        tests/cpu/test_view_tracking_infrastructure.py
git commit -m "feat(view): thread view_func/rev_view_func through _make_view"
```

---

## Task 4: Engine-level grad rebase when accumulating onto a view

**Files:**
- Modify: `src/candle/_C/_autograd_engine.pyx:293-330`
- Test: `tests/cpu/test_view_tracking_infrastructure.py`

- [ ] **Step 1: Add the failing rebase test**

Append to `tests/cpu/test_view_tracking_infrastructure.py`:

```python
def test_grad_on_view_with_rev_view_func_rebases_onto_base():
    """If a view tensor with _rev_view_func receives gradient during backward,
    the engine must rebase that grad through _rev_view_func and accumulate it
    onto _base, leaving the view's own .grad as None.
    """
    base = candle.zeros((4,))
    base.requires_grad = True

    # Construct a fake view: shape (2,), stride identity, that adds grad
    # back as a (4,)-shaped tensor of half-magnitude per entry.
    from candle._backends.common.view import _make_view
    fwd = lambda new_base: new_base
    rev = lambda grad_view: candle.zeros((4,))  # zeros: simplest deterministic rebase
    view = _make_view(
        base,
        shape=(2,),
        stride=(1,),
        offset=0,
        op="identity_truncate",
        view_func=fwd,
        rev_view_func=rev,
    )
    view.requires_grad = True

    # Drive a backward pass that lands a non-trivial grad on the view tensor.
    g_view = candle.ones((2,))
    base.grad = None
    view._accumulate_grad_node = None  # avoid hook interference

    # Use the engine's accumulate path directly through the public backward()
    # entry point on a tiny graph: y = view; y.backward(g_view).
    view.backward(g_view)

    # Grad should be on the base, not on the view.
    assert view.grad is None
    assert base.grad is not None
    # The rev_view_func returned zeros((4,)), so base.grad should be zeros.
    import numpy as np
    np.testing.assert_array_equal(base.grad.numpy(), np.zeros((4,)))
```

- [ ] **Step 2: Run, expect FAIL**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/cpu/test_view_tracking_infrastructure.py::test_grad_on_view_with_rev_view_func_rebases_onto_base -v --tb=long
```

Expected: failure — likely `view.grad` is not `None`, or `base.grad` is `None`.

- [ ] **Step 3: Modify the engine's leaf accumulation path**

Open `src/candle/_C/_autograd_engine.pyx` and find `_accumulate_tensor_grad` (currently around line 293). The current logic accumulates into `tensor.grad` directly. Insert a rebase block immediately after the `apply_hooks` call but before the `should_accumulate_into_grad` decision:

```cython
        # PyTorch-aligned view-rebase: if the leaf is a view that carries a
        # _rev_view_func, redirect grad accumulation onto its base.
        cdef object base
        cdef object rev_func
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

The recursion is safe because `base._base is None` for a flat (non-nested) view, and the rebase block early-exits with `return`. For nested views (`view of view of base`) the recursion walks the chain.

- [ ] **Step 4: Rebuild Cython**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 python setup.py build_ext --inplace
```

- [ ] **Step 5: Run the rebase test**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/cpu/test_view_tracking_infrastructure.py::test_grad_on_view_with_rev_view_func_rebases_onto_base -v --tb=long
```

Expected: PASS.

- [ ] **Step 6: Run full view-tracking infra test file**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/cpu/test_view_tracking_infrastructure.py -v
```

Expected: 7 PASS.

- [ ] **Step 7: Run the autograd regression sweep**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/cpu/test_autograd_api.py tests/cpu/test_autograd_function.py \
                   tests/cpu/test_view_dispatch.py tests/cpu/test_tensor_view.py \
                   tests/contract/test_autograd_create_graph.py -q
```

Expected: green. Since no existing view op sets `_rev_view_func`, the new branch never fires for them, so behaviour is unchanged.

- [ ] **Step 8: Commit**

```bash
git add src/candle/_C/_autograd_engine.pyx \
        tests/cpu/test_view_tracking_infrastructure.py
git commit -m "feat(autograd): rebase grad through _rev_view_func when leaf is a view"
```

---

## Task 5: Codegen step — emit `src/candle/_generated/view_funcs.py` from a static inventory

**Files:**
- Create: `tools/autograd/gen_view_funcs.py`
- Modify: `tools/autograd/gen_autograd.py`
- Create: `src/candle/_generated/view_funcs.py` (auto-generated)
- Modify: `tests/contract/test_codegen_roundtrip.py`
- Test: `tests/contract/test_codegen_roundtrip.py`

The 1A pass produces a data-only inventory file. Sub-batch 1B will fill its `_view_func` / `_rev_view_func` callables with real bodies; today the inventory is just a list of view-op names and a placeholder kind enum.

- [ ] **Step 1: Write `gen_view_funcs.py`**

Create `tools/autograd/gen_view_funcs.py` with the exact content below. The op list mirrors PyTorch's `VIEW_FUNCTIONS` keys for ops candle already implements (audit: `view`, `reshape`, `permute`, `transpose`, `squeeze`, `unsqueeze`, `expand`, `narrow`, `select`, `slice`, `as_strided`, `flatten`, `t`, `unfold`, `view_as_real`, `view_as_complex`, plus candle-specific `contiguous` / `movedim` / `unflatten` recognised as views in candle today).

```python
"""Generate src/candle/_generated/view_funcs.py — the inventory of view ops.

Sub-batch 1A: emit data-only inventory.  Sub-batch 1B will plug per-op
forward/reverse callables.
"""
from __future__ import annotations


# Mirrors PyTorch tools/autograd/gen_inplace_or_view_type.py::VIEW_FUNCTIONS
# restricted to ops candle currently implements as views.  Sub-batch 1B
# extends this with per-op view_func / rev_view_func bodies.
VIEW_OPS = (
    "view",
    "reshape",
    "permute",
    "transpose",
    "squeeze",
    "unsqueeze",
    "expand",
    "narrow",
    "select",
    "slice",
    "as_strided",
    "flatten",
    "t",
    "unfold",
    "view_as_real",
    "view_as_complex",
    "contiguous",
    "movedim",
    "unflatten",
)


_HEADER = '''\
"""Auto-generated view-op inventory — DO NOT EDIT.

Generated by tools/autograd/gen_view_funcs.py.
Sub-batch 1A populates the static inventory; 1B fills in the per-op
view_func / rev_view_func bodies.
"""
from __future__ import annotations

'''


def gen_view_funcs() -> str:
    """Return the source text of src/candle/_generated/view_funcs.py."""
    parts = [_HEADER]
    parts.append("VIEW_OPS = (\n")
    for op in VIEW_OPS:
        parts.append(f"    {op!r},\n")
    parts.append(")\n")
    return "".join(parts)
```

- [ ] **Step 2: Wire it into `tools/autograd/gen_autograd.py`**

Open `tools/autograd/gen_autograd.py`. Inside `main()`, find the `files = {...}` dict (currently around line 23) and add a new entry:

```python
    from .gen_view_funcs import gen_view_funcs

    files = {
        "functions.py": gen_functions(infos),
        "variable_type.py": gen_variable_type(infos),
        "registration.py": gen_registration(infos),
        "_functions_cy.pyx": gen_functions_pyx(infos),
        "_variable_type_cy.pyx": gen_variable_type_pyx(infos),
        "view_funcs.py": gen_view_funcs(),
    }
```

- [ ] **Step 3: Run the generator and commit the new generated file**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 python -m tools.autograd.gen_autograd
```

Expected: log line `view_funcs.py — written (...)`. `src/candle/_generated/view_funcs.py` is created. Other files are unchanged from baseline.

- [ ] **Step 4: Verify the generated file content**

```bash
head -20 src/candle/_generated/view_funcs.py
```

Expected: header docstring + `VIEW_OPS = (` tuple containing all 19 op names.

- [ ] **Step 5: Extend roundtrip test to cover the new file**

Open `tests/contract/test_codegen_roundtrip.py`. Find `_GENERATED_FILES = (...)` near the top. Add `"view_funcs.py"` to the tuple:

```python
_GENERATED_FILES = (
    "functions.py",
    "variable_type.py",
    "registration.py",
    "_functions_cy.pyx",
    "_variable_type_cy.pyx",
    "view_funcs.py",
)
```

- [ ] **Step 6: Run the roundtrip test**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/contract/test_codegen_roundtrip.py -v --tb=short
```

Expected: still red (because the 5 pre-existing files still drift). But the failure message must now also include `view_funcs.py` only if there is a discrepancy on it — and there should NOT be: the new file roundtrips cleanly because it is purely generator-derived.

Verify this: in the failure summary, only the 3 pre-existing drifted files (`functions.py`, `variable_type.py`, `registration.py`) appear, NOT `view_funcs.py`.

- [ ] **Step 7: Run the focused test that imports candle to ensure nothing breaks**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -c "from candle._generated import view_funcs; print(len(view_funcs.VIEW_OPS))"
```

Expected: prints `19`.

- [ ] **Step 8: Commit**

```bash
git add tools/autograd/gen_view_funcs.py tools/autograd/gen_autograd.py \
        src/candle/_generated/view_funcs.py \
        tests/contract/test_codegen_roundtrip.py
git commit -m "feat(codegen): emit _generated/view_funcs.py inventory of view ops"
```

---

## Task 6: Verify full gate

**Files:**
- Verify only

- [ ] **Step 1: Pylint must remain 10/10**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  pylint src/candle/ --rcfile=.github/pylint.conf
```

Expected: `Your code has been rated at 10.00/10`. If anything new in this sub-batch trips pylint, fix it (likely `# pylint: disable=too-many-arguments` on the extended `_make_view` signature, or `unused-argument` on the placeholder fwd/rev parameters in tests).

- [ ] **Step 2: Full CPU + contract gate**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/cpu/ tests/contract/ -q --tb=short
```

Expected: same number of green tests as the sub-batch 1A baseline plus the 7 new tests in `test_view_tracking_infrastructure.py`. Roundtrip test in `test_codegen_roundtrip.py` is still red (expected — the 110 legacy ops haven't moved yet).

- [ ] **Step 3: Confirm the only red test is the roundtrip drift test**

```bash
PYTHONPATH="$PWD/src" conda run -n candle311 \
  python -m pytest tests/cpu/ tests/contract/ -q --tb=line 2>&1 | grep -E "FAILED|ERROR"
```

Expected: only `tests/contract/test_codegen_roundtrip.py::test_gen_autograd_roundtrips_checked_in_generated_sources` shows as `FAILED`.

- [ ] **Step 4: Inspect the diff one last time for stray edits**

```bash
git log --oneline upstream/main..HEAD
git diff --stat upstream/main..HEAD
```

Expected: 5 commits (Tasks 2-5 plus the spec/docs commit `b14a5e9` already there). Diff stat shows files: `_tensor_impl.pxd`, `_tensor_impl.pyx`, `common/view.py`, `_autograd_engine.pyx`, `gen_view_funcs.py`, `gen_autograd.py`, `view_funcs.py`, `test_view_tracking_infrastructure.py`, `test_codegen_roundtrip.py`, `2026-05-04-...-design.md`, `2026-05-04-...-plan.md`. No other files changed.

- [ ] **Step 5: Push and open PR**

```bash
git push -u origin view-tracking-infrastructure
gh pr create --repo candle-org/candle --head lvyufeng:view-tracking-infrastructure --base main \
  --title "feat(autograd): build PyTorch-aligned view-tracking infrastructure (1A of 4)" \
  --body "$(cat <<'EOF'
## Summary
- Adds `_view_func` / `_rev_view_func` cdef public attributes on `TensorImpl`.
- Threads `view_func` / `rev_view_func` kwargs through `_make_view`.
- Engine-level rebase: when a view tensor receives gradient during backward, walk it through `_rev_view_func` and accumulate on `_base`.
- Adds `tools/autograd/gen_view_funcs.py` codegen step emitting `_generated/view_funcs.py` (data-only inventory in 1A).
- Roundtrip contract test now also covers the new generated file.

## Why
This is sub-batch 1A of 4 in the codegen-legacy-ops torch-alignment migration (see `docs/superpowers/specs/2026-05-04-codegen-legacy-ops-torch-alignment-design.md`). It builds the infrastructure that lets later sub-batches retire the hand-added Backward Node classes for view ops in favour of PyTorch-style automatic gradient rebasing.

## Test plan
- [x] `tests/cpu/test_view_tracking_infrastructure.py` — 7 new tests pin `_view_func` / `_rev_view_func` storage, propagation through `_make_view`, and engine-level grad rebase.
- [x] `tests/contract/test_codegen_roundtrip.py` extended to cover the new generated `view_funcs.py`.
- [x] All existing autograd / view tests still green.
- [x] Pylint 10.00/10.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR opens, returns URL.

---

## Self-review

Spec requirements covered by this plan:

| Spec section | Plan task | Status |
|---|---|---|
| `gen_inplace_or_view_type.py` (Route A infrastructure) | n/a — deferred to 1B; 1A only adds the inventory data file (`gen_view_funcs.py`) | scoped down |
| `gen_view_funcs.py` codegen | Task 5 | ✅ |
| `_view_func` / `_rev_view_func` cdef fields on `TensorBase` | Task 2 | ✅ |
| `as_view` ctor populates the new fields | Task 3 (extends `_make_view`, candle's analogue of torch `as_view`) | ✅ |
| Engine-level grad rebase | Task 4 | ✅ |
| Roundtrip contract test extended | Task 5 Step 5 | ✅ |
| New `tests/cpu/test_view_tracking.py` | Task 2-4 — combined into `test_view_tracking_infrastructure.py` | ✅ (renamed for scope clarity) |

**Scope-down note (intentional):** the spec listed `gen_inplace_or_view_type.py` as a 1A deliverable, but on closer audit candle does not have a separate "ADInplaceOrView" dispatch key today (PyTorch does) and inserting one is its own significant infrastructure change. For 1A we land just enough — the cdef attributes, the `_make_view` plumbing, the engine rebase, and the inventory file. Sub-batch 1B will use this surface to migrate ops without needing a parallel `ADInplaceOrView` dispatch key. If a follow-up batch wants strict torch-equivalent dispatch-key semantics, it can be added after 1B / 1C / 1D land.

Type / signature consistency check:

- `_make_view(base, shape, stride, offset, op, source=None, *, creation_kind=None, view_func=None, rev_view_func=None)` — used in Task 3, consumed by tests in Task 3 and Task 4.
- `TensorImpl._view_func`, `TensorImpl._rev_view_func` — declared in Task 2 (.pxd), initialised in Task 2 (.pyx), populated by `_make_view` in Task 3, read by the autograd engine in Task 4.
- `gen_view_funcs()` returns `str` — used in Task 5 Step 2 inside the `files` dict.

No placeholders. No "TBD". Every task has the actual code it expects to see written.

---

## Out of scope (explicit non-deliverables)

- Migrating any of the 6 view ops onto `_view_func` / `_rev_view_func`. That is sub-batch 1B.
- Removing any hand-added Backward Node classes from `_generated/`. That is sub-batch 1B.
- Yaml entries for `broadcast_to` / `tile` / `moveaxis` / `take_along_dim` / `repeat_interleave`. That is sub-batch 1C.
- Composite forward for `diff`. That is sub-batch 1D.
- Any of the other 98 legacy ops. Out of scope of the entire 1A-1D series; each family gets its own spec + plan.

