# Generated Autograd Drift Convergence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the signature drift between `_variable_type_cy.pyx` (codegen from `derivatives.yaml`) and the hand-edited `variable_type.py` / `functions.py`, so that the compiled Cython generated autograd path can safely handle more ops at runtime without `_save()` keyword mismatches or argument count errors.

**Architecture:** Fix this in two waves. Wave 1 fixes the 6 actively-broken ops whose Cython `_save()` calls pass wrong keyword arguments to the Python backward nodes in `functions.py`. The fix is to update `derivatives.yaml` entries so that the codegen output matches the hand-edited Python surface, then regenerate all 5 generated files. Wave 2 migrates the registration of the fixed ops from `_VT_PY` (Python fallback) back to `_VT` (compiled candidate). Each wave is independently shippable.

**Tech Stack:** Python codegen (`tools/autograd/`), YAML derivative specs, generated Cython/Python autograd wrappers, pytest, conda `mindnlp` environment

---

## Background: the 9 mismatched ops

| Op | CY first arg | PY first arg | `_save()` mismatch? |
|----|-------------|-------------|---------------------|
| `cumsum` | `self_` | `input` | Yes — CY passes `self_=`, node expects `input_=` |
| `gather` | `self_` | `input` | Yes — CY passes `self_=`, node expects `input_=` |
| `prod` | `self_` | `input` | Yes — CY passes `self_=`, node expects `input_=` |
| `repeat` | `self_` | `input` | Yes — CY passes `self_=`, node expects `input_=` |
| `sort` | `self_` | `self` | Yes — CY passes `indices=`, node expects `result1=` |
| `topk` | `self_` | `self` | Yes — CY passes `indices=`, node expects `result1=` |
| `avg_pool2d` | `self_` | `self` | No — backward node compatible |
| `index_select` | `self_` | `input` | No — backward node has shim |
| `max_pool2d` | `self_` | `self` | No — backward node compatible |

**Wave 1 targets the 6 "Yes" ops.** The 3 "No" ops are lower priority since they work at runtime despite the signature difference.

---

## Scope guardrails

- Do NOT hand-edit `src/candle/_generated/*.py` or `*.pyx` directly. All changes go through `derivatives.yaml` + codegen.
- Do NOT change the 110 Python-only legacy wrappers in this plan. That is a separate future effort.
- Do NOT change `gen_registration.py` or the registration split structure. Only move ops between `_VT_PY` and `_VT` sections in the already-generated `registration.py`.
- If regeneration changes unrelated ops, verify those changes are benign before committing.

---

### Task 1: Pin the 6 broken ops with targeted regression tests

**Files:**
- Create: `tests/contract/test_generated_drift_convergence.py`
- Test: `tests/contract/test_generated_drift_convergence.py`

**Step 1: Write failing tests that encode the current mismatch**

Create `tests/contract/test_generated_drift_convergence.py`:

```python
"""Tests that the generated Cython forward wrappers pass correct _save() kwargs
to the Python backward nodes. These fail when derivatives.yaml and the hand-edited
Python surface are out of sync."""
import re
from pathlib import Path

_GEN = Path(__file__).parent.parent.parent / "src" / "candle" / "_generated"


def _read(name):
    return (_GEN / name).read_text(encoding="utf-8")


def _extract_save_kwargs(text, class_name):
    """Extract the keyword argument names from a backward node's _save() method."""
    pattern = rf"class {class_name}\(.*?\n(.*?)(?=\nclass |\Z)"
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return set()
    body = m.group(1)
    save_match = re.search(r"def _save\(self, \*, ([^)]*)\)", body)
    if not save_match:
        return set()
    return {p.split("=")[0].strip() for p in save_match.group(1).split(",")}


def _extract_save_call_kwargs(text, func_name):
    """Extract the keyword argument names from a forward wrapper's grad_fn._save() call."""
    pattern = rf"def {func_name}\(.*?\n(.*?)(?=\ndef |\Z)"
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return set()
    body = m.group(1)
    save_match = re.search(r"grad_fn\._save\(([^)]*)\)", body)
    if not save_match:
        return set()
    return {p.split("=")[0].strip() for p in save_match.group(1).split(",")}


def test_cumsum_save_kwargs_match():
    fn = _read("functions.py")
    cy = _read("_variable_type_cy.pyx")
    node_kwargs = _extract_save_kwargs(fn, "CumsumBackward0")
    cy_call_kwargs = _extract_save_call_kwargs(cy, "cumsum_autograd_post")
    assert cy_call_kwargs.issubset(node_kwargs), (
        f"CY cumsum_autograd_post passes {cy_call_kwargs} but node accepts {node_kwargs}"
    )


def test_gather_save_kwargs_match():
    fn = _read("functions.py")
    cy = _read("_variable_type_cy.pyx")
    node_kwargs = _extract_save_kwargs(fn, "GatherBackward0")
    cy_call_kwargs = _extract_save_call_kwargs(cy, "gather_autograd_post")
    assert cy_call_kwargs.issubset(node_kwargs), (
        f"CY gather_autograd_post passes {cy_call_kwargs} but node accepts {node_kwargs}"
    )


def test_prod_save_kwargs_match():
    fn = _read("functions.py")
    cy = _read("_variable_type_cy.pyx")
    node_kwargs = _extract_save_kwargs(fn, "ProdBackward0")
    cy_call_kwargs = _extract_save_call_kwargs(cy, "prod_autograd_post")
    assert cy_call_kwargs.issubset(node_kwargs), (
        f"CY prod_autograd_post passes {cy_call_kwargs} but node accepts {node_kwargs}"
    )


def test_repeat_save_kwargs_match():
    fn = _read("functions.py")
    cy = _read("_variable_type_cy.pyx")
    node_kwargs = _extract_save_kwargs(fn, "RepeatBackward0")
    cy_call_kwargs = _extract_save_call_kwargs(cy, "repeat_autograd_post")
    assert cy_call_kwargs.issubset(node_kwargs), (
        f"CY repeat_autograd_post passes {cy_call_kwargs} but node accepts {node_kwargs}"
    )


def test_sort_save_kwargs_match():
    fn = _read("functions.py")
    cy = _read("_variable_type_cy.pyx")
    node_kwargs = _extract_save_kwargs(fn, "SortBackward0")
    cy_call_kwargs = _extract_save_call_kwargs(cy, "sort_autograd_post")
    assert cy_call_kwargs.issubset(node_kwargs), (
        f"CY sort_autograd_post passes {cy_call_kwargs} but node accepts {node_kwargs}"
    )


def test_topk_save_kwargs_match():
    fn = _read("functions.py")
    cy = _read("_variable_type_cy.pyx")
    node_kwargs = _extract_save_kwargs(fn, "TopkBackward0")
    cy_call_kwargs = _extract_save_call_kwargs(cy, "topk_autograd_post")
    assert cy_call_kwargs.issubset(node_kwargs), (
        f"CY topk_autograd_post passes {cy_call_kwargs} but node accepts {node_kwargs}"
    )
```

**Step 2: Run the tests to verify failure**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/contract/test_generated_drift_convergence.py -v --tb=short
```

Expected: 6 failures showing `_save()` kwarg mismatches.

**Step 3: Commit checkpoint**

```bash
git add tests/contract/test_generated_drift_convergence.py
git commit -m "test: pin 6 generated autograd _save() kwarg mismatches"
```

---

### Task 2: Fix `derivatives.yaml` entries for the 6 broken ops

**Files:**
- Modify: `tools/autograd/derivatives.yaml`
- Test: `tests/contract/test_generated_drift_convergence.py`

**Step 1: Understand the root cause for each op**

The mismatch happens because:
- `derivatives.yaml` uses PyTorch-native parameter names (e.g. `self` which becomes `self_` in codegen)
- The hand-edited Python `variable_type.py` was rewritten to use `input` as the first arg for some ops
- The hand-edited Python `functions.py` backward nodes were rewritten to use `input_` in `_save()`
- The Cython codegen still follows `derivatives.yaml` and uses `self_`

**The correct fix direction:** Update `derivatives.yaml` so that the codegen output matches what the hand-edited Python backward nodes expect. This means:
- For `cumsum`, `gather`, `prod`, `repeat`: the yaml entry's schema should use `input` instead of `self` as the first tensor arg, so codegen produces `input_` which matches the backward node's `_save(*, input_=None)`
- For `sort`, `topk`: the yaml entry's saved outputs should use `result1` (matching the backward node) instead of letting codegen produce `indices`

**Step 2: Update each entry in `derivatives.yaml`**

For each of the 6 ops, find the entry in `tools/autograd/derivatives.yaml` and update the schema line so that the first tensor argument name and saved output names match what the hand-edited backward node expects.

**Important:** Read the current Python `functions.py` backward node for each op FIRST to understand exactly what `_save()` parameter names it uses. Then adjust the yaml entry to produce those names.

For example, if `GatherBackward0._save(*, index=None, input_=None)` in `functions.py`, then the yaml entry for `gather` should have `input` as its first tensor arg (not `self`), so codegen produces `input_` → `grad_fn._save(index=index, input_=input_)`.

**Step 3: Regenerate all 5 generated files**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m tools.autograd.gen_autograd
```

Expected: All 5 files update. `_variable_type_cy.pyx` and `variable_type.py` should now produce matching `_save()` calls for the 6 ops.

**Step 4: Verify the `_save()` kwargs now match**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/contract/test_generated_drift_convergence.py -v --tb=short
```

Expected: 6 tests pass.

**Step 5: Verify no unrelated regressions in the regenerated files**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/contract/test_generated_registration_coverage.py -v --tb=short
```

Expected: All registration coverage tests still pass.

**Step 6: Run CPU autograd tests for the 6 ops**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/cpu/test_autograd_ops.py -k "Gather or Repeat or Cumsum or Prod or Sort or Topk" -v --tb=short
```

Expected: All pass.

**Step 7: Commit checkpoint**

```bash
git add tools/autograd/derivatives.yaml src/candle/_generated/functions.py src/candle/_generated/variable_type.py src/candle/_generated/_variable_type_cy.pyx src/candle/_generated/_functions_cy.pyx src/candle/_generated/registration.py
git commit -m "fix: align derivatives.yaml with hand-edited backward node _save() signatures"
```

---

### Task 3: Move the 6 fixed ops back to compiled registration path

**Files:**
- Modify: `src/candle/_generated/registration.py`
- Test: `tests/cpu/test_autograd_ops.py`
- Test: `tests/contract/test_generated_drift_convergence.py`

**Step 1: Change registration for the 6 ops from `_VT_PY` to `_VT`**

In `src/candle/_generated/registration.py`, find the registration lines for `cumsum`, `gather`, `prod`, `repeat`, `sort`, `topk` and change `_VT_PY.` back to `_VT.`.

**Step 2: Build compiled extensions**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python setup.py build_ext --inplace
```

Expected: Build succeeds.

**Step 3: Run the 6 ops through CPU autograd tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/cpu/test_autograd_ops.py -k "Gather or Repeat or Cumsum or Prod or Sort or Topk" -v --tb=short
```

Expected: All pass — now running through the compiled Cython path.

**Step 4: Run broader CPU test suite to check for regressions**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/cpu/test_autograd_ops.py tests/cpu/test_backward_round2.py tests/cpu/test_transformer_backward.py -v --tb=short
```

Expected: No new failures compared to upstream main.

**Step 5: Commit checkpoint**

```bash
git add src/candle/_generated/registration.py
git commit -m "refactor: move 6 drift-fixed ops back to compiled registration path"
```

---

### Task 4: Full verification

**Files:**
- Verify only

**Step 1: Build**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python setup.py build_ext --inplace
```

**Step 2: Run contract tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/contract/test_generated_drift_convergence.py tests/contract/test_generated_registration_coverage.py -v --tb=short
```

**Step 3: Run CPU + contract full suite**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/cpu/ tests/contract/ -v --tb=short
```

**Step 4: Run MPS suite**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/mps/ -v --tb=short
```

**Step 5: Run pylint**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  pylint src/candle/ --rcfile=.github/pylint.conf
```

Expected: 10.00/10.

**Step 6: Final commit**

```bash
git add -A
git commit -m "feat: converge generated autograd drift for 6 signature-mismatched ops"
```

---

## Notes for whoever executes this plan

1. **The fix direction is: make codegen match the hand-edited Python backward nodes**, not the other way around. The Python backward nodes (`functions.py`) are the runtime truth that CPU tests validate against. The Cython forward wrappers (`_variable_type_cy.pyx`) must produce `_save()` calls compatible with those nodes.

2. **Do not hand-edit `_generated/*.py` or `*.pyx`**. Only edit `derivatives.yaml` and `tools/autograd/gen_*.py`, then regenerate.

3. **If regeneration changes more than the 6 target ops**, diff carefully. Some changes may be benign (e.g., whitespace, alias reordering). Others may indicate that the yaml edit affected a shared overload. Review each diff hunk.

4. **The 3 compatible-but-mismatched ops** (`avg_pool2d`, `index_select`, `max_pool2d`) are not in scope for this plan. They work at runtime despite the signature difference and can be fixed in a follow-up.

5. **The 110 Python-only legacy wrappers** are not in scope. They remain on `_VT_PY` and are a separate future effort.

6. **After this plan lands**, the compiled Cython path will handle 6 more ops at runtime, reducing the `_VT_PY` fallback surface.
