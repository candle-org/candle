# Generated Autograd Drift Convergence and Registration Follow-up Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-establish `tools/autograd/derivatives.yaml` plus the code generators as the single source of truth for generated autograd wrappers, then safely enable runtime registration to use `_variable_type_cy` only for the generated-safe subset without breaking legacy/manual wrappers.

**Architecture:** Treat the current problem as two coupled but separable issues: (1) drift between `derivatives.yaml` and the hand-edited/generated Python artifacts under `src/candle/_generated/`; (2) registration currently assuming one monolithic `_VT` surface even though compiled `_variable_type_cy.pyx` only covers the codegen-derived subset. First, inventory and codify the drift with tests. Second, move Python-only manual additions back into the generator inputs or explicitly classify them as legacy. Third, split registration into generated-safe compiled registrations vs Python-only legacy registrations. Keep changes generic and PyTorch-compatible; do not add app-specific exceptions.

**Tech Stack:** Python codegen (`tools/autograd/`), YAML derivative specs, generated autograd wrappers, Cython-generated wrapper modules, pytest, dispatch registration

---

## Task 1: Pin the current drift and registration gap in tests

**Files:**
- Modify: `tests/contract/test_gen_registration_filter.py`
- Create if missing: `tests/contract/test_generated_registration_coverage.py`
- Test: `tests/contract/test_generated_registration_coverage.py`

**Step 1: Write a failing coverage test for registration vs `_variable_type_cy.pyx`**

Add a test file `tests/contract/test_generated_registration_coverage.py` with these checks:

```python
import re
from pathlib import Path


def _read(path):
    return Path(path).read_text()


def test_registration_symbols_exist_in_either_compiled_or_python_surface():
    root = Path(__file__).resolve().parents[2]
    reg = _read(root / "src" / "candle" / "_generated" / "registration.py")
    vt_py = _read(root / "src" / "candle" / "_generated" / "variable_type.py")
    vt_cy = _read(root / "src" / "candle" / "_generated" / "_variable_type_cy.pyx")

    reg_names = set(re.findall(r"_VT\.([A-Za-z0-9_]+)", reg))
    py_defs = set(re.findall(r"^def ([A-Za-z0-9_]+)", vt_py, re.MULTILINE))
    cy_defs = set(re.findall(r"^def ([A-Za-z0-9_]+)", vt_cy, re.MULTILINE))

    missing = sorted(name for name in reg_names if name not in py_defs and name not in cy_defs)
    assert missing == []
```

Add a second test that explicitly demonstrates today’s drift:

```python
def test_compiled_variable_type_surface_matches_generated_safe_registration_subset():
    root = Path(__file__).resolve().parents[2]
    reg = _read(root / "src" / "candle" / "_generated" / "registration.py")
    vt_cy = _read(root / "src" / "candle" / "_generated" / "_variable_type_cy.pyx")

    reg_names = set(re.findall(r"_VT\.([A-Za-z0-9_]+)", reg))
    cy_defs = set(re.findall(r"^def ([A-Za-z0-9_]+)", vt_cy, re.MULTILINE))

    assert "sum_to_size_autograd_post" not in cy_defs
    assert "sum_to_size_autograd_post" in reg_names
```

**Step 2: Run the tests to verify failure**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/contract/test_generated_registration_coverage.py -v --tb=short
```

Expected:
- The first test may fail on broken alias entries in `registration.py`
- The second test should pass and document the concrete drift (`sum_to_size_autograd_post` missing from `_variable_type_cy.pyx`)

**Step 3: Add one alias-specific regression test**

In the same file, add:

```python
def test_registration_does_not_reference_generic_alias_without_backing_wrapper():
    root = Path(__file__).resolve().parents[2]
    reg = _read(root / "src" / "candle" / "_generated" / "registration.py")
    vt_py = _read(root / "src" / "candle" / "_generated" / "variable_type.py")

    reg_names = set(re.findall(r"_VT\.([A-Za-z0-9_]+)", reg))
    py_defs = set(re.findall(r"^def ([A-Za-z0-9_]+)", vt_py, re.MULTILINE))

    assert "add_autograd" not in reg_names or "add_autograd" in py_defs
```

This should fail today if `registration.py` still points at generic names that only have overload-specific implementations.

**Step 4: Re-run the focused tests**

Run the same command again and confirm the failures are still for the intended reasons.

**Step 5: Commit checkpoint**

```bash
git add tests/contract/test_generated_registration_coverage.py
git commit -m "test: pin generated registration coverage gaps"
```

---

## Task 2: Inventory and codify Python-only manual additions

**Files:**
- Modify: `tests/contract/test_generated_registration_coverage.py`
- Modify: `tools/autograd/derivatives.yaml`
- Modify: `tools/autograd/gen_variable_type.py`
- Modify: `tools/autograd/gen_functions.py`
- Test: `tests/contract/test_generated_registration_coverage.py`

**Step 1: Write a failing test for known Python-only manual wrappers**

Add a classification test with a hard-coded minimal seed list derived from the current inventory:

```python
def test_known_python_only_manual_wrappers_are_tracked_explicitly():
    known_manual = {
        "sum_to_size_autograd_post",
        "diff_autograd",
        "diff_autograd_post",
    }
    root = Path(__file__).resolve().parents[2]
    vt_py = _read(root / "src" / "candle" / "_generated" / "variable_type.py")

    py_defs = set(re.findall(r"^def ([A-Za-z0-9_]+)", vt_py, re.MULTILINE))
    for name in known_manual:
        assert name in py_defs
```

Then add a second assertion documenting the intended future state:

```python
def test_manual_wrapper_inventory_must_be_empty_after_drift_convergence():
    assert True, "replace with real inventory check once Task 2 lands"
```

This is a placeholder red test you will strengthen immediately after inventory logic is implemented.

**Step 2: Run the focused tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/contract/test_generated_registration_coverage.py -k "manual_wrapper" -v --tb=short
```

Expected:
- The inventory placeholder should force a failure until you codify the real distinction.

**Step 3: Decide op-by-op source of truth**

For the first pass, classify only the highest-impact drifted additions:
- `sum_to_size`
- `diff`
- any other wrappers currently in `variable_type.py` but absent from `_variable_type_cy.pyx` and absent from `derivatives.yaml`

Use this rule:
- If the op is conceptually generated from derivatives and should exist in both Python/Cython outputs, add it to `derivatives.yaml`
- If it is truly outside the generator model, mark it as legacy/manual and keep it out of the compiled-safe section

**Step 4: Add `sum_to_size` to the generator source of truth**

Update `tools/autograd/derivatives.yaml` with a representation of `sum_to_size` that matches current behavior. If the generator cannot currently express “post-only wrapper” semantics, extend the generator minimally to support it rather than keeping `sum_to_size` as a hidden manual append.

Minimum acceptable outcome for this task:
- `sum_to_size_autograd_post` comes from the generator inputs instead of manual Python drift
- both `variable_type.py` and `_variable_type_cy.pyx` gain the same generated symbol after regeneration

**Step 5: Add `diff` to the same source of truth or explicitly mark it legacy**

Do the same for `diff`. If you cannot safely express it in the current generator shape, add a deliberate “legacy/manual inventory” mechanism and document it in code/tests so drift is explicit, not accidental.

**Step 6: Regenerate generated artifacts**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m tools.autograd.gen_autograd
```

Expected:
- `src/candle/_generated/variable_type.py` and `_variable_type_cy.pyx` update in lockstep for the migrated entries

**Step 7: Replace the placeholder inventory test with a real one**

Update the placeholder so it checks that every “manual” wrapper is either:
- now present in both generated outputs, or
- explicitly listed in a small `LEGACY_MANUAL_WRAPPERS` set in the test/module

**Step 8: Re-run coverage tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/contract/test_generated_registration_coverage.py -v --tb=short
```

Expected:
- Improved coverage signal with fewer undocumented drift cases.

**Step 9: Commit checkpoint**

```bash
git add tools/autograd/derivatives.yaml tools/autograd/gen_variable_type.py tools/autograd/gen_functions.py src/candle/_generated/variable_type.py src/candle/_generated/_variable_type_cy.pyx tests/contract/test_generated_registration_coverage.py
git commit -m "refactor: move manual generated wrappers back to codegen inputs"
```

---

## Task 3: Fix registration aliasing for overload-generic names

**Files:**
- Modify: `tools/autograd/gen_registration.py`
- Modify: `tools/autograd/gen_variable_type.py`
- Modify: `src/candle/_generated/registration.py`
- Test: `tests/contract/test_generated_registration_coverage.py`

**Step 1: Write a failing alias test**

Add tests for known generic alias cases:

```python
def test_registration_generic_aliases_have_backing_wrapper_symbols():
    root = Path(__file__).resolve().parents[2]
    reg = _read(root / "src" / "candle" / "_generated" / "registration.py")
    vt_py = _read(root / "src" / "candle" / "_generated" / "variable_type.py")

    reg_names = set(re.findall(r"_VT\.([A-Za-z0-9_]+)", reg))
    py_defs = set(re.findall(r"^def ([A-Za-z0-9_]+)", vt_py, re.MULTILINE))

    for name in ("add_autograd", "sub_autograd", "mul_autograd", "div_autograd", "pow_autograd"):
        assert name not in reg_names or name in py_defs
```

**Step 2: Run the test to verify failure**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/contract/test_generated_registration_coverage.py -k "generic_aliases_have_backing_wrapper_symbols" -v --tb=short
```

Expected:
- Failure for at least one generic alias (e.g. `add_autograd`) if registration still references names that only exist as overload-specific wrappers.

**Step 3: Fix alias strategy at the generator level**

Pick one consistent rule and implement it end-to-end:

**Preferred rule:** ensure `gen_variable_type.py` always emits canonical generic aliases for overloaded ops when registration wants generic names.

Example target generated lines at the end of `variable_type.py` and `_variable_type_cy.pyx`:

```python
# Canonical overload aliases
add_autograd = add_tensor_autograd
add_autograd_post = add_tensor_autograd_post
mul_autograd = mul_tensor_autograd
mul_autograd_post = mul_tensor_autograd_post
```

If a generic alias is semantically ambiguous, then `gen_registration.py` should register the correct overload-specific name instead of inventing a generic one.

**Step 4: Regenerate artifacts**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m tools.autograd.gen_autograd
```

**Step 5: Re-run the alias tests**

Run the same command from Step 2.

Expected:
- PASS.

**Step 6: Commit checkpoint**

```bash
git add tools/autograd/gen_registration.py tools/autograd/gen_variable_type.py src/candle/_generated/registration.py src/candle/_generated/variable_type.py src/candle/_generated/_variable_type_cy.pyx tests/contract/test_generated_registration_coverage.py
git commit -m "fix: align generated registration aliases with wrapper outputs"
```

---

## Task 4: Split registration into generated-safe vs legacy/manual sections

**Files:**
- Modify: `tools/autograd/gen_registration.py`
- Modify: `src/candle/_generated/registration.py`
- Test: `tests/contract/test_generated_registration_coverage.py`
- Test: `tests/cpu/test_declarative_autograd.py`

**Step 1: Write a failing generated-safe split test**

Add a test that checks registration source selection is explicit:

```python
def test_registration_splits_compiled_safe_and_python_legacy_sections():
    root = Path(__file__).resolve().parents[2]
    text = _read(root / "src" / "candle" / "_generated" / "registration.py")
    assert "_VT_CY" in text
    assert "_VT_PY" in text
```

**Step 2: Run the test to verify failure**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/contract/test_generated_registration_coverage.py -k "splits_compiled_safe_and_python_legacy_sections" -v --tb=short
```

Expected:
- FAIL because registration still assumes a single `_VT` source.

**Step 3: Change `gen_registration.py` to emit two sources**

Target generated pattern:

```python
def register_generated_autograd_kernels():
    from .._dispatch.registration import register_autograd_kernels, register_autograd_post_kernels
    from . import variable_type as _VT_PY
    try:
        from . import _variable_type_cy as _VT_CY
    except ImportError:
        _VT_CY = None
```

Then emit registrations in two sections:

### Section A — generated-safe
Use `_VT_CY` if available, otherwise `_VT_PY`:

```python
    _VT = _VT_CY if _VT_CY is not None else _VT_PY
    register_autograd_kernels('abs', default=_VT.abs_autograd, ...)
```

### Section B — legacy/manual
Always use `_VT_PY`:

```python
    register_autograd_post_kernels('sum_to_size', _VT_PY.sum_to_size_autograd_post)
    register_autograd_kernels('diff', default=_VT_PY.diff_autograd, ...)
```

The generated-safe/legacy split should be data-driven from the inventory created in Tasks 1–3, not hard-coded ad hoc throughout the file.

**Step 4: Regenerate registration.py**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m tools.autograd.gen_autograd
```

**Step 5: Add runtime smoke tests**

In `tests/cpu/test_declarative_autograd.py`, add a focused import/runtime test:

```python
def test_registration_imports_when_compiled_variable_type_is_present():
    import candle._generated.registration as reg
    assert hasattr(reg, "register_generated_autograd_kernels")
```

And add a second one specifically for a legacy/manual op:

```python
def test_sum_to_size_post_registration_still_resolves_with_compiled_modules_present():
    import candle._generated.variable_type as vt
    assert hasattr(vt, "sum_to_size_autograd_post")
```

**Step 6: Re-run focused tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/contract/test_generated_registration_coverage.py -v --tb=short
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/cpu/test_declarative_autograd.py -k "registration_imports_when_compiled_variable_type_is_present or sum_to_size_post_registration_still_resolves" -v --tb=short
```

Expected:
- PASS.
- No import-time `AttributeError` from `_variable_type_cy` missing Python-only names.

**Step 7: Commit checkpoint**

```bash
git add tools/autograd/gen_registration.py src/candle/_generated/registration.py tests/contract/test_generated_registration_coverage.py tests/cpu/test_declarative_autograd.py
git commit -m "refactor: split generated registration into compiled and legacy paths"
```

---

## Task 5: Re-enable runtime preference for `_variable_type_cy` safely

**Files:**
- Modify: `tests/cpu/test_declarative_autograd.py`
- Verify only otherwise

**Step 1: Restore the runtime preference smoke tests**

Re-introduce focused tests equivalent to the previous deferred Task 6, but now against the split registration model:

```python
def test_registration_prefers_compiled_generated_variable_type_for_generated_safe_ops():
    import candle._generated._variable_type_cy as cy_vt
    assert hasattr(cy_vt, "exp_autograd")


def test_registration_keeps_python_fallback_for_legacy_ops():
    import candle._generated.variable_type as py_vt
    assert hasattr(py_vt, "sum_to_size_autograd_post")
```

**Step 2: Run those tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/cpu/test_declarative_autograd.py -k "prefers_compiled_generated_variable_type_for_generated_safe_ops or keeps_python_fallback_for_legacy_ops" -v --tb=short
```

Expected:
- PASS.

**Step 3: Build compiled modules in place**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python setup.py build_ext --inplace
```

Expected:
- `_generated/_variable_type_cy*.so` and `_generated/_functions_cy*.so` build successfully.

**Step 4: Run focused end-to-end declarative tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/cpu/test_declarative_autograd.py -k "generated_cython or registration or generated_package" -v --tb=short
```

Expected:
- PASS for the generated-module and registration smoke tests.

**Step 5: Commit checkpoint**

```bash
git add tests/cpu/test_declarative_autograd.py
git commit -m "test: verify compiled generated registration split"
```

---

## Task 6: Verify the narrowed runtime-switch scope end-to-end

**Files:**
- Verify only

**Step 1: Run build verification**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python setup.py build_ext --inplace
```

Expected: success.

**Step 2: Run contract tests covering codegen + registration**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/contract/test_generated_registration_coverage.py -v --tb=short
```

Expected: PASS.

**Step 3: Run focused CPU tests for generated autograd infra**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/cpu/test_declarative_autograd.py -v --tb=short
```

Expected: no new import-time or registration-time failures.

**Step 4: Run pylint gate**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  pylint src/candle/ --rcfile=.github/pylint.conf
```

Expected:
- 10.00/10 in the current environment, with only config compatibility warnings if using older pylint.

**Step 5: Commit final checkpoint**

```bash
git add tests/contract/test_generated_registration_coverage.py tests/cpu/test_declarative_autograd.py tools/autograd/derivatives.yaml tools/autograd/gen_registration.py tools/autograd/gen_variable_type.py tools/autograd/gen_functions.py src/candle/_generated/registration.py src/candle/_generated/variable_type.py src/candle/_generated/_variable_type_cy.pyx src/candle/_generated/functions.py src/candle/_generated/_functions_cy.pyx
git commit -m "refactor: converge generated autograd registration with Cython surface"
```

---

## Notes for whoever executes this plan

1. **Do not start by flipping `_VT` globally to `_variable_type_cy`.** That was already tried and breaks on Python-only legacy wrappers like `sum_to_size_autograd_post`.
2. **Do not hand-edit `src/candle/_generated/*.py` or `*.pyx` unless you are simultaneously changing the generator inputs or generator code.** The whole point of this follow-up is to stop that drift.
3. **Prefer bringing wrappers back into `derivatives.yaml` over growing ad hoc split logic.** Split registration only after the generator/source-of-truth story is cleaned up enough.
4. **If an op truly cannot fit the current generator model, make that explicit as legacy/manual.** Hidden drift is worse than a small explicit `LEGACY_MANUAL_WRAPPERS` set.
5. **Expect some registration references to be generic aliases over overloaded wrappers.** Fix those at the generator/alias layer, not with one-off `hasattr` checks sprinkled through runtime code.
6. **Re-run `gen_autograd` after any generator or yaml change.** Never reason from stale generated files.
