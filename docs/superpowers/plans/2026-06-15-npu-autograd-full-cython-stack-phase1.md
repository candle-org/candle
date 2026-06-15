# NPU Autograd Full-Cython Stack — Phase 1 (Generator Fix) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the generated Cython backward Node classes (`_functions_cy.pyx`) actually wire into the autograd engine by emitting `def backward` (not the currently-broken `def apply`) and switching `_variable_type_cy.pyx` to import them as `_F` — so every generated backward formula runs as compiled Cython instead of Python.

**Architecture:** The autograd engine calls `node.backward(grad)`. Today the generated `.pyx` nodes define `def apply(self, grad)` (which the engine never calls) and their saved-tensor retrieval is also broken (`self.saved_tensors[...]` indexes a method). Meanwhile `_variable_type_cy.pyx` imports the **Python** `functions.py` as `_F`, so the engine only ever sees Python nodes. Phase 1 fixes the `.pyx` node generator to mirror the correct Python generator body (emitting `def backward`, correct saved-tensor retrieval, `_saved_fields` population), regenerates the checked-in files, and flips the `_F` import in `_variable_type_cy.pyx` to `_functions_cy`. After this, every non-skipped op's backward runs in compiled Cython — no `cdef class` conversion yet (that is a later phase), just a correct compiled `class X(Node):`.

**Tech Stack:** Python code generator (`tools/autograd/gen_functions.py`, `tools/autograd/gen_variable_type.py`), Cython (`.pyx` compiled to `.so`), pytest, Candle NPU backend.

**Spec:** `docs/superpowers/specs/2026-06-15-npu-autograd-full-cython-stack-design.md`

**Worktree:** `npu-autograd-cython` (already created and built)

---

## File Structure

- **Modify:** `tools/autograd/gen_functions.py` — `_gen_one_node_pyx` function (lines ~2004–2171). Rewrite the `.pyx` node body to emit `def backward(self, grad)` and correct saved-tensor retrieval, mirroring the working `_gen_one_node` Python generator. The `.pyx` version keeps its two distinguishing features: (a) `_ensure_refs()` call at the top of `backward` to populate cached module globals, (b) `_backward_dispatch_keyset(self._raw_keyset)` single-arg form and the `with _grad_context(keyset):` / `cython-safe-formula` gating already present.
- **Modify:** `tools/autograd/gen_variable_type.py` — `_ensure_refs` body emitted for `_variable_type_cy.pyx` (line ~437). Change `from . import functions as _f` to `from . import _functions_cy as _f`.
- **Regenerate (checked-in artifacts):** `src/candle/_generated/_functions_cy.pyx`, `src/candle/_generated/_variable_type_cy.pyx`.
- **Create:** `tests/npu/cython/test_generated_cy_nodes_wired.py` — proves generated `.pyx` nodes are the ones the engine calls, and that backward correctness holds.

---

### Task 1: Write the RED test proving `.pyx` nodes are unwired

**Files:**
- Create: `tests/npu/cython/test_generated_cy_nodes_wired.py`

- [ ] **Step 1: Write the failing test**

Create `tests/npu/cython/test_generated_cy_nodes_wired.py`:

```python
"""Phase 1: prove generated Cython backward nodes are wired into the engine.

Before this phase, _variable_type_cy.pyx imports the PYTHON functions.py as _F,
so the engine only ever sees Python node classes. After the fix, it imports the
compiled _functions_cy module, and a backward node's __module__ is
'candle._generated._functions_cy'.
"""
import numpy as np


def test_generated_backward_node_runs_in_cython_module(npu_device):
    """A simple op (no hand-written NPU fast node) must attach a node whose
    class lives in the compiled _functions_cy module, not the Python functions
    module."""
    import candle as torch

    x = torch.ones(2, 2, device=npu_device, dtype=torch.float32, requires_grad=True)
    # abs has no hand-written NPU fast node, so it uses the generated node.
    y = x.abs()
    assert y.requires_grad
    assert y.grad_fn is not None
    module = type(y.grad_fn).__module__
    assert module == "candle._generated._functions_cy", (
        f"expected generated Cython node, got {module!r} "
        "(engine is still using Python functions.py nodes)"
    )


def test_generated_backward_node_correctness(npu_device):
    """Backward through a generated Cython node must produce correct gradients."""
    import candle as torch

    x = torch.ones(2, 2, device=npu_device, dtype=torch.float32, requires_grad=True)
    y = x.abs()
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.cpu().numpy(), np.ones((2, 2)), rtol=1e-6)


def test_backward_has_backward_method_not_apply(npu_device):
    """The generated node class must define `backward` (what the engine calls),
    not just `apply`."""
    import candle as torch

    x = torch.ones(2, 2, device=npu_device, dtype=torch.float32, requires_grad=True)
    y = x.abs()
    node_cls = type(y.grad_fn)
    assert "backward" in node_cls.__dict__, (
        f"{node_cls.__name__} must define backward() in its own __dict__"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311
PYTHONPATH=$(pwd)/src python -m pytest tests/npu/cython/test_generated_cy_nodes_wired.py -v
```
Expected: FAIL. `test_generated_backward_node_runs_in_cython_module` fails because the node's `__module__` is `candle._generated.functions` (Python), not `candle._generated._functions_cy`. The correctness test may pass or fail depending on the broken `apply` path.

- [ ] **Step 3: Commit the RED test**

```bash
git add tests/npu/cython/test_generated_cy_nodes_wired.py
git commit -m "test(npu): add RED test for generated Cython backward node wiring

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Fix `_gen_one_node_pyx` to emit `def backward` with correct saved-tensor retrieval

**Files:**
- Modify: `tools/autograd/gen_functions.py` (function `_gen_one_node_pyx`, lines ~2004–2171)

The current `_gen_one_node_pyx` emits `def apply(self, grad)` and retrieves saved tensors via `self.saved_tensors[self._saved_{name}_idx]` — both wrong (the engine calls `.backward`, and `saved_tensors` is a method). The Python generator `_gen_one_node` (lines ~1481–1639) is correct: it emits `def backward(self, grad)`, calls `self.saved_tensors()` once into `_saved`, and indexes `_saved`. This task rewrites `_gen_one_node_pyx` to mirror `_gen_one_node` exactly, while preserving the two `.pyx`-specific differences.

- [ ] **Step 1: Read the current `_gen_one_node_pyx` and `_gen_one_node` to confirm the diff**

Read `tools/autograd/gen_functions.py` lines 2004–2171 (`_gen_one_node_pyx`) and lines 1481–1639 (`_gen_one_node`). Confirm:
- `_gen_one_node_pyx` line 2021 emits `class {cls_name}(_Node):` — change to `class {cls_name}(Node):` is NOT needed; `_Node` is imported as `Node as _Node` in the header (line 1683). Leave `(_Node)`.
- Line 2055 emits `def apply(self, grad):` — change to `def backward(self, grad):`.
- Line 2057 emits `keyset = _backward_dispatch_keyset(self._raw_keyset)` — the Python version (line 1533) emits `_backward_dispatch_keyset(self._raw_keyset, self._active_keyset)`. Keep the `.pyx` single-arg form only if that matches the cached `_backward_dispatch_keyset` signature; the safe change is to mirror the Python version's two-arg call.

- [ ] **Step 2: Rewrite `_gen_one_node_pyx`'s `backward` method body**

In `tools/autograd/gen_functions.py`, replace the `# apply method` block (lines ~2054–2170) of `_gen_one_node_pyx` with a body that mirrors `_gen_one_node`. The replacement, keeping `.pyx`-specific `_ensure_refs()` and the `cython-safe-formula` gating, is:

```python
    # backward method (the engine calls node.backward(grad))
    lines.append("\n    def backward(self, grad):")
    lines.append("        _ensure_refs()")
    lines.append("        keyset = _backward_dispatch_keyset(self._raw_keyset, self._active_keyset)")

    # Retrieve saved tensors — mirror the Python generator exactly
    optional_tensor_names = {a.name for a in info.args if a.is_optional_tensor}
    if all_saved:
        lines.append("        _saved = self.saved_tensors()")
    for name in saved_inputs:
        arg = next((a for a in info.args if a.name == name), None)
        if arg and arg.is_tensor_list:
            continue
        local = _safe_local(name)
        local_names.add(local)
        if name in optional_tensor_names:
            lines.append(f"        {local} = _saved[self._saved_{name}_idx] if self._saved_{name}_idx is not None else None")
        else:
            lines.append(f"        {local} = _saved[self._saved_{name}_idx] if self._saved_{name}_idx is not None else None")
    for name in saved_outputs:
        local = _safe_local(name)
        local_names.add(local)
        lines.append(f"        {local} = _saved[self._saved_{name}_idx] if self._saved_{name}_idx is not None else None")

    # Retrieve unsaved tensor args referenced in formulas (unchanged from current pyx gen)
    tensor_args = [a for a in info.args if a.is_tensor or a.is_optional_tensor or a.is_tensor_list]
    tensor_arg_indices = {a.name: i for i, a in enumerate(tensor_args)}
    saved_set = set(saved_outputs)
    for name in saved_inputs:
        arg = next((a for a in info.args if a.name == name), None)
        if arg and arg.is_tensor_list:
            continue
        saved_set.add(name)
    import re as _re
    _ident_pat = _re.compile(r"\b([a-zA-Z_]\w*)\b")
    formula_referenced = set()
    for d in info.derivatives:
        formula_referenced |= set(_ident_pat.findall(d.formula))
    tensor_arg_names = {a.name for a in tensor_args}
    unsaved_referenced = (formula_referenced & tensor_arg_names) - saved_set
    for name in sorted(unsaved_referenced):
        arg = next((a for a in info.args if a.name == name), None)
        local = _safe_local(name)
        local_names.add(local)
        if arg and arg.is_tensor_list:
            lines.append(f"        {local} = list(self.inputs)")
        else:
            idx = tensor_arg_indices[name]
            lines.append(f"        {local} = self.inputs[{idx}]")

    # Retrieve non-tensor args
    for arg in non_tensor_args:
        local = _safe_local(arg.name)
        local_names.add(local)
        lines.append(f"        {local} = self._{arg.name}")

    diff_inputs = info.differentiable_inputs
    grad_vars: dict = {}
    if any('grad_input_mask' in d.formula for d in info.derivatives):
        mask_parts = []
        for arg in diff_inputs:
            local_name = _safe_local(arg.name)
            if arg.is_optional_tensor:
                mask_parts.append(f"({local_name} is not None)")
            else:
                mask_parts.append("True")
        mask_expr = f"[{', '.join(mask_parts)}]"
        if _is_cython_safe_formula(mask_expr, local_names):
            lines.append(f"        grad_input_mask = {mask_expr}")
            local_names.add("grad_input_mask")
    if info.derivatives:
        lines.append("        with _grad_context(keyset):")

    for deriv in info.derivatives:
        formula = transpile(deriv.formula)
        formula = _rewrite_keyword_refs(formula, {arg.name for arg in info.args})
        safe_formula = _is_cython_safe_formula(formula, local_names)
        if len(deriv.var_names) == 1:
            var_name = deriv.var_names[0]
            grad_var = f"grad_{var_name}"
            grad_vars[var_name] = grad_var
            local_names.add(grad_var)
            if safe_formula:
                if _should_guard_not_implemented_formula(formula, var_name, info):
                    needed = _grad_needed_expr(var_name, info)
                    lines.append(f"            {grad_var} = {formula} if {needed} else None")
                else:
                    lines.append(f"            {grad_var} = {formula}")
            else:
                if _should_guard_not_implemented_formula(formula, var_name, info):
                    needed = _grad_needed_expr(var_name, info)
                    lines.append(f"            {grad_var} = _cy_not_implemented(\"{cls_name}: {var_name}\") if {needed} else None")
                else:
                    lines.append(f"            {grad_var} = _cy_not_implemented(\"{cls_name}: {var_name}\")")
        else:
            grad_var_names = [f"grad_{v}" for v in deriv.var_names]
            for v, gv in zip(deriv.var_names, grad_var_names):
                grad_vars[v] = gv
                local_names.add(gv)
            if safe_formula:
                lhs = ", ".join(grad_var_names)
                lines.append(f"            {lhs} = {formula}")
            else:
                for v, gv in zip(deriv.var_names, grad_var_names):
                    lines.append(f"            {gv} = _cy_not_implemented(\"{cls_name}: {v}\")")

    diff_inputs = info.differentiable_inputs
    if len(diff_inputs) == 0:
        lines.append("        return (grad,)")
    elif len(diff_inputs) == 1 and diff_inputs[0].is_tensor_list:
        grad_var = grad_vars.get(diff_inputs[0].name, "None")
        lines.append(f"        return {grad_var}")
    else:
        ret_parts = []
        for arg in diff_inputs:
            if arg.name in grad_vars:
                ret_parts.append(grad_vars[arg.name])
            elif arg.name in info.non_differentiable:
                ret_parts.append("None")
            else:
                ret_parts.append("None")
        lines.append(f"        return ({', '.join(ret_parts)},)")
```

Note: the only intentional differences from the Python `_gen_one_node` are (a) `_ensure_refs()` as the first line, (b) `safe_formula` gating that falls back to `_cy_not_implemented`, and (c) the `grad_input_mask` is only emitted when cython-safe. These match the existing `.pyx` generator's intent.

- [ ] **Step 3: Also fix the `_save` method to populate `_saved_fields`**

The current `.pyx` `_save` block (lines ~2039–2053) does NOT populate `_saved_fields`, but the Python generator does (lines 1525–1528) — and `Node.__getattr__` (in `_autograd_node.pyx`) reads `_saved_fields` to resolve `_saved_<name>` accesses. Without this, formulas referencing saved tensors as `_saved_self` (via `__getattr__`) break. Mirror the Python generator by appending, after the `super().save_for_backward(*tensors)` line in the `.pyx` `_save` block:

```python
        # Populate _saved_fields so Node.__getattr__ resolves _raw_saved_* / _saved_*
        for name in all_saved:
            lines.append(f"        if self._saved_{name}_idx is not None:")
            lines.append(f"            self._saved_fields[{name!r}] = self._saved_tensors_list[self._saved_{name}_idx]")
```

- [ ] **Step 4: Run the generator to regenerate the checked-in `.pyx` files**

Run:
```bash
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311
PYTHONPATH=$(pwd)/src python -m tools.autograd.gen_autograd
```
Expected: prints `_functions_cy.pyx — written (...)` and `_variable_type_cy.pyx — unchanged` (the variable-type file is unchanged in this task).

- [ ] **Step 5: Sanity-check the regenerated node has `def backward`**

Run:
```bash
grep -c "def backward" src/candle/_generated/_functions_cy.pyx
grep -c "def apply" src/candle/_generated/_functions_cy.pyx
```
Expected: `def backward` count is now ~684; `def apply` count is 0.

- [ ] **Step 6: Commit the generator fix + regenerated functions**

```bash
git add tools/autograd/gen_functions.py src/candle/_generated/_functions_cy.pyx
git commit -m "refactor(autograd): generated .pyx nodes emit def backward, not apply

The engine calls node.backward(grad); the generated _functions_cy nodes
previously defined apply() and indexed saved_tensors as if it were a list
(it is a method), so they were never usable. Mirror the working Python
generator body: def backward, single saved_tensors() call into _saved,
and _saved_fields population so Node.__getattr__ resolves saved tensors.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Switch `_variable_type_cy.pyx` `_F` import to `_functions_cy`

**Files:**
- Modify: `tools/autograd/gen_variable_type.py` (line ~437, the `_ensure_refs` body for the `.pyx`)
- Regenerate: `src/candle/_generated/_variable_type_cy.pyx`

- [ ] **Step 1: Change the generator's `.pyx` `_F` import**

In `tools/autograd/gen_variable_type.py`, find the `_ensure_refs` body string emitted for `_variable_type_cy.pyx` (the block containing `from . import functions as _f` around line 437). Change:

```python
        from . import functions as _f
        _F = _f
```
to:
```python
        from . import _functions_cy as _f
        _F = _f
```

Leave the Python `variable_type.py` header (line 154, `from . import functions as _F`) unchanged — the Python module still uses Python nodes.

- [ ] **Step 2: Regenerate `_variable_type_cy.pyx`**

Run:
```bash
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311
PYTHONPATH=$(pwd)/src python -m tools.autograd.gen_autograd
```
Expected: `_variable_type_cy.pyx — written (...)`.

- [ ] **Step 3: Verify the import line changed in the checked-in file**

Run:
```bash
grep "from . import functions as _f\|from . import _functions_cy as _f" src/candle/_generated/_variable_type_cy.pyx
```
Expected: only `from . import _functions_cy as _f` appears.

- [ ] **Step 4: Rebuild Cython extensions**

Run:
```bash
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311
PYTHONPATH=$(pwd)/src python setup.py build_ext --inplace 2>&1 | tail -5
```
Expected: build completes, `copying ... _functions_cy...so` and `_variable_type_cy...so` appear in the tail.

- [ ] **Step 5: Commit the import switch**

```bash
git add tools/autograd/gen_variable_type.py src/candle/_generated/_variable_type_cy.pyx
git commit -m "refactor(autograd): wire _variable_type_cy to compiled _functions_cy nodes

_variable_type_cy.pyx previously imported the Python functions.py as _F,
so the engine only ever saw Python backward nodes. Switch _F to the
compiled _functions_cy module so generated nodes actually run in Cython.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Run the Phase 1 test and confirm GREEN

**Files:**
- Test: `tests/npu/cython/test_generated_cy_nodes_wired.py`

- [ ] **Step 1: Run the Phase 1 test**

Run:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311
PYTHONPATH=$(pwd)/src python -m pytest tests/npu/cython/test_generated_cy_nodes_wired.py -v --tb=short
```
Expected: all 3 tests PASS. `test_generated_backward_node_runs_in_cython_module` confirms `__module__ == "candle._generated._functions_cy"`.

- [ ] **Step 2: If any test fails, debug the generated node**

If a test fails (e.g. `_saved_fields` KeyError, or a formula returns wrong type), inspect the specific generated node:
```bash
grep -A 20 "class AbsBackward0" src/candle/_generated/_functions_cy.pyx
```
Compare against the Python `AbsBackward0` in `src/candle/_generated/functions.py`. Fix the discrepancy in `_gen_one_node_pyx`, regenerate, rebuild, re-test. Do NOT edit the checked-in `.pyx` directly — always fix the generator.

---

### Task 5: Full test-suite validation (correctness gate)

**Files:**
- No changes; validation only.

- [ ] **Step 1: Run the contract codegen roundtrip test**

Run:
```bash
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311
PYTHONPATH=$(pwd)/src python -m pytest tests/contract/test_codegen_roundtrip.py -v --tb=short
```
Expected: PASS. The checked-in `_functions_cy.pyx` and `_variable_type_cy.pyx` must exactly match fresh generator output. If this fails, regenerate and re-commit the artifacts (the generator and checked-in files drifted).

- [ ] **Step 2: Run the full NPU test suite**

Run:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311
PYTHONPATH=$(pwd)/src python -m pytest tests/npu/ -v --tb=short 2>&1 | tail -40
```
Expected: no regressions vs `main`. Record any pre-existing failures (the 310B flip+argsort crash documented in memory is pre-existing RED and not a regression). If a previously-green test now fails, the generated Cython node for that op has a bug — fix in the generator, regenerate, rebuild, re-test.

- [ ] **Step 3: Run CPU + contract tests**

Run:
```bash
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311
PYTHONPATH=$(pwd)/src python -m pytest tests/cpu/ tests/contract/ -v --tb=short 2>&1 | tail -40
```
Expected: PASS. The generated nodes are device-agnostic formulas; CPU must still work.

- [ ] **Step 4: Run pylint**

Run:
```bash
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311
PYTHONPATH=$(pwd)/src python -m pylint src/candle/ --rcfile=.github/pylint.conf 2>&1 | tail -15
```
Expected: PASS (pylint is the merge gate).

---

### Task 6: Benchmark — confirm no regression, capture Phase 1 baseline

**Files:**
- Create: `benchmarks/_profile_qwen2_backward_phase1.py`

- [ ] **Step 1: Create the Phase 1 backward benchmark**

Create `benchmarks/_profile_qwen2_backward_phase1.py` (adapted from the in-worktree profile script):

```python
"""Phase 1 baseline: Qwen2 backward median after wiring generated Cython nodes.

Run inside torchnpu311 with PYTHONPATH=<worktree>/src.
"""
import sys, os, time
sys.path.insert(0, "/home/jenkins/lvyufeng/candle")
from compat.transformers.conftest import apply_all_patches
apply_all_patches()
import candle as torch
sys.modules["torch"] = torch
from transformers import Qwen2Config, Qwen2ForCausalLM

CONFIG = dict(vocab_size=128, hidden_size=64, intermediate_size=128,
              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
              max_position_embeddings=64, attention_dropout=0.0, use_cache=False)
device = torch.Device("npu:0"); dtype = torch.float16
config = Qwen2Config(**CONFIG); torch.manual_seed(20260608)
model = Qwen2ForCausalLM(config).to(device).to(dtype); model.train()
input_ids = torch.arange(0, 8, device=device, dtype=torch.int64).reshape(1, 8) % 128
labels = (input_ids + 1) % 128
attention_mask = torch.ones((1, 8), device=device, dtype=torch.int64)

for _ in range(5):
    for p in model.parameters():
        if getattr(p, "grad", None) is not None: p.grad = None
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    out.loss.backward()
torch.npu.synchronize()

def fresh_forward():
    for p in model.parameters():
        if getattr(p, "grad", None) is not None: p.grad = None
    return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)

times = []
for _ in range(20):
    out = fresh_forward()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out.loss.backward()
    torch.npu.synchronize()
    times.append(time.perf_counter() - t0)
times.sort()
print(f"PHASE1 BACKWARD: min={min(times)*1000:.2f}ms median={times[10]*1000:.2f}ms max={max(times)*1000:.2f}ms")
print(f"BASELINE (pre-phase1) was: median 36.4ms")
```

- [ ] **Step 2: Run the benchmark**

Run:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate torchnpu311
cd /home/jenkins/lvyufeng/candle
PYTHONPATH=/home/jenkins/lvyufeng/candle/.claude/worktrees/npu-autograd-cython/src \
  python benchmarks/_profile_qwen2_backward_phase1.py
```
Expected: prints `PHASE1 BACKWARD: ... median=Xms`. Record X. The acceptance bar for Phase 1 is **no regression** (median ≤ ~36ms). A small improvement is possible (compiled nodes avoid some Python overhead) but not required; performance comes in later phases.

- [ ] **Step 3: Commit the benchmark**

```bash
git add benchmarks/_profile_qwen2_backward_phase1.py
git commit -m "bench(npu): add Phase 1 Qwen2 backward benchmark

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Open the Phase 1 PR

**Files:**
- No code changes; PR creation.

- [ ] **Step 1: Rebase onto upstream main**

Run:
```bash
git fetch upstream main
git rebase upstream/main
```
Resolve any conflicts (likely none — generated files and generator are isolated). If `_functions_cy.pyx` conflicts, regenerate it after resolving the generator conflict.

- [ ] **Step 2: Push to origin**

Run:
```bash
git push -u origin npu-autograd-cython
```

- [ ] **Step 3: Create the PR**

Run:
```bash
gh pr create --repo candle-org/candle \
  --head lvyufeng:npu-autograd-cython --base main \
  --title "refactor(autograd): wire generated Cython backward nodes into the engine (Phase 1)" \
  --body "## Phase 1: generator fix + _F switch

Fixes the generated \`_functions_cy.pyx\` backward nodes so they actually run:
- Generated nodes emitted \`def apply\` but the engine calls \`node.backward\`; emit \`def backward\`.
- \`saved_tensors\` was indexed as a list but is a method; call it once into \`_saved\`.
- \`_save\` did not populate \`_saved_fields\`, so \`Node.__getattr__\` could not resolve saved tensors.
- \`_variable_type_cy.pyx\` imported the Python \`functions.py\` as \`_F\`; switch to compiled \`_functions_cy\`.

After this, every non-skipped op's backward runs as compiled Cython instead of Python.

**Baseline:** backward median 36.4ms (no regression expected; perf gains land in later phases).
**Validation:** full \`tests/npu/\`, \`tests/cpu/\`, \`tests/contract/\` green; pylint green; codegen roundtrip green.

Spec: \`docs/superpowers/specs/2026-06-15-npu-autograd-full-cython-stack-design.md\`

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 4: Once pylint passes on CI, merge (squash), then clean up**

Per project rules, pylint is the merge gate. After merge:
```bash
# From the main repo root (not the worktree)
git worktree remove .claude/worktrees/npu-autograd-cython
git branch -d npu-autograd-cython
git push origin --delete npu-autograd-cython
git checkout main && git pull upstream main && git push origin main
```
Then continue to Phase 2 (separate plan, separate worktree).

---

## Self-Review Notes

**Spec coverage for Phase 1:**
- "Generator fix" (Component 1) → Tasks 2, 3.
- "`_F` switch" (Component 1) → Task 3.
- "RED tests before production code" (spec testing strategy) → Task 1.
- "Full NPU + CPU + contract suites pass" (acceptance) → Task 5.
- "Backward median stays ≤ baseline" (Phase 1 acceptance) → Task 6.
- Phases 2–4 (routing, fast nodes, kernels) are deliberately out of scope — each gets its own plan after Phase 1 lands.

**Phases 2–4 are explicitly deferred** because they build on a working generator output and are independently shippable. This plan produces working, tested software (every generated backward formula runs in Cython) on its own.
