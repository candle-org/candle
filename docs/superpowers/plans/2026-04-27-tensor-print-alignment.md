# Tensor Print Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite Candle’s tensor printing stack so `src/candle/_tensor_str.py` becomes the primary torch-style implementation and dense/main-path repr behavior closely matches local torch.

**Architecture:** Port the main structure of `torch/_tensor_str.py` into `src/candle/_tensor_str.py`, adapt torch runtime hooks to Candle equivalents, and shrink `src/candle/_printing.py` into a compatibility wrapper. Keep unsupported special branches on explicit local fallbacks so the batch stays bounded to printing.

**Tech Stack:** Python, Candle tensor API, pytest, local torch reference implementation

---

## File map

- **Modify:** `src/candle/_tensor_str.py`
  - Becomes the canonical owner of tensor print options and formatting.
  - Hosts `PRINT_OPTS`, `set_printoptions`, `get_printoptions`, `printoptions`, `_Formatter`, `_scalar_str`, `_vector_str`, `_tensor_str_with_formatter`, `_tensor_str`, and `_str`.

- **Modify:** `src/candle/_printing.py`
  - Reduced to a compatibility wrapper that delegates into `_tensor_str.py`.
  - Must not own independent print-option state after the rewrite.

- **Modify:** `tests/cpu/test_tensor_print.py`
  - Expanded to compare Candle output directly against local torch output for dense/main-path behavior and printoption state.

- **Possibly modify:** `src/candle/__init__.py`
  - Only if export wiring needs adjustment after the `_tensor_str.py` rewrite.

---

### Task 1: Port print-option ownership into `_tensor_str.py`

**Files:**
- Modify: `src/candle/_tensor_str.py`
- Modify: `src/candle/_printing.py`
- Test: `tests/cpu/test_tensor_print.py`

- [ ] **Step 1: Write the failing printoptions ownership tests**

```python
import candle as torch
from candle import _tensor_str


def test_printoptions_state_lives_in_tensor_str_module():
    prev = torch.get_printoptions()
    try:
        torch.set_printoptions(precision=2, linewidth=70)
        opts = torch.get_printoptions()
        assert opts["precision"] == 2
        assert opts["linewidth"] == 70
        assert _tensor_str.get_printoptions()["precision"] == 2
        assert _tensor_str.get_printoptions()["linewidth"] == 70
    finally:
        torch.set_printoptions(**prev)


def test_printoptions_context_restores_state():
    prev = torch.get_printoptions()
    with torch.printoptions(precision=2, sci_mode=True):
        inside = torch.get_printoptions()
        assert inside["precision"] == 2
        assert inside["sci_mode"] is True
    assert torch.get_printoptions() == prev
```

- [ ] **Step 2: Run the targeted tests to verify the gap**

Run:

```bash
conda run -n candle311 python -m pytest tests/cpu/test_tensor_print.py -q --tb=short -k "printoptions_state_lives_in_tensor_str_module or printoptions_context_restores_state"
```

Expected: at least one FAIL because `_tensor_str.py` is still only a thin wrapper and does not own the torch-style printing stack.

- [ ] **Step 3: Replace `_tensor_str.py` with torch-style print-option scaffolding**

Use `torch/_tensor_str.py` as the source structure. The resulting Candle file should define the option state in `_tensor_str.py` itself and keep `_str()` in the same module.

```python
# src/candle/_tensor_str.py
# mypy: allow-untyped-defs
import contextlib
import dataclasses
import math
from typing import Any

import candle as torch
from candle import inf


@dataclasses.dataclass
class __PrinterOptions:
    precision: int = 4
    threshold: float = 1000
    edgeitems: int = 3
    linewidth: int = 80
    sci_mode: bool | None = None


PRINT_OPTS = __PrinterOptions()


def set_printoptions(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    profile=None,
    sci_mode=None,
):
    if profile is not None:
        if profile == "default":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
        elif profile == "short":
            PRINT_OPTS.precision = 2
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 2
            PRINT_OPTS.linewidth = 80
        elif profile == "full":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = inf
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
    if precision is not None:
        PRINT_OPTS.precision = precision
    if threshold is not None:
        PRINT_OPTS.threshold = threshold
    if edgeitems is not None:
        PRINT_OPTS.edgeitems = edgeitems
    if linewidth is not None:
        PRINT_OPTS.linewidth = linewidth
    PRINT_OPTS.sci_mode = sci_mode


def get_printoptions() -> dict[str, Any]:
    return dataclasses.asdict(PRINT_OPTS)


@contextlib.contextmanager

def printoptions(**kwargs):
    old_kwargs = get_printoptions()
    set_printoptions(**kwargs)
    try:
        yield
    finally:
        set_printoptions(**old_kwargs)
```

- [ ] **Step 4: Reduce `_printing.py` to a pure compatibility wrapper**

`_printing.py` should stop owning separate state and instead delegate to `_tensor_str.py`.

```python
# src/candle/_printing.py
from ._tensor_str import get_printoptions, printoptions, set_printoptions, _str


def format_tensor(tensor, tensor_contents=None):
    return _str(tensor, tensor_contents=tensor_contents)
```

- [ ] **Step 5: Run the targeted tests again**

Run:

```bash
conda run -n candle311 python -m pytest tests/cpu/test_tensor_print.py -q --tb=short -k "printoptions_state_lives_in_tensor_str_module or printoptions_context_restores_state"
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/candle/_tensor_str.py src/candle/_printing.py tests/cpu/test_tensor_print.py
git commit -m "refactor(print): move print option ownership into _tensor_str"
```

---

### Task 2: Port the dense/main-path formatter from torch

**Files:**
- Modify: `src/candle/_tensor_str.py`
- Test: `tests/cpu/test_tensor_print.py`

- [ ] **Step 1: Write failing dense-format comparison tests**

```python
import candle as candle_torch
import torch as ref_torch


def test_dense_vector_repr_matches_local_torch():
    c = candle_torch.tensor([1.23456, 2.0, -3.5], dtype=candle_torch.float32)
    r = ref_torch.tensor([1.23456, 2.0, -3.5], dtype=ref_torch.float32)
    assert repr(c) == repr(r)


def test_dense_matrix_repr_matches_local_torch():
    c = candle_torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=candle_torch.float32)
    r = ref_torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ref_torch.float32)
    assert repr(c) == repr(r)


def test_complex_repr_matches_local_torch():
    c = candle_torch.tensor([1 + 2j, -3 + 0.5j], dtype=candle_torch.complex64)
    r = ref_torch.tensor([1 + 2j, -3 + 0.5j], dtype=ref_torch.complex64)
    assert repr(c) == repr(r)
```

- [ ] **Step 2: Run the dense-format tests to capture failure**

Run:

```bash
conda run -n candle311 python -m pytest tests/cpu/test_tensor_print.py -q --tb=short -k "dense_vector_repr_matches_local_torch or dense_matrix_repr_matches_local_torch or complex_repr_matches_local_torch"
```

Expected: FAIL because Candle still uses the simplified NumPy-based formatting path.

- [ ] **Step 3: Port `_Formatter`, `_scalar_str`, `_vector_str`, and `_tensor_str_with_formatter`**

Port the main formatting pipeline from local `torch/_tensor_str.py`, adapting only the runtime hooks from torch to Candle.

Required structure to add in `src/candle/_tensor_str.py`:

```python
def tensor_totype(t):
    dtype = torch.double
    return t.to(dtype=dtype)


class _Formatter:
    def __init__(self, tensor):
        self.floating_dtype = tensor.dtype.is_floating_point
        self.int_mode = True
        self.sci_mode = False
        self.max_width = 1

        with torch.no_grad():
            tensor_view = tensor.reshape(-1)

        if not self.floating_dtype:
            for value in tensor_view:
                value_str = f"{value}"
                self.max_width = max(self.max_width, len(value_str))
        else:
            nonzero_finite_vals = torch.masked_select(
                tensor_view, torch.isfinite(tensor_view) & tensor_view.ne(0)
            )
            if nonzero_finite_vals.numel() == 0:
                return
            nonzero_finite_abs = tensor_totype(nonzero_finite_vals.abs())
            nonzero_finite_min = tensor_totype(nonzero_finite_abs.min())
            nonzero_finite_max = tensor_totype(nonzero_finite_abs.max())
            for value in nonzero_finite_vals:
                if value != torch.ceil(value):
                    self.int_mode = False
                    break
            self.sci_mode = (
                nonzero_finite_max / nonzero_finite_min > 1000.0
                or nonzero_finite_max > 1.0e8
                or nonzero_finite_min < 1.0e-4
                if PRINT_OPTS.sci_mode is None
                else PRINT_OPTS.sci_mode
            )
```

Also port the paired formatting helpers from torch:

```python
def _scalar_str(self, formatter1, formatter2=None):
    ...


def _vector_str(self, indent, summarize, formatter1, formatter2=None):
    ...


def _tensor_str_with_formatter(self, indent, summarize, formatter1, formatter2=None):
    ...
```

Adaptation rules:
- keep the control flow and formatting policy from torch,
- replace torch runtime helpers only where Candle needs a valid equivalent,
- do not add unrelated runtime features.

- [ ] **Step 4: Implement `_tensor_str` and `_str` using the ported formatter path**

Replace the wrapper-only `_str` with torch-style orchestration.

```python
def _tensor_str(self, indent):
    if self.numel() == 0:
        return "[]"
    summarize = self.numel() > PRINT_OPTS.threshold
    formatter = _Formatter(get_summarized_data(self) if summarize else self)
    return _tensor_str_with_formatter(self, indent, summarize, formatter)


def _str(self, *, tensor_contents=None):
    if tensor_contents is None:
        tensor_contents = _tensor_str(self, 0)
    suffixes = []
    if self.dtype != torch.get_default_dtype() or self.device.type == "meta":
        suffixes.append(f"dtype={self.dtype}")
    if self.device.type != torch.get_default_device().type:
        suffixes.append(f"device='{self.device}'" if self.device.type == "npu" else f"device='{self.device.type}'")
    if self.requires_grad:
        suffixes.append("requires_grad=True")
    if self.grad_fn is not None:
        suffixes.append(f"grad_fn=<{type(self.grad_fn).__name__}>")
    return f"tensor({tensor_contents}{', ' + ', '.join(suffixes) if suffixes else ''})"
```

- [ ] **Step 5: Re-run the dense-format tests**

Run:

```bash
conda run -n candle311 python -m pytest tests/cpu/test_tensor_print.py -q --tb=short -k "dense_vector_repr_matches_local_torch or dense_matrix_repr_matches_local_torch or complex_repr_matches_local_torch"
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/candle/_tensor_str.py tests/cpu/test_tensor_print.py
git commit -m "feat(print): port torch dense tensor formatting pipeline"
```

---

### Task 3: Add summarization, linewidth, and sci_mode parity tests

**Files:**
- Modify: `tests/cpu/test_tensor_print.py`
- Modify: `src/candle/_tensor_str.py`

- [ ] **Step 1: Add failing printoption behavior tests**

```python
import candle as candle_torch
import torch as ref_torch


def test_threshold_summarization_matches_local_torch():
    prev_c = candle_torch.get_printoptions()
    prev_r = ref_torch.get_printoptions()
    try:
        candle_torch.set_printoptions(threshold=5)
        ref_torch.set_printoptions(threshold=5)
        c = candle_torch.arange(10)
        r = ref_torch.arange(10)
        assert repr(c) == repr(r)
    finally:
        candle_torch.set_printoptions(**prev_c)
        ref_torch.set_printoptions(**prev_r)


def test_linewidth_wrapping_matches_local_torch():
    prev_c = candle_torch.get_printoptions()
    prev_r = ref_torch.get_printoptions()
    try:
        candle_torch.set_printoptions(linewidth=30)
        ref_torch.set_printoptions(linewidth=30)
        c = candle_torch.arange(16).reshape(4, 4)
        r = ref_torch.arange(16).reshape(4, 4)
        assert repr(c) == repr(r)
    finally:
        candle_torch.set_printoptions(**prev_c)
        ref_torch.set_printoptions(**prev_r)


def test_scientific_mode_matches_local_torch():
    prev_c = candle_torch.get_printoptions()
    prev_r = ref_torch.get_printoptions()
    try:
        candle_torch.set_printoptions(sci_mode=True, precision=2)
        ref_torch.set_printoptions(sci_mode=True, precision=2)
        c = candle_torch.tensor([1.0e-5, 2.0e6])
        r = ref_torch.tensor([1.0e-5, 2.0e6])
        assert repr(c) == repr(r)
    finally:
        candle_torch.set_printoptions(**prev_c)
        ref_torch.set_printoptions(**prev_r)
```

- [ ] **Step 2: Run the targeted printoption behavior tests**

Run:

```bash
conda run -n candle311 python -m pytest tests/cpu/test_tensor_print.py -q --tb=short -k "threshold_summarization_matches_local_torch or linewidth_wrapping_matches_local_torch or scientific_mode_matches_local_torch"
```

Expected: FAIL until summarization and wrapping helpers fully match torch behavior.

- [ ] **Step 3: Port the remaining dense helpers from `torch/_tensor_str.py`**

Port the summarization helpers that the formatter pipeline depends on.

Add the torch-style helpers to `src/candle/_tensor_str.py`:

```python
def get_summarized_data(self):
    ...


def _tensor_str(self, indent):
    ...
```

Make sure the implementation covers:
- summarization through `threshold` and `edgeitems`,
- recursive formatting for N-D tensors,
- line wrapping based on `PRINT_OPTS.linewidth`,
- `sci_mode=None` auto-selection behavior.

Keep the implementation as close to torch as possible; do not invent new formatting policy.

- [ ] **Step 4: Re-run the targeted printoption behavior tests**

Run:

```bash
conda run -n candle311 python -m pytest tests/cpu/test_tensor_print.py -q --tb=short -k "threshold_summarization_matches_local_torch or linewidth_wrapping_matches_local_torch or scientific_mode_matches_local_torch"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/candle/_tensor_str.py tests/cpu/test_tensor_print.py
git commit -m "feat(print): align summarization and scientific formatting"
```

---

### Task 4: Handle suffixes and controlled fallback branches

**Files:**
- Modify: `src/candle/_tensor_str.py`
- Modify: `tests/cpu/test_tensor_print.py`

- [ ] **Step 1: Add suffix and fallback tests**

```python
import candle as candle_torch
import torch as ref_torch


def test_requires_grad_suffix_matches_local_torch():
    c = candle_torch.tensor([1.0], requires_grad=True)
    r = ref_torch.tensor([1.0], requires_grad=True)
    assert repr(c) == repr(r)


def test_meta_tensor_repr_matches_local_torch():
    c = candle_torch.empty((2, 2), device="meta")
    r = ref_torch.empty((2, 2), device="meta")
    assert repr(c) == repr(r)
```

If there are special tensor categories Candle cannot fully print like torch yet, add stability tests instead of exact-equality tests:

```python
def test_special_tensor_print_path_does_not_crash():
    t = make_special_case_tensor()
    rep = repr(t)
    assert isinstance(rep, str)
    assert rep
```

- [ ] **Step 2: Run the suffix/fallback tests**

Run:

```bash
conda run -n candle311 python -m pytest tests/cpu/test_tensor_print.py -q --tb=short -k "requires_grad_suffix_matches_local_torch or meta_tensor_repr_matches_local_torch or special_tensor_print_path_does_not_crash"
```

Expected: FAIL for any remaining suffix or special-branch mismatch.

- [ ] **Step 3: Implement suffix parity and local fallbacks**

In `src/candle/_tensor_str.py`, finish the `_str` suffix logic and add explicit fallback branches only where Candle lacks the runtime needed to match torch exactly.

The required output policy is:

```python
if self.dtype != torch.get_default_dtype() or self.device.type == "meta":
    suffixes.append(f"dtype={self.dtype}")
if self.device.type != torch.get_default_device().type:
    ...
if self.requires_grad:
    suffixes.append("requires_grad=True")
if self.grad_fn is not None:
    suffixes.append(f"grad_fn=<{type(self.grad_fn).__name__}>")
```

If a torch branch cannot be modeled exactly in Candle today, keep the fallback local to that branch and document it with one short inline comment stating the missing runtime capability.

- [ ] **Step 4: Run the full print test file**

Run:

```bash
conda run -n candle311 python -m pytest tests/cpu/test_tensor_print.py -q --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/candle/_tensor_str.py tests/cpu/test_tensor_print.py
git commit -m "feat(print): align tensor repr suffixes and fallback branches"
```

---

### Task 5: Run printing-safe regression suites

**Files:**
- Test: `tests/cpu/test_tensor_print.py`
- Test: `tests/common/test_c_core.py`
- Test: `tests/cpu/test_tensor_api_contract.py`
- Test: `tests/contract/test_storage_contract.py`

- [ ] **Step 1: Run the print-specific suite**

Run:

```bash
conda run -n candle311 python -m pytest tests/cpu/test_tensor_print.py -q --tb=short
```

Expected: PASS.

- [ ] **Step 2: Run the common core regression suite**

Run:

```bash
conda run -n candle311 python -m pytest tests/common/test_c_core.py -q --tb=short
```

Expected: PASS.

- [ ] **Step 3: Run the tensor API contract suite**

Run:

```bash
conda run -n candle311 python -m pytest tests/cpu/test_tensor_api_contract.py -q --tb=short
```

Expected: PASS.

- [ ] **Step 4: Run the storage contract suite**

Run:

```bash
conda run -n candle311 python -m pytest tests/contract/test_storage_contract.py -q --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/candle/_tensor_str.py src/candle/_printing.py tests/cpu/test_tensor_print.py
git commit -m "test(print): verify tensor repr alignment regression suites"
```

---

## Self-review

- **Spec coverage:**
  - Primary `_tensor_str.py` ownership: covered by Task 1 and Task 2
  - `_printing.py` compatibility reduction: covered by Task 1
  - Dense/main-path fidelity: covered by Task 2 and Task 3
  - Controlled fallback branches: covered by Task 4
  - Focused print regressions and broader safety suites: covered by Task 5

- **Placeholder scan:**
  - No `TODO`/`TBD` placeholders in steps
  - Each code-changing step includes concrete code to add or port
  - Each test step includes exact commands and expected outcomes

- **Type consistency:**
  - Plan consistently uses `_tensor_str.py` as the primary owner and `_printing.py` as the wrapper
  - Test names, helper names, and file paths match across tasks

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-27-tensor-print-alignment.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?