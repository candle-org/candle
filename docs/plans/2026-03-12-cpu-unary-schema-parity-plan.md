# CPU Unary Schema Parity Batch 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add contract parity coverage and schema-level error message alignment for unary elementwise CPU ops.

**Architecture:** Create one parameterized contract test file for unary ops using the parity harness, then add schema error overrides or validation tweaks only where PyTorch error messages differ.

**Tech Stack:** Python, pytest, candle dispatch/schema system.

---

### Task 1: Add unary parity contract tests

**Files:**
- Create: `tests/contract/test_training_core_unary_parity.py`

**Step 1: Write the failing test**

Create `tests/contract/test_training_core_unary_parity.py` with a parameterized list of unary ops and a helper that calls `run_training_core_parity_case`. For each op, compare forward values/dtypes, and for known error cases set `expect_error=True` with `check_error_message=True`.

Example skeleton:
```python
import candle as torch
import pytest
from .helpers import run_training_core_parity_case

UNARY_OPS = [
    "abs", "neg", "exp", "log", "sqrt", "sin", "cos", "tan", "tanh", "sigmoid",
    "floor", "ceil", "round", "trunc", "frac", "log2", "log10", "exp2",
    "rsqrt", "reciprocal", "sign", "signbit", "isnan", "isinf", "isfinite",
    "sinh", "cosh", "asinh", "acosh", "atanh", "erf", "erfc", "softplus",
    "relu6", "gelu", "silu", "mish", "square",
]

@pytest.mark.parametrize("op_name", UNARY_OPS)
def test_unary_forward_parity(op_name):
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name=op_name,
        candle_fn=lambda x: getattr(torch, op_name)(x),
        torch_fn=lambda x: getattr(real_torch, op_name)(x),
        candle_inputs=lambda: (torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32),),
        torch_inputs=lambda: (real_torch.tensor([1.0, -2.0, 3.0], dtype=real_torch.float32),),
    )

    assert result["dtype_match"] is True
    assert result["shape_match"] is True
    assert result["value_match"] is True
```

Add targeted error parity cases (e.g., `log`/`sqrt` with integer inputs if torch errors, or domain mismatch if applicable) only once you observe mismatches.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_unary_parity.py -v`
Expected: FAIL for missing error overrides or behavior mismatches.

---

### Task 2: Align schema error messages for mismatched unary ops

**Files:**
- Modify: `src/candle/_dispatch/schemas.py`
- Modify: `src/candle/_dispatch/schema.py` (only if validation logic is required)

**Step 1: Write the failing test**

Use the failing cases from Task 1.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_unary_parity.py -v`
Expected: FAIL with mismatched error messages.

**Step 3: Write minimal implementation**

Add `register_error_overrides` for unary ops whose call‑site error messages differ. If a type/shape validation tweak is needed, add a targeted validator in `schema.py` and route it when `op_short_name` matches the op.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_unary_parity.py -v`
Expected: PASS.

**Step 5: Commit**

`git add tests/contract/test_training_core_unary_parity.py src/candle/_dispatch/schemas.py src/candle/_dispatch/schema.py`
`git commit -m "test: add unary op contract parity"`

---

### Task 3: Contract gate

**Step 1: Run contract tests**

Run: `PYTHONPATH=src pytest tests/contract/ -v --tb=short`
Expected: PASS.

**Step 2: Commit gate evidence if needed**

No code changes expected; if new adjustments were made, commit them separately.

