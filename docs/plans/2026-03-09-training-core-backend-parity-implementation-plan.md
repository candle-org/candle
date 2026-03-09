# Training Core Backend Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a PyTorch-oracle parity harness and close the first-wave backend consistency gaps for training-core operators, starting with binary arithmetic, reshape/view, and reduction semantics.

**Architecture:** Keep Candle's existing schema-first dispatch architecture intact. Add parity contracts in `tests/contract/` that compare Candle directly against real PyTorch, then patch the smallest shared semantic layer possible before touching backend-specific kernels. Use a minimal training smoke gate only as an integration check, not as the primary discovery tool.

**Tech Stack:** Python, pytest, Candle dispatch/autograd/functionalize stack, numpy-backed CPU kernels, NPU/MPS/CUDA backend kernels, real PyTorch as oracle

---

### Task 1: Build the shared training-core parity harness

**Files:**
- Create: `tests/contract/test_training_core_parity_harness.py`
- Modify: `tests/contract/helpers.py`
- Reference: `tests/contract/test_harness.py`
- Reference: `tests/cpu/test_ops_cpu.py`

**Step 1: Write the failing harness test**

Add a focused contract test that proves the harness can:

- run Candle and real PyTorch on the same scalar/binary op case,
- compare output dtype and values,
- compare exception type when a case is invalid.

Use a tiny helper shape first, for example `add(int64, float32)` and `mean(int64)`.

Suggested skeleton:

```python
import candle as torch
import torch as real_torch

from .helpers import run_training_core_parity_case


def test_parity_harness_compares_forward_dtype_and_value():
    result = run_training_core_parity_case(
        op_name="add",
        candle_fn=lambda a, b: torch.add(a, b),
        torch_fn=lambda a, b: real_torch.add(a, b),
        candle_inputs=lambda: (
            torch.tensor([1, 2, 3], dtype=torch.int64),
            torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([1, 2, 3], dtype=real_torch.int64),
            real_torch.tensor([0.5, 0.5, 0.5], dtype=real_torch.float32),
        ),
    )
    assert result["dtype_match"] is True
    assert result["value_match"] is True
```

**Step 2: Run the new test to verify it fails**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_parity_harness.py -v --tb=short`
Expected: FAIL because `run_training_core_parity_case` does not exist yet.

**Step 3: Write the minimal shared helper implementation**

In `tests/contract/helpers.py`, add a reusable helper that:

- constructs Candle and real-PyTorch inputs separately,
- executes both functions,
- returns a structured result with dtype/value/shape match booleans,
- optionally compares exceptions,
- uses `numpy.testing.assert_allclose` with configurable tolerances.

Do not add backward or alias coverage yet. Keep Task 1 limited to forward + error scaffolding.

**Step 4: Re-run the harness test**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_parity_harness.py -v --tb=short`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/contract/test_training_core_parity_harness.py tests/contract/helpers.py
git commit -m "test(contract): add training core parity harness scaffold"
```

---

### Task 2: Extend the parity harness for backward and alias-aware cases

**Files:**
- Modify: `tests/contract/helpers.py`
- Modify: `tests/contract/test_training_core_parity_harness.py`
- Reference: `tests/contract/test_inplace_view_rules.py`
- Reference: `tests/contract/test_autograd_contract.py`
- Reference: `src/candle/_tensor.py`

**Step 1: Write failing tests for backward and view/inplace support**

Add two harness-level tests:

- one that compares gradients for a simple differentiable case like `sum(relu(x * y))`,
- one that compares alias/version-sensitive behavior for a view case, such as `base.view(...); view.add_(...)` raising or mutating like PyTorch.

Suggested skeleton:

```python
def test_parity_harness_compares_gradients():
    ...


def test_parity_harness_can_check_view_inplace_behavior():
    ...
```

**Step 2: Run the new tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_parity_harness.py -k "gradient or view" -v --tb=short`
Expected: FAIL because the helper does not yet compare grads or alias outcomes.

**Step 3: Implement minimal helper support**

Extend `tests/contract/helpers.py` with optional flags such as:

- `check_backward=True`
- `check_alias=True`
- `compare_error_substring=True`

For backward cases:

- call `.backward()` on both Candle and real PyTorch outputs,
- collect leaf grads,
- compare grad dtype, shape, and value.

For alias/view cases:

- expose whether output is a view,
- compare mutation side effects on the base tensor,
- compare exception class and key message fragment when inplace-on-view is rejected.

**Step 4: Re-run the targeted harness tests**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_parity_harness.py -v --tb=short`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/contract/helpers.py tests/contract/test_training_core_parity_harness.py
git commit -m "test(contract): extend training core parity harness for grads and alias checks"
```

---

### Task 3: Add failing binary arithmetic parity contracts

**Files:**
- Create: `tests/contract/test_training_core_binary_parity.py`
- Reference: `tests/cpu/test_ops_cpu.py`
- Reference: `src/candle/_functional.py`
- Reference: `src/candle/_backends/cpu/ops.py`
- Reference: `src/candle/_backends/npu/ops.py`
- Reference: `src/candle/_backends/mps/ops.py`

**Step 1: Write failing parity contracts for binary arithmetic**

Start with the highest-risk cases already hinted at by existing CPU parity tests:

- `add(int64, float32)` dtype promotion
- `mul(int64, float32)` dtype promotion
- `div(int64, float32)` result dtype and values
- `true_divide(int64, int32)` result dtype and values
- `add(bool, int64)` promotion
- broadcasted `add` with shape mismatch success and failure cases
- inplace `add_` scalar/tensor update semantics

Each contract should use the shared harness and run against Candle backends that are available in the environment.

**Step 2: Run the binary parity tests to verify current failures**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_binary_parity.py -v --tb=short`
Expected: FAIL on at least one backend, likely from dtype promotion or backend-specific drift.

**Step 3: Fix shared semantic issues first**

Check shared wrappers before patching per-backend kernels:

- `src/candle/_functional.py`
- any common dtype utility in `src/candle/_dtype.py`

If the failure is backend-specific, patch the smallest affected backend implementation:

- CPU: `src/candle/_backends/cpu/ops.py`
- NPU: `src/candle/_backends/npu/ops.py`
- MPS: `src/candle/_backends/mps/ops.py`

Implementation constraints:

- match real PyTorch dtype promotion,
- preserve no-CPU-fallback rule for device backends,
- do not broaden scope to unrelated binary ops in the same commit.

**Step 4: Re-run the binary parity tests**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_binary_parity.py -v --tb=short`
Expected: PASS.

**Step 5: Run adjacent regressions**

Run: `PYTHONPATH=src pytest tests/cpu/test_ops_cpu.py -k "dtype_promotion or true_divide or add_bool" -v --tb=short`
Expected: PASS.

**Step 6: Commit**

```bash
git add tests/contract/test_training_core_binary_parity.py src/candle/_functional.py src/candle/_dtype.py src/candle/_backends/cpu/ops.py src/candle/_backends/npu/ops.py src/candle/_backends/mps/ops.py
git commit -m "fix(parity): align training core binary arithmetic with torch"
```

---

### Task 4: Add failing reshape/view parity contracts

**Files:**
- Create: `tests/contract/test_training_core_view_parity.py`
- Reference: `tests/contract/test_functionalize_view_writeback.py`
- Reference: `tests/contract/test_inplace_view_rules.py`
- Reference: `src/candle/_tensor.py`
- Reference: `src/candle/_dispatch/dispatcher.py`
- Reference: `src/candle/_dispatch/functionalize.py`
- Reference: `src/candle/_backends/autograd.py`

**Step 1: Write failing parity contracts for reshape/view semantics**

Cover the first-wave training-relevant cases:

- `reshape` vs `view` on contiguous input
- `view` on non-contiguous input when PyTorch rejects it
- inplace mutation through a view updates the base when allowed
- inplace on a restricted leaf/view path raises the same exception class as PyTorch
- backward through `reshape`, `view`, `transpose`, `squeeze`, and `unsqueeze`

Use the parity harness for both forward and backward checks. For error cases, compare exception class and a stable substring, not the full line unless already contract-locked elsewhere.

**Step 2: Run the view parity tests to verify current failures**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_view_parity.py -v --tb=short`
Expected: FAIL on one or more alias/version/backward cases.

**Step 3: Implement the smallest semantic fixes**

Expected code touch points:

- `src/candle/_tensor.py` for `_check_inplace`, view metadata, version updates
- `src/candle/_dispatch/dispatcher.py` for mutating-arg checks and version bumping
- `src/candle/_dispatch/functionalize.py` for writeback behavior
- `src/candle/_backends/autograd.py` for view backward wrappers

Keep fixes tightly scoped to the failing parity cases.

**Step 4: Re-run the view parity tests**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_view_parity.py -v --tb=short`
Expected: PASS.

**Step 5: Run adjacent regressions**

Run: `PYTHONPATH=src pytest tests/contract/test_inplace_view_rules.py tests/contract/test_functionalize_view_writeback.py tests/cpu/test_tensor_view.py -v --tb=short`
Expected: PASS.

**Step 6: Commit**

```bash
git add tests/contract/test_training_core_view_parity.py src/candle/_tensor.py src/candle/_dispatch/dispatcher.py src/candle/_dispatch/functionalize.py src/candle/_backends/autograd.py
git commit -m "fix(parity): align training core view semantics with torch"
```

---

### Task 5: Add failing reduction parity contracts

**Files:**
- Create: `tests/contract/test_training_core_reduction_parity.py`
- Reference: `tests/cpu/test_ops_cpu.py`
- Reference: `src/candle/_functional.py`
- Reference: `src/candle/_backends/cpu/ops.py`
- Reference: `src/candle/_backends/npu/ops.py`
- Reference: `src/candle/_backends/mps/ops.py`

**Step 1: Write failing parity contracts for reductions**

Cover first-wave reduction semantics:

- `mean(int64)` raises like PyTorch
- `mean(int64, dtype=float32)` succeeds and matches values/dtype
- `sum(int8, dtype=int64)` accumulates in target dtype
- `sum/mean` with negative dim and tuple dims
- `keepdim=True/False`
- backward for differentiable floating-point `sum` and `mean`

Add backend-tolerance configuration only where numerics require it. Do not weaken dtype or semantic assertions.

**Step 2: Run the reduction parity tests to verify current failures**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_reduction_parity.py -v --tb=short`
Expected: FAIL, likely on MPS/NPU dtype handling and error semantics.

**Step 3: Implement minimal fixes**

Likely touch points:

- `src/candle/_functional.py` for wrapper-level dtype/error behavior
- `src/candle/_backends/cpu/ops.py::sum_` and `mean_`
- `src/candle/_backends/npu/ops.py::sum_` and `mean`
- `src/candle/_backends/mps/ops.py::sum_` and `mean_`

Important constraints:

- `sum(dtype=...)` must accumulate in target dtype when PyTorch does,
- integer `mean` without explicit dtype must error like PyTorch,
- do not add silent device fallback for unsupported reductions.

**Step 4: Re-run the reduction parity tests**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_reduction_parity.py -v --tb=short`
Expected: PASS.

**Step 5: Run adjacent regressions**

Run: `PYTHONPATH=src pytest tests/cpu/test_ops_cpu.py -k "mean_int or sum_dtype" tests/cpu/test_top_level_ops.py -v --tb=short`
Expected: PASS.

**Step 6: Update known-kernel issues if any backend limitation remains accepted**

If a backend still has a documented temporary limitation that is explicitly tolerated, add an entry to `docs/known-kernel-issues.md`.

**Step 7: Commit**

```bash
git add tests/contract/test_training_core_reduction_parity.py src/candle/_functional.py src/candle/_backends/cpu/ops.py src/candle/_backends/npu/ops.py src/candle/_backends/mps/ops.py docs/known-kernel-issues.md
git commit -m "fix(parity): align training core reduction semantics with torch"
```

---

### Task 6: Add a minimal training-core torch parity smoke gate

**Files:**
- Create: `tests/contract/test_training_core_smoke_parity.py`
- Reference: `tests/npu/test_npu_golden_training_loop.py`
- Reference: `tests/cpu/test_optim.py`
- Reference: `src/candle/nn/functional.py`
- Reference: `src/candle/optim/sgd.py`

**Step 1: Write the failing smoke parity test**

Add a compact training loop that runs both Candle and real PyTorch on the same model/data/seed for a few steps:

- one `Linear` layer,
- one activation (`relu` or `gelu`),
- one reduction loss (`mean` of squared error),
- one optimizer (`SGD` first, no momentum),
- 3 to 5 steps only.

Assert:

- all losses are finite,
- loss direction is consistent,
- parameter values after final step stay within tolerance of real PyTorch,
- gradients are populated at each step.

Keep this test CPU-first if needed for determinism. Make device expansion follow-up work, not part of this task.

**Step 2: Run the smoke test to verify it fails or exposes drift**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_smoke_parity.py -v --tb=short`
Expected: FAIL until the parity harness and first-wave semantics are fully aligned.

**Step 3: Fix any remaining shared training-core drift**

Patch only the smallest remaining semantic gaps revealed by the smoke test. Do not turn the smoke test into a catch-all long-tail debugging bucket.

**Step 4: Re-run the smoke test**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_smoke_parity.py -v --tb=short`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/contract/test_training_core_smoke_parity.py src/candle/nn/functional.py src/candle/optim/sgd.py
git commit -m "test(contract): add training core torch parity smoke gate"
```

---

### Task 7: Run the focused contract gate for phase 1

**Files:**
- No code changes required unless regressions are found

**Step 1: Run the phase-1 focused contract suite**

Run:

```bash
PYTHONPATH=src pytest \
  tests/contract/test_training_core_parity_harness.py \
  tests/contract/test_training_core_binary_parity.py \
  tests/contract/test_training_core_view_parity.py \
  tests/contract/test_training_core_reduction_parity.py \
  tests/contract/test_training_core_smoke_parity.py \
  -v --tb=short
```

Expected: PASS.

**Step 2: Run adjacent regression contracts**

Run:

```bash
PYTHONPATH=src pytest \
  tests/contract/test_inplace_view_rules.py \
  tests/contract/test_functionalize_view_writeback.py \
  tests/contract/test_autograd_contract.py \
  tests/contract/test_dispatch_contract.py \
  -v --tb=short
```

Expected: PASS.

**Step 3: Commit any final doc or regression-fix updates**

```bash
git add <final-files>
git commit -m "test(contract): finalize phase-1 training core parity gate"
```

---

### Task 8: Optional phase-1.5 follow-up for linear/activation parity

**Files:**
- Create: `tests/contract/test_training_core_linear_activation_parity.py`
- Reference: `src/candle/nn/functional.py`
- Reference: `src/candle/_backends/autograd.py`

**Step 1: Write failing linear/activation parity tests**

Start with:

- `linear` forward parity
- `relu`, `gelu`, `silu` forward parity
- backward parity for `linear + relu`

**Step 2: Run tests to confirm failure where present**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_linear_activation_parity.py -v --tb=short`
Expected: FAIL if remaining drift exists.

**Step 3: Implement minimal fixes**

Patch shared wrappers or autograd/backend kernels only where needed.

**Step 4: Re-run tests**

Run: `PYTHONPATH=src pytest tests/contract/test_training_core_linear_activation_parity.py -v --tb=short`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/contract/test_training_core_linear_activation_parity.py src/candle/nn/functional.py src/candle/_backends/autograd.py src/candle/_backends/cpu/ops.py src/candle/_backends/npu/ops.py src/candle/_backends/mps/ops.py
git commit -m "fix(parity): align training core linear and activation semantics with torch"
```
