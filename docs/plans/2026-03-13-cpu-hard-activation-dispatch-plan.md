# CPU Hard Activation Dispatch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Route `torch.nn.functional.hardswish`, `torch.nn.functional.hardsigmoid`, and `torch.nn.functional.softsign` through their registered backend ops and add focused CPU/contract tests that lock down forward parity and dispatch routing.

**Architecture:** Keep this batch tightly scoped to the public functional API path plus focused tests. The schemas and CPU backend kernels already exist. This batch should prove the public API reaches the registered backend path instead of staying on composite Python implementations.

**Tech Stack:** Python, pytest, Candle dispatch stack, CPU backend kernels, PyTorch forward parity reference.

---

### Task 1: Add focused CPU regression tests for hard activation routing and forward parity

**Files:**
- Modify: `tests/cpu/test_nn_functional.py`

**Step 1: Write the failing tests**

Add focused tests for:
- `F.hardswish(x)` matches PyTorch numerically on a simple float input
- `F.hardsigmoid(x)` matches PyTorch numerically on a simple float input
- `F.softsign(x)` matches PyTorch numerically on a simple float input
- `F.hardswish(...)` routes through `dispatch("hardswish", ...)`
- `F.hardsigmoid(...)` routes through `dispatch("hardsigmoid", ...)`
- `F.softsign(...)` routes through `dispatch("softsign", ...)`

Use local `import torch as torch_ref` in the parity tests and compare `out.numpy()` to `ref.detach().numpy()` with `np.testing.assert_allclose(...)`.

**Step 2: Run the targeted CPU slice and verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/cpu/test_nn_functional.py -k "hardswish or hardsigmoid or softsign" -v --tb=short
```

Expected: the dispatch-routing tests fail because `nn.functional` still uses composite implementations.

---

### Task 2: Add one focused contract parity test file for the hard activation trio

**Files:**
- Create: `tests/contract/test_training_core_hard_activation_parity.py`

**Step 1: Write contract parity tests using the existing harness**

Add one parity-harness style forward test per op:
- `hardswish`
- `hardsigmoid`
- `softsign`

Each test should assert:
- `result["dtype_match"] is True`
- `result["shape_match"] is True`
- `result["value_match"] is True`

**Step 2: Run the targeted contract slice**

Run:

```bash
PYTHONPATH=src pytest tests/contract/test_training_core_hard_activation_parity.py -v --tb=short
```

Expected: pass once the public functional path and tests are correct.

---

### Task 3: Route the public functional APIs through dispatch

**Files:**
- Modify: `src/candle/nn/functional.py`

**Step 1: Replace the composite implementations with dispatch**

Update:
- `hardswish()` to call `dispatch("hardswish", input.device.type, input)`
- `hardsigmoid()` to call `dispatch("hardsigmoid", input.device.type, input)`
- `softsign()` to call `dispatch("softsign", input.device.type, input)`

Keep the public signatures unchanged.

**Step 2: Re-run the targeted CPU and contract tests**

Run:

```bash
PYTHONPATH=src pytest tests/cpu/test_nn_functional.py -k "hardswish or hardsigmoid or softsign" -v --tb=short
PYTHONPATH=src pytest tests/contract/test_training_core_hard_activation_parity.py -v --tb=short
```

Expected: both targeted slices pass.

---

### Task 4: Run adjacent regression coverage

**Step 1: Run the full functional CPU file**

Run:

```bash
PYTHONPATH=src pytest tests/cpu/test_nn_functional.py -v --tb=short
```

**Step 2: Run the required contract gate**

Run:

```bash
PYTHONPATH=src pytest tests/contract/ -v --tb=short
```

**Step 3: Run the broader CPU gate**

Run:

```bash
PYTHONPATH=src pytest tests/cpu/ -v --tb=short
```

---

### Task 5: Commit the batch cleanly

**Files:**
- Modify: `src/candle/nn/functional.py`
- Modify: `tests/cpu/test_nn_functional.py`
- Create: `tests/contract/test_training_core_hard_activation_parity.py`
- Add: `docs/plans/2026-03-13-cpu-hard-activation-dispatch-plan.md`

**Step 1: Inspect diff**

Run:

```bash
git status --short
git diff -- src/candle/nn/functional.py tests/cpu/test_nn_functional.py tests/contract/test_training_core_hard_activation_parity.py docs/plans/2026-03-13-cpu-hard-activation-dispatch-plan.md
```

**Step 2: Commit**

```bash
git add src/candle/nn/functional.py tests/cpu/test_nn_functional.py tests/contract/test_training_core_hard_activation_parity.py docs/plans/2026-03-13-cpu-hard-activation-dispatch-plan.md
git commit -m "fix: route hard activations through backend kernels"
```
