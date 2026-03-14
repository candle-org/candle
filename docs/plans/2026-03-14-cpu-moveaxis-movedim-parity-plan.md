# Moveaxis Movedim Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining CPU-side `moveaxis`/`movedim` parity gaps by adding focused coverage for normal behavior, aligning schema-level invalid-dimension errors with PyTorch, and fixing incorrect pending shapes in pipeline/meta mode.

**Architecture:** Keep this batch mechanism-focused. Reuse the existing public API, schemas, CPU kernels, and autograd registrations. Add red tests first in CPU, contract, and pipeline suites; then make the smallest possible changes in schema validation and meta inference so runtime behavior and pending-shape metadata match PyTorch semantics.

**Tech Stack:** Python, pytest, Candle dispatch/schema/meta system, NumPy-backed CPU kernels.

---

### Task 1: Add focused CPU top-level parity tests

**Files:**
- Modify: `tests/cpu/test_top_level_ops.py`
- Test: `tests/cpu/test_top_level_ops.py`

**Step 1: Write the failing test**

Add tests that cover:
- `torch.movedim(x, 0, 2)` returns shape `(3, 4, 2)` with expected values.
- `torch.moveaxis(x, 0, 2)` matches `torch.movedim(x, 0, 2)` exactly.
- `x.movedim((0, 2), (2, 0))` returns shape `(4, 3, 2)` with expected values.
- `x.moveaxis((0, 2), (2, 0))` matches `x.movedim((0, 2), (2, 0))` exactly.

**Step 2: Run test to verify current status**

Run: `PYTHONPATH=src pytest tests/cpu/test_top_level_ops.py -k "moveaxis or movedim" -v --tb=short`
Expected: pass or expose any unexpected functional gap.

**Step 3: Keep or refine only if needed**

If functional coverage is green immediately, keep the tests as regression coverage and move on.

**Step 4: Re-run test**

Run the same command and confirm clean output.

**Step 5: Commit**

Do not commit yet; batch commit after schema/meta fixes are green.

### Task 2: Add failing pipeline/meta regression test

**Files:**
- Modify: `tests/cpu/test_pipeline.py`
- Test: `tests/cpu/test_pipeline.py`

**Step 1: Write the failing test**

Add a focused test that executes under `with torch.pipeline():` and asserts pending tensor shapes are already permuted correctly for:
- `torch.movedim(x, 0, 2)` -> pending shape `(3, 4, 2)`
- `torch.moveaxis(x, (0, 2), (2, 0))` -> pending shape `(4, 3, 2)`

Also assert the shapes remain the same after flush.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/cpu/test_pipeline.py -k "movedim or moveaxis" -v --tb=short`
Expected: FAIL because `infer_movedim` currently returns the input shape unchanged.

**Step 3: Write minimal implementation**

Update `src/candle/_backends/meta/infer.py` so `infer_movedim` computes the permuted output shape and contiguous stride, with source/destination normalization matching runtime semantics.

**Step 4: Run test to verify it passes**

Run the same command and confirm PASS.

**Step 5: Commit**

Do not commit yet; batch commit after contract parity is green.

### Task 3: Add failing schema-level error-contract tests

**Files:**
- Modify: `tests/contract/test_schema_dim_validation.py`
- Modify: `src/candle/_dispatch/schema.py`
- Test: `tests/contract/test_schema_dim_validation.py`

**Step 1: Write the failing test**

Add exact Torch-alignment tests using `assert_torch_error(...)` for both `dispatch("movedim", ...)` and `dispatch("moveaxis", ...)` covering:
- duplicate source dims: `([0, 0], [1, 2])`
- source/destination length mismatch: `([0, 1], [2])`
- out-of-range dim: `(3, 0)` on a rank-3 tensor

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/contract/test_schema_dim_validation.py -k "movedim or moveaxis" -v --tb=short`
Expected: FAIL because current behavior leaks NumPy/backend exceptions and texts.

**Step 3: Write minimal implementation**

In `src/candle/_dispatch/schema.py`:
- add a validator for `movedim`/`moveaxis` source and destination arguments
- normalize int/list/tuple forms into comparable dim lists
- reject bools and non-int entries with Torch-style tuple-of-ints errors where applicable
- reject duplicate normalized dims with Torch-style `movedim: repeated dim in `source`` / `destination` messages
- reject source/destination length mismatch with Torch-style `Invalid source or destination dims` message
- reject out-of-range dims with the existing Torch-style `Dimension out of range...` text

Keep the validator scoped only to these ops.

**Step 4: Run test to verify it passes**

Run the same command and confirm PASS.

**Step 5: Commit**

Do not commit yet; batch commit after broader verification.

### Task 4: Run focused and broader verification

**Files:**
- Verify: `tests/cpu/test_top_level_ops.py`
- Verify: `tests/cpu/test_pipeline.py`
- Verify: `tests/contract/test_schema_dim_validation.py`
- Verify: `tests/contract/`
- Verify: `tests/cpu/`

**Step 1: Run focused regression commands**

Run:
- `PYTHONPATH=src pytest tests/cpu/test_top_level_ops.py -k "moveaxis or movedim" -v --tb=short`
- `PYTHONPATH=src pytest tests/cpu/test_pipeline.py -k "movedim or moveaxis" -v --tb=short`
- `PYTHONPATH=src pytest tests/contract/test_schema_dim_validation.py -k "movedim or moveaxis" -v --tb=short`

**Step 2: Run required contract gate**

Run: `PYTHONPATH=src pytest tests/contract/ -v --tb=short`
Expected: PASS

**Step 3: Run broader CPU gate**

Run: `PYTHONPATH=src pytest tests/cpu/ tests/contract/ -v --tb=short`
Expected: PASS

**Step 4: Commit**

```bash
git add docs/plans/2026-03-14-cpu-moveaxis-movedim-parity-plan.md   tests/cpu/test_top_level_ops.py   tests/cpu/test_pipeline.py   tests/contract/test_schema_dim_validation.py   src/candle/_dispatch/schema.py   src/candle/_backends/meta/infer.py
git commit -m "fix: align moveaxis/movedim parity"
```
