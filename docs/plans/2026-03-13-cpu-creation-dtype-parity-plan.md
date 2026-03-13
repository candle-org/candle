# CPU Creation Dtype Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Align `torch.tensor(...)` and `torch.as_tensor(...)` creation dtype fallback with Candle's current repo-local contract: when `dtype` is omitted, creation should respect `torch.get_default_dtype()`, while `torch.as_tensor(existing_tensor)` should preserve the source tensor's dtype and device unless an override is provided.

**Architecture:** Keep this batch narrowly focused on the public creation API layer in `src/candle/_creation.py`. Drive the behavior through CPU regressions first, then implement the smallest possible fallback changes in the creation helpers. Preserve existing bool inference and explicit `dtype=` / `device=` override behavior.

**Tech Stack:** Python, pytest, Candle creation helpers, CPU regression tests.

---

### Task 1: Capture repo-local default-dtype behavior in CPU tests

**Files:**
- Modify: `tests/cpu/test_dtype_device.py`

**Step 1: Add focused regression tests**

Cover the behaviors this batch needs to lock down:
- `torch.tensor([1, 2, 3])` follows `torch.get_default_dtype()` when `dtype=None`
- `torch.as_tensor([1, 2, 3])` follows `torch.get_default_dtype()` when `dtype=None`
- `torch.as_tensor(existing_tensor)` preserves the source tensor's dtype/device when no overrides are requested
- Existing bool creation behavior remains unchanged as a guard

Example shape:

```python
with torch.dtype(torch.float64):
    x = torch.tensor([1, 2, 3])
    assert x.dtype == torch.float64
```

**Step 2: Run the focused test slice and confirm the red state if needed**

Run:

```bash
PYTHONPATH=src pytest tests/cpu/test_dtype_device.py -k "set_default_dtype_affects_tensor_and_as_tensor_creation or as_tensor_existing_tensor_preserves_dtype" -v --tb=short
```

Expected before the fix: creation paths still hard-code `float32` instead of consulting the default dtype.

---

### Task 2: Implement minimal creation fallback changes

**Files:**
- Modify: `src/candle/_creation.py`

**Step 1: Update `tensor()` default-dtype fallback**

When `dtype is None`, use `get_default_dtype()` instead of a hard-coded `torch.float32` fallback.

**Step 2: Update `as_tensor()` behavior**

- If `data` is already a Candle `Tensor` and there are no overrides, return it unchanged.
- If `data` is already a Candle `Tensor` and `dtype` or `device` is overridden, call `data.to(...)` with those overrides.
- If `data` is raw Python/NumPy data and `dtype is None`, use `get_default_dtype()` as the fallback.

**Step 3: Re-run the focused regressions**

Run:

```bash
PYTHONPATH=src pytest tests/cpu/test_dtype_device.py -k "set_default_dtype_affects_tensor_and_as_tensor_creation or as_tensor_existing_tensor_preserves_dtype" -v --tb=short
```

Expected after the fix: the focused regression slice passes.

---

### Task 3: Verify adjacent CPU and contract coverage

**Step 1: Run the broader CPU suite**

Run:

```bash
PYTHONPATH=src pytest tests/cpu/ -v --tb=short
```

Expected: CPU regression coverage stays green.

**Step 2: Run the required contract gate**

Run:

```bash
PYTHONPATH=src pytest tests/contract/ -v --tb=short
```

Expected: contract suite stays green.

---

### Task 4: Commit the batch cleanly

**Files:**
- Modify: `src/candle/_creation.py`
- Modify: `tests/cpu/test_dtype_device.py`
- Modify: `tests/cpu/test_top_level_ops.py`
- Add: `docs/plans/2026-03-13-cpu-creation-dtype-parity-plan.md`

**Step 1: Inspect the final diff**

Run:

```bash
git status --short
git diff -- src/candle/_creation.py tests/cpu/test_dtype_device.py tests/cpu/test_top_level_ops.py docs/plans/2026-03-13-cpu-creation-dtype-parity-plan.md
```

**Step 2: Commit the implementation and plan**

Suggested commits:

```bash
git commit -m "fix: respect default dtype in tensor creation"
git commit -m "docs: add creation dtype parity plan"
```

**Step 3: Push and update the upstream PR**

Run:

```bash
git push -u origin feat/cpu-creation-indexing-parity
```

Then ensure the upstream PR summary matches the corrected scope: default-dtype fallback and `as_tensor(existing_tensor)` preservation, not raw integer `int64` inference parity.
