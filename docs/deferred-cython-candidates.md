# Deferred Cython Migration Candidates

## Status: Tracking only — not yet scheduled for implementation

These candidates were identified during the post-autograd Cython migration wave
but deferred because the functional wrapper and tensor API layers were higher priority.

---

## 1. Dispatcher Python coordination layer

**Files:** `src/candle/_dispatch/dispatcher.py`, `src/candle/_dispatch/registry.py`

**Hot-path functions to consider:**
- `_state_stack()` / `current_dispatch_keyset()` / `_push_dispatch_context` / `_pop_dispatch_context`
- `_bump_versions` / `_check_inplace_targets` / `_mutating_args`
- `_PendingOp.execute` / `_FunctionalizePendingOp.execute`

**Already Cython:** `_dispatcher_core.pyx` handles the core dispatch loop.
The Python layer still owns TLS state, version bumping, and pending-op execution.

**Estimated value:** Medium-high. Per-op TLS stack and version-bump loops.
**Estimated complexity:** Medium. TLS state management needs careful handling.

---

## 2. nn.Module.__call__ hook orchestration

**File:** `src/candle/nn/module.py` (lines 41-69)

**Hot-path area:** The `__call__` method iterates forward pre-hooks and post-hooks
on every module forward pass.

**Already Cython:** Nothing.

**Estimated value:** Medium. Every nn.Module forward traverses this, but hook dicts
are typically empty in production so the overhead is mainly Python method dispatch.
**Estimated complexity:** Low-medium. The hook loop is self-contained.

---

## 3. Common nn layer wrappers

**Files:** `src/candle/nn/linear.py`, `src/candle/nn/conv.py`, `src/candle/nn/batchnorm.py`

**Estimated value:** Low-medium. These are thin wrappers over `_functional.py` which
is already Cython-backed. The remaining overhead is the Python method call itself.
**Estimated complexity:** Low.

---

## 4. Remaining _backends/autograd.py backward glue

**File:** `src/candle/_backends/autograd.py`

**Already migrated:** Regular wrapper factories (Task 24 equivalent in the autograd PR).
**Remaining:** Irregular backward paths (rrelu, view/inplace-sensitive, norm specials).

**Estimated value:** Medium on backward-heavy workloads.
**Estimated complexity:** Medium-high due to per-op special cases.

---

## Prerequisites before starting any of these

1. The post-autograd-cython PR must be merged first
2. Benchmark before/after each slice to confirm the migration actually removes measurable overhead
3. Lock contracts for the targeted hot paths before moving code
