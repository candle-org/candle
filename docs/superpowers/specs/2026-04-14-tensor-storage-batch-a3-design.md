# Tensor / Storage Batch A3 Design

Date: 2026-04-14
Status: Drafted after A2 completion
Parent spec: `docs/superpowers/specs/2026-04-13-candle-runtime-rebuild-design.md`
Previous batch: `docs/superpowers/specs/2026-04-14-tensor-storage-batch-a2-design.md`

## 1. Batch Definition

- **Subsystem:** S1 Tensor / Storage runtime core
- **Batch:** A3
- **Purpose:** move `Tensor.data` shallow metadata-copy truth out of the Python `Tensor` shell and into the Cython `TensorImpl`, aligning Candle more closely with PyTorch's `TensorImpl::shallow_copy_from()` model while staying strictly inside the Tensor / Storage subsystem

A3 is a narrow residual S1 batch. It does not introduce new Tensor features. It tightens runtime ownership so that `Tensor.data` stops being a Python-side fact source for storage/shape/stride/offset mutation.

## 2. Why A3 Exists

A2 moved important version-sensitive and common view-sensitive runtime truth into `TensorImpl`. However, one important Tensor mutation path still remains owned by the Python shell:

- `src/candle/_tensor.py:199-215` still performs direct `data`-setter runtime mutation in Python by assigning `_storage`, `stride`, `offset`, and bumping version state from the shell

That leaves Candle in a partially converged state:

- `detach()` is already Cython-centered
- common view attachment is already more Cython-centered
- but `Tensor.data = other` still mutates runtime truth from Python

A3 exists to eliminate that remaining mismatch.

## 3. Source Alignment Basis

A3 is explicitly anchored to **PyTorch source structure**, while also using **torch_npu as a reference for NPU runtime ownership direction**.

### 3.1 PyTorch alignment point

PyTorch documents Tensor metadata shallow-copy ownership in `c10/core/TensorImpl.h:2029-2066`.

That note establishes two distinct runtime patterns:

1. `detach()` uses `shallow_copy_and_detach()`
2. `set_data(tensor)` uses `shallow_copy_from()`

The key architectural point is not only which public API exists. The key point is **where runtime truth lives**:

- storage pointer
- sizes
- strides
- storage offset
- related metadata-copy behavior

all belong to `TensorImpl`-level shallow-copy machinery, not to a Python shell wrapper.

### 3.2 torch_npu reference point

torch_npu is relevant to A3 as a **runtime ownership reference**, but **not** as a plugin architecture template for Candle.

The important lesson from torch_npu is that NPU runtime-sensitive truth is kept in native-facing runtime owners rather than in long-term Python wrapper state. However, Candle should **not** mirror torch_npu's external plugin-style layering as the long-term architecture target.

For Candle, the intended structural direction is:

- NPU remains a first-class built-in Candle device
- NPU storage should converge toward the same kind of framework-owned device-storage role that CUDA has in Candle
- NPU Tensor / Storage runtime truth should live inside Candle's own Tensor / Storage runtime model, not in a torch_npu-style add-on ownership layer

A3 itself does not finish that NPU-storage convergence. But it must remain consistent with that direction by continuing to centralize Tensor / Storage runtime truth inside Candle-owned runtime objects.

## 4. Problem Statement

The problem A3 addresses is not that Candle lacks a `data` setter. The problem is that Candle still lets the Python shell act as a runtime truth source for shallow metadata replacement.

That creates three risks:

1. **Owner drift risk** — future fixes to `data`-related semantics will keep landing in `_tensor.py`
2. **Correctness drift risk** — metadata-copy truth can diverge from the Cython-centered runtime model established by A2
3. **Phase-coupling risk** — later autograd work will inherit an unstable split where some Tensor metadata mutation paths are runtime-owned and others are shell-owned

A3 therefore treats `Tensor.data` as a **runtime-owner convergence problem**, not as a surface API problem.

## 5. Scope

### 5.1 In scope

A3 covers:

- `Tensor.data` setter runtime truth
- shallow metadata-copy behavior for:
  - storage
  - size/shape
  - stride
  - storage offset
  - device cache refresh
  - dtype cache refresh
- version-sensitive behavior directly tied to this metadata-copy path, to the extent needed to keep existing Tensor / Storage contract behavior stable
- narrow contract tests proving the new ownership boundary

### 5.2 Out of scope

A3 must not redesign or substantially modify:

- autograd graph/runtime
- dispatcher/schema behavior
- backend-specific execution code
- view-family redesign
- `_view_meta` architecture beyond what is strictly needed for `data`-setter correctness
- Python-side public API shape of `Tensor.data`
- full PyTorch `set_data` autograd parity across all edge cases

A3 is intentionally about **fact-source migration first**, not full semantic completion of all future autograd-sensitive cases.

## 6. Allowed Files

A3 should primarily modify only:

- `src/candle/_tensor.py`
- `src/candle/_cython/_tensor_impl.pyx`
- `src/candle/_cython/_tensor_impl.pxd`
- narrowly targeted contract tests under `tests/contract/`

Only if strictly necessary for Tensor / Storage-local support:

- `src/candle/_storage.py`
- `src/candle/_cython/_storage.pyx`

## 7. Forbidden Files

A3 must not modify:

- `src/candle/_dispatch/**`
- `src/candle/autograd/**`
- `src/candle/_backends/**`
- `src/candle/_functional.py`

If A3 appears to require those changes, that is evidence of scope leak and should be deferred to a later batch.

## 8. Design Overview

A3 follows the same owner-convergence rule as A2:

> Tensor runtime metadata truth should live in `TensorImpl`, not in the Python `Tensor` shell.

After A3:

- Python `Tensor.data` remains the public API entry point
- Python still performs user-facing validation and error reporting
- Cython `TensorImpl` becomes the fact source for the actual shallow metadata-copy mutation

This mirrors the PyTorch split between public Variable/Tensor surface and `TensorImpl` shallow-copy ownership.

## 9. Runtime Ownership After A3

### 9.1 Python shell responsibilities after A3

Python `Tensor` may still:

- expose the public `data` property and setter
- validate that the replacement object is a `Tensor`
- validate user-visible compatibility constraints (shape, dtype, or other currently supported public checks)
- raise user-facing errors

Python `Tensor` should no longer directly mutate runtime truth for the `data` setter by assigning storage/stride/offset/version fields itself.

### 9.2 Cython responsibilities after A3

`src/candle/_cython/_tensor_impl.pyx` should more clearly own:

- shallow metadata-copy from one tensor runtime object to another
- storage replacement for the `data` setter path
- size/stride/storage-offset refresh for that path
- device/dtype runtime cache refresh tied to that path
- version-sensitive runtime mutation tied to the same path

### 9.3 Intended Cython helper shape

A3 should introduce one narrow helper in `TensorImpl`, with a name such as:

- `cy_set_data_runtime_truth_from(self, other)`

or

- `cy_shallow_copy_runtime_truth_from(self, other)`

The exact helper name is less important than the ownership rule: the helper should be the single fact source for the runtime mutation.

## 10. Alignment Semantics

### 10.1 PyTorch-aligned conceptual mapping

After A3, Candle's ownership story should read cleanly against PyTorch:

- `detach()` ↔ `shallow_copy_and_detach()`-style runtime truth
- `data` setter ↔ `shallow_copy_from()`-style runtime truth

A3 does **not** claim full PyTorch implementation parity. It claims the correct **mechanical ownership direction**.

### 10.2 What A3 does not attempt yet

PyTorch's broader `set_data` behavior interacts with autograd-specific constraints and metadata-change policy. A3 does not pull all of that into scope.

Instead, A3 makes one controlled move:

- the Tensor / Storage runtime metadata-copy path stops being shell-owned
- later autograd batches can refine higher-level semantics on top of a more correct runtime owner boundary

## 11. Version Behavior Rule

A3 must preserve Candle's currently intended contract behavior for the `data` setter unless an existing contract test proves that behavior is already wrong.

In practice, this means:

- no double version bumps
- no silent loss of version updates
- no new Python/Cython split where both layers partially own the bump rule

The version-sensitive rule for this path should be expressed in one Cython-centered helper.

## 12. Test Strategy

### 12.1 General principle

Use existing contract tests as the primary rails. Add only focused tests that make the A3 ownership migration observable.

### 12.2 Primary rails

Use existing tests in:

- `tests/contract/test_tensor_storage_owner_contract.py`
- `tests/contract/test_tensor_alias_version_contract.py`

### 12.3 Required focused coverage

A3 should ensure there is explicit coverage for:

1. `Tensor.data` routing through runtime-owned shallow metadata-copy truth
2. storage pointer after `data` replacement matching the source tensor
3. shape / stride / offset after `data` replacement matching the source tensor where the public contract permits it
4. device / dtype runtime cache remaining correct after the replacement
5. version behavior remaining single-sourced and stable

If those checks are not already present, add narrow contract tests for them.

### 12.4 A3 validation rails

Before closing A3, run at minimum:

- `tests/contract/test_tensor_storage_owner_contract.py`
- `tests/contract/test_tensor_alias_version_contract.py`
- `tests/contract/test_inplace_view_rules.py`
- `tests/contract/test_storage_contract.py`

## 13. Success Criteria

A3 is complete only when:

1. `src/candle/_tensor.py` no longer directly performs `data`-setter runtime truth mutation by assigning storage/stride/offset/version fields itself
2. `TensorImpl` is the single fact source for shallow metadata-copy truth on the `data` path
3. Candle's runtime ownership story maps more cleanly to PyTorch's `shallow_copy_from()` model
4. A3 validation rails pass
5. no dispatcher/autograd/backend scope creep is introduced

## 14. Relationship to Later Work

A3 does not finish S1. It prepares later work by removing one more shell-owned metadata mutation path.

In particular:

- later autograd batches can refine `set_data`-adjacent semantics on top of a more stable runtime owner
- later Tensor residual batches can continue shrinking shell-owned truth only where real runtime ownership still leaks upward
- later storage-focused S1 work should keep pushing NPU storage toward the same built-in Candle device-storage model used for CUDA, rather than toward a torch_npu-style plugin ownership split
- later backend/NPU work can rely on a cleaner Tensor / Storage metadata owner boundary

## 15. Final Design Summary

A3 is a narrow Tensor / Storage residual batch that aligns Candle more directly to PyTorch's TensorImpl shallow-copy model.

- it targets `Tensor.data` as the next owner-convergence step
- it moves shallow metadata-copy runtime truth into Cython `TensorImpl`
- it keeps Python as the public shell for validation and errors only
- it uses existing contract tests plus a few focused rails to prove the migration
- it stays strictly inside S1 Tensor / Storage scope
