# Tensor / Storage Batch A2 Design

Date: 2026-04-14
Status: Drafted after A1 completion
Parent spec: `docs/superpowers/specs/2026-04-13-candle-runtime-rebuild-design.md`
Previous batch: `docs/superpowers/specs/2026-04-13-tensor-storage-batch-a1-design.md`

## 1. Batch Definition

- **Subsystem:** S1 Tensor / Storage runtime core
- **Batch:** A2
- **Purpose:** move version-sensitive and view-sensitive runtime truth out of the Python `Tensor` shell and into the Cython `TensorImpl`, while staying strictly inside the Tensor / Storage subsystem

A2 is the first Tensor/Storage batch that performs real correctness-convergence work rather than only declaring boundaries. It is still intentionally narrow: it does not redesign dispatcher, autograd runtime, backend behavior, or NPU execution.

## 2. Why A2 Exists

A1 established the long-term ownership boundary for Tensor/Storage and prevented new Python-side owner growth. However, several runtime-sensitive behaviors still remain temporarily in the Python shell:

- version-sensitive behavior in `detach`, `set_`, and related Tensor-side mutation paths
- view-sensitive behavior tied to `_base` and `_view_meta`
- Tensor shell logic that still acts as a partial fact source for version/view truth rather than merely forwarding to Cython

A2 exists to begin the actual owner migration that A1 prepared for.

## 3. Problem Statement

The problem A2 addresses is not “Tensor APIs are missing.” The problem is that version/view correctness still relies too heavily on Python-shell logic.

That creates three risks:

1. **Correctness risk** — version truth and view truth can drift apart across Python and Cython paths
2. **Architecture drift risk** — future fixes will keep landing in `_tensor.py` because it is the easiest place to patch
3. **Phase coupling risk** — later autograd and dispatcher work will inherit unstable Tensor/Storage truth

A2 therefore treats version/view-sensitive runtime state as a **fact-source problem**, not merely a method-location problem.

## 4. Scope

### 4.1 In scope

A2 covers two tightly related sub-batches:

- **A2a:** version-sensitive runtime truth
- **A2b:** view-sensitive runtime truth

Together, they cover:

- `_version_counter`
- `detach`
- `set_`
- Tensor-side mutation-sensitive runtime truth that directly affects version correctness
- `_base`
- `_view_meta`
- view-family helper truth to the extent needed to make `_base` / `_view_meta` more Cython-centered

### 4.2 Out of scope

A2 must not redesign or substantially modify:

- dispatcher or schema behavior
- autograd graph/runtime
- backend-specific kernels or execution logic
- NPU execution ownership
- fake/meta/functionalization compatibility layers
- performance-driven optimizations unrelated to owner convergence

## 5. Allowed Files

A2 should primarily modify only:

- `src/candle/_tensor.py`
- `src/candle/_cython/_tensor_impl.pyx`
- `src/candle/_cython/_tensor_impl.pxd`

Only if strictly necessary for Tensor/Storage-local correctness or support:

- `src/candle/_storage.py`
- `src/candle/_cython/_storage.pyx`
- narrowly targeted contract tests under `tests/contract/`

## 6. Forbidden Files

A2 must not modify:

- `src/candle/_dispatch/**`
- `src/candle/autograd/**`
- `src/candle/_backends/**`
- `src/candle/_functional.py`
- backend runtime code outside Tensor/Storage-local support

If A2 appears to require one of those changes, that is evidence of scope leak and must be deferred or split into a later batch.

## 7. Design Overview

A2 is a **correctness-convergence** batch, not a feature batch. The guiding rule is:

> Tensor/version/view truth should live where runtime metadata already lives: in `TensorImpl`, not in the Python shell.

The Python `Tensor` class should continue to expose public APIs and user-facing argument handling, but it should stop being the place where version-sensitive and view-sensitive runtime facts are established.

## 8. A2a — Version-Sensitive Runtime Truth

### 8.1 Goal

Move the core truth for version-sensitive Tensor behavior closer to `TensorImpl`, especially for:

- `detach`
- `_version_counter`
- `set_`
- Tensor-side mutation-sensitive metadata updates that directly affect version correctness

### 8.2 Python shell responsibilities after A2a

Python `Tensor` may still:

- expose `detach()` and `set_()` as public methods
- normalize user-facing arguments
- provide user-facing error formatting

Python `Tensor` should no longer be the primary owner of version-sensitive runtime truth.

### 8.3 Cython responsibilities after A2a

`_cython/_tensor_impl.pyx` should more clearly own:

- version-counter truth
- detach-derived runtime truth
- version-sensitive metadata mutation paths used by Tensor-side operations

### 8.4 A2a intended effect

A2a does not need to move every related method wholesale into Cython. It needs to move the **fact source** and the most runtime-sensitive semantics into Cython so that Python becomes a thin shell over the same truth.

## 9. A2b — View-Sensitive Runtime Truth

### 9.1 Goal

Move the core truth for view-sensitive Tensor behavior closer to `TensorImpl`, especially for:

- `_base`
- common view base/version attachment truth
- view-family helper truth where Python still manually assembles derived runtime facts
- the boundary between Cython-owned view runtime truth and Python-owned user-facing `_view_meta` augmentation

### 9.2 Python shell responsibilities after A2b

Python `Tensor` may still:

- expose `view`, `as_strided`, `transpose`, and related APIs
- normalize shape/dim arguments
- raise user-facing errors
- attach user-facing `_view_meta["op"]` detail and similar shell-level metadata augmentation

Python `Tensor` should no longer be the main fact source for `_base` and version-sharing truth on common view paths, but `_view_meta` assembly may still remain partially in the Python shell for now.

### 9.3 Cython responsibilities after A2b

`_cython/_tensor_impl.pyx` should more clearly own:

- base-view linkage truth
- common view base/version attachment truth
- attachment of view-derived objects to the same version-sharing/runtime truth model

This batch does not require all `_view_meta` construction to move into Cython. Some user-facing `_view_meta` payload assembly may remain in the Python shell after A2.

### 9.4 A2b intended effect

A2b is not a full view-system rewrite. It is a controlled move toward a single runtime truth center for view-derived tensors.

## 10. A2a / A2b Dependency Relationship

A2a should land before A2b.

Reason:

- view truth must be compatible with version truth
- once version-sensitive runtime ownership is more Cython-centered, view metadata can attach to a more stable base
- doing A2b first would increase the chance of duplicating temporary logic

So A2 is designed as one spec, but it should be implemented in two bounded phases:

1. **A2a first**
2. **A2b second**
3. **A2 total validation last**

## 11. Test Strategy

### 11.1 General principle

Use existing contract tests as the primary correctness rails. Add new tests only when a real gap exists.

### 11.2 A2a primary rails

Use existing tests in `tests/contract/test_tensor_alias_version_contract.py`:

- `test_detach_shares_version_counter_with_source`
- `test_set_bumps_version_counter_once`
- `test_set_on_view_bumps_shared_version_counter_once`
- `test_setitem_bumps_version_counter`
- `test_dispatch_setitem_bumps_version_counter_exactly_once`

Only add focused tests if the current suite cannot distinguish the intended owner migration from a regression.

### 11.3 A2b primary rails

Use existing tests in `tests/contract/test_tensor_alias_version_contract.py`:

- `test_as_strided_view_shares_version_counter_with_source`
- `test_diagonal_view_shares_version_counter_with_source`
- `test_movedim_view_shares_version_counter_with_source`
- `test_moveaxis_view_shares_version_counter_with_source`
- `test_expand_view_shares_version_counter_with_source`
- `test_broadcast_to_view_shares_version_counter_with_source`
- `test_split_view_shares_version_counter_with_source`
- `test_chunk_view_shares_version_counter_with_source`
- `test_unary_inplace_preserves_view_aliasing`

Only add focused `_base` / `_view_meta` tests if the current suite is too indirect for a concrete migration step.

### 11.4 A2 total rails

Before closing A2, run at minimum:

- `tests/contract/test_tensor_alias_version_contract.py`
- `tests/contract/test_inplace_view_rules.py`
- `tests/contract/test_tensor_storage_owner_contract.py`
- `tests/contract/test_storage_contract.py`

## 12. Success Criteria

### 12.1 A2a success

A2a is complete only when:

1. `detach` no longer relies primarily on Python-shell runtime truth
2. `set_` is more clearly anchored to Cython-owned runtime truth
3. version-sensitive Tensor truth is more centralized in `TensorImpl`
4. the A2a primary rails pass
5. no dispatcher/autograd/backend scope creep was introduced

### 12.2 A2b success

A2b is complete only when:

1. `_base` and `_view_meta` truth are more centralized in `TensorImpl`
2. view-derived tensors still satisfy the existing version-sharing contracts
3. Python-shell view-sensitive owner logic is reduced further
4. the A2b primary rails pass
5. no dispatcher/autograd/backend scope creep was introduced

### 12.3 A2 total success

A2 is complete only when:

1. A2a and A2b both land cleanly
2. the A2 total rails pass
3. Python `Tensor` shell owns less version/view-sensitive runtime truth than it did after A1
4. `_cython/_tensor_impl.pyx` is a more explicit truth center for version/view runtime state
5. A2 remains strictly inside the Tensor/Storage subsystem

## 13. Relationship to Later Work

A2 does **not** finish Tensor/Storage work. It prepares later work by making Tensor runtime truth more stable.

In particular:

- later autograd batches should inherit more reliable version/view semantics
- later dispatcher batches should consume a more stable Tensor runtime model
- later backend/NPU batches should not have to guess where Tensor runtime truth lives

A2 is therefore a local correctness-convergence batch with system-wide downstream value.

## 14. Final Design Summary

A2 is a strict Tensor/Storage-only batch that covers both version-sensitive and view-sensitive runtime truth in one design, but implements them in two bounded stages.

- **A2a** converges version/detach/set_-related runtime truth toward Cython `TensorImpl`
- **A2b** converges `_base` / `_view_meta` / view-family runtime truth toward Cython `TensorImpl`
- existing alias/version contract tests serve as the main safety rails
- success is measured by owner convergence and correctness stability, not by feature count or performance
