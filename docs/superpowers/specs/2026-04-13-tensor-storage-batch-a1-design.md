# Tensor / Storage Batch A1 Design

Date: 2026-04-13
Status: First bounded implementation batch under the Candle runtime rebuild program
Parent spec: `docs/superpowers/specs/2026-04-13-candle-runtime-rebuild-design.md`

## 1. Batch Definition

- **Subsystem:** S1 Tensor / Storage runtime core
- **Batch:** A1
- **Purpose:** define and enforce long-term ownership boundaries for Tensor/Storage runtime state before attempting larger semantic or behavioral migration

This batch is intentionally narrow. It is a boundary-definition batch, not a broad correctness or performance batch.

## 2. Why This Batch Comes First

Everything else in Candle depends on tensor/storage runtime truth:

- autograd depends on version/view/alias truth
- dispatcher depends on tensor metadata and mutation semantics
- NPU execution depends on storage/device/runtime boundaries

If these boundaries remain fuzzy, later phases will either drift architecturally or require repeated rework.

## 3. Current Problem Statement

Current Candle already uses a Cython tensor core, but ownership is still split:

- `src/candle/_tensor.py` still contains runtime-critical owner logic
- tensor/storage metadata propagation is not fully shell-vs-core separated
- version / alias / view runtime responsibilities are not yet fully centralized
- storage ownership boundaries are not explicit enough for later dispatcher/autograd/NPU convergence

The problem in this batch is **not** "Tensor is totally wrong." The problem is that the long-term ownership boundary is not yet explicit or enforced.

## 4. Batch Goal

At the end of A1, the project must have a clear and enforced answer to all of the following:

1. What state belongs to the Cython tensor core?
2. What state belongs to the storage core?
3. What responsibilities remain in Python `Tensor` shell code?
4. Which currently Python-owned responsibilities are scheduled for migration in A2+?
5. What new Python-side runtime owner logic is now forbidden?

## 5. Allowed Files

Only the following implementation files may be modified in A1:

- `src/candle/_tensor.py`
- `src/candle/_cython/_tensor_impl.pyx`
- `src/candle/_storage.py`
- `src/candle/_cython/_storage.pyx`

Allowed supporting files:

- targeted contract tests relevant to tensor/storage ownership boundaries
- design/plan documents for this batch

## 6. Forbidden Files

A1 must not modify any of the following runtime subsystems:

- `src/candle/_dispatch/**`
- `src/candle/autograd/**`
- `src/candle/_backends/npu/**`
- `src/candle/_functional.py`
- backend runtime implementations outside storage/tensor core

Exception: read-only inspection is allowed during implementation, but not edits.

## 7. Long-Term Ownership Boundary for Tensor / Storage

### 7.1 Cython Tensor core must own

- tensor identity state
- shape / stride / offset runtime state
- device / dtype cached runtime state
- dispatch-key-relevant cached runtime state
- version-counter attachment point
- base/view relationship attachment point
- autograd-metadata attachment point

### 7.2 Storage core must own

- underlying data pointer ownership
- typed storage identity
- pointer-lifetime-related core data
- storage-backed dtype/device ownership facts

### 7.3 Python Tensor shell may own

- public method surface
- argument normalization for user-facing APIs
- high-level wrapper behavior that does not become a runtime owner
- compatibility glue that forwards immediately into Cython/runtime layers

### 7.4 Python Tensor shell must not grow to own

- new version/alias/view core state
- new persistent runtime device/dtype/storage ownership state
- new execution or lifecycle ownership
- new bridge-layer runtime caches intended to become permanent

## 8. Expected A1 Deliverables

A1 should produce:

1. an explicit field/responsibility split between `_tensor.py` and `_tensor_impl.pyx`
2. an explicit field/responsibility split between `_storage.py` and `_storage.pyx`
3. removal or reduction of obvious Python runtime owner responsibilities where safe and tightly scoped
4. a migration inventory for A2 showing which Python-owned pieces are still intentionally temporary
5. targeted tests or assertions that make boundary drift harder

## 9. Non-Goals

A1 does **not** attempt to:

- redesign autograd graph/runtime
- redesign dispatcher or schema behavior
- touch NPU execution ownership
- chase performance improvements
- resolve every existing tensor correctness issue
- add advanced override/fake/meta compatibility

If a proposed edit primarily serves one of these goals, it belongs to a later batch and must be deferred.

## 10. Validation

Validation for A1 is intentionally narrow.

### Structural validation

Confirm that:

- no new Python runtime owner has been introduced
- modified logic moved toward Cython/storage ownership rather than away from it
- file responsibilities are clearer after the batch than before

### Behavioral validation

Run only the smallest relevant tests for:

- tensor construction / storage attachment
- version/view boundary checks if touched
- any targeted contract tests created or updated for ownership boundaries

No broad autograd, dispatcher, or backend validation should be introduced here unless directly required by touched tensor/storage behavior.

## 11. Exit Criteria

A1 is complete only when:

1. the Tensor/Storage owner boundary is documented and reflected in code structure
2. no forbidden subsystem files were modified
3. Python shell responsibilities are no broader than before
4. the next migration batch (A2) has a precise list of remaining runtime-owned responsibilities to move
5. the batch can hand off cleanly without ambiguity about what belongs where

## 12. A2 Preview

A2 will build on A1 by converging the most correctness-critical runtime semantics in this subsystem:

- version counter behavior
- alias semantics
- base/view metadata truth
- detach-related ownership correctness

A1 should make that batch easier, narrower, and less ambiguous — not partially absorb it.

## 13. Concrete A2 Migration Inventory

A2 should target the following runtime-sensitive responsibilities that remain temporary after A1:

- version-sensitive Python Tensor shell behavior that still reaches into runtime-owned state
- remaining alias/view boundary helpers that still live in `_tensor.py`
- any storage-shell logic that still behaves like a low-level owner rather than a public API wrapper
