# Training Core Backend Parity Design

## Context

Candle already has a meaningful torch-like mechanism baseline: schema-first operator registration, dispatch keys, autograd, functionalize, AMP, and multiple backend entry points. The current gap is no longer just operator count. The higher-risk issue is backend semantic drift: the same high-frequency training op can behave differently across CPU, NPU, MPS, and CUDA, or differ from real PyTorch in dtype promotion, reduction semantics, aliasing, autograd, or error behavior.

For the next phase, priority should shift from broadening operator surface area to hardening backend consistency for the training-critical subset. This aligns with the repository's documented 0.1 scope: stable single-card NPU training core path first, with CPU as development baseline and no CPU fallback from device backends.

## Goal

Build a contract-first parity layer for training-core high-frequency operators so Candle backends converge to PyTorch semantics instead of only converging to each other.

Success means:

- PyTorch is the only semantic source of truth for the targeted operator subset.
- Candle `CPU/NPU/MPS/CUDA` all pass the same parity contracts, within explicit numeric tolerances where needed.
- A minimal training loop gate stays green while parity coverage expands.
- Backend bugs are exposed as backend bugs, not hidden by CPU fallback or backend-specific behavior drift.

## Confirmed Scope

This phase covers training-core high-frequency operators only.

Included operator families:

- `linear`
- `activation`
- `reduction`
- `broadcast` and binary arithmetic needed by training
- `reshape/view`
- `indexing`
- `optimizer-primitives`

Representative first-wave operators inside those families:

- `add`, `sub`, `mul`, `div`, `true_divide`
- `matmul`, `mm`, `addmm`
- `relu`, `sigmoid`, `tanh`, `gelu`, `silu`
- `sum`, `mean`, `prod`, `norm`
- `reshape`, `view`, `transpose`, `squeeze`, `unsqueeze`, `permute`, `flatten`
- `getitem`, `setitem`, `gather`, `scatter`, `index_select`, `masked_select`
- `add_`, `sub_`, `mul_`, `copy_`, `zero_`, and scalar-mixed optimizer updates

Explicitly out of scope for this phase:

- LLM-specific fused kernels and serving features
- long-tail operator expansion outside the training core subset
- FSDP, DeviceMesh, Tensor Parallel, and other distributed high-level features
- true compile/JIT/export acceleration work
- sparse tensor and quantized tensor system design

## Non-Negotiable Principle

Backend consistency must align to real PyTorch behavior, not merely to Candle CPU behavior.

That means:

- `torch` is the oracle for forward values, output dtype, shape, broadcast behavior, alias semantics, autograd behavior, optimizer update behavior, and error class.
- Candle CPU is a diagnostic baseline only. It is useful for isolating whether a failure comes from a device backend or from shared eager/dispatch/autograd logic, but it is not the acceptance target.
- Differences between Candle backends are acceptable only when they remain within explicit numeric tolerance while preserving PyTorch semantics.

## Architectural Direction

### 1. Contract-First, Family-Oriented Governance

Work should be organized by operator family, not by backend and not by individual failing model scripts.

Each family gets a parity contract that covers the semantic dimensions that matter for training:

- forward numerical result
- output dtype and promotion behavior
- shape, stride, and view semantics where applicable
- broadcast behavior
- autograd gradients
- inplace/view alias and version-counter behavior
- error class and key error semantics

This avoids the failure mode where one backend gets patched for one model case while the same operator family remains inconsistent elsewhere.

### 2. Two-Layer Verification Model

Verification should be split into two layers.

Layer A is the primary gate: family-level parity contracts under `tests/contract/`. These compare Candle behavior directly against real PyTorch for targeted operators and semantic cases.

Layer B is a minimal training smoke gate. This validates that the family contracts actually protect the training path: `forward -> loss -> backward -> optimizer.step`. The smoke gate is not the place to discover detailed operator drift; it exists to prevent regressions in the integrated training loop.

### 3. Shared Parity Harness Instead of One-Off Tests

The contract layer should use a shared parity harness rather than many hand-written backend-specific tests.

The harness should accept:

- operator under test
- sample inputs
- target comparison dimensions
- expected alias/view flags
- per-backend numeric tolerance overrides
- expected exception type when the case is invalid

The harness then runs:

1. real PyTorch on CPU as semantic oracle,
2. Candle on each enabled backend,
3. forward and backward comparisons,
4. optional aliasing/version checks,
5. optional error checks.

This keeps the cost of adding new parity cases low and makes backend drift visible in a uniform format.

## Operator Family Breakdown

### 1. Broadcast and Binary Arithmetic

This is the first family to harden because it underpins linear layers, activations, reductions, and optimizer updates.

Primary semantics to lock down:

- dtype promotion matches PyTorch
- scalar-tensor and tensor-tensor combinations behave identically
- broadcast rules and failure cases match PyTorch
- mixed-device behavior remains explicitly unsupported where Candle does not support it
- inplace updates mutate correctly and preserve version semantics

### 2. Reshape and View

This family is the second priority because many training failures that look like operator bugs are actually alias/view semantic bugs.

Primary semantics:

- `reshape` vs `view` behavior follows PyTorch
- view-producing ops preserve alias relationships
- inplace writes on views obey PyTorch-style restrictions
- version counters change when they should
- backward through views matches PyTorch

### 3. Reduction

Reduction semantics are a common source of drift across backends and dtypes.

Primary semantics:

- integer and bool reduction dtype behavior matches PyTorch
- `dim`, tuple dims, negative dims, and `keepdim` match PyTorch
- numeric tolerances are backend-appropriate but semantically correct
- backward behavior matches PyTorch for differentiable reductions

### 4. Linear and Activation

After arithmetic, views, and reductions are stable, linear and activation operators can be hardened on top of those foundations.

Primary semantics:

- `matmul/mm/addmm/linear` parity
- activation forward parity for common training activations
- activation backward parity
- interaction with broadcast and dtype promotion remains torch-aligned

### 5. Indexing

Indexing sits later in the sequence because it depends heavily on alias rules, broadcast, and shape semantics already being stable.

Primary semantics:

- `getitem/setitem` and simple advanced indexing parity
- gather/scatter/index_select semantics
- masking semantics
- gradient behavior where differentiable

### 6. Optimizer Primitives

This family finalizes the training loop by ensuring optimizer update kernels behave like PyTorch.

Primary semantics:

- inplace arithmetic used by optimizers matches PyTorch
- scalar-mixed updates do not drift in dtype behavior
- parameter mutation and version behavior remain correct
- `zero_grad` and repeated step behavior stay stable

## Execution Order

The order of work should be:

1. `broadcast/binary`
2. `reshape/view`
3. `reduction`
4. `linear/activation`
5. `indexing`
6. `optimizer-primitives`

This ordering is intentional. The first three families are semantic infrastructure for the latter three. If they are unstable, later fixes will be noisy and likely need to be redone.

## Test Strategy

### Contract Layer

Add a new training-core parity contract suite under `tests/contract/`.

Suggested file layout:

- `tests/contract/test_training_core_parity_harness.py`
- `tests/contract/test_training_core_binary_parity.py`
- `tests/contract/test_training_core_view_parity.py`
- `tests/contract/test_training_core_reduction_parity.py`
- `tests/contract/test_training_core_linear_activation_parity.py`
- `tests/contract/test_training_core_indexing_parity.py`
- `tests/contract/test_training_core_optimizer_parity.py`

These files should compare Candle to real PyTorch directly, not just compare Candle backends to each other.

### Minimal Training Gate

Add a compact integration gate that covers a minimal training loop, for example:

- `Linear -> activation -> reduction(loss)`
- backward pass
- simple optimizer step
- repeat for multiple steps

This gate should validate:

- loss remains finite
- gradients are populated where expected
- parameter updates stay numerically aligned with PyTorch within tolerance
- repeated steps do not introduce backend-specific drift

## Failure Classification

Every parity failure should be classified into one of five buckets:

- semantic mismatch
- autograd mismatch
- view/inplace alias mismatch
- numeric tolerance overflow
- backend unsupported or missing implementation

This classification matters because the remediation path is different:

- semantic mismatch often points to shared wrapper or dispatch logic
- autograd mismatch points to backward registration or wrapper logic
- alias mismatch points to view/inplace/version mechanics
- numeric overflow points to backend kernel precision or accumulation behavior
- unsupported points to explicit scope gaps or missing kernels

## Remediation Rules

- Never use CPU fallback to hide a device backend bug.
- If a native device kernel is wrong, fix it or replace it with an on-device composite implementation.
- If a backend limitation is known and temporarily tolerated, record it in `docs/known-kernel-issues.md` with the required fields.
- If the bug is shared across backends, fix the common functional/dispatch/autograd layer before patching backend kernels.

## First-Phase Deliverables

At the end of phase 1, the repository should have:

- a shared training-core PyTorch parity harness,
- family-level parity contracts for the first three families (`broadcast/binary`, `reshape/view`, `reduction`),
- a minimal training smoke gate tied to torch parity expectations,
- a failure-classification workflow for backend parity bugs,
- updated kernel-issue documentation for any accepted temporary backend-specific gaps.

## Acceptance Criteria

This phase is complete only if all of the following are true:

- the targeted training-core parity contracts pass against real PyTorch,
- supported backends differ only within approved numeric tolerances,
- no targeted device-path parity gap is hidden by CPU fallback,
- the minimal training loop gate remains green,
- newly discovered backend limitations are documented explicitly instead of being left implicit.

## Follow-Up After Design Approval

Once implementation starts:

1. create an execution plan with task-level sequencing,
2. use TDD for each operator family,
3. start from failing parity tests against real PyTorch,
4. fix the smallest shared semantic layer possible before backend-specific patches,
5. run targeted contracts plus the training smoke gate before each merge point.
