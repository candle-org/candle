# Candle Runtime Rebuild Design

Date: 2026-04-13
Status: Drafted from source-driven analysis of PyTorch + torch_npu + current Candle

## 1. Background and Goal

Candle has reached the point where local fixes are no longer enough. The current implementation mixes Python shell logic, Python runtime ownership, Cython acceleration fragments, and backend-specific execution paths in ways that make long-term parity with `torch + torch_npu` increasingly fragile.

The goal of this design is **not** to copy PyTorch code. The goal is to realign Candle's long-term architecture to the same **mechanism boundaries** as `torch + torch_npu` while preserving Candle's Python package surface and implementing its runtime core in **Cython + backend-native kernels/bindings**.

Target outcome:

- Python remains the public API and semantic shell
- Cython becomes the steady-state runtime owner for hot-path mechanisms that PyTorch implements in C++
- backend execution substrates remain backend-specific, but integrate under a unified owner model
- Candle's runtime is rebuilt in bounded phases rather than via one-shot repo-wide rewrite

## 2. Analysis Baseline

This design is based on direct source inspection of:

- PyTorch runtime core (`c10/core`, `aten/src/ATen/core`, `torch/csrc/autograd`, `torch/utils/_python_dispatch.py`, `torch/overrides.py`, `torch/_subclasses/fake_tensor.py`)
- torch_npu runtime and extension layers (`torch_npu/csrc`, `torch_npu/npu`, `third_party/op-plugin/op_plugin/utils`)
- current Candle implementation (`src/candle/_tensor.py`, `src/candle/_dispatch/dispatcher.py`, `src/candle/_backends/npu/aclnn.py`, `src/candle/_cython/_aclnn_ffi.pyx`)

The design therefore uses **source-driven ownership alignment**, not guesses or API-only comparison.

## 3. Mechanism Map

### 3.1 Object runtime layer

PyTorch centers tensor runtime ownership in `TensorImpl`, `StorageImpl`, dispatch key state, and autograd attachment points. `StorageImpl` owns the underlying backing buffer and allocator relationships; `TensorImpl` owns tensor identity, metadata, and dispatch state; autograd metadata is attached as runtime state, not as Python-owned side tables.

For Candle, the corresponding long-term runtime owner must be Cython.

### 3.2 Device runtime layer

PyTorch/torch_npu place allocator, stream/event, and lifetime bookkeeping in native runtime layers. Python wraps these concepts but does not own them. torch_npu follows this pattern: Python `Stream`/`Event` are wrappers around `_C` base classes; allocator and memory bookkeeping live in native code.

For Candle, device runtime bookkeeping must move to Cython/backend-native owners.

### 3.3 Dispatcher/operator layer

PyTorch uses schema, dispatch keys, runtime operator tables, boxed/unboxed boundaries, and redispatch as a first-class execution model. Schema alias/mutation/out semantics are runtime facts, not loose conventions. Candle already has dispatcher and keyset concepts, but too much runtime logic still lives in Python.

For Candle, the dispatcher shell may remain Python, but the runtime core must move to Cython.

### 3.4 Autograd layer

PyTorch autograd is a runtime graph system: `AutogradMeta`, `Node`, `Edge(input_nr)`, `SavedVariable`, `output_nr`, `mark_dirty`, version checks, and forward-AD metadata all exist as core runtime mechanisms. They are not optional extensions.

For Candle, autograd correctness-critical state must converge into Cython runtime ownership.

### 3.5 Override / fake / functionalization layer

PyTorch distinguishes API-level override (`__torch_function__`) from dispatcher-level override (`__torch_dispatch__`), and treats fake/meta/functionalization as legitimate architecture layers rather than test-only helpers. Candle does not need to fully implement these immediately, but the rebuilt runtime must preserve room for them.

### 3.6 torch_npu extension layer

torch_npu does not leave NPU execution ownership in Python. Python provides shell/configuration/context surfaces, while native code owns bindings, allocator, stream/event, command execution, and op-plugin substrate. This is the direct reference for Candle's NPU target state.

## 4. Current Candle Architectural Drift

### 4.1 Tensor/Storage drift

Candle already has a Cython `TensorImpl`, but `_tensor.py` still owns too much runtime-critical logic, including metadata propagation and some mutation-sensitive behavior. This leaves object ownership split across Python and Cython.

### 4.2 Autograd drift

Candle has substantial autograd code in Cython, but key correctness mechanisms remain fragmented across tensor helpers, dispatcher behavior, and autograd runtime components. This makes version/view/output-slot semantics harder to reason about and evolve.

### 4.3 Dispatcher drift

Candle has keyset and pending-op concepts, but the dispatcher still performs too much runtime work in Python. This diverges from the role the dispatcher plays in PyTorch.

### 4.4 NPU drift

`src/candle/_backends/npu/aclnn.py` is still a Python execution owner. `_aclnn_ffi.pyx` accelerates primitives, but the owner structure remains Python-centered. This is the clearest mismatch with torch_npu.

### 4.5 Automation drift risk

Without strong automation constraints, Claude Code will tend to:

- expand scope across multiple subsystems in one batch
- preserve temporary bridges too long
- reintroduce Python owners because they are easier to patch
- mix architectural cleanup with feature work

This design therefore includes a governance model, not just an implementation model.

## 5. Target Architecture

## 5.1 Python shell responsibilities

Python may continue to own:

- public API entrypoints
- Tensor method surface
- functional API shell
- user-facing argument normalization
- high-level context managers and flags
- strategy selection where appropriate
- high-level glue for override/fake/meta layers

Python must **not** own steady-state runtime lifecycle for tensor metadata, storage ownership, autograd graph state, dispatcher hot path, or NPU execution infrastructure.

### 5.2 Cython runtime core responsibilities

Cython becomes the steady-state owner for:

- tensor identity and metadata runtime
- storage ownership core and typed storage mechanics
- version / alias / view relationships
- dispatch key state and dispatcher hot path machinery
- autograd metadata, edge/node/output-slot storage, and backward runtime spine
- backend bridge entrypoints
- device-lifetime bookkeeping needed on hot paths
- NPU execution ownership backbone

### 5.3 Backend/native substrate responsibilities

Backend-specific layers continue to own:

- CPU kernels
- MPS/CUDA/NPU runtime APIs
- hardware allocator/event/stream specifics
- NPU op_api/aclnn actual execution primitives

But these backend layers integrate under Cython-owned runtime boundaries, not Python-owned ones.

## 6. Ownership Decision Matrix

### Must converge to Cython ownership

- Tensor identity / shape / stride / offset / dtype / device runtime state
- Storage / typed storage core ownership
- version counters, alias/view runtime relationships
- autograd metadata attachment and graph metadata
- saved tensor version/output-slot metadata
- dispatcher hot path, keyset synthesis, redispatch core
- NPU bindings / init-finish owner
- NPU descriptor / workspace / executor / queue / PTA owner
- device-lifetime bookkeeping such as record-stream-style mechanisms

### May remain in Python shell form

- top-level API entrypoints
- Tensor method wrappers
- user-facing functional shells
- user-facing context managers and flags
- high-level override/fake/meta orchestration
- testing and diagnostics glue

### Forbidden long-term structures

- Python execution owner for NPU (`aclnn.py`-style long-term center)
- Python-only dispatcher hot path as steady-state design
- Python-owned runtime alias/version/view core
- long-lived dual owner layers after migration is complete
- new `*_bridge.py` files that become permanent owner layers

## 7. Subsystem Breakdown

### S1. Tensor / Storage runtime core

Foundational object model: tensor identity, storage ownership, version, alias, view metadata, device/dtype metadata.

### S2. Autograd runtime core

Autograd metadata, graph model, saved tensors, output slots, version checks, forward-AD core.

### S3. Dispatcher / operator system

Schema semantics, keyset routing, redispatch, operator core, functional shell boundary.

### S4. NPU execution backbone

Bindings, descriptors, workspace, executor, queue, PTA, custom kernel and fallback integration.

### S5. Advanced extension layers

Override, fake/meta, functionalization, compile/export-friendly boundaries.

## 8. Dependency Order

The required rebuild order is:

1. S1 Tensor/Storage
2. S2 Autograd
3. S3 Dispatcher/operator
4. S4 NPU backbone
5. S5 Advanced layers

This order is mandatory because:

- autograd depends on tensor/storage runtime truth
- dispatcher correctness depends on tensor/runtime semantics and partially on autograd semantics
- NPU execution backbone consumes dispatcher and tensor/runtime boundaries
- advanced layers depend on all earlier runtime boundaries being coherent

## 9. Rebuild Program

### Phase 0
Source-driven mechanism mapping and architecture decisions.

### Phase 1
Tensor/Storage runtime convergence.

### Phase 2
Autograd runtime convergence.

### Phase 3
Dispatcher/operator convergence.

### Phase 4
NPU execution backbone rebuild and deletion of Python execution ownership.

### Phase 5
Advanced extension compatibility and future-proofing.

Each phase must be executed in bounded batches, with a single primary subsystem per batch.

## 10. Claude Code Governance Model

Long-term automated development must be constrained by executable guardrails, not only prose instructions.

### Required governance principles

- default to planning, not direct unrestricted implementation
- each batch must declare subsystem, allowed files, forbidden files, validations, and exit criteria
- edits must be blocked outside the current batch whitelist
- Python execution owner expansion must be blocked
- scope expansion across multiple runtime subsystems in one batch must be blocked
- pre-compact and stop hooks must preserve structured state handoff

### Required configuration direction

A project-level `.claude/settings.json` should eventually enforce:

- planning-first permission mode
- destructive Bash restrictions
- edit-time whitelist enforcement per approved batch
- post-edit architecture drift checks
- bounded validation per batch
- compaction/stop state capture

## 11. Immediate First Step

The first implementation batch must be **Tensor/Storage A1**, which defines runtime ownership boundaries without attempting broad behavioral migration. This is the minimum safe starting point for the rebuild program.

## 12. Success Criteria

This design is successful only when:

1. Python is reduced to shell-level responsibilities for runtime-critical systems
2. Cython becomes the steady-state runtime owner for the core mechanisms listed above
3. each subsystem is rebuilt in dependency order without long-lived dual ownership
4. the NPU execution substrate is no longer Python-owned
5. Claude Code can execute bounded batches under enforced automation constraints without architectural drift
