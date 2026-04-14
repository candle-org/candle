# Whole Storage Mechanism Realignment Master Design

Date: 2026-04-14
Status: Drafted after storage mechanism re-scope
Parent spec: `docs/superpowers/specs/2026-04-13-candle-runtime-rebuild-design.md`
Replaces narrower storage master direction: `docs/superpowers/specs/2026-04-14-storage-runtime-realignment-master-design.md`

## 1. Master Spec Definition

- **Subsystem:** S1 Tensor / Storage runtime core
- **Primary focus:** whole Storage mechanism
- **Spec type:** master spec, not a single implementation batch
- **Purpose:** define Candle’s full Storage alignment target across Python surface, runtime owner model, sharing/serialization behavior, and built-in device storage specialization for CPU / CUDA / NPU

This spec is intentionally broader than the earlier storage-surface-first direction. It exists because the user wants the **whole storage mechanism** aligned, not only `_storage.py` cleanup, and explicitly wants NPU storage included in the target architecture from the start.

This document does **not** require implementation in one uninterrupted batch. It requires that the final target be written down in one coherent place before more storage work continues.

## 2. Why This Spec Exists

A1-A3 improved Tensor / Storage convergence, especially on the Tensor side:

- A1 defined Tensor / Storage ownership boundaries
- A2 moved important version-sensitive and view-sensitive Tensor runtime truth into `TensorImpl`
- A3 moved `Tensor.data` runtime truth onto the Cython-backed runtime path

But the next structural problem is larger than a single residual Tensor path.

The project still lacks a fully specified answer to:

- what Candle’s Storage Python surface should look like
- what Candle’s real Storage owner model should be
- how sharing / serialization / multiprocessing should attach to Storage ownership
- how CPU / CUDA / NPU storage should fit into one built-in Candle storage architecture

Without that, further work risks drifting into:

- Python-shell-heavy storage logic
- CPU-first designs that later treat CUDA/NPU as special exceptions
- NPU designs that mimic torch_npu plugin layering instead of a built-in Candle storage model
- Tensor follow-up work built on unstable storage boundaries

This spec exists to stop that drift.

## 3. Source Alignment Basis

### 3.1 `torch/storage.py` — Python storage surface reference

PyTorch’s `torch/storage.py` is the primary reference for Candle’s Python storage surface.

It establishes the public Python-visible storage layer:

- `_StorageBase`
- `UntypedStorage`
- `TypedStorage`
- Python-visible repr / conversion / wrapper behavior
- public methods such as `cpu()`, `cuda()`, `share_memory_()`, `to()`, `clone()`, and typed/untyped accessors

The key lesson is not that Candle must copy every exact symbol or implementation detail. The key lesson is that the Python storage file has a **constrained public-shell role**.

It is not the final owner of storage runtime truth.

### 3.2 `c10/core/StorageImpl.h` — runtime owner model reference

PyTorch’s actual storage owner model is defined at the `StorageImpl` layer.

That layer directly owns or defines:

- backing `DataPtr`
- allocator attachment
- device type / device placement
- `nbytes`
- resizable state
- mutable pointer access rules
- storage identity semantics

The key architectural lesson is:

> aligning to PyTorch storage requires more than matching `torch/storage.py`; it requires a runtime owner model that is structurally compatible with `StorageImpl`-style truth.

### 3.3 PyTorch sharing / serialization / reductions paths

PyTorch storage behavior is not fully captured by Python classes alone.

Storage sharing, multiprocessing reductions, rebuild paths, and serialization all rely on storage identity and owner metadata. That means Candle’s storage target cannot be correct unless it also defines how Storage participates in:

- `share_memory_`
- multiprocessing reductions / rebuild
- serialization / checkpoint rebuild
- typed/untyped storage reconstruction

### 3.4 torch_npu — runtime ownership direction only

torch_npu is a secondary reference for **runtime ownership direction**, not for overall architecture.

The important lesson from torch_npu is that NPU storage is still built on a storage-owner concept (`NPUStorageImpl : StorageImpl`) rather than being an unrelated external data wrapper.

That means the right Candle conclusion is:

- NPU storage must live inside Candle’s own storage architecture
- torch_npu behavior and native ownership are useful references
- torch_npu’s plugin-style position is **not** Candle’s target architecture

### 3.5 Architectural rule for Candle

Candle’s long-term rule is:

- CPU / CUDA / NPU storage all belong to one Candle-owned storage mechanism
- device-specific behavior may differ by backend/runtime implementation
- but they must fit the same owner model and surface boundary story
- NPU storage is a built-in device specialization, not a plugin-owned side channel

## 4. Problem Statement

The storage problem is not simply that `_storage.py` contains too much code. The problem has four coupled layers.

### 4.1 Python-surface drift

`src/candle/_storage.py` currently mixes together:

- public storage API surface
- lifecycle helpers
- sharing bookkeeping
- pinned host lifetime behavior
- device-specific storage details
- typed/untyped runtime glue

That is broader than the role PyTorch gives `torch/storage.py`.

### 4.2 Runtime-owner drift

Candle does not yet have a fully clarified storage owner model equivalent in role to `StorageImpl`.

That means there is not yet one stable fact source for:

- pointer ownership
- allocation lifetime
- resizable behavior
- device placement truth
- storage identity semantics
- typed/untyped backing linkage

### 4.3 device-architecture drift

Without a full mechanism spec, storage design can drift into a fragmented model:

- CPU storage treated as the “real” storage path
- CUDA storage treated as a special backend path
- NPU storage treated as a plugin-like or add-on path

That is specifically not the target architecture.

### 4.4 Tensor-coupling drift

Further Tensor convergence depends on Storage boundaries being stable.

If Storage architecture remains underspecified, later Tensor shell reduction will keep inheriting unstable assumptions about ownership and rebuild semantics.

## 5. Scope

### 5.1 In scope

This master spec defines the target architecture for the whole Storage mechanism, including:

- Python storage public surface responsibilities
- runtime storage owner model
- typed/untyped storage role boundaries
- pointer / lifetime / allocator / device-placement truth
- storage sharing / multiprocessing / serialization participation
- built-in device storage unification across CPU / CUDA / NPU
- the dependency relationship between Storage convergence and later Tensor shell reduction
- decomposition into later implementation batches

### 5.2 Out of scope

This master spec does not itself implement or redesign:

- dispatcher/schema behavior
- autograd graph/runtime
- backend kernel implementations unrelated to storage architecture
- the full later Tensor shell convergence work

This document defines the whole storage target and the downstream handoff, but remains a Storage-first architecture spec.

## 6. Target Architecture

### 6.1 Python storage surface target

`src/candle/_storage.py` should converge toward the role played by `torch/storage.py`.

That means it may continue to provide:

- public storage classes and user-facing entry points
- Python-visible repr / convenience methods
- public wrapper methods genuinely part of the storage API
- thin wrappers around runtime-owned storage state

It should stop being the place where long-term storage owner truth is established.

### 6.2 Runtime storage owner target

Candle needs a stable storage runtime owner layer that plays the same architectural role that `StorageImpl` plays in PyTorch.

That owner layer should carry or define the actual truth for:

- backing pointer ownership
- allocation lifetime
- allocator attachment
- `nbytes`
- resizable state
- device placement
- storage identity semantics
- typed/untyped linkage truth
- share / rebuild relevant owner metadata

Python storage wrappers may expose that behavior, but should not remain the primary fact source for it.

### 6.3 Typed / untyped storage target

Candle should preserve a user-visible distinction between typed and untyped storage only where that distinction is meaningfully part of the PyTorch-compatible API.

But the runtime truth for typed/untyped linkage should not rely on Python wrappers inventing or maintaining that ownership relationship.

The target is:

- Python wrappers expose the public distinction
- runtime owners define the actual backing relationship

### 6.4 Sharing / serialization target

Storage sharing and storage serialization must be treated as first-class parts of the storage mechanism.

That means Candle’s storage architecture must define how storage owner truth participates in:

- `share_memory_`
- shared-memory bookkeeping
- multiprocessing reductions / rebuild
- serialization / checkpoint rebuild
- typed storage reconstruction from untyped backing

These behaviors must be built on the same storage owner model, not on a separate ad hoc Python-only layer.

### 6.5 Built-in device storage target

Candle’s storage architecture should converge on one rule across device types:

- CPU storage is a built-in storage specialization
- CUDA storage is a built-in storage specialization
- NPU storage is a built-in storage specialization

This does **not** mean the implementations are identical.

It means all three belong to the same storage mechanism and owner model.

NPU is therefore not a plugin-owned storage architecture. It is a Candle-owned storage specialization that uses NPU-specific runtime behavior internally.

## 7. Required Boundary Rules

### 7.1 Rule: Python exposes, runtime owns

Whenever storage behavior is runtime-sensitive, the default rule should be:

> Python exposes the API; runtime/Cython owns the truth.

### 7.2 Rule: no new owner truth in `_storage.py`

Future storage work must not add new lifecycle-sensitive owner logic to `_storage.py` unless PyTorch clearly keeps equivalent logic at the Python storage layer.

### 7.3 Rule: one storage architecture, many device specializations

Candle must not evolve separate storage architectures for CPU / CUDA / NPU.

It may evolve device-specialized implementations, but they must belong to one unified built-in storage mechanism.

### 7.4 Rule: NPU is built-in, not plugin storage

Future NPU storage work must not introduce or preserve a plugin-owned architectural split.

Using torch_npu as a runtime behavior and ownership reference is allowed.

Using torch_npu’s plugin position as Candle’s architecture target is not allowed.

### 7.5 Rule: Tensor follows Storage

Further Tensor shell reduction should be built on a stabilized storage architecture.

Storage boundary work is therefore upstream of later Tensor residual batches.

## 8. Decomposition Into Future Implementation Batches

This master spec is intended to feed multiple implementation batches.

### 8.1 Batch B1: Python storage surface alignment

Purpose:

- align `src/candle/_storage.py` more closely to `torch/storage.py`
- reduce Python-shell lifecycle and bookkeeping exposure
- preserve only the public Python storage surface that Candle should continue exposing

### 8.2 Batch B2: storage owner core alignment

Purpose:

- define and migrate toward a clearer Candle storage owner model aligned in role to `StorageImpl`
- stabilize pointer / lifetime / allocator / resizable / device truth
- reduce Python-layer owner truth around typed/untyped backing

### 8.3 Batch B3: sharing / serialization / reduction alignment

Purpose:

- unify storage sharing and rebuild behavior under the storage owner model
- align multiprocessing reductions and serialization rebuild paths with the new owner truth
- eliminate ad hoc Python-only storage lifecycle assumptions

### 8.4 Batch B4: built-in device storage unification

Purpose:

- make the Candle-owned storage model explicitly uniform across CPU / CUDA / NPU
- ensure NPU storage follows the same built-in storage architecture class as CUDA storage
- push any remaining plugin-style drift out of the storage layer

### 8.5 Tensor handoff batch

Once the first storage batches land, later Tensor residual work can continue shrinking `_tensor.py` and related shell-level logic on top of the stabilized storage boundary.

## 9. Relationship to Tensor Work

This spec does not make Tensor the implementation target of the next batch.

Instead, it establishes the ordering rule:

1. define the whole storage target first
2. land storage batches on top of that target
3. continue Tensor shell convergence after Storage architecture stabilizes

So the relationship is explicit:

- Storage-first in architecture
- Tensor-next in residual shell reduction

## 10. Testing Strategy

### 10.1 General principle

Use tests to lock both:

- the public Python storage surface
- the owner-boundary rules that prevent runtime truth from drifting back into Python shell code
- the cross-path correctness of sharing / serialization / rebuild behavior

### 10.2 Primary rails

The most relevant existing rails are:

- `tests/contract/test_storage_contract.py`
- `tests/contract/test_tensor_storage_owner_contract.py`
- multiprocessing / serialization tests that exercise storage rebuild behavior

### 10.3 Additional rails allowed

Future storage batches may add focused contract tests only when current tests do not adequately capture:

- Python public surface alignment
- typed/untyped boundary behavior
- owner-model invariants
- serialization/rebuild invariants
- device-storage specialization invariants

### 10.4 NPU confidence rails

NPU-related storage verification should remain architecture-focused:

- validate built-in storage interface/owner expectations
- validate rebuild/share behavior where relevant
- avoid turning storage batches into broad backend-kernel campaigns

## 11. Success Criteria for the Master Spec

This whole-storage line of work is successful only when:

1. Candle has an explicit storage target aligned to `torch/storage.py` at the Python surface level
2. Candle has a runtime storage owner model aligned in role to `StorageImpl`
3. storage sharing / serialization / multiprocessing behavior are built on that owner model
4. Python storage shell code is no longer the long-term fact source for runtime-sensitive storage truth
5. Candle has a clear built-in storage architecture spanning CPU / CUDA / NPU
6. NPU storage is aligned as first-class Candle storage rather than plugin-owned storage
7. later Tensor shell reduction can proceed on top of a stabilized Storage architecture

## 12. Final Design Summary

This v2 master spec defines Candle’s complete Storage mechanism target in one place.

- `torch/storage.py` is the primary reference for the Python storage surface
- `StorageImpl` is the primary reference for the runtime owner model
- torch sharing / serialization paths are part of the storage architecture target, not side details
- torch_npu is a secondary reference for runtime ownership direction only
- Candle’s long-term rule is one built-in storage mechanism across CPU / CUDA / NPU
- later Tensor shell reduction should follow that Storage stabilization
