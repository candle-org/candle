# Candle v2 Greenfield Runtime Design

## Goal

Build a new Candle v2 runtime that is free to replace the current Tensor, dispatcher, autograd, and NPU backend internals. The old Candle codebase remains a compatibility reference and migration source, but v2 is a new implementation path whose hard performance goals are:

1. NPU single-operator latency is at least as fast as torch_npu for every benchmarked L0 operator.
2. NPU composite/model latency is faster than torch_npu for targeted transformer workloads.
3. Python-visible APIs converge back to PyTorch compatibility after the v2 hot path proves the performance model.

This is not another incremental Cython wrapper tranche. It is a new runtime core designed around NPU execution efficiency, graph/fusion defaults, and C++ ownership of hot-path metadata and scheduling.

## Non-goals

- Do not preserve the current Python/Cython dispatcher structure when it conflicts with performance.
- Do not require users to opt into graph or fusion APIs for the default fast path.
- Do not add CPU fallback for NPU correctness or performance.
- Do not attempt full PyTorch API coverage in the first implementation phase.
- Do not delete the existing `src/candle/` runtime until v2 has replacement coverage and measured superiority.

## Architecture Overview

Candle v2 introduces a parallel implementation under a new namespace while the old runtime remains intact:

```text
src/candle2/
  __init__.py
  tensor.py                  # Thin Python API shell over native TensorImpl
  functional.py              # PyTorch-like functional wrappers
  nn/                        # Minimal modules that call v2 dispatcher
  _runtime/
    bindings/                # pybind11/nanobind module entry points
    core/                    # TensorImpl, Storage, dtype, device, errors
    dispatcher/              # C++ operator registry and schemas
    autograd/                # C++ autograd graph/tape engine
    npu/                     # ACL/ACLNN runtime, allocator, kernels, plans
    graph/                   # Capture, fusion, replay, shape-specialized executors
    codegen/                 # Optional generated op/schema/kernel tables
```

The old `src/candle/` package is not removed in this phase. It can later dispatch selected paths to v2 behind an environment gate such as `CANDLE_RUNTIME=v2`, then eventually make v2 the default after compatibility and performance gates pass.

## Implementation Language

The runtime core should be C++17 or C++20 exposed to Python through pybind11 or nanobind.

Rationale:

- CANN/ACL/ACLNN are C ABI libraries, so C++ can bind them directly without ctypes/Cython indirection.
- Tensor metadata, storage ownership, descriptor caches, workspace pools, and stream/event lifetimes need deterministic native ownership.
- pybind/nanobind gives a thin Python shell while keeping eager execution hot paths in native code.
- Cython may remain as compatibility glue, but v2 should not depend on Cython for core performance architecture.

Rust is not rejected globally, but it is not the recommended first runtime language because the CANN ABI, existing build context, and Python extension integration are more direct in C++.

## Tensor and Storage Model

The v2 Tensor is a thin Python handle over a native TensorImpl:

```cpp
struct TensorImpl {
  Storage storage;
  DType dtype;
  Device device;
  SmallVector<int64_t, 6> sizes;
  SmallVector<int64_t, 6> strides;
  int64_t storage_offset;
  Layout layout;
  AutogradMeta* autograd;
};
```

Requirements:

- View operations are metadata-first and do not copy data.
- Storage offset is part of the native descriptor model, not a Python-side afterthought.
- Kernel launch descriptors always receive the correct effective pointer/offset semantics.
- Contiguous materialization is explicit and measurable, not hidden inside Python composites.
- Tensor birth, storage wrapping, view creation, dtype/device metadata, and version counters are native-owned.

The initial Tensor surface only needs methods required by the Phase 1 hot path: shape, stride, dtype, device, storage_offset, contiguous, reshape, view/slice, transpose/permute where needed, arithmetic, matmul, softmax, rms_norm/layer_norm, and minimal `.cpu()`/debug conversion for tests.

## Dispatcher Model

The v2 dispatcher is a native registry:

```text
(op name, dispatch key, dtype/layout constraints, autograd key) -> kernel function pointer
```

Requirements:

- Python wrappers call into the native dispatcher once; dispatch does not bounce back into Python.
- Schema metadata lives in native tables or generated C++ tables.
- Runtime validation remains intentional, but fast-path checks are native and shape-specialized where possible.
- Composite kernels may exist, but they expand in C++ or graph IR, never as Python operator chains on NPU.
- Dispatcher calls can be captured into graph/fusion plans without reconstructing Python call stacks.

## NPU Backend Model

The NPU backend is divided into three layers:

1. **NpuKernel** — a direct wrapper around one ACLNN kernel or small native primitive.
2. **NpuPlan** — a shape/dtype/stride-specialized plan that owns descriptors, executor lookup/cache keys, workspace requirements, and scalar constants.
3. **NpuGraphExecutor** — a multi-op static-shape executor for fused or captured workloads.

Required backend services:

- ACL runtime/context initialization.
- Stream-local allocator and event management.
- Workspace pool keyed by stream and size class.
- Descriptor/executor cache that never reuses stale tensor addresses incorrectly.
- Scalar tensor cache and scalar descriptor cache.
- Accurate effective pointer/storage-offset handling.
- Kernel submission counters, allocation counters, workspace allocation counters, and timing hooks for performance gates.

Native ACLNN kernels remain preferred for L0 when they match or beat torch_npu. When ACLNN has a correctness bug, v2 may use an on-device composite or a fused custom path, but the workaround must remain on NPU and be documented.

## Graph and Fusion Model

Composite/model performance must not rely on Python eager chains. v2 treats graph/fusion as a default optimization path, not an optional user API.

Initial graph targets:

- Fused RoPE: avoid `slice -> contiguous -> neg -> cat -> mul -> add` eager chains.
- Fused MLP: linear + activation + linear, with forward and backward variants.
- Attention block: qkv projection, RoPE, scores, softmax, value projection.
- Transformer block: norm, attention, residual, FFN, residual.

Graph execution requirements:

- Static-shape capture and replay for transformer hot paths.
- Shape-specialized plan cache with safe invalidation.
- Tensor address rebinding at replay time, avoiding stale bound descriptors.
- No Python work inside replay.
- Fallback to eager native kernels when a graph pattern does not match.

## Autograd Model

The v2 autograd engine is native:

- TensorImpl owns or references AutogradMeta.
- Backward graph uses native Node and Edge structures.
- Saved tensors preserve storage/view metadata correctly.
- View backward, broadcast/reduce backward, matmul backward, norm backward, and elementwise backward run through the native dispatcher.
- Training hot paths can be graph-captured or explicitly fused after forward parity is proven.

Initial autograd coverage should focus on transformer training-critical paths instead of full PyTorch autograd breadth.

## Migration Strategy

### Phase 0 — Native runtime skeleton

Deliverables:

- Build system for a native v2 extension.
- `candle2.Tensor` backed by native TensorImpl.
- Minimal CPU backend only for smoke/correctness tests.
- NPU runtime initialization, stream, allocator, storage, and `.cpu()` debug copy.

Acceptance:

- Can allocate an NPU tensor, inspect metadata, and copy small debug tensors to CPU.
- No old Candle dispatcher is used inside v2 hot paths.

### Phase 1 — L0 NPU transformer hot path

Implement the minimum L0 operator set needed for transformer inference/training benchmarks:

- Creation: empty, zeros, ones, randn where needed.
- Views/layout: slice, reshape, transpose/permute, contiguous, clone/copy.
- Elementwise: add, sub, mul, div, neg, gelu, silu, sigmoid/tanh as needed.
- Linear algebra: matmul, bmm, addmm.
- Reductions/normalization: softmax, log_softmax, rms_norm, layer_norm, sum/mean as needed.
- Data: embedding and gather/scatter paths used by transformer workloads.

Acceptance:

- Each benchmarked L0 op has Candle v2 / torch_npu median ratio <= 1.0.
- Reports include latency distribution, allocation count, workspace count, and submit count.
- Any slower op blocks phase completion.

### Phase 2 — Default graph/fusion superiority

Deliverables:

- Fused RoPE default path.
- Fused MLP default path.
- Attention block graph executor.
- Transformer block graph executor.

Acceptance:

- L1 composite blocks outperform torch_npu equivalent PyTorch code.
- L2 selected model benchmarks outperform torch_npu.
- Users do not need to call a special graph API for the default fast path.

### Phase 3 — Native autograd hot path

Deliverables:

- Native autograd engine for Phase 1/2 ops.
- Forward/backward graph replay where profitable.
- Training benchmarks for MLP and attention blocks.

Acceptance:

- Backward hot paths are at least torch_npu parity for L0 and faster for targeted L1/L2 blocks.

### Phase 4 — Compatibility bridge

Deliverables:

- Optional route from old `candle` API into v2 for covered ops.
- Environment gate such as `CANDLE_RUNTIME=v2`.
- Compatibility tests that compare old Candle/PyTorch/v2 behavior.

Acceptance:

- Covered user-facing APIs preserve PyTorch-compatible behavior.
- v2 becomes default only after performance and compatibility gates pass.

## Validation Gates

Every phase must produce machine-readable benchmark artifacts.

Required gates:

- L0 single-op gate: v2 median <= torch_npu median for every covered op.
- L1 composite gate: v2 median < torch_npu median for covered blocks.
- L2 model gate: v2 total latency < torch_npu total latency for selected models.
- Correctness gate: focused numerical parity tests for each covered op/block.
- No CPU fallback gate: tests and instrumentation confirm NPU tensors do not route through CPU computation.
- Regression gate: allocation count, workspace allocation count, and kernel submit count do not regress silently.

## Existing Work Handling

The current `npu-l0-parity-tranche` work should be treated as diagnostic input, not as the final architecture. Its benchmark gates, known kernel issue notes, and profiling data remain useful. Its incremental fallback patches should not become the long-term strategy if they increase eager kernel count or materialization overhead.

Pending uncommitted fixes in that branch should either be:

1. Committed only as short-lived correctness unblockers for current benchmark visibility, or
2. Abandoned in favor of v2 once the implementation plan begins.

Do not mix the greenfield runtime implementation with broad old-runtime cleanup in the same PR.

## First Implementation Slice

The first implementation plan should target Phase 0 plus a tiny Phase 1 vertical slice:

1. Native extension skeleton.
2. Native TensorImpl and NPU Storage.
3. NPU empty/ones creation.
4. NPU add and mul via ACLNN.
5. Benchmark comparing v2 add/mul against torch_npu with allocation/workspace counters.

This slice proves the new architecture can bypass old dispatcher overhead before adding more operators.
