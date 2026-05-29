# NPU Performance Core Redesign

## Goals

Candle NPU has three hard requirements:

1. Single-operator performance must not be lower than torch_npu.
2. Composite/model performance must be better than torch_npu.
3. Python-visible APIs must remain fully PyTorch compatible.

Performance work is not limited to Cython. C, C++, Rust, generated bindings, Cython glue, or replacement internal architectures are all acceptable when they keep the code clear and improve performance. Existing Cython/Python hot paths may be replaced when they block these goals.

## Non-goals

- Do not change `import candle as torch` semantics to a non-PyTorch lazy framework.
- Do not introduce CPU fallback for NPU correctness or performance.
- Do not preserve existing Cython structure for compatibility with old internals.
- Do not expose graph/fusion APIs as a requirement for user code to get the default fast path.

## Acceptance Gates

### L0: Single-operator parity gate

Every benchmarked NPU operator must be at least as fast as the corresponding torch_npu operator under the same shapes, dtype, synchronization mode, and warmup policy.

Initial L0 coverage:

- elementwise: add, sub, mul, div
- activation: gelu, silu, sigmoid, tanh
- normalization: rms_norm, layer_norm, group_norm
- reduction: sum, mean, softmax, log_softmax
- linear algebra: matmul, bmm, addmm
- memory/layout: contiguous, clone, copy, view-safe reshape paths
- indexing/data: embedding, gather/scatter paths used by transformer workloads
- backward hot paths for the above where torch_npu exposes equivalent behavior

Required output per op:

- Candle median latency
- torch_npu median latency
- p10/p90 or min/max distribution
- sync and no-sync/chained submission modes where meaningful
- kernel count or ACLNN submit count
- allocation count and workspace allocation count
- ratio: Candle / torch_npu

Gate: no L0 operator may remain slower than torch_npu without being marked as a blocking regression.

### L1: Composite block superiority gate

Composite static blocks must outperform torch_npu on equivalent PyTorch code.

Initial L1 coverage:

- FFN block: linear + activation + linear
- attention block: qkv projection, attention score, softmax, value projection
- transformer block: layer norm, attention, residual, FFN, residual
- repeated transformer blocks
- selected elementwise-heavy chains that expose host launch overhead

Required output:

- Candle eager latency
- Candle captured/replayed latency when graph path applies
- torch_npu latency
- Candle eager / torch_npu ratio
- Candle graph / torch_npu ratio
- graph replay speedup over Candle eager
- graph capture overhead and amortization count

Gate: L1 captured/replayed path must be faster than torch_npu while preserving API-visible PyTorch semantics.

### L2: End-to-end model superiority gate

End-to-end model benchmarks must show Candle faster than torch_npu for forward, backward, and total iteration time.

Initial L2 coverage:

- MLP from `benchmarks/perf_candle_vs_torch_npu.py`
- transformer block from `benchmarks/perf_candle_vs_torch_npu.py`
- ResNet-style block from `benchmarks/perf_candle_vs_torch_npu.py`

Required output:

- forward latency
- backward latency
- total forward + backward latency
- Candle eager and graph/fused path where applicable
- torch_npu baseline
- numerical parity checks at benchmark setup time

Gate: composite/model Candle path must be faster than torch_npu. The stretch target remains 1.5x torch_npu performance on the transformer-oriented workloads, but the hard floor is strict superiority for composed workloads.

## Architecture

The Python API remains PyTorch-compatible and is not the performance core. Hot training-loop work should cross into native code quickly and stay there.

```text
Python torch-compatible API
  -> thin dispatch/autograd boundary
  -> native NPU performance core
      -> single-op fast path
      -> graph/fused composite path
  -> ACLNN/ACLRT/HCCL runtime
```

The native performance core should be structured around explicit layers rather than Cython object dispatch:

1. API boundary adapters
2. Tensor/storage/device handles
3. native dispatcher
4. op descriptor/executor cache
5. workspace and allocator pools
6. graph capture/replay/fusion executor
7. autograd scheduling and backward graph integration
8. runtime FFI layer for ACLNN/ACLRT/HCCL

Implementation language is chosen per layer:

- C or generated C is preferred for the thinnest ACLNN/ACLRT FFI and ABI-stable shims.
- C++ is preferred for dispatcher, graph executor, autograd engine, ownership, queues, and caches.
- Rust is acceptable for IR/optimizer components if it improves safety without complicating Python/native integration.
- Cython is acceptable as temporary glue, not as the long-term hot-path object scheduler.

## Single-op Fast Path

The single-op path exists to satisfy the L0 parity gate.

```text
Tensor op
  -> native dispatcher
  -> cached schema/device/dtype/shape decision
  -> cached ACLNN descriptor/executor when safe
  -> pooled output/storage/workspace allocation
  -> ACLNN launch on current stream
  -> minimal Tensor wrapper return
```

Key requirements:

- Avoid Python allocation, Python tuple construction, and repeated dynamic lookup on hot paths.
- Avoid stale pointer bugs in executor caches by keying reusable plans on stable shape/dtype/stride metadata while rebinding tensor addresses at launch time.
- Reuse temporary workspace buffers across compatible calls.
- Make allocator fast paths O(1) under steady state; defer expensive event draining outside the per-allocation critical path.
- Preserve current-stream semantics and stream safety.
- Keep native kernels as normal default paths, not optional flags.

## Composite Graph Path

The composite path exists to satisfy L1/L2 superiority gates.

Candle already has low-level ACL graph capture/replay through CANN `aclmdlRI`. That mechanism records submitted work, but it does not optimize or fuse Candle operations. The redesign should build a higher-level execution layer above it.

```text
PyTorch-compatible eager calls
  -> internal trace/capture of a static block
  -> stable input/output buffer binding
  -> graph replay plan
  -> optional fusion or launch coalescing
  -> default replay on subsequent compatible invocations
```

First target: static shapes and stable control flow. Dynamic shape support can be added later through multiple cached plans or runtime update APIs.

Graph plans must track:

- input tensor identities or storage slots
- output slots
- shapes, strides, dtype, device
- stream and capture mode
- mutation and aliasing constraints
- workspace pool requirements
- autograd participation and backward graph capture status

The graph path must be internal. User code should not have to opt into non-PyTorch semantics to benefit from it.

## Autograd Strategy

The existing Cython autograd engine can remain as a correctness reference during migration, but it should not define the final performance architecture.

Short term:

- Count backward nodes, emitted NPU kernels, allocation count, and Python callback count in L2 benchmarks.
- Remove proven backward recomputation on NPU hot paths, especially sigmoid/tanh/softmax/log_softmax and normalization backward where saved forward results or stats avoid extra kernels.
- Capture forward + backward static blocks where possible.

Medium term:

- Move autograd scheduling, dependency counting, ready queues, and graph task execution into a native engine if Python callback overhead remains measurable after graph replay.
- Keep Python-visible autograd behavior aligned with PyTorch: `grad_fn`, hooks, retain graph, create graph, anomaly behavior, saved tensor hooks, views, and in-place semantics.

Native autograd is a performance tool, not an API change.

## Benchmark Plan

Phase 0 must land before major rewrites.

1. Extend L0 op benchmarks with torch_npu baselines and JSON output.
2. Extend pipeline/static block benchmarks with equivalent torch_npu implementations.
3. Add graph/eager mode fields to benchmark output.
4. Add instrumentation counters:
   - ACLNN submit count
   - graph replay count
   - allocator malloc/free count
   - workspace allocation count
   - host sync count
   - Python backward callback count where available
5. Store benchmark artifacts in a stable machine-readable format so regressions can be diffed.

Benchmarks are not only reporting tools. They are the acceptance gates for PRs that claim NPU performance improvements.

## Migration Plan

1. Establish benchmark gates and current ratios.
2. Rank L0 regressions by model impact and absolute latency.
3. Replace the slowest single-op paths with native fast paths until L0 parity is reached.
4. Build graph/captured callable support for L1 static blocks.
5. Integrate backward capture and remove backward recomputation on hot paths.
6. Reassess whether a native C++ autograd engine is needed for remaining L2 gaps.
7. Delete old Cython/Python hot paths after native replacements are proven by benchmarks and compatibility tests.

## Compatibility and Correctness

Every performance path must preserve PyTorch-visible behavior. Required checks:

- existing CPU/contract tests when shared semantics change
- NPU tests for touched ops
- no CPU fallback contract tests
- torch_npu numerical comparisons in benchmarks for representative inputs
- compatibility tests for affected API areas when API semantics are touched

Optimization must not weaken schema validation, device placement, stream semantics, or no-fallback guarantees.

## Risks

- Graph replay can accidentally assume static aliasing or mutation behavior that PyTorch allows to vary. Mitigation: conservative plan guards and fallback to NPU eager path, not CPU.
- Executor caching can reuse stale tensor addresses. Mitigation: cache plans/descriptors separately from per-call addresses.
- Native rewrites can fragment code. Mitigation: explicit layer boundaries and generated bindings for repetitive ACLNN surfaces.
- Chasing single-op parity can distract from model superiority. Mitigation: rank L0 work by L1/L2 impact and keep graph path as the main model-speed strategy.
