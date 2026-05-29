# NPU L0 Operator Parity Batch 1 Design

## Goal

Bring Candle NPU's first tranche of high-impact single-operator benchmarks to parity with torch_npu, while preserving full PyTorch-compatible API behavior and NPU no-CPU-fallback guarantees.

This phase targets L0 parity first. L1/L2 model benchmarks remain observation gates in this phase; model-level superiority is deferred to the later graph/fusion phase after the underlying op latencies are no longer the dominant bottleneck.

## Current baseline

Baseline was collected on this machine using:

- Candle env: `/home/jenkins/anaconda3/envs/candle311`
- torch_npu env: `/home/jenkins/anaconda3/envs/torchnpu311`
- CANN env: `/usr/local/Ascend/cann-9.0.0/set_env.sh`

Representative L0 fp16 inference gaps:

| Operator | Current Candle / torch_npu |
|---|---:|
| rms_norm | ~11241x slower |
| mul | ~3.82x slower |
| silu | ~2.43x slower |
| add | ~2.43x slower |
| matmul_qkv | ~2.02x slower |
| softmax | ~1.16x slower |
| rope | Candle error: `GetWorkspaceSize failed: 561103 [op=neg, device=npu]` |
| cross_entropy | Candle error: `NPU mul requires matching dtypes` |

Representative L1/L2 gaps:

- L1 A2s: ~12.78x slower
- L1 B1s: ~10.72x slower
- L1 A1: ~8.45x slower
- L2 xfmr total: ~12.17x slower
- L2 mlp total: ~9.91x slower

## Scope

### In scope

1. Ensure L0 benchmarks measure the intended native Candle NPU operator paths rather than accidental Python composites.
2. Add or repair native NPU routing for the first high-impact tranche:
   - `rms_norm`
   - `add`
   - `mul`
   - `silu`
   - `matmul_qkv` / bmm attention cases
   - `softmax`
3. Fix correctness errors blocking L0 gate completeness:
   - `rope`
   - `cross_entropy`
4. Use benchmark gates as acceptance checks for each operator batch.
5. Keep every optimization as a default path, not an optional fast flag.

### Out of scope

- Full graph/fusion replay for L1/L2 composed workloads.
- C++ autograd engine replacement.
- Broad dispatcher rewrite.
- CPU fallback for NPU limitations.
- API shortcuts that diverge from PyTorch semantics.

## Architecture

Phase 1 keeps the Python API stable and attacks the NPU hot path from closest-to-benchmark to lowest-level implementation.

```text
L0 benchmark case
  -> torch-compatible Candle API
  -> dispatch/autograd wrapper
  -> NPU backend op
  -> _C/_npu_ops.pyx fast path or ACLNN binding
  -> ACLNN/ACLRT launch
```

Each operator fix must identify which layer is the bottleneck:

1. Benchmark case is measuring the wrong thing.
2. Public API is not routing to the native op.
3. NPU backend is falling back to a Python composite.
4. `_C/_npu_ops.pyx` fast path exists but has avoidable host overhead.
5. ACLNN binding itself is missing, broken, or using an unfavorable call shape.
6. Allocator/workspace/tensor-wrapper overhead dominates the operator.

Only after the failing layer is identified should implementation changes be made.

## Batch 1A: rms_norm native route

The current `rms_norm` benchmark uses a Python expression:

```python
x * torch_mod.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
```

That measures a composite chain, not Candle's native NPU `rms_norm` path. Since Candle already has ACLNN `rms_norm` / `rms_norm_grad` bindings, the first task is to align benchmark and API routing.

Required behavior:

- Add an L0 benchmark case that calls the same public Candle API path users should call for RMSNorm.
- Ensure the Candle path reaches the NPU native `rms_norm` backend instead of expanding into `pow + mean + rsqrt + mul`.
- Ensure torch_npu baseline uses equivalent PyTorch/torch_npu behavior.
- Preserve the composite expression only as a separate benchmark if needed, named clearly as a composite chain.

Acceptance:

- Native `rms_norm` benchmark runs without error.
- Its L0 ratio is reported separately from the composite chain.
- If native Candle is still slower than torch_npu, the remaining gap is attributable to the native binding/host overhead rather than the benchmark measuring the wrong implementation.

## Batch 1B: add/mul/silu host overhead

Current ratios show host overhead dominates small elementwise ops:

- add: ~2.43x slower
- mul: ~3.82x slower
- silu: ~2.43x slower

These ops already have Cython fast paths in `_C/_npu_ops.pyx`. Phase 1B profiles and tightens those paths.

Investigation order:

1. Verify benchmark dispatch reaches `fast_add`, `fast_mul`, and `fast_silu`.
2. Measure per-call costs for:
   - descriptor/executor creation
   - workspace allocation/free
   - output storage allocation
   - tensor wrapper construction
   - stream synchronization
3. Compare Cython fast path with Python ACLNN wrapper path to verify the fast path is actually faster.
4. Implement the smallest native-path change that removes the measured bottleneck.

Likely implementation areas:

- `_C/_npu_ops.pyx` binary/unary helpers
- `_C/_aclnn_ffi.pyx` executor binding usage
- NPU allocator fast path and workspace reuse
- backend registration paths that choose fast ops

Acceptance:

- `add`, `mul`, and `silu` L0 fp16 inference ratios are each `<= 1.0`, or a documented blocker identifies why ACLNN cannot match torch_npu with the current public runtime.
- No correctness regressions in affected NPU tests.

## Batch 1C: matmul/bmm/softmax

Current representative ratios:

- matmul_qkv: ~2.02x slower
- bmm attention scores/output: ~1.6-1.7x slower in broader baseline
- softmax: ~1.16x slower

Investigation order:

1. Confirm operands are already contiguous or identify unnecessary contiguity conversions.
2. Confirm batched matmul uses one ACLNN batched matmul where possible rather than Python loops.
3. Check shape flattening and descriptor construction overhead for repeated static shapes.
4. Check softmax backend route and whether it is using ACLNN native softmax or a composite.

Acceptance:

- matmul_qkv, bmm attention cases, and softmax ratios each reach `<= 1.0`, or have a measured blocker recorded for Phase 2 graph/fusion.

## Batch 1D: correctness blockers

Two L0 cases currently error:

1. `rope`: `GetWorkspaceSize failed: 561103 [op=neg, device=npu]`
2. `cross_entropy`: `NPU mul requires matching dtypes`

These must be fixed before the single-op gate can be meaningful.

Required behavior:

- `rope` must run fully on NPU without CPU fallback.
- `cross_entropy` must preserve torch-compatible dtype behavior and avoid internal mismatched NPU `mul` inputs.
- Fix source behavior, not the benchmark, unless the benchmark is calling an invalid torch-incompatible path.

Acceptance:

- Both cases produce successful L0 benchmark rows.
- Numerical parity checks are added where practical for the affected op behavior.

## Testing and validation

For each batch:

1. Add or update focused tests before implementation.
2. Run affected tests.
3. Run targeted L0 benchmark for the touched operators with:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
CONDA_SH=/home/jenkins/anaconda3/etc/profile.d/conda.sh \
CANN_SET_ENV=/usr/local/Ascend/cann-9.0.0/set_env.sh \
CANDLE_CONDA_ENV=candle311 \
TORCH_NPU_CONDA_ENV=torchnpu311 \
conda run -n candle311 --no-capture-output \
  python -m benchmarks.op_benchmark_npu.run \
  --ops <ops> --dtype fp16 --scenario infer --mode fwd \
  --warmup 5 --iters 30 --json-output <artifact>.json
```

4. Run L1/L2 observation probes after each batch to ensure no major regressions.

Phase 1 is complete when:

- No Batch 1 L0 operator is slower than torch_npu without a documented blocker.
- `rope` and `cross_entropy` no longer error in L0.
- L1/L2 observation results improve or remain stable.

## Risks

- Optimizing the benchmark path but not the real model path. Mitigation: trace dispatch for both benchmark and L1/L2 cases.
- Executor caching with stale tensor addresses. Mitigation: cache only safe descriptors/plans and rebind addresses at launch time.
- Workspace reuse causing stream safety bugs. Mitigation: stream-scoped ownership and targeted tests.
- Native RMSNorm API mismatch. Mitigation: keep public API torch-compatible and isolate benchmark-only variants clearly.
- Spending too long on micro-op parity when graph fusion will dominate. Mitigation: stop after the first high-impact tranche and reassess L1/L2.
