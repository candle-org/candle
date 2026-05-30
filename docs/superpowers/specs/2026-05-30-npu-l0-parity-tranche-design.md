# NPU L0 Parity Tranche Design

## Goal

Make every benchmarked L0 single-operator on NPU at least as fast as the corresponding torch_npu operator under the same shapes, dtype, synchronization mode, and warmup policy.

This tranche is the prerequisite for L1/L2 model superiority work. It does not attempt graph/fusion or model-level optimization.

## Acceptance Gate

Final artifact:

```text
results/npu_perf_gate/l0_parity_final.json
```

Must satisfy:

- Exit code 0 from the gate command.
- All target L0 rows status `ok`.
- Every target op `median_ratio <= 1.0` (Candle / torch_npu).
- `rope` and `cross_entropy` included and passing.
- `rms_norm_native` included and passing.
- No CPU fallback.
- Affected correctness tests pass.
- Full required validation pass for any `src/candle/` changes.

Gate command:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
CONDA_SH=/home/jenkins/anaconda3/etc/profile.d/conda.sh \
CANN_SET_ENV=/usr/local/Ascend/cann-9.0.0/set_env.sh \
CANDLE_CONDA_ENV=candle311 \
TORCH_NPU_CONDA_ENV=torchnpu311 \
conda run -n candle311 --no-capture-output \
  python -m benchmarks.op_benchmark_npu.run \
  --mode fwd \
  --dtype fp16 \
  --scenario infer \
  --warmup 5 \
  --iters 30 \
  --json-output results/npu_perf_gate/l0_parity_final.json \
  --fail-on-ratio \
  --max-ratio 1.0
```

## Current Baseline

Fresh L0 fp16 fwd/infer benchmark (2026-05-30, batch=1, seq=2048, warmup=3, iters=10):

| Op | Candle median ms | torch_npu median ms | Ratio | Status |
|---|---:|---:|---:|---|
| rms_norm | 2352.6771 | 0.2084 | 11289.24x | ok (composite, not native) |
| embedding | 0.2923 | 0.0820 | 3.56x | ok |
| mul | 0.2678 | 0.0808 | 3.31x | ok |
| silu | 0.2570 | 0.1130 | 2.27x | ok |
| add | 0.1688 | 0.0797 | 2.12x | ok |
| matmul_qkv | 0.6404 | 0.3150 | 2.03x | ok |
| dropout | 0.4815 | 0.2400 | 2.01x | ok |
| bmm_attn_scores | 0.4707 | 0.2875 | 1.64x | ok |
| bmm_attn_output | 0.5397 | 0.3432 | 1.57x | ok |
| matmul_ffn_up | 1.1416 | 0.7680 | 1.49x | ok |
| matmul_ffn_down | 1.1529 | 0.8517 | 1.35x | ok |
| softmax | 1.0832 | 0.9412 | 1.15x | ok |
| rope | — | 0.2314 | N/A | error: GetWorkspaceSize failed: 561103 [op=neg] |
| cross_entropy | — | 0.8642 | N/A | error: NPU mul requires matching dtypes |

Notes:
- `rms_norm` benchmark currently measures composite `pow + mean + rsqrt + mul`, not native RMSNorm.
- Excluding `rms_norm`, average ratio is approximately 2.05x slower.
- No successful op currently meets the <= 1.0 parity gate.

## Target Ops

All of the following must reach `median_ratio <= 1.0`:

- `rms_norm` (native route)
- `add`
- `mul`
- `silu`
- `softmax`
- `matmul_qkv`
- `bmm_attn_scores`
- `bmm_attn_output`
- `matmul_ffn_up`
- `matmul_ffn_down`
- `embedding`
- `dropout`
- `rope`
- `cross_entropy`

## Non-Goals

- L1/L2 graph replay or fusion.
- C++ autograd engine replacement.
- Full dispatcher rewrite.
- Application-specific benchmark hacks.
- CPU fallback.
- Optional performance flags.
- Schema weakening.

## Architecture

Each op fix follows this diagnostic pipeline:

```text
L0 benchmark case
  -> torch-compatible Candle public API
  -> dispatch / functional routing
  -> NPU backend op
  -> _C/_npu_ops.pyx fast path or ACLNN binding
  -> ACLNN/ACLRT launch
  -> JSON gate: Candle median <= torch_npu median
```

Each operator fix must identify which layer is the bottleneck before implementation:

1. Benchmark case is measuring the wrong thing.
2. Public API is not routing to the native op.
3. NPU backend is falling back to a Python composite.
4. `_C/_npu_ops.pyx` fast path exists but has avoidable host overhead.
5. ACLNN binding itself is missing, broken, or using an unfavorable call shape.
6. Allocator/workspace/tensor-wrapper overhead dominates the operator.

Only after the failing layer is identified should implementation changes be made.

## Batch 1: Benchmark Routing and Instrumentation

### rms_norm

Current benchmark measures composite `x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)`.

Required changes:

- Split benchmark into `rms_norm_native` (calls Candle default user API for RMSNorm) and `rms_norm_composite` (diagnostic only).
- Gate uses `rms_norm_native` result.
- If Candle lacks a clear public native RMSNorm route, add one that is torch-compatible.
- Add route evidence: native case must reach NPU `rms_norm` backend or `_C` native path.

### Path evidence for all target ops

Add lightweight routing verification for key ops:

- Confirm `add`, `mul`, `silu` reach `_C/_npu_ops.pyx` fast paths.
- Confirm `softmax` reaches native ACLNN softmax.
- Confirm `matmul_qkv` / bmm cases reach batched ACLNN.
- Confirm `embedding` reaches native ACLNN embedding.
- Confirm `dropout` reaches native ACLNN dropout.

Evidence can be test assertions, benchmark debug metadata, or dispatch trace.

## Batch 2: Correctness Blockers

### rope

Error: `GetWorkspaceSize failed: 561103 [op=neg, device=npu]`

Benchmark uses `torch_mod.cat((-t2, t1), dim=-1)`, so failure is unary `neg` on NPU.

Resolution path:

1. Determine if NPU `neg` is a general CANN bug for this dtype/shape/stride.
2. If general bug:
   - Implement on-device composite workaround (e.g., `mul(x, -1)` or `sub(0, x)`).
   - Keep native `neg` entry point guarded.
   - Document in `docs/known-kernel-issues.md`.
3. If shape/stride specific:
   - Fix view/contiguous handling in the RoPE composite to avoid the broken path.
   - All computation must remain on NPU.
4. Acceptance: `rope` row status `ok`, median ratio enters gate.

### cross_entropy

Error: `NPU mul requires matching dtypes [op=mul, device=npu]`

Resolution path:

1. Trace internal composite to find dtype mismatch source.
2. Add torch-compatible dtype promotion/cast at functional/composite boundary.
3. Do not weaken schema validation.
4. Acceptance: `cross_entropy` row status `ok`, numerical parity with torch_npu within acceptable tolerance.

## Batch 3: Small-Op Host Overhead

Target ops: `add`, `mul`, `silu`.

Current ratios: 3.31x, 2.27x, 2.12x slower.

Investigation order:

1. Verify benchmark dispatch reaches `fast_add`, `fast_mul`, `fast_silu` in `_C/_npu_ops.pyx`.
2. If not reaching fast path, fix registration/routing first.
3. If already on fast path, measure per-call costs:
   - Descriptor/executor creation.
   - Workspace allocation/free.
   - Output storage allocation.
   - Tensor wrapper construction.
   - Stream synchronization.
4. Implement smallest default-path change that removes measured bottleneck:
   - Reusable descriptor/executor plan (key on shape/dtype/stride, rebind addresses at launch).
   - Workspace pool (stream-scoped).
   - Allocator drain reduction.
   - Direct device pointer access.
   - C-level Tensor construction fast path.

Acceptance: each op `median_ratio <= 1.0`.

## Batch 4: Matmul, BMM, Softmax, Embedding, Dropout

### matmul / bmm

Current ratios: 1.35x to 2.03x slower.

Strategy:

- Confirm batched ACLNN path (not Python loops).
- Confirm flatten-leading-batch-dims path covers benchmark shapes.
- Eliminate unnecessary contiguous/view conversions.
- Reuse Batch 3 native path optimizations for per-call overhead.

### softmax

Current ratio: 1.15x slower.

Strategy:

- Confirm native ACLNN softmax route.
- Eliminate host overhead / descriptor overhead.
- Do not introduce graph path for this 15% gap.

### embedding

Current ratio: 3.56x slower.

Strategy:

- Confirm native ACLNN embedding route.
- Check gather/index path, dtype/index placement, output allocation.

### dropout

Current ratio: 2.01x slower.

Strategy:

- Confirm native ACLNN dropout route.
- Check random mask generation, in-place mask, sync behavior.

Acceptance: all ops `median_ratio <= 1.0`.

## Testing Strategy

### Correctness tests

- `tests/npu/test_rope_l0_parity.py`
- `tests/npu/test_cross_entropy_npu_dtype.py`
- `tests/npu/test_rms_norm_native_route.py`

Or appended to existing test files if appropriate.

### Route tests

- Verify `rms_norm_native` case does not call composite expression.
- Verify benchmark JSON distinguishes native vs composite.

### Performance gates

Each batch saves before/after artifact:

```text
results/npu_perf_gate/<batch>/before.json
results/npu_perf_gate/<batch>/after.json
```

### Regression tests

For any `src/candle/` change:

```bash
pylint src/candle/ --rcfile=.github/pylint.conf
python -m pytest tests/cpu/ tests/contract/ -v --tb=short
```

Plus affected NPU tests.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Optimize benchmark, not real path | Benchmark must use public Candle API; route evidence required |
| Executor cache reuses stale pointer | Cache shape/dtype/stride only; rebind addresses at launch |
| Workspace reuse stream safety | Stream-scoped ownership; per-current-stream reuse only |
| dtype fix breaks torch semantics | Write dtype behavior tests first; fix at promotion/cast layer |
| CANN kernel bug silently masked | Keep native entry guarded; document in known-kernel-issues.md |
| L0 focus delays L1/L2 | Run lightweight L1/L2 observation after each batch (not a merge gate) |

## Implementation Boundaries

### Allowed modifications

- `benchmarks/op_benchmark_npu/*`
- `tests/cpu/*benchmark*`
- `tests/npu/*`
- `src/candle/_backends/npu/*`
- `src/candle/_C/_npu_ops.pyx`
- `src/candle/_C/_aclnn_ffi.pyx`
- `src/candle/_functional.py` / generated API routes (torch-compatible semantics only)
- `docs/known-kernel-issues.md`
- New spec/plan docs

### Not allowed

- Import torch in `src/candle/`
- Shape/op special cases for benchmarks
- CPU fallback
- Optional performance flags
- Schema weakening
- Delete broken native kernel entry without guarded path and issue doc
- Require non-torch API for fast path

## Success Definition

This tranche succeeds when the final gate command exits 0 with all target ops at `median_ratio <= 1.0`, no error rows, no CPU fallback, and all affected tests passing.
