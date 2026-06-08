# NPU Transformer / SDPA Performance Playbook

This document is a reusable checklist for implementing and optimizing PyTorch-compatible Transformer attention on Candle NPU. It captures lessons from the NPU MLP optimization work: benchmark first, add tests first, optimize generic operators/backends, preserve autograd semantics, and only claim performance wins with fresh evidence.

## Objective

- Implement/support `torch.nn.functional.scaled_dot_product_attention` (SDPA) for NPU.
- Support autograd for training workloads, not only inference.
- Match or beat `torch_npu` for Transformer forward, backward, and total step time on representative shapes.
- Keep Candle as a generic PyTorch compatibility layer: no model-specific or benchmark-shape hacks.

## Non-negotiable constraints

- No `torch` imports in `src/candle`; PyTorch is only for tests/benchmarks.
- No CPU fallback for NPU tensors.
- Keep all NPU computation on device: ACLNN native kernels first, then on-device composites only if native kernels are absent or broken.
- Preserve dispatch/schema/autograd semantics, including `create_graph`, saved tensor hooks, public `next_functions`, retain-grad, and leaf `.grad` behavior.
- Unsupported SDPA feature combinations must fall back to the normal generic path or raise a PyTorch-compatible error; they must not silently compute incorrect results.

## Development sequence

### 1. Establish a benchmark before changing source

Create or extend a benchmark that measures both Candle and `torch_npu` with identical:

- batch size
- number of heads
- sequence length(s)
- head dimension
- dtype, especially float16/bfloat16
- dropout probability and training/eval mode
- `is_causal`
- attention mask dtype/shape/broadcast semantics
- explicit `scale` if provided
- synchronization boundaries

The benchmark must report:

- forward median/min/p10/p90
- backward median/min/p10/p90
- total median/min/p10/p90
- profiler top rows when available

Always run Candle with:

```bash
PYTHONPATH=<your-worktree>/src
```

The `candle311` env otherwise may import a fixed editable install from another worktree.

### 2. Profile and identify the real gap

Before implementing, determine whether time is spent in:

- native/fused SDPA kernel(s)
- matmul/bmm hot paths
- softmax/logsumexp
- dropout/random mask generation
- causal or additive mask handling
- transposes/views/contiguous materialization
- autograd node overhead
- leaf grad accumulation or clone/materialization
- allocator/runtime/deferred executor cleanup
- Python dispatch or wrapper overhead

Do not implement based on intuition alone. Record the hot rows and pick the next optimization from evidence.

### 3. Add RED tests first

For every source behavior change, add tests before production code and verify they fail for the expected reason.

Recommended tests:

- SDPA forward correctness vs PyTorch for supported combinations.
- SDPA backward correctness for query/key/value and optional mask-related behavior.
- NPU device placement: outputs and gradients stay on NPU.
- Routing/performance structure:
  - native/fused NPU SDPA path avoids Python ACLNN wrapper when intended;
  - backward uses fused/native helper or tested Cython composite, not a slow Python formula;
  - no accidental `to(cpu)`/CPU fallback;
  - no avoidable `contiguous`/`transpose` redispatch on hot shapes.
- Autograd semantics:
  - `retain_grad`
  - saved tensor hooks
  - public `next_functions`
  - `create_graph=True` fallback/behavior
  - input subset backward/grad behavior if relevant.

### 4. Choose implementation path

Priority order:

1. Native fused ACLNN SDPA forward and backward kernels.
2. Native large kernels for subpieces, with Cython/C `_C` bindings and PTA executor cache.
3. Generic on-device composite using existing optimized matmul/bmm, softmax, dropout, mask, and view ops.
4. Never CPU fallback for NPU.

If a native kernel is broken:

- keep the native entry point guarded/commented for future re-enable;
- document the issue in `docs/known-kernel-issues.md`;
- implement the workaround as a generic on-device composite.

### 5. Cython/NPU fast-path checklist

Use these techniques from previous successful NPU work:

- Exact base `TensorImpl` guards before bypassing dispatcher.
- Direct cached fields: `_shape_tuple`, `_stride_tuple`, `_device_index`, `_dtype_code`, `_storage._untyped._device_ptr`, `_c_offset`, `_itemsize`.
- Cached stream/runtime/allocator helpers.
- Fast large-pool tensor wrapper for fresh outputs.
- PTA executor cache for stable native kernels.
- PTA hash keys include every semantic attribute:
  - op name
  - query/key/value descriptors
  - output descriptors
  - dtype(s)
  - scale
  - dropout probability
  - training flag if it changes behavior
  - causal flag
  - mask presence/type/shape/stride/dtype
  - softmax stats/logsumexp descriptor if present
  - alias discriminator bits where relevant
  - stream
- Keep descriptors alive until deferred executor cleanup.
- Defer workspace frees through runtime.
- Do not make optimized paths optional flags; performance path is default when safe.

### 6. Autograd checklist

SDPA train-mode support must preserve graph semantics:

- Forward fast path attaches an autograd node or routes through generated autograd safely.
- Backward uses native fused SDPA backward if available.
- If composite backward is needed, all computations remain on NPU and use generic ops.
- Saved tensor handling respects mutation/version checks and saved tensor hooks.
- Public `next_functions` remains PyTorch-compatible.
- Internal caches may avoid creating public `AccumulateGrad` objects, but public introspection must materialize them.
- Fresh NPU gradients may be marked as owned only when safe for leaf `.grad` stealing.
- Non-contiguous or expanded-view gradients must not be marked owned unless leaf storage semantics remain correct.

### 7. Validation gate before PR

For `src/candle/` changes, run:

```bash
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311 && \
PYTHONPATH=<your-worktree>/src python setup.py build_ext --inplace

source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311 && \
PYTHONPATH=<your-worktree>/src python -m pytest tests/npu/ -v --tb=short

source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311 && \
PYTHONPATH=<your-worktree>/src python -m pytest tests/cpu/ tests/contract/ -v --tb=short

source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311 && \
PYTHONPATH=<your-worktree>/src python -m pylint src/candle/ --rcfile=.github/pylint.conf
```

Also run the Transformer/SDPA benchmark multiple times. Do not claim parity from one noisy run.

## Escalation rule

If eager generic operator optimization cannot reach torch_npu parity, do not add model-specific fusion. Move to a generic graph/capture or backend fused-execution design that can apply to Transformer patterns broadly.
