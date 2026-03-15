# Known Kernel Issues

This document tracks native kernel bugs and their on-device composite workarounds across GPU/NPU backends. It serves as the regression testing checklist after platform upgrades (CANN SDK, CUDA toolkit, macOS/Metal).

## How to Use This Document

- **After a platform upgrade**: Test each `open` entry by re-enabling the native kernel and running the associated test. If the native kernel works, mark the entry as `resolved` and remove the composite workaround.
- **When adding a workaround**: Add a new entry here with all required fields.

## Note on torch_npu

The official Ascend PyTorch backend (`torch_npu`) does not implement the 910B fallback ops natively either. It registers a blanket `PrivateUse1` fallback that moves tensors to CPU via `at::native::cpu_fallback` (see `VariableFallbackKernel.cpp` line 259). Candle instead uses on-device ACLNN small-op composites so that tensors never leave the NPU.

## Issue Table

| Op | Backend | Error Description | Composite Workaround | Platform Version | Status |
|----|---------|-------------------|----------------------|------------------|--------|
| `cartesian_prod` | npu | Current implementation required `Tensor.to("cpu")` to enumerate values, which violates the no-CPU-fallback rule. | Implemented via reshape+repeat+tile+stack composite (all on-device). | CANN 8.3 / Candle `0.1.x` | resolved |
| `block_diag` | npu | Current implementation required `Tensor.to("cpu")` to materialize block contents, which violates the no-CPU-fallback rule. | Implemented via zeros_create+memcpy_d2d composite (all on-device). | CANN 8.3 / Candle `0.1.x` | resolved |
| `contiguous` (stride=0 from expand) | npu | NPU `contiguous` does not correctly materialize expanded tensors with stride=0 dimensions. Copies raw storage bytes instead of respecting strides, producing garbage. | Avoid expand+contiguous; use reshape+repeat instead. | CANN 8.5 / Ascend910B3 | open |
| `repeat_interleave` (tensor `repeats`) | npu | Current implementation required reading NPU `repeats` values on CPU to build gather indices, which violates the no-CPU-fallback rule. | Use integer `repeats` only for now; fail explicitly for tensor-valued repeats until an on-device index builder exists. | CANN 8.3 / Candle `0.1.x` | open |
| `baddbmm` (tensor `alpha`/`beta`) | npu | Current implementation required reading NPU tensor scalars for `alpha`/`beta` on CPU, which violates the no-CPU-fallback rule. | Use Python numeric `alpha`/`beta` only for now; fail explicitly for tensor-valued scalars until an on-device scalar path is implemented. | CANN 8.3 / Candle `0.1.x` | open |
| `std` (dim=None) | npu (910B) | `aclnnVar` all-reduce (no dim) returns error 161002 on 910B series. Per-dim reduction works fine. | Flatten to `(1, N)` then `var(dim=1)` + `sqrt`. Gate: `_use_soc_fallback("std")`. | CANN 8.5 / Ascend910B | open |
| `nansum` | npu (910B) | `aclnnReduceNansum` returns error 161002 on 910B series. | Replace NaN with 0 via `where(isnan(x), 0, x)` then `sum`. Gate: `_use_soc_fallback("nansum")`. | CANN 8.5 / Ascend910B | open |
| `instance_norm` | npu (910B) | `aclnnInstanceNorm` returns error 161002 on 910B series. | Manual composite: per-instance mean/var → normalize → affine. Gate: `_use_soc_fallback("instance_norm")`. | CANN 8.5 / Ascend910B | open |
| `adaptive_avg_pool2d` | npu (910B) | `cubeMathType=1` state from matmul contaminates subsequent `aclnnAdaptiveAvgPool2d`, producing wrong results. | Manual composite: reshape + mean over spatial blocks. Gate: `_use_soc_fallback("adaptive_avg_pool2d")`. | CANN 8.5 / Ascend910B | open |
| `avg_pool2d` | npu (910B) | `aclnnAvgPool2d` returns error 161002 on 910B series. | Depthwise conv2d with uniform `1/(kH*kW)` weights. Gate: `_use_soc_fallback("avg_pool2d")`. | CANN 8.5 / Ascend910B | open |
| `upsample_nearest1d` | npu (910B) | `aclnnUpsampleNearest1d` produces incorrect results on 910B series. | Manual repeat-based upsampling on device. Gate: `_use_soc_fallback("upsample_nearest1d")`. | CANN 8.5 / Ascend910B | open |
| `einsum` | npu (910B) | `aclnnEinsum` returns error 161002 on 910B series. | Parse equation string, decompose into transpose/reshape/matmul/sum. Gate: `_use_soc_fallback("einsum")`. | CANN 8.5 / Ascend910B | open |
| `isinf` | npu (910B) | `aclnnIsInf` returns error 161001 (unavailable) on 910B series. | Composite: `!isfinite(x) & isfinite(1/x)`. Gate: `_use_soc_fallback("isinf")`. | CANN 8.5 / Ascend910B | open |
| `isinf` | npu (310B) | `aclnnIsInf` returns error 161001 (unavailable) on 310B series. | Same composite as 910B: `!isfinite(x) & isfinite(1/x)`. Gate: `_use_soc_fallback("isinf")`. | CANN 8.5 / Ascend310B | open |
| `matmul` | npu (310B) | `aclnnMatmul` returns error 561103 for float32 inputs on 310B; float16 works natively. | Cast float32 inputs to float16, run native `aclnnMatmul`, cast result back to float32. Gate: `_use_soc_fallback("matmul")`. | CANN 8.5 / Ascend310B | open |
| `addmm` | npu (310B) | `aclnnAddmm` returns error 561103 for float32 inputs on 310B; float16 works natively. | Cast float32 inputs to float16, run native `aclnnAddmm`, cast result back to float32. Gate: `_use_soc_fallback("addmm")`. | CANN 8.5 / Ascend310B | open |
| `mv` | npu (310B) | `aclnnMv` returns error 561103 for float32 inputs on 310B; float16 works natively. | Cast float32 inputs to float16, run native `aclnnMv`, cast result back to float32. Gate: `_use_soc_fallback("mv")`. | CANN 8.5 / Ascend310B | open |
| `dot` | npu (310B) | `aclnnDot` returns error 561103 for all dtypes (float16/float32) on 310B. | Composite: `mul(a, b)` then `sum()`. Gate: `_use_soc_fallback("dot")`. | CANN 8.5 / Ascend310B | open |
| `im2col` | npu (910B) | `aclnnIm2col` returns error 561103 on 910B series. | Manual sliding-window extraction via slice + reshape + cat. Gate: `_use_soc_fallback("im2col")`. | CANN 8.5 / Ascend910B | open |
| `frac` | npu (910A) | `aclnnFrac` returns error 561000 (unsupported) on 910A. | Composite: `a - trunc(a)`. Gate: `_use_soc_fallback("frac")`. | CANN 8.5 / Ascend910A | open |
| `gather` | npu (910A) | `aclnnGather` returns error 561103 on multi-dimensional inputs on 910A. | Scatter-based one-hot + sum composite (reuses `_gather_310b_fallback`). Gate: `_use_soc_fallback("gather")`. | CANN 8.5 / Ascend910A | open |

<!--
Entry template:
| `getitem` (bool mask) | npu | ACLNN index fails for bool-mask advanced indexing (aclnnIndexGetWorkspaceSize 161001). | Fail explicitly; no CPU fallback. | CANN 8.3 / Candle `0.1.x` | open |
| `aten.op_name` | mps/cuda/npu | Brief error description | `op_a` + `op_b` composite | macOS XX.X / CANN X.X / CUDA XX.X | open/resolved |
-->
