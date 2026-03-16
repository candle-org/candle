# Known Kernel Issues

This file documents confirmed ACLNN kernel bugs and missing ops per chip.
Each entry includes: op name, backend, error code, workaround, and platform version.

All entries were verified by running `tests/npu/310b/` locally on the target hardware.

---

## 310B

| Op | ACLNN kernel | Error | Workaround | Verified on |
|---|---|---|---|---|
| `isinf` | `aclnnIsInf` | 161001 (unavailable) | composite: `~isfinite & ~isnan` | CANN 8.x |
| `dot` | `aclnnDot` | 561103 | composite: `(a * b).sum()` | CANN 8.x |
| `remainder` | `aclnnRemainderTensorTensor` | 161002 | composite: `a - b * floor(a/b)` + where | CANN 8.x |
| `where` | `aclnnSWhere` | 561000 | composite: `cond * x + ~cond * y` | CANN 8.x |
| `atan2` | `aclnnAtan2` | 561103 | composite: `atan(a/b)` with quadrant correction | CANN 8.x |
| `lerp` | `aclnnLerps` | 561103 | composite: `a + weight * (b - a)` | CANN 8.x |
| `softplus` | `aclnnSoftplus` | 561103 | composite: `log(1 + exp(x))` | CANN 8.x |
| `isclose` | `aclnnIsClose` | 561103 | composite: `abs(a-b) <= atol + rtol*abs(b)` | CANN 8.x |
| `flip` | `aclnnFlip` | 561000 | composite: slice-and-concat per dim | CANN 8.x |
| `argsort` | `aclnnTopk` (used internally) | 561103 | composite: sort + return indices | CANN 8.x |
| `sort` | `aclnnTopk` | 561103 | composite: bitonic sort via small ops | CANN 8.x |
| `topk` | `aclnnTopk` | 561103 | composite: sort + slice | CANN 8.x |
| `diag` | `aclnnDiag` | 561103 | composite: index select on diagonal | CANN 8.x |
| `gather` | `aclnnGather` | 561103 | composite: loop over dim slices | CANN 8.x |
| `take_along_dim` | `aclnnGather` | 561103 | composite: same as gather | CANN 8.x |
| `layer_norm` (float32) | `aclnnLayerNorm` | 561103 | cast to float16, run, cast back | CANN 8.x |
| `mish` | `aclnnMish` | 561103 | composite: `x * tanh(softplus(x))` | CANN 8.x |
| `batch_norm` | `aclnnBatchNorm` | 161002 | composite: manual mean/var normalize | CANN 8.x |
| `avg_pool2d` | `aclnnAvgPool2d` | 161002 | composite: unfold + mean | CANN 8.x |
| `adaptive_avg_pool2d` | `aclnnAdaptiveAvgPool2d` | cubeMathType contamination | composite: interpolate via avg_pool2d | CANN 8.x |
| `einsum` | `aclnnEinsum` | untested | composite: matmul/permute/sum patterns | CANN 8.x |
| `matmul` (float32) | `aclnnMatmul` | float32 unsupported | cast inputs to float16 | CANN 8.x |
| `addmm` (float32) | `aclnnAddmm` | float32 unsupported | cast inputs to float16 | CANN 8.x |
| `mv` (float32) | `aclnnMv` | float32 unsupported | cast inputs to float16 | CANN 8.x |

## 910B

| Op | ACLNN kernel | Error | Workaround | Verified on |
|---|---|---|---|---|
| `std` | `aclnnVar` (all-reduce) | 161002 | composite: manual var via mean | CANN 8.x |
| `nansum` | `aclnnReduceNansum` | 161002 | composite: `where(isnan, 0, x).sum()` | CANN 8.x |
| `instance_norm` | `aclnnInstanceNorm` | 161002 | composite: layer_norm per channel | CANN 8.x |
| `avg_pool2d` | `aclnnAvgPool2d` | 161002 | composite: unfold + mean | CANN 8.x |
| `adaptive_avg_pool2d` | `aclnnAdaptiveAvgPool2d` | cubeMathType=1 state corruption | composite: interpolate via avg_pool2d | CANN 8.x |
| `upsample_nearest1d` | `aclnnUpsampleNearest1d` | broken | composite: reshape to 4D + upsample_nearest2d | CANN 8.x |
| `einsum` | `aclnnEinsum` | 161002 | composite: matmul/permute/sum patterns | CANN 8.x |
| `isinf` | `aclnnIsInf` | 161001 (unavailable) | composite: `~isfinite & ~isnan` | CANN 8.x |
| `im2col` | `aclnnIm2col` | 561103 | composite: unfold | CANN 8.x |

## 910A

| Op | ACLNN kernel | Error | Workaround | Verified on |
|---|---|---|---|---|
| `allclose` | 6-op composite | ACLNN 561000 under executor pool pressure | composite: `isclose(...).all()` | CANN 8.x |
