# Known Kernel Issues

This document tracks native kernel bugs and their on-device composite workarounds across GPU/NPU backends. It serves as the regression testing checklist after platform upgrades (CANN SDK, CUDA toolkit, macOS/Metal).

## How to Use This Document

- **After a platform upgrade**: Test each `open` entry by re-enabling the native kernel and running the associated test. If the native kernel works, mark the entry as `resolved` and remove the composite workaround.
- **When adding a workaround**: Add a new entry here with all required fields.

## Issue Table

| Op | Backend | Error Description | Composite Workaround | Platform Version | Status |
|----|---------|-------------------|----------------------|------------------|--------|
| `cartesian_prod` | npu | Current implementation required `Tensor.to("cpu")` to enumerate values, which violates the no-CPU-fallback rule. | No workaround yet; fail explicitly until a true on-device composition is implemented. | CANN 8.3 / Candle `0.1.x` | open |
| `block_diag` | npu | Current implementation required `Tensor.to("cpu")` to materialize block contents, which violates the no-CPU-fallback rule. | No workaround yet; fail explicitly until a true on-device composition is implemented. | CANN 8.3 / Candle `0.1.x` | open |
| `repeat_interleave` (tensor `repeats`) | npu | Current implementation required reading NPU `repeats` values on CPU to build gather indices, which violates the no-CPU-fallback rule. | Use integer `repeats` only for now; fail explicitly for tensor-valued repeats until an on-device index builder exists. | CANN 8.3 / Candle `0.1.x` | open |
| `baddbmm` (tensor `alpha`/`beta`) | npu | Current implementation required reading NPU tensor scalars for `alpha`/`beta` on CPU, which violates the no-CPU-fallback rule. | Use Python numeric `alpha`/`beta` only for now; fail explicitly for tensor-valued scalars until an on-device scalar path is implemented. | CANN 8.3 / Candle `0.1.x` | open |

<!--
Entry template:
| `getitem` (bool mask) | npu | ACLNN index fails for bool-mask advanced indexing (aclnnIndexGetWorkspaceSize 161001). | Fail explicitly; no CPU fallback. | CANN 8.3 / Candle `0.1.x` | open |
| `aten.op_name` | mps/cuda/npu | Brief error description | `op_a` + `op_b` composite | macOS XX.X / CANN X.X / CUDA XX.X | open/resolved |
-->
