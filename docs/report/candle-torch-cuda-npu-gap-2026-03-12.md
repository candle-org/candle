# Candle vs torch CUDA/NPU Gap Report
> Date: 2026-03-12
> Goal: Static scan of Candle to compare gaps vs torch CUDA and torch NPU, with actionable gap lists

---
## 1. Scope and Method
- Scope: schema registration, backend kernel registration, autograd registration, functional API, and CPU round-trips in NPU ops.
- Sources: `src/candle/_dispatch/schemas.py`, `src/candle/_backends/*/__init__.py`, `src/candle/_backends/autograd.py`, `src/candle/_functional.py`, `src/candle/_backends/npu/ops.py`, `docs/known-kernel-issues.md`.
- Note: Static analysis only, independent of local CUDA/NPU runtime availability.

## 2. Coverage Summary
| Item | Count | Notes |
|---|---:|---|
| Schema ops | 384 | Ops registered in `schemas.py` |
| CPU kernels | 384 | Registered in `cpu/__init__.py` |
| MPS kernels | 380 | Registered in `mps/__init__.py` |
| NPU kernels | 380 | Registered in `npu/__init__.py` |
| CUDA kernels | 7 | Registered in `cuda/__init__.py` |
| Autograd kernels | 300 | Registered in `autograd.py` |
| Functional API defs | 254 | def count in `candle/_functional.py` |

## 3. Backend Coverage Gaps (relative to CPU)
### 3.1 CUDA Low Coverage
- CPU to CUDA missing: 377 ops
- CUDA currently registers only 7 ops: `add`, `to`, `tensor`, `zeros`, `ones`, `empty`, `full`.
- Conclusion: large structural gap vs torch.cuda kernel coverage.

### 3.2 NPU/MPS Missing
- CPU to NPU missing: 4 ops
- CPU to MPS missing: 4 ops
- NPU and MPS both miss these 4 ops:
  - adaptive_max_pool3d, max_unpool1d, max_unpool2d, max_unpool3d

## 4. Autograd Coverage Gap
- Schema to Autograd missing: 84 ops
- Full list:
  - _adadelta_step, _adagrad_step, _adam_step, _adamax_step, _adamw_step, _asgd_step, _nadam_step, _radam_step, _rmsprop_step, _rprop_step, _sgd_step, _sparse_adam_step, adaptive_max_pool3d, all, allclose, any, arange, argmax, argmin, argsort, argwhere, bernoulli_, bincount, bitwise_and, bitwise_not, bitwise_or, bitwise_xor, bucketize, cartesian_prod, cauchy_, count_nonzero, dropout, empty, eq, equal, erfinv_, exponential_, eye, fill_, full, ge, geometric_, gt, histc, histogram, isclose, isfinite, isin, isinf, isnan, isneginf, isposinf, isreal, le, linspace, log_normal_, logical_and, logical_not, logical_or, logical_xor, logspace, lt, max_unpool1d, max_unpool2d, max_unpool3d, ne, nonzero, normal_, one_hot, ones, rand, randint, randint_, randn, random_, randperm, range, searchsorted, tensor, tril_indices, triu_indices, uniform_, unique, zeros

## 5. Functional API Coverage Gap
- Schema to Functional missing: 154 ops

### 5.1 Category Summary
- optim_steps: 12
  - _adadelta_step, _adagrad_step, _adam_step, _adamax_step, _adamw_step, _asgd_step, _nadam_step, _radam_step, _rmsprop_step, _rprop_step, _sgd_step, _sparse_adam_step
- pooling: 15
  - adaptive_avg_pool1d, adaptive_avg_pool2d, adaptive_avg_pool3d, adaptive_max_pool1d, adaptive_max_pool2d, adaptive_max_pool3d, avg_pool1d, avg_pool2d, avg_pool3d, max_pool1d, max_pool2d, max_pool3d, max_unpool1d, max_unpool2d, max_unpool3d
- conv: 6
  - conv1d, conv2d, conv3d, conv_transpose1d, conv_transpose2d, conv_transpose3d
- fft: 16
  - fft_fft, fft_fft2, fft_fftn, fft_fftshift, fft_hfft, fft_ifft, fft_ifft2, fft_ifftn, fft_ifftshift, fft_ihfft, fft_irfft, fft_irfft2, fft_irfftn, fft_rfft, fft_rfft2, fft_rfftn
- linalg: 29
  - linalg_cholesky, linalg_cond, linalg_det, linalg_eig, linalg_eigh, linalg_eigvals, linalg_eigvalsh, linalg_householder_product, linalg_inv, linalg_lstsq, linalg_lu, linalg_lu_factor, linalg_lu_solve, linalg_matrix_exp, linalg_matrix_norm, linalg_matrix_power, linalg_matrix_rank, linalg_multi_dot, linalg_norm, linalg_pinv, linalg_slogdet, linalg_solve, linalg_solve_triangular, linalg_svd, linalg_svdvals, linalg_tensorinv, linalg_tensorsolve, linalg_vander, linalg_vector_norm
- special: 21
  - special_digamma, special_entr, special_erfcx, special_erfinv, special_gammainc, special_gammaincc, special_gammaln, special_i0, special_i0e, special_i1, special_i1e, special_log_ndtr, special_logit, special_multigammaln, special_ndtr, special_ndtri, special_polygamma, special_sinc, special_xlog1py, special_xlogy, special_zeta
- norms: 7
  - batch_norm, group_norm, instance_norm, layer_norm, linalg_matrix_norm, linalg_norm, linalg_vector_norm
- activations: 8
  - celu, elu, hardshrink, hardsigmoid, hardswish, leaky_relu, mish, silu
- sampling: 9
  - bernoulli_, cauchy_, exponential_, geometric_, log_normal_, normal_, randint_, random_, uniform_
- indexing: 2
  - getitem, setitem
- nn: 7
  - batch_norm, col2im, ctc_loss, dropout, embedding, grid_sample, im2col
- uncategorized: 26
  - affine_grid, contiguous, div_, erfinv_, fill_, gelu, log_softmax, max_, min_, normalize, one_hot, pad, prelu, rrelu, selu, softmax, softshrink, softsign, sub_, threshold, uniform, upsample_bicubic2d, upsample_bilinear2d, upsample_linear1d, upsample_nearest1d, upsample_nearest2d

## 6. NPU CPU Round-Trip Hotspots
- These functions use numpy or D2H/H2D paths (memcpy_d2h, _sync_to_cpu, _copy_cpu_to_npu):
  - _adam_step_op, _build_dft_matrices, _expand_ellipsis, _is_basic_index_key, _is_int_index, _npu_advanced_setitem, _npu_assign_to_view, _npu_basic_getitem_view, _read_bool_scalar, _read_int64_scalar, _to_npu_index_tensor, addcdiv, addcmul, avg_pool3d_op, bincount_aclnn, bincount_op, col2im_op, conv3d_op, conv_transpose3d_op, ctc_loss_op, det_op, diagonal_op, histc_op, histogram_op, im2col_op, index_put_, linalg_matrix_rank_op, nanmedian_op, nanquantile_op, one_hot, quantile_op, random_, scatter_
- Total: 33

## 7. Known Issues (docs/known-kernel-issues.md)
- | `cartesian_prod` | npu | Current implementation required `Tensor.to("cpu")` to enumerate values, which violates the no-CPU-fallback rule. | No workaround yet; fail explicitly until a true on-device composition is implemented. | CANN 8.3 / Candle `0.1.x` | open |
- | `block_diag` | npu | Current implementation required `Tensor.to("cpu")` to materialize block contents, which violates the no-CPU-fallback rule. | No workaround yet; fail explicitly until a true on-device composition is implemented. | CANN 8.3 / Candle `0.1.x` | open |
- | `repeat_interleave` (tensor `repeats`) | npu | Current implementation required reading NPU `repeats` values on CPU to build gather indices, which violates the no-CPU-fallback rule. | Use integer `repeats` only for now; fail explicitly for tensor-valued repeats until an on-device index builder exists. | CANN 8.3 / Candle `0.1.x` | open |
- | `baddbmm` (tensor `alpha`/`beta`) | npu | Current implementation required reading NPU tensor scalars for `alpha`/`beta` on CPU, which violates the no-CPU-fallback rule. | Use Python numeric `alpha`/`beta` only for now; fail explicitly for tensor-valued scalars until an on-device scalar path is implemented. | CANN 8.3 / Candle `0.1.x` | open |
- | `aten.op_name` | mps/cuda/npu | Brief error description | `op_a` + `op_b` composite | macOS XX.X / CANN X.X / CUDA XX.X | open/resolved |

## 8. Priority Action Table (P0/P1/P2)
| Priority | Goal | Main gaps | Actions | Verification |
|---|---|---|---|---|
| P0 | Device semantics and minimal training path | CUDA coverage 7/384 and numpy paths<br>NPU has 33 CPU round-trips<br>Autograd missing 84 ops (pool/dropout/indexing etc.) | CUDA: replace numpy paths with on-device kernels, start from training-minimal set<br>NPU: remove CPU round-trips in §6, keep on-device composites only<br>Autograd: implement backward for training-critical ops or define non-differentiable behavior | `pytest tests/contract/ -v --tb=short`<br>Device-semantic checks and minimal training smoke test |
| P1 | API surface parity | Functional missing 154 ops (conv/pool/fft/linalg/special etc.)<br>CUDA/NPU coverage still behind CPU | Add functional wrappers and align signatures<br>Expand CUDA/NPU kernel coverage (training path first)<br>Fill remaining autograd ops | Contract tests and API parity checks |
| P2 | Performance and ecosystem | Perf tuning, memory stats, AMP, profiling, docs completeness | Kernel tuning and fusion<br>Memory stats, RNG, AMP, profiler APIs<br>Support matrix and known-issues hygiene | Benchmarks and regression stats, doc review |

## 9. Summary
- CUDA: current implementation is not a real GPU path and needs on-device kernels for core training ops.
- NPU: main risk is CPU round-trips; prioritize removing them with on-device composites.
- Autograd/Functional: close missing lists with training-critical ops first.
