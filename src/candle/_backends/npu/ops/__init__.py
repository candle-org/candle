from ._helpers import (
    _unwrap_storage, _wrap_tensor, _numel, _dtype_itemsize,
    _cast_tensor_dtype, _broadcast_shape, _broadcast_shape_checked,
    _npu_broadcast_to, _npu_arange_1d, _use_soc_fallback,
    _npu_add_scalar_, _npu_linear_index, npu_index_put_impl,
    _matmul_out_shape, _normalize_tensor_sequence_args,
    _iter_indices, _broadcast_index, _batch_offset,
    _unary_op, _binary_op,
    _normalize_reduction_dims, _reduce_out_shape,
    _reduce_dim_sizes, _broadcast_dims_to_out,
    _scalar_to_npu_tensor, _scalar_to_npu_tensor_no_add, _nan_like,
    # Re-export commonly used imports so op functions can use them
    bool_dtype, int32_dtype, int64_dtype, float_dtype,
    npu_typed_storage_from_ptr, reshape,
    aclnn, npu_runtime, npu_state, ops_soc,
)
import ctypes

from .math import (
    add, sub, mul, div,
    add_, sub_, mul_, div_,
    abs, neg, sign, signbit, square,
    isfinite, isinf, isnan, isposinf, isneginf,
    exp, log, sqrt, rsqrt, sin, cos, tan, tanh, sigmoid,
    sinh, cosh, erf, erfc, floor, ceil, round, trunc, frac,
    log2, log10, exp2, expm1, log1p,
    asin, acos, atan, asinh, acosh, atanh,
    atan2, pow, floor_divide,
)

from .comparison import (
    eq, ne, le, lt, gt, ge,
    logical_and, logical_or, logical_not, logical_xor,
    bitwise_not, bitwise_and, bitwise_or, bitwise_xor,
    equal, allclose, isclose,
)

from .reduce import (
    argmax, argmin, median, kthvalue, searchsorted, unique,
    amax, amin, count_nonzero, all_, any_,
    min_, max_, maximum, minimum, fmin, fmax,
    cumsum, cumprod, cummax, argsort, sort, topk,
    sum_, mean, var_, std_, norm_, prod_,
    cummin_op, logsumexp_op, renorm_op, nansum,
    aminmax_op, nanmean_op, argwhere_op,
    quantile_op, nanquantile_op, nanmedian_op,
    aminmax_aclnn,
)

from .shape import (
    flatten_op, contiguous, flip, roll, rot90,
    repeat, tile, repeat_interleave,
    tril, triu, tril_indices, triu_indices,
    diag, cartesian_prod, block_diag,
    broadcast_to_op, movedim_op, moveaxis_op,
    unflatten_op, diagonal_op, one_hot,
    scatter, nonzero,
    cat, concatenate, stack, pad_sequence,
    chunk, split, vsplit, hsplit, dsplit,
    unbind, hstack, vstack, row_stack, dstack, column_stack,
    getitem, setitem,
    gather, index_select, take, take_along_dim, masked_select,
    narrow, select, expand,
    masked_fill, masked_fill_,
    index_put_, index_put, index_copy_, index_fill_, index_add_,
    scatter_, scatter_add_, masked_scatter_,
    unfold,
)

from .activation import (
    relu, relu_, relu6, softplus, hardtanh,
    silu, gelu, leaky_relu, elu, mish, prelu,
    selu_op, celu_op, threshold_op,
    hardshrink_op, softshrink_op, hardswish_op, hardsigmoid_op,
    softsign_op, rrelu_op,
    softmax, log_softmax,
    embedding,
    dropout,
)

from .norm import (
    layer_norm, batch_norm, group_norm,
    instance_norm, rms_norm, normalize_op,
)

from .linalg import (
    matmul, dot, mv, outer, mm_op, bmm_op,
    addmm, baddbmm, einsum_,
    linalg_qr, linalg_inv, linalg_vector_norm_op,
    linalg_norm_op, linalg_matrix_norm_op,
    linalg_multi_dot_op, linalg_matrix_power_op, linalg_vander_op,
    linalg_cholesky_op, linalg_cond_op, linalg_det_op, linalg_slogdet_op,
    linalg_eig_op, linalg_eigh_op, linalg_eigvals_op, linalg_eigvalsh_op,
    linalg_householder_product_op, linalg_lstsq_op,
    linalg_lu_op, linalg_lu_factor_op, linalg_lu_solve_op,
    linalg_matrix_exp_op, linalg_matrix_rank_op,
    linalg_pinv_op, linalg_solve_op, linalg_solve_triangular_op,
    linalg_svd_op, linalg_svdvals_op,
    linalg_tensorinv_op, linalg_tensorsolve_op,
    matrix_power_op, det_op, inner_op, tensordot_op,
    trace_op, cross_op, dist_op, cdist_op,
)

from .conv import (
    conv2d, conv1d, conv_transpose2d, conv_transpose1d,
    conv3d_op, conv_transpose3d_op,
    max_pool2d, max_pool3d, avg_pool2d,
    adaptive_avg_pool2d, adaptive_max_pool2d,
    adaptive_avg_pool3d_op, avg_pool3d_op,
    adaptive_avg_pool1d_op, avg_pool1d_op, max_pool1d_op, adaptive_max_pool1d_op,
    upsample_nearest2d, upsample_bilinear2d,
    upsample_bicubic2d_op, upsample_linear1d_op, upsample_nearest1d_op,
    im2col_op, col2im_op,
    grid_sample_op, affine_grid_op,
    pad, ctc_loss_op,
)

from .random import (
    randperm, zero_, uniform_, normal_, randint_, random_,
    bernoulli_, exponential_, log_normal_, cauchy_, geometric_,
    fill_, clamp_, copy_, erfinv_, reciprocal_,
)

from .elementwise import (
    where, lerp, addcmul, addcdiv,
    logaddexp, logaddexp2, hypot,
    remainder, fmod,
    clamp, clamp_min, clamp_max,
    heaviside_op, uniform_op, isreal_op,
    isin_op, bucketize_op, diff_op,
    bincount_op, bincount_aclnn, histc_op, histogram_op,
)

from .special import (
    special_digamma, special_erfinv, special_gammaln, special_sinc,
    special_entr_op, special_erfcx_op, special_logit_op,
    special_ndtr_op, special_log_ndtr_op,
    special_xlogy_op, special_xlog1py_op, special_multigammaln_op,
    special_i0_op, special_i0e_op,
    special_i1_op, special_i1e_op, special_ndtri_op,
    special_polygamma_op, special_zeta_op,
    special_gammainc_op, special_gammaincc_op,
    fft_fft_op, fft_ifft_op, fft_rfft_op, fft_irfft_op,
    fft_fft2_op, fft_ifft2_op, fft_rfft2_op, fft_irfft2_op,
    fft_fftn_op, fft_ifftn_op, fft_rfftn_op, fft_irfftn_op,
    fft_hfft_op, fft_ihfft_op, fft_fftshift_op, fft_ifftshift_op,
)

from .optim import (
    _adam_step_op, _adamw_step_op, _sgd_step_op,
    _adagrad_step_op, _rmsprop_step_op, _adadelta_step_op,
    _adamax_step_op, _asgd_step_op, _nadam_step_op,
    _radam_step_op, _rprop_step_op, _sparse_adam_step_op,
)

