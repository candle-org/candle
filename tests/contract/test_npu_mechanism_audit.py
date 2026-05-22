import re
from pathlib import Path

import candle
from candle._dispatch.keys import DispatchKey
from candle._dispatch.registry import registry


_REPO_ROOT = Path(candle.__file__).resolve().parents[2]


def _source(path):
    return (_REPO_ROOT / path).read_text(encoding="utf-8")


def _npu_forward_ops():
    ops = set()
    for name, entry in registry._ops.items():
        if DispatchKey.NPU in entry.kernels:
            ops.add(name.split("::")[-1])
    return ops


def _npu_autograd_ops():
    ops = set()
    for name, entry in registry._ops.items():
        if DispatchKey.AutogradNPU in entry.kernels or DispatchKey.Autograd in entry.kernels:
            ops.add(name.split("::")[-1])
    return ops


def test_cython_and_python_npu_dispatch_key_constants_stay_in_sync():
    dispatch_src = _source("src/candle/_C/_dispatch.pyx")
    core_src = _source("src/candle/_C/_dispatcher_core.pyx")

    expected = {
        "NPU": int(DispatchKey.NPU).bit_length() - 1,
        "AUTOGRAD_NPU": int(DispatchKey.AutogradNPU).bit_length() - 1,
    }
    for name, shift in expected.items():
        pattern = rf"DEF _DK_{name}\s*=\s*1\s*<<\s*{shift}\b"
        assert re.search(pattern, dispatch_src), f"_dispatch.pyx {name} key constant drifted"
        assert re.search(pattern, core_src), f"_dispatcher_core.pyx {name} key constant drifted"


def _function_source(src, name):
    pattern = rf"^def {name}\(.*?(?=^def |\Z)"
    match = re.search(pattern, src, flags=re.MULTILINE | re.DOTALL)
    assert match is not None, f"missing function {name}"
    return match.group(0)


def test_npu_autograd_overrides_do_not_use_cpu_fallbacks():
    autograd_src = _source("src/candle/_backends/autograd.py")
    npu_override_src = autograd_src.split("# NPU ACLNN fused backward kernels", 1)[1]

    forbidden = ["from .cpu", "import numpy", "_to_numpy", "_from_numpy"]
    for marker in forbidden:
        assert marker not in npu_override_src


def test_npu_forward_paths_do_not_copy_registered_ops_to_cpu():
    checked = {
        "src/candle/_backends/npu/ops/comparison.py": ["allclose"],
        "src/candle/_backends/npu/ops/elementwise.py": ["hypot"],
        "src/candle/_backends/npu/ops/reduce.py": ["fmin", "fmax"],
    }
    forbidden = ['.to("cpu")', ".to('cpu')", "_copy_npu_to_cpu"]

    for path, names in checked.items():
        src = _source(path)
        for name in names:
            body = _function_source(src, name)
            for marker in forbidden:
                assert marker not in body


def test_npu_operator_parity_shims_delegate_to_cython():
    elementwise_src = _source("src/candle/_backends/npu/ops/elementwise.py")
    reduce_src = _source("src/candle/_backends/npu/ops/reduce.py")
    activation_src = _source("src/candle/_backends/npu/ops/activation.py")
    math_src = _source("src/candle/_backends/npu/ops/math.py")

    hypot_body = _function_source(elementwise_src, "hypot")
    assert "_fast_hypot_impl" in hypot_body
    assert "sqrt(add(mul(" not in hypot_body

    for name, fast_name in {"fmin": "_fast_fmin_impl", "fmax": "_fast_fmax_impl"}.items():
        body = _function_source(reduce_src, name)
        assert fast_name in body
        assert "where(" not in body

    activation_fast_names = {
        "relu6": "_fast_relu6_impl",
        "selu_op": "_fast_selu_impl",
        "celu_op": "_fast_celu_impl",
        "threshold_op": "_fast_threshold_impl",
        "hardshrink_op": "_fast_hardshrink_impl",
        "softshrink_op": "_fast_softshrink_impl",
        "hardswish_op": "_fast_hardswish_impl",
        "hardsigmoid_op": "_fast_hardsigmoid_impl",
        "softsign_op": "_fast_softsign_impl",
        "rrelu_op": "_fast_rrelu_impl",
    }
    forbidden = ["return where(", "return clamp(", "= where(", "= clamp(", "return mul(", "return div(", "return add(", "return sub("]
    for name, fast_name in activation_fast_names.items():
        body = _function_source(activation_src, name)
        assert fast_name in body
        for marker in forbidden:
            assert marker not in body

    for name, fast_name in {"frac": "_fast_frac_impl", "reciprocal": "_fast_reciprocal_impl"}.items():
        body = _function_source(math_src, name)
        assert fast_name in body
        for marker in forbidden:
            assert marker not in body


def test_bulk_npu_parity_shims_delegate_to_cython():
    math_src = _source("src/candle/_backends/npu/ops/math.py")
    comparison_src = _source("src/candle/_backends/npu/ops/comparison.py")
    special_src = _source("src/candle/_backends/npu/ops/special.py")

    expectations = {
        math_src: {
            "square": "_fast_square_impl",
            "trunc": "_fast_trunc_impl",
            "isposinf": "_fast_isposinf_impl",
            "isneginf": "_fast_isneginf_impl",
            "isinf": "_fast_isinf_impl",
            "isnan": "_fast_isnan_impl",
        },
        comparison_src: {
            "isclose": "_fast_isclose_impl",
        },
        special_src: {
            "special_sinc": "_fast_special_sinc_impl",
            "special_erfcx_op": "_fast_special_erfcx_impl",
            "special_logit_op": "_fast_special_logit_impl",
            "special_ndtr_op": "_fast_special_ndtr_impl",
            "special_log_ndtr_op": "_fast_special_log_ndtr_impl",
            "special_xlogy_op": "_fast_special_xlogy_impl",
            "special_xlog1py_op": "_fast_special_xlog1py_impl",
        },
    }
    for src, mapping in expectations.items():
        for name, fast_name in mapping.items():
            body = _function_source(src, name)
            assert fast_name in body, f"{name} does not delegate to {fast_name}"


def test_npu_bulk_fast_helpers_have_no_python_fallback_bodies():
    math_src = _source("src/candle/_backends/npu/ops/math.py")

    expectations = {
        "square": "_fast_square_impl",
        "isinf": "_fast_isinf_impl",
        "isnan": "_fast_isnan_impl",
        "isposinf": "_fast_isposinf_impl",
        "isneginf": "_fast_isneginf_impl",
        "trunc": "_fast_trunc_impl",
    }
    forbidden = [
        "return _unary_op(",
        "return _binary_op(",
        "aclnn.",
        "_wrap_tensor(",
        "npu_runtime._alloc_device",
        "_unwrap_storage(",
    ]
    for name, fast_name in expectations.items():
        body = _function_source(math_src, name)
        assert fast_name in body, f"{name} does not delegate to {fast_name}"
        for marker in forbidden:
            assert marker not in body


def test_npu_special_gamma_and_erfinv_wrappers_delegate_to_cython():
    special_src = _source("src/candle/_backends/npu/ops/special.py")
    random_src = _source("src/candle/_backends/npu/ops/random.py")

    special_expectations = {
        "special_digamma": "_fast_digamma_impl",
        "special_erfinv": "_fast_erfinv_impl",
        "special_gammaln": "_fast_lgamma_impl",
    }
    forbidden = [
        "return _unary_op(",
        "return _binary_op(",
        "aclnn.",
        "_wrap_tensor(",
        "npu_runtime._alloc_device",
        "_unwrap_storage(",
    ]
    for name, fast_name in special_expectations.items():
        body = _function_source(special_src, name)
        assert fast_name in body, f"{name} does not delegate to {fast_name}"
        for marker in forbidden:
            assert marker not in body

    erfinv_body = _function_source(random_src, "erfinv_")
    assert "_fast_erfinv_impl" in erfinv_body
    for marker in forbidden:
        assert marker not in erfinv_body


def test_npu_comparison_thin_wrappers_delegate_to_cython():
    comparison_src = _source("src/candle/_backends/npu/ops/comparison.py")
    expectations = {
        "eq": "_fast_eq_impl",
        "ne": "_fast_ne_impl",
        "le": "_fast_le_impl",
        "lt": "_fast_lt_impl",
        "gt": "_fast_gt_impl",
        "ge": "_fast_ge_impl",
        "logical_and": "_fast_logical_and_impl",
        "logical_or": "_fast_logical_or_impl",
        "logical_xor": "_fast_logical_xor_impl",
        "bitwise_and": "_fast_bitwise_and_impl",
        "bitwise_or": "_fast_bitwise_or_impl",
        "bitwise_xor": "_fast_bitwise_xor_impl",
    }
    forbidden = ["return _binary_op(", "aclnn.", "_unwrap_storage(", "_wrap_tensor(", "npu_runtime._alloc_device"]
    for name, fast_name in expectations.items():
        body = _function_source(comparison_src, name)
        assert fast_name in body, f"{name} does not delegate to {fast_name}"
        for marker in forbidden:
            assert marker not in body


def test_npu_thin_binary_wrappers_delegate_to_cython():
    math_src = _source("src/candle/_backends/npu/ops/math.py")
    elementwise_src = _source("src/candle/_backends/npu/ops/elementwise.py")
    reduce_src = _source("src/candle/_backends/npu/ops/reduce.py")
    expectations = {
        math_src: {
            "sub": "_fast_sub_impl",
            "div": "_fast_div_impl",
            "_pow_tensor_scalar_op": "_fast_pow_tensor_scalar_impl",
            "pow": "_fast_pow_impl",
            "floor_divide": "_fast_floor_divide_impl",
        },
        elementwise_src: {
            "logaddexp": "_fast_logaddexp_impl",
            "logaddexp2": "_fast_logaddexp2_impl",
            "fmod": "_fast_fmod_impl",
            "remainder": "_fast_remainder_impl",
        },
        reduce_src: {
            "maximum": "_fast_maximum_impl",
            "minimum": "_fast_minimum_impl",
            "min_": "_fast_minimum_impl",
            "max_": "_fast_maximum_impl",
        },
    }
    for src, mapping in expectations.items():
        for name, fast_name in mapping.items():
            body = _function_source(src, name)
            assert fast_name in body, f"{name} does not delegate to {fast_name}"
            assert "return _binary_op(" not in body
            assert "_binary_op(" not in body, f"{name} still references _binary_op"
            assert "aclnn." not in body, f"{name} still references aclnn"


def test_npu_existing_fast_helpers_have_no_python_fallback_bodies():
    math_src = _source("src/candle/_backends/npu/ops/math.py")
    comparison_src = _source("src/candle/_backends/npu/ops/comparison.py")
    elementwise_src = _source("src/candle/_backends/npu/ops/elementwise.py")

    expectations = {
        math_src: {
            "add": "_fast_add_impl",
            "mul": "_fast_mul_impl",
        },
        comparison_src: {
            "bitwise_not": "_fast_bitwise_not_impl",
        },
        elementwise_src: {
            "addcmul": "_fast_addcmul_impl",
            "addcdiv": "_fast_addcdiv_impl",
        },
    }
    forbidden = [
        "return _unary_op(",
        "return _binary_op(",
        "aclnn.",
        "_wrap_tensor(",
        "npu_runtime._alloc_device",
        "_unwrap_storage(",
        "_to_numpy(",
    ]
    for src, mapping in expectations.items():
        for name, fast_name in mapping.items():
            body = _function_source(src, name)
            assert fast_name in body, f"{name} does not delegate to {fast_name}"
            for marker in forbidden:
                assert marker not in body


def test_npu_unary_math_wrappers_delegate_to_cython():
    math_src = _source("src/candle/_backends/npu/ops/math.py")
    comparison_src = _source("src/candle/_backends/npu/ops/comparison.py")

    math_expectations = {
        "abs": "_fast_abs_impl",
        "neg": "_fast_neg_impl",
        "sign": "_fast_sign_impl",
        "exp": "_fast_exp_impl",
        "log": "_fast_log_impl",
        "sqrt": "_fast_sqrt_impl",
        "rsqrt": "_fast_rsqrt_impl",
        "sin": "_fast_sin_impl",
        "cos": "_fast_cos_impl",
        "tan": "_fast_tan_impl",
        "tanh": "_fast_tanh_impl",
        "sigmoid": "_fast_sigmoid_impl",
        "sinh": "_fast_sinh_impl",
        "cosh": "_fast_cosh_impl",
        "erf": "_fast_erf_impl",
        "erfc": "_fast_erfc_impl",
        "floor": "_fast_floor_impl",
        "ceil": "_fast_ceil_impl",
        "round": "_fast_round_impl",
        "log2": "_fast_log2_impl",
        "log10": "_fast_log10_impl",
        "exp2": "_fast_exp2_impl",
        "asinh": "_fast_asinh_impl",
        "acosh": "_fast_acosh_impl",
        "atanh": "_fast_atanh_impl",
        "atan": "_fast_atan_impl",
        "asin": "_fast_asin_impl",
        "acos": "_fast_acos_impl",
        "expm1": "_fast_expm1_impl",
        "log1p": "_fast_log1p_impl",
        "signbit": "_fast_signbit_impl",
        "isfinite": "_fast_isfinite_impl",
    }
    forbidden = ["return _unary_op(", "aclnn.", "_wrap_tensor(", "npu_runtime._alloc_device", "_unwrap_storage("]
    for name, fast_name in math_expectations.items():
        body = _function_source(math_src, name)
        assert fast_name in body, f"{name} does not delegate to {fast_name}"
        for marker in forbidden:
            assert marker not in body

    logical_not_body = _function_source(comparison_src, "logical_not")
    assert "_fast_logical_not_impl" in logical_not_body
    for marker in forbidden:
        assert marker not in logical_not_body


def test_npu_soc_guarded_fast_helpers_do_not_keep_native_python_fallbacks():
    activation_src = _source("src/candle/_backends/npu/ops/activation.py")
    elementwise_src = _source("src/candle/_backends/npu/ops/elementwise.py")
    math_src = _source("src/candle/_backends/npu/ops/math.py")

    expectations = {
        activation_src: {
            "softplus": "_fast_softplus_impl",
            "hardtanh": "_fast_hardtanh_impl",
            "mish": "_fast_mish_impl",
            "dropout": "_fast_dropout_impl",
        },
        elementwise_src: {
            "where": "_fast_where_impl",
            "lerp": "_fast_lerp_tensor_impl",
        },
        math_src: {
            "atan2": "_fast_atan2_impl",
        },
    }
    forbidden = [
        "return _unary_op(",
        "return _binary_op(",
        "aclnn.",
        "npu_runtime._alloc_device",
        "_unwrap_storage(",
        "_wrap_tensor(",
    ]
    for src, mapping in expectations.items():
        for name, fast_name in mapping.items():
            body = _function_source(src, name)
            assert fast_name in body, f"{name} does not delegate to {fast_name}"
            for marker in forbidden:
                assert marker not in body


def test_npu_activation_native_wrappers_delegate_to_cython():
    activation_src = _source("src/candle/_backends/npu/ops/activation.py")
    expectations = {
        "relu": "_fast_relu_impl",
        "silu": "_fast_silu_impl",
        "gelu": "_fast_gelu_impl",
        "leaky_relu": "_fast_leaky_relu_impl",
        "elu": "_fast_elu_impl",
        "prelu": "_fast_prelu_impl",
        "softmax": "_fast_softmax_impl",
        "log_softmax": "_fast_log_softmax_impl",
        "embedding": "_fast_embedding_impl",
    }
    forbidden = ["return _unary_op(", "aclnn.", "_wrap_tensor(", "npu_runtime._alloc_device", "_unwrap_storage("]
    for name, fast_name in expectations.items():
        body = _function_source(activation_src, name)
        assert fast_name in body, f"{name} does not delegate to {fast_name}"
        for marker in forbidden:
            assert marker not in body

def test_npu_clamp_wrappers_delegate_to_cython():
    elementwise_src = _source("src/candle/_backends/npu/ops/elementwise.py")
    expectations = {
        "clamp": "_fast_clamp_impl",
        "clamp_min": "_fast_clamp_min_impl",
        "clamp_max": "_fast_clamp_max_impl",
    }
    forbidden = ["aclnn.", "npu_runtime._alloc_device", "_unwrap_storage(", "_wrap_tensor("]
    for name, fast_name in expectations.items():
        body = _function_source(elementwise_src, name)
        assert fast_name in body, f"{name} does not delegate to {fast_name}"
        for marker in forbidden:
            assert marker not in body


def test_npu_inplace_arithmetic_wrappers_delegate_to_cython():
    math_src = _source("src/candle/_backends/npu/ops/math.py")
    expectations = {
        "add_": "_fast_add_inplace_impl",
        "mul_": "_fast_mul_inplace_impl",
        "sub_": "_fast_sub_inplace_impl",
        "div_": "_fast_div_inplace_impl",
    }
    forbidden = ["aclnn.", "npu_runtime._alloc_device", "_unwrap_storage(", "_wrap_tensor("]
    for name, fast_name in expectations.items():
        body = _function_source(math_src, name)
        assert fast_name in body, f"{name} does not delegate to {fast_name}"
        for marker in forbidden:
            assert marker not in body


def test_npu_inplace_copy_and_clamp_wrappers_delegate_to_cython():
    random_src = _source("src/candle/_backends/npu/ops/random.py")
    expectations = {
        "fill_": "_fast_fill_inplace_impl",
        "copy_": "_fast_copy_inplace_impl",
        "clamp_": "_fast_clamp_inplace_impl",
    }
    forbidden = ["aclnn.", "npu_runtime._alloc_device", "_unwrap_storage(", "_wrap_tensor("]
    for name, fast_name in expectations.items():
        body = _function_source(random_src, name)
        assert fast_name in body, f"{name} does not delegate to {fast_name}"
        for marker in forbidden:
            assert marker not in body


def test_npu_zero_and_reciprocal_inplace_have_no_python_fallback_bodies():
    random_src = _source("src/candle/_backends/npu/ops/random.py")
    forbidden = ["aclnn.", "npu_runtime._alloc_device", "_unwrap_storage(", "_wrap_tensor("]

    zero_body = _function_source(random_src, "zero_")
    assert "fill_(" in zero_body, "zero_ should delegate to fill_"
    for marker in forbidden:
        assert marker not in zero_body, f"zero_ still references {marker}"

    reciprocal_body = _function_source(random_src, "reciprocal_")
    assert "_fast_copy_inplace_impl" in reciprocal_body, \
        "reciprocal_ does not delegate to _fast_copy_inplace_impl"
    for marker in forbidden:
        assert marker not in reciprocal_body, f"reciprocal_ still references {marker}"


def test_npu_random_integer_inplace_wrappers_delegate_floor_to_cython():
    random_src = _source("src/candle/_backends/npu/ops/random.py")
    forbidden = ["aclnn.", "npu_runtime._alloc_device", "_unwrap_storage(", "_wrap_tensor("]
    for name in ["randint_", "random_"]:
        body = _function_source(random_src, name)
        assert "_fast_floor_inplace_impl" in body, \
            f"{name} does not delegate floor to _fast_floor_inplace_impl"
        for marker in forbidden:
            assert marker not in body, f"{name} still references {marker}"


def test_npu_log_normal_inplace_exp_delegates_to_cython():
    random_src = _source("src/candle/_backends/npu/ops/random.py")
    body = _function_source(random_src, "log_normal_")
    assert "_fast_exp_inplace_impl" in body, \
        "log_normal_ does not delegate exp to _fast_exp_inplace_impl"
    forbidden = ["aclnn.", "npu_runtime._alloc_device", "_unwrap_storage(", "_wrap_tensor("]
    for marker in forbidden:
        assert marker not in body, f"log_normal_ still references {marker}"


def test_npu_exponential_inplace_delegates_to_cython():
    random_src = _source("src/candle/_backends/npu/ops/random.py")
    body = _function_source(random_src, "exponential_")
    for fast_name in [
        "_fast_log_inplace_impl",
        "_fast_neg_inplace_impl",
        "_fast_mul_inplace_impl",
    ]:
        assert fast_name in body, f"exponential_ does not delegate to {fast_name}"
    forbidden = ["aclnn.", "npu_runtime._alloc_device", "_unwrap_storage(", "_wrap_tensor("]
    for marker in forbidden:
        assert marker not in body, f"exponential_ still references {marker}"


def test_npu_bernoulli_inplace_delegates_to_device_ops_and_cython_copy():
    random_src = _source("src/candle/_backends/npu/ops/random.py")
    body = _function_source(random_src, "bernoulli_")
    for expected in ["lt(", "_cast_tensor_dtype(", "_fast_copy_inplace_impl"]:
        assert expected in body, f"bernoulli_ does not delegate to {expected}"
    forbidden = ["aclnn.", "npu_runtime._alloc_device", "_unwrap_storage(", "_wrap_tensor("]
    for marker in forbidden:
        assert marker not in body, f"bernoulli_ still references {marker}"


def test_npu_geometric_inplace_delegates_to_cython():
    random_src = _source("src/candle/_backends/npu/ops/random.py")
    body = _function_source(random_src, "geometric_")
    for fast_name in [
        "_fast_log_inplace_impl",
        "_fast_div_inplace_impl",
        "_fast_ceil_inplace_impl",
    ]:
        assert fast_name in body, f"geometric_ does not delegate to {fast_name}"
    forbidden = ["aclnn.", "npu_runtime._alloc_device", "_unwrap_storage(", "_wrap_tensor("]
    for marker in forbidden:
        assert marker not in body, f"geometric_ still references {marker}"


def test_npu_cauchy_inplace_delegates_to_cython():
    random_src = _source("src/candle/_backends/npu/ops/random.py")
    body = _function_source(random_src, "cauchy_")
    for fast_name in [
        "_fast_sub_inplace_impl",
        "_fast_mul_inplace_impl",
        "_fast_tan_inplace_impl",
        "_fast_add_inplace_impl",
    ]:
        assert fast_name in body, f"cauchy_ does not delegate to {fast_name}"
    forbidden = ["aclnn.", "npu_runtime._alloc_device", "_unwrap_storage(", "_wrap_tensor("]
    for marker in forbidden:
        assert marker not in body, f"cauchy_ still references {marker}"


def test_npu_random_inplace_shims_have_no_dispatch_redundant_device_guard():
    random_src = _source("src/candle/_backends/npu/ops/random.py")
    for name in ["zero_", "uniform_", "normal_", "reciprocal_"]:
        body = _function_source(random_src, name)
        assert 'a.device.type != "npu"' not in body, (
            f"{name} still has dispatch-redundant device guard"
        )


def test_npu_relu_inplace_delegates_to_cython():
    activation_src = _source("src/candle/_backends/npu/ops/activation.py")
    body = _function_source(activation_src, "relu_")
    assert "_fast_relu_inplace_impl" in body, (
        "relu_ does not delegate to _fast_relu_inplace_impl"
    )
    forbidden = [
        "aclnn.",
        "npu_runtime._alloc_device",
        "npu_runtime.memcpy_d2d",
        "_unwrap_storage(",
    ]
    for marker in forbidden:
        assert marker not in body, f"relu_ still references {marker}"


def test_npu_single_tensor_unary_shims_have_no_dispatch_redundant_device_guard():
    targets = [
        ("src/candle/_backends/npu/ops/activation.py", "softplus"),
        ("src/candle/_backends/npu/ops/math.py", "isnan"),
        ("src/candle/_backends/npu/ops/elementwise.py", "clamp"),
        ("src/candle/_backends/npu/ops/elementwise.py", "clamp_min"),
        ("src/candle/_backends/npu/ops/elementwise.py", "clamp_max"),
        ("src/candle/_backends/npu/ops/_helpers.py", "_unary_op"),
    ]
    for path, name in targets:
        src = _source(path)
        body = _function_source(src, name)
        assert 'a.device.type != "npu"' not in body, (
            f"{path}::{name} still has dispatch-redundant device guard"
        )


def test_npu_reduce_single_tensor_shims_have_no_dispatch_redundant_device_guard():
    reduce_src = _source("src/candle/_backends/npu/ops/reduce.py")
    targets = [
        "argmax",
        "argmin",
        "median",
        "kthvalue",
        "amax",
        "amin",
        "count_nonzero",
        "all_",
        "any_",
        "unique",
        "sum_",
        "cumsum",
        "cumprod",
        "cummax",
        "argsort",
        "sort",
        "topk",
        "nansum",
    ]
    for name in targets:
        body = _function_source(reduce_src, name)
        assert 'a.device.type != "npu"' not in body, (
            f"reduce.py::{name} still has dispatch-redundant device guard"
        )


def test_npu_shape_single_tensor_shims_have_no_dispatch_redundant_device_guard():
    shape_src = _source("src/candle/_backends/npu/ops/shape.py")
    targets = [
        "_slice_along_dim",
        "contiguous",
        "flip",
        "roll",
        "rot90",
        "repeat",
        "repeat_interleave",
        "tril",
        "triu",
        "diag",
        "scatter",
        "nonzero",
    ]
    for name in targets:
        body = _function_source(shape_src, name)
        assert 'a.device.type != "npu"' not in body, (
            f"shape.py::{name} still has dispatch-redundant device guard"
        )


def test_npu_conv_single_tensor_shims_have_no_dispatch_redundant_device_guard():
    conv_src = _source("src/candle/_backends/npu/ops/conv.py")
    targets = [
        "pad",
    ]
    for name in targets:
        body = _function_source(conv_src, name)
        assert 'input.device.type != "npu"' not in body, (
            f"conv.py::{name} still has dispatch-redundant device guard on `input`"
        )


def test_npu_multi_input_shims_drop_dispatched_first_arg_guard():
    """Multi-input NPU shims must not include the dispatcher-routed first arg in
    cross-device parity checks. The dispatcher has already validated the primary
    arg before the shim runs, so `<primary>.device.type != "npu" or` is dead
    code on the normal path. Cross-input parity for secondary args is still
    required.
    """
    expectations = [
        ("src/candle/_backends/npu/ops/linalg.py", "matmul", "a"),
        ("src/candle/_backends/npu/ops/linalg.py", "dot", "a"),
        ("src/candle/_backends/npu/ops/linalg.py", "mv", "a"),
        ("src/candle/_backends/npu/ops/linalg.py", "outer", "a"),
        ("src/candle/_backends/npu/ops/reduce.py", "searchsorted", "sorted_sequence"),
        ("src/candle/_backends/npu/ops/comparison.py", "equal", "a"),
        ("src/candle/_backends/npu/ops/_helpers.py", "_binary_op_slow", "a"),
    ]
    for path, name, primary in expectations:
        body = _function_source(_source(path), name)
        forbidden = f'{primary}.device.type != "npu" or'
        assert forbidden not in body, (
            f"{path}::{name} still has dispatch-redundant first-arg device guard: "
            f"{forbidden}"
        )


def test_npu_multi_input_shims_drop_standalone_primary_arg_guard():
    """Multi-input NPU shims must not include a standalone `if <primary>.device.type
    != "npu":` guard on the dispatcher-routed primary arg. The dispatcher has
    already validated the primary's device key before the shim runs, so the
    standalone guard is dead code on the normal path. Cross-input parity
    checks on secondary args remain required.
    """
    expectations = [
        # (path, function, primary-arg local name)
        ("src/candle/_backends/npu/ops/elementwise.py", "where", "x"),
        ("src/candle/_backends/npu/ops/shape.py", "pad_sequence", "first"),
    ]
    for path, name, primary in expectations:
        body = _function_source(_source(path), name)
        forbidden = f'if {primary}.device.type != "npu":'
        assert forbidden not in body, (
            f"{path}::{name} still has dispatch-redundant standalone primary-arg "
            f"device guard: `{forbidden}`"
        )


def test_npu_sequence_input_shims_skip_dispatched_primary_in_parity_loop():
    """`cartesian_prod` and `block_diag` are dispatched by
    `tensors[0].device.type`. Their cross-input parity loop must iterate over
    `tensors[1:]` so the dispatch-redundant device/dtype check against
    `tensors[0]` (which is also `first` for the dtype comparison and would be
    vacuous) is skipped. The dim check on `first` must happen standalone.
    """
    shape_src = _source("src/candle/_backends/npu/ops/shape.py")
    for name in ("cartesian_prod", "block_diag"):
        body = _function_source(shape_src, name)
        assert "for t in tensors[1:]:" in body, (
            f"shape.py::{name} should iterate cross-input parity over "
            f"`tensors[1:]` so it skips the dispatched primary `tensors[0]`"
        )


def test_npu_pad_sequence_lifts_dispatch_redundant_validation_out_of_memcpy_loop():
    """`pad_sequence` is dispatched by `seqs[0].device.type`. The validation
    block for `seqs[0]` (device dispatch-redundant, dtype vacuous, trailing
    shape vacuous) must be lifted out of the `enumerate(seqs)` memcpy loop and
    iterated over `seqs[1:]` instead. The memcpy loop still needs index `i`
    for offsets so it remains over `enumerate(seqs)`.
    """
    shape_src = _source("src/candle/_backends/npu/ops/shape.py")
    body = _function_source(shape_src, "pad_sequence")
    assert "for t in seqs[1:]:" in body, (
        "shape.py::pad_sequence should validate cross-input parity in a "
        "separate `for t in seqs[1:]:` pre-loop so the dispatched primary "
        "`seqs[0]` is skipped"
    )


def test_npu_std_sqrt_delegates_through_cython_sqrt_shim():
    reduce_src = _source("src/candle/_backends/npu/ops/reduce.py")
    body = _function_source(reduce_src, "std_")
    forbidden = ["_unary_op(", "aclnn.sqrt"]
    for marker in forbidden:
        assert marker not in body, (
            f"reduce.py::std_ still references {marker}; must delegate to Cython-backed sqrt"
        )


def test_npu_cython_fast_helpers_have_no_dispatch_redundant_device_guard():
    src = _source("src/candle/_C/_npu_ops.pyx")
    # Standalone single-tensor `if a.device.type != "npu":` guards inside
    # `fast_*` helpers are dispatch-redundant — the helpers are only entered
    # through Python NPU shims that the dispatcher already routed by NPU
    # device key. Multi-tensor guards (cond/x/y on where, weight on lerp,
    # third-input c on addcmul/addcdiv, a-and-weight on prelu) validate
    # cross-input device parity and remain.
    standalone_guard = re.compile(
        r'^\s*if a\.device\.type != "npu":\s*$', flags=re.M
    )
    for match in standalone_guard.finditer(src):
        line_start = src.rfind("\n", 0, match.start()) + 1
        line_no = src.count("\n", 0, line_start) + 1
        raise AssertionError(
            f"_npu_ops.pyx:{line_no} still has dispatch-redundant single-tensor "
            f"device guard: {src[line_start:match.end()].rstrip()}"
        )


def test_core_npu_training_ops_have_forward_and_autograd_registration():
    forward_ops = _npu_forward_ops()
    autograd_ops = _npu_autograd_ops()
    required = {
        "add",
        "mul",
        "matmul",
        "relu",
        "sum",
        "mean",
        "reshape",
        "view",
        "transpose",
        "permute",
        "slice",
    }

    assert required <= forward_ops
    assert required <= autograd_ops


def test_npu_forward_autograd_registration_inventory_is_explicit():
    forward_ops = _npu_forward_ops()
    autograd_ops = _npu_autograd_ops()
    missing_autograd = forward_ops - autograd_ops

    expected_missing = {
        "_adadelta_step",
        "_adagrad_step",
        "_adam_step",
        "_adamax_step",
        "_adamw_step",
        "_asgd_step",
        "_nadam_step",
        "_radam_step",
        "_rmsprop_step",
        "_rprop_step",
        "_sgd_step",
        "_sparse_adam_step",
        "allclose",
        "arange",
        "argmax",
        "argmin",
        "argsort",
        "argwhere",
        "as_strided_copy",
        "bincount",
        "bitwise_and",
        "bitwise_not",
        "bitwise_or",
        "bitwise_xor",
        "bucketize",
        "cartesian_prod",
        "empty",
        "empty_like",
        "equal",
        "erfinv_",
        "expand_copy",
        "eye",
        "flatten",
        "full",
        "full_like",
        "histogram",
        "isclose",
        "isfinite",
        "isinf",
        "isin",
        "isneginf",
        "isposinf",
        "isreal",
        "linspace",
        "logical_and",
        "logical_not",
        "logical_or",
        "logical_xor",
        "logspace",
        "movedim",
        "narrow",
        "ones",
        "ones_like",
        "rand",
        "rand_like",
        "randint",
        "randint_",
        "randint_like",
        "randn",
        "randn_like",
        "randperm",
        "range",
        "reciprocal_",
        "searchsorted",
        "slice_copy",
        "squeeze",
        "tensor",
        "tril_indices",
        "triu_indices",
        "unflatten",
        "zeros",
        "zeros_like",
    }

    assert missing_autograd == expected_missing
    assert len(forward_ops) == 396
    assert len(autograd_ops & forward_ops) == 324
