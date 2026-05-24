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


def test_npu_special_unary_wrappers_collapse_to_cython_aliases():
    """Audit: `special_digamma`, `special_erfinv`, and `special_gammaln`
    in `_backends/npu/ops/special.py` are 2-line `def` wrappers whose bodies
    only return their module-level Cython helpers. Their schemas each take a
    single tensor input, identical to the imported helper signatures, so the
    wrappers add Python call frames without changing behavior.

    Collapse them to module-level aliases:

      special_digamma = _fast_digamma_impl
      special_erfinv = _fast_erfinv_impl
      special_gammaln = _fast_lgamma_impl

    Keep the names intact so `_backends/npu/__init__.py` registrations and the
    internal `special_multigammaln_op()` call site continue to work.
    """
    special_src = _source("src/candle/_backends/npu/ops/special.py")

    expectations = {
        "special_digamma": "_fast_digamma_impl",
        "special_erfinv": "_fast_erfinv_impl",
        "special_gammaln": "_fast_lgamma_impl",
    }
    for name, fast_name in expectations.items():
        assert f"def {name}(" not in special_src, (
            f"`{name}` is still defined as a `def` wrapper in `special.py`. "
            f"Replace it with the module-level alias `{name} = {fast_name}`."
        )
        assert f"{name} = {fast_name}" in special_src, (
            f"`special.py` must bind `{name}` to `{fast_name}` via a "
            "module-level alias so the registration/import surface stays "
            "unchanged."
        )


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


def test_npu_adam_step_delegates_through_cython_fast_adam_step():
    """`optim.py::_adam_step_op` must delegate to the Cython `fast_adam_step`
    helper. The Python shim should not reference `aclnn.apply_adam_w_v2`
    directly — backend implementation lives in `_C/_npu_ops.pyx`.
    """
    optim_src = _source("src/candle/_backends/npu/ops/optim.py")
    body = _function_source(optim_src, "_adam_step_op")
    assert "aclnn.apply_adam_w_v2" not in body, (
        "optim.py::_adam_step_op still calls aclnn.apply_adam_w_v2 directly; "
        "should delegate to the Cython fast_adam_step helper"
    )


def test_npu_trace_delegates_through_cython_fast_trace():
    """`linalg.py::trace_op` must delegate to the Cython `fast_trace` helper.
    The Python shim should not reference `aclnn.strace` directly — the
    aclnnTrace orchestration belongs in `_C/_npu_ops.pyx`.
    """
    linalg_src = _source("src/candle/_backends/npu/ops/linalg.py")
    body = _function_source(linalg_src, "trace_op")
    assert "aclnn.strace" not in body, (
        "linalg.py::trace_op still calls aclnn.strace directly; "
        "should delegate to the Cython fast_trace helper"
    )


def test_npu_linalg_inv_delegates_through_cython_fast_inverse():
    """`linalg.py::linalg_inv` must delegate to the Cython `fast_inverse`
    helper. The Python shim should not reference `aclnn.inverse` directly —
    the aclnnInverse orchestration belongs in `_C/_npu_ops.pyx`.
    """
    linalg_src = _source("src/candle/_backends/npu/ops/linalg.py")
    body = _function_source(linalg_src, "linalg_inv")
    assert "aclnn.inverse" not in body, (
        "linalg.py::linalg_inv still calls aclnn.inverse directly; "
        "should delegate to the Cython fast_inverse helper"
    )


def test_npu_helpers_no_unused_unary_op_python_fallback():
    """The Python-side `_unary_op` orchestration helper in
    `_backends/npu/ops/_helpers.py` is dead code — every unary op was
    migrated to a Cython `fast_*` helper. Remove the helper and its
    import sites so the surface area matches what is actually in use.
    """
    helpers_src = _source("src/candle/_backends/npu/ops/_helpers.py")
    assert "def _unary_op(" not in helpers_src, (
        "_helpers.py still defines the unused _unary_op Python helper; "
        "all NPU unary orchestration now flows through Cython fast_* helpers"
    )


def test_npu_init_has_no_duplicate_registry_register_calls():
    """`src/candle/_backends/npu/__init__.py` must not register the same
    op name twice — a later `registry.register(name, "npu", impl)` call
    silently overrides the earlier one, leaving dead code and dead
    imports behind. Pick exactly one implementation per op.
    """
    init_src = _source("src/candle/_backends/npu/__init__.py")
    names = re.findall(r'registry\.register\(\s*"([^"]+)"\s*,\s*"npu"', init_src)
    seen = {}
    for name in names:
        seen[name] = seen.get(name, 0) + 1
    duplicates = sorted(n for n, c in seen.items() if c > 1)
    assert not duplicates, (
        f"duplicate NPU registry.register entries for: {duplicates}; "
        "remove the dead earlier registration so each op has a single entry"
    )


def test_npu_helpers_no_unused_reduction_dim_helpers():
    """`_reduce_dim_sizes` and `_broadcast_dims_to_out` in
    `_backends/npu/ops/_helpers.py` are dead code — only imported by
    `reduce.py` and `ops/__init__.py`, never actually called. Remove
    them and their imports to keep the shared helper surface honest.
    """
    helpers_src = _source("src/candle/_backends/npu/ops/_helpers.py")
    for fn in ("_reduce_dim_sizes", "_broadcast_dims_to_out"):
        assert f"def {fn}(" not in helpers_src, (
            f"_helpers.py still defines unused {fn}; it has no live callers"
        )


def test_npu_ops_modules_do_not_import_linalg_only_helpers():
    """`_matmul_out_shape`, `_iter_indices`, `_broadcast_index`, and
    `_batch_offset` are only called from `_backends/npu/ops/linalg.py`.
    Other ops modules should not list them in their `_helpers` import
    block — the unused names just clutter the import surface.
    """
    dead_names = ("_matmul_out_shape", "_iter_indices",
                  "_broadcast_index", "_batch_offset")
    consumer_modules = (
        "src/candle/_backends/npu/ops/__init__.py",
        "src/candle/_backends/npu/ops/activation.py",
        "src/candle/_backends/npu/ops/conv.py",
        "src/candle/_backends/npu/ops/elementwise.py",
        "src/candle/_backends/npu/ops/norm.py",
        "src/candle/_backends/npu/ops/optim.py",
        "src/candle/_backends/npu/ops/random.py",
        "src/candle/_backends/npu/ops/special.py",
    )
    offenders = []
    for path in consumer_modules:
        src = _source(path)
        for name in dead_names:
            if re.search(rf"\b{name}\b", src):
                offenders.append(f"{path}: {name}")
    assert not offenders, (
        "These NPU ops modules still reference linalg-only helpers in their "
        "imports — drop the unused names:\n  " + "\n  ".join(offenders)
    )




def test_npu_ops_modules_do_not_import_unused_binary_op_helper():
    """`_binary_op` is a thin wrapper around the Cython `fast_binary_op`
    that exists only for the helper's own internal use (and one fallback
    shim in `_C/_npu_ops_fallback.py`). The ops modules import the name
    but never call it — drop the dead imports so the helper surface stays
    honest.
    """
    consumer_modules = (
        "src/candle/_backends/npu/ops/__init__.py",
        "src/candle/_backends/npu/ops/activation.py",
        "src/candle/_backends/npu/ops/conv.py",
        "src/candle/_backends/npu/ops/elementwise.py",
        "src/candle/_backends/npu/ops/linalg.py",
        "src/candle/_backends/npu/ops/math.py",
        "src/candle/_backends/npu/ops/norm.py",
        "src/candle/_backends/npu/ops/optim.py",
        "src/candle/_backends/npu/ops/random.py",
        "src/candle/_backends/npu/ops/shape.py",
        "src/candle/_backends/npu/ops/special.py",
    )
    offenders = []
    for path in consumer_modules:
        src = _source(path)
        if re.search(r"\b_binary_op\b", src):
            offenders.append(path)
    assert not offenders, (
        "These NPU ops modules still reference _binary_op but never call it — "
        "drop the unused import:\n  " + "\n  ".join(offenders)
    )


def test_npu_helpers_no_unused_scalar_to_npu_tensor_no_add_helper():
    """`_scalar_to_npu_tensor_no_add` lives in `_helpers.py` and is
    imported by every ops module, but nothing actually calls it. Drop
    the helper entirely so the import surface stays in sync with what
    is used.
    """
    helpers_src = _source("src/candle/_backends/npu/ops/_helpers.py")
    assert "def _scalar_to_npu_tensor_no_add" not in helpers_src, (
        "_helpers.py still defines _scalar_to_npu_tensor_no_add; it has no live callers"
    )


def test_npu_ops_modules_do_not_import_reduce_only_helpers():
    """`_normalize_reduction_dims` and `_reduce_out_shape` are only called
    from `_backends/npu/ops/reduce.py` (and `_helpers.py` internal use).
    Other ops modules should not list them in their `_helpers` import
    block — the unused names just clutter the import surface.
    """
    dead_names = ("_normalize_reduction_dims", "_reduce_out_shape")
    consumer_modules = (
        "src/candle/_backends/npu/ops/__init__.py",
        "src/candle/_backends/npu/ops/activation.py",
        "src/candle/_backends/npu/ops/conv.py",
        "src/candle/_backends/npu/ops/elementwise.py",
        "src/candle/_backends/npu/ops/linalg.py",
        "src/candle/_backends/npu/ops/norm.py",
        "src/candle/_backends/npu/ops/optim.py",
        "src/candle/_backends/npu/ops/random.py",
        "src/candle/_backends/npu/ops/shape.py",
        "src/candle/_backends/npu/ops/special.py",
    )
    offenders = []
    for path in consumer_modules:
        src = _source(path)
        for name in dead_names:
            if re.search(rf"\b{name}\b", src):
                offenders.append(f"{path}: {name}")
    assert not offenders, (
        "These NPU ops modules still reference reduce-only helpers — "
        "drop the unused names:\n  " + "\n  ".join(offenders)
    )


def test_npu_ops_modules_do_not_import_unused_index_put_or_tensor_seq_helpers():
    """`npu_index_put_impl` is only called from `_backends/npu/ops/conv.py`
    and `_normalize_tensor_sequence_args` is only called from
    `_backends/npu/ops/shape.py`. Other ops modules should not list them in
    their `_helpers` import block — the unused names just clutter the import
    surface.
    """
    cases = (
        (
            "npu_index_put_impl",
            (
                "src/candle/_backends/npu/ops/__init__.py",
                "src/candle/_backends/npu/ops/activation.py",
                "src/candle/_backends/npu/ops/elementwise.py",
                "src/candle/_backends/npu/ops/linalg.py",
                "src/candle/_backends/npu/ops/norm.py",
                "src/candle/_backends/npu/ops/optim.py",
                "src/candle/_backends/npu/ops/random.py",
                "src/candle/_backends/npu/ops/shape.py",
                "src/candle/_backends/npu/ops/special.py",
            ),
        ),
        (
            "_normalize_tensor_sequence_args",
            (
                "src/candle/_backends/npu/ops/__init__.py",
                "src/candle/_backends/npu/ops/activation.py",
                "src/candle/_backends/npu/ops/conv.py",
                "src/candle/_backends/npu/ops/elementwise.py",
                "src/candle/_backends/npu/ops/linalg.py",
                "src/candle/_backends/npu/ops/norm.py",
                "src/candle/_backends/npu/ops/optim.py",
                "src/candle/_backends/npu/ops/random.py",
                "src/candle/_backends/npu/ops/special.py",
            ),
        ),
    )
    offenders = []
    for name, consumer_modules in cases:
        for path in consumer_modules:
            src = _source(path)
            if re.search(rf"\b{name}\b", src):
                offenders.append(f"{path}: {name}")
    assert not offenders, (
        "These NPU ops modules still reference single-consumer helpers — "
        "drop the unused names:\n  " + "\n  ".join(offenders)
    )


def test_npu_ops_modules_do_not_import_unused_npu_add_scalar_helper():
    """`_npu_add_scalar_` is only called from `_backends/npu/ops/random.py`.
    Other ops modules should not list it in their `_helpers` import block —
    the unused name just clutters the import surface.
    """
    consumer_modules = (
        "src/candle/_backends/npu/ops/__init__.py",
        "src/candle/_backends/npu/ops/activation.py",
        "src/candle/_backends/npu/ops/conv.py",
        "src/candle/_backends/npu/ops/elementwise.py",
        "src/candle/_backends/npu/ops/linalg.py",
        "src/candle/_backends/npu/ops/norm.py",
        "src/candle/_backends/npu/ops/optim.py",
        "src/candle/_backends/npu/ops/shape.py",
        "src/candle/_backends/npu/ops/special.py",
    )
    offenders = []
    for path in consumer_modules:
        src = _source(path)
        if re.search(r"\b_npu_add_scalar_\b", src):
            offenders.append(path)
    assert not offenders, (
        "These NPU ops modules still reference `_npu_add_scalar_` — "
        "drop the unused import:\n  " + "\n  ".join(offenders)
    )


def test_npu_ops_modules_do_not_import_unused_broadcast_shape_helpers():
    """`_broadcast_shape_checked` is only used inside `_helpers.py` itself
    and is not called by any op module. `_broadcast_shape` is only called
    from `comparison.py`, `elementwise.py`, and `linalg.py`. Other ops
    modules should not list these names in their `_helpers` import block —
    the unused names just clutter the import surface.
    """
    cases = (
        (
            "_broadcast_shape_checked",
            (
                "src/candle/_backends/npu/ops/__init__.py",
                "src/candle/_backends/npu/ops/activation.py",
                "src/candle/_backends/npu/ops/conv.py",
                "src/candle/_backends/npu/ops/elementwise.py",
                "src/candle/_backends/npu/ops/linalg.py",
                "src/candle/_backends/npu/ops/math.py",
                "src/candle/_backends/npu/ops/norm.py",
                "src/candle/_backends/npu/ops/optim.py",
                "src/candle/_backends/npu/ops/random.py",
                "src/candle/_backends/npu/ops/reduce.py",
                "src/candle/_backends/npu/ops/shape.py",
                "src/candle/_backends/npu/ops/special.py",
            ),
        ),
        (
            "_broadcast_shape",
            (
                "src/candle/_backends/npu/ops/__init__.py",
                "src/candle/_backends/npu/ops/activation.py",
                "src/candle/_backends/npu/ops/conv.py",
                "src/candle/_backends/npu/ops/math.py",
                "src/candle/_backends/npu/ops/norm.py",
                "src/candle/_backends/npu/ops/optim.py",
                "src/candle/_backends/npu/ops/random.py",
                "src/candle/_backends/npu/ops/reduce.py",
                "src/candle/_backends/npu/ops/shape.py",
                "src/candle/_backends/npu/ops/special.py",
            ),
        ),
    )
    offenders = []
    for name, consumer_modules in cases:
        for path in consumer_modules:
            src = _source(path)
            if re.search(rf"\b{name}\b", src):
                offenders.append(f"{path}: {name}")
    assert not offenders, (
        "These NPU ops modules still reference broadcast-shape helpers "
        "that they don't call — drop the unused names:\n  "
        + "\n  ".join(offenders)
    )


def test_npu_ops_modules_do_not_import_unused_npu_broadcast_to_helper():
    """`_npu_broadcast_to` is only called from `_backends/npu/ops/conv.py`,
    `elementwise.py`, `linalg.py`, and `shape.py`. Other ops modules
    should not list it in their `_helpers` import block — the unused name
    just clutters the import surface.
    """
    consumer_modules = (
        "src/candle/_backends/npu/ops/__init__.py",
        "src/candle/_backends/npu/ops/activation.py",
        "src/candle/_backends/npu/ops/math.py",
        "src/candle/_backends/npu/ops/norm.py",
        "src/candle/_backends/npu/ops/optim.py",
        "src/candle/_backends/npu/ops/random.py",
        "src/candle/_backends/npu/ops/reduce.py",
        "src/candle/_backends/npu/ops/special.py",
    )
    offenders = []
    for path in consumer_modules:
        src = _source(path)
        if re.search(r"\b_npu_broadcast_to\b", src):
            offenders.append(path)
    assert not offenders, (
        "These NPU ops modules still reference `_npu_broadcast_to` — "
        "drop the unused import:\n  " + "\n  ".join(offenders)
    )


def test_npu_helpers_no_unused_nan_like_helper():
    """`_nan_like` lives in `_helpers.py` and is imported by a few ops
    modules, but nothing actually calls it. Drop the helper entirely so
    the import surface stays in sync with what is used.
    """
    helpers_src = _source("src/candle/_backends/npu/ops/_helpers.py")
    assert "def _nan_like" not in helpers_src, (
        "_helpers.py still defines _nan_like; it has no live callers"
    )


def test_npu_ops_modules_do_not_import_unused_npu_linear_index_helper():
    """`_npu_linear_index` is only called from `_dispatch/functionalize.py`,
    which imports it directly from `_helpers`. Other ops modules should
    not list it in their `_helpers` import block — the unused name just
    clutters the import surface.
    """
    consumer_modules = (
        "src/candle/_backends/npu/ops/activation.py",
        "src/candle/_backends/npu/ops/conv.py",
        "src/candle/_backends/npu/ops/elementwise.py",
        "src/candle/_backends/npu/ops/linalg.py",
        "src/candle/_backends/npu/ops/norm.py",
        "src/candle/_backends/npu/ops/optim.py",
        "src/candle/_backends/npu/ops/random.py",
        "src/candle/_backends/npu/ops/shape.py",
        "src/candle/_backends/npu/ops/special.py",
    )
    offenders = []
    for path in consumer_modules:
        src = _source(path)
        if re.search(r"\b_npu_linear_index\b", src):
            offenders.append(path)
    assert not offenders, (
        "These NPU ops modules still reference `_npu_linear_index` — "
        "drop the unused import:\n  " + "\n  ".join(offenders)
    )


def test_npu_ops_modules_do_not_import_unused_cast_soc_or_scalar_helpers():
    """`_cast_tensor_dtype`, `_use_soc_fallback`, and `_scalar_to_npu_tensor`
    are all heavily used helpers — but a few ops modules list them in their
    `_helpers` import block without ever calling them locally. The
    `_backends/npu/ops/__init__.py` re-exports them as well, but no external
    consumer reaches them via `npu_ops.<name>` (Cython hot paths and tests
    import directly from `_helpers`). Drop the dead listings so the import
    surface stays in sync with what is used.
    """
    cases = (
        (
            "_cast_tensor_dtype",
            (
                "src/candle/_backends/npu/ops/__init__.py",
                "src/candle/_backends/npu/ops/math.py",
                "src/candle/_backends/npu/ops/optim.py",
                "src/candle/_backends/npu/ops/shape.py",
            ),
        ),
        (
            "_use_soc_fallback",
            (
                "src/candle/_backends/npu/ops/__init__.py",
                "src/candle/_backends/npu/ops/optim.py",
                "src/candle/_backends/npu/ops/special.py",
            ),
        ),
        (
            "_scalar_to_npu_tensor",
            (
                "src/candle/_backends/npu/ops/__init__.py",
            ),
        ),
    )
    offenders = []
    for name, consumer_modules in cases:
        for path in consumer_modules:
            src = _source(path)
            if re.search(rf"\b{name}\b", src):
                offenders.append(f"{path}: {name}")
    assert not offenders, (
        "These NPU ops modules still reference unused helpers — "
        "drop the unused imports:\n  " + "\n  ".join(offenders)
    )


def test_npu_ops_modules_do_not_import_unused_ops_soc_helper():
    """`ops_soc` is only used from `_backends/npu/ops/shape.py` (and inside
    `_helpers.py` itself). Other ops modules and the package `__init__`
    list it in their `_helpers` import block but never call it — drop the
    unused listings so the import surface stays in sync with what is used.
    """
    consumer_modules = (
        "src/candle/_backends/npu/ops/__init__.py",
        "src/candle/_backends/npu/ops/activation.py",
        "src/candle/_backends/npu/ops/conv.py",
        "src/candle/_backends/npu/ops/elementwise.py",
        "src/candle/_backends/npu/ops/linalg.py",
        "src/candle/_backends/npu/ops/math.py",
        "src/candle/_backends/npu/ops/norm.py",
        "src/candle/_backends/npu/ops/optim.py",
        "src/candle/_backends/npu/ops/random.py",
        "src/candle/_backends/npu/ops/reduce.py",
        "src/candle/_backends/npu/ops/special.py",
    )
    offenders = []
    for path in consumer_modules:
        src = _source(path)
        if re.search(r"\bops_soc\b", src):
            offenders.append(path)
    assert not offenders, (
        "These NPU ops modules still reference `ops_soc` — "
        "drop the unused import:\n  " + "\n  ".join(offenders)
    )


def test_npu_ops_modules_do_not_import_unused_backend_infrastructure():
    """The `_helpers` re-exports backend infrastructure (`aclnn`,
    `npu_runtime`, `npu_state`, `npu_typed_storage_from_ptr`, `reshape`)
    for ops modules that need direct kernel access. Several ops modules
    list these names without ever calling them — drop the dead listings
    so the import surface stays in sync with what is used.
    """
    cases = (
        (
            "aclnn",
            (
                "src/candle/_backends/npu/ops/__init__.py",
                "src/candle/_backends/npu/ops/activation.py",
                "src/candle/_backends/npu/ops/math.py",
                "src/candle/_backends/npu/ops/optim.py",
                "src/candle/_backends/npu/ops/special.py",
            ),
        ),
        (
            "npu_runtime",
            (
                "src/candle/_backends/npu/ops/__init__.py",
                "src/candle/_backends/npu/ops/activation.py",
                "src/candle/_backends/npu/ops/math.py",
                "src/candle/_backends/npu/ops/optim.py",
            ),
        ),
        (
            "npu_state",
            (
                "src/candle/_backends/npu/ops/__init__.py",
                "src/candle/_backends/npu/ops/activation.py",
                "src/candle/_backends/npu/ops/math.py",
                "src/candle/_backends/npu/ops/optim.py",
                "src/candle/_backends/npu/ops/special.py",
            ),
        ),
        (
            "npu_typed_storage_from_ptr",
            (
                "src/candle/_backends/npu/ops/activation.py",
                "src/candle/_backends/npu/ops/math.py",
                "src/candle/_backends/npu/ops/optim.py",
            ),
        ),
        (
            "reshape",
            (
                "src/candle/_backends/npu/ops/__init__.py",
                "src/candle/_backends/npu/ops/optim.py",
            ),
        ),
    )
    offenders = []
    for name, consumer_modules in cases:
        for path in consumer_modules:
            src = _source(path)
            if re.search(rf"\b{name}\b", src):
                offenders.append(f"{path}: {name}")
    assert not offenders, (
        "These NPU ops modules still reference unused backend infrastructure — "
        "drop the unused imports:\n  " + "\n  ".join(offenders)
    )


def test_npu_ops_package_init_does_not_import_unused_ctypes():
    """`src/candle/_backends/npu/ops/__init__.py` carries a bare
    `import ctypes` from older code paths that have all moved to the
    Cython helpers. Nothing in the file references `ctypes.*` anymore —
    drop the dead import so the package surface stays honest.
    """
    src = _source("src/candle/_backends/npu/ops/__init__.py")
    assert not re.search(r"\bctypes\b", src), (
        "src/candle/_backends/npu/ops/__init__.py still references `ctypes` "
        "but never calls it — drop the dead `import ctypes`."
    )


def test_npu_helpers_does_not_carry_unused_module_level_aliases():
    """`src/candle/_backends/npu/ops/_helpers.py` carries a bare
    ``import ctypes`` at module scope from older code paths; the only
    ``ctypes`` reference is inside ``_scalar_to_npu_tensor``, which
    already does a function-local ``import ctypes``. Drop the dead
    module-level import so the helper surface stays honest.
    """
    src = _source("src/candle/_backends/npu/ops/_helpers.py")
    assert not re.search(r"^import ctypes\b", src, flags=re.MULTILINE), (
        "src/candle/_backends/npu/ops/_helpers.py still carries a "
        "module-level `import ctypes` but the only use is inside "
        "_scalar_to_npu_tensor, which imports ctypes locally — drop "
        "the dead module-level import."
    )


def test_npu_reduce_does_not_import_unused_reshape_alias():
    """`src/candle/_backends/npu/ops/reduce.py` imports the ``reshape``
    alias from ``._helpers`` but never calls it — every reshape site in
    the file goes through the function-local
    ``from ...common import view as view_backend`` followed by
    ``view_backend.reshape(...)``. Drop the dead listing.
    """
    src = _source("src/candle/_backends/npu/ops/reduce.py")
    in_helpers_block = False
    helpers_block_imports = []
    for line in src.splitlines():
        if line.strip().startswith("from ._helpers import"):
            in_helpers_block = True
        if in_helpers_block:
            helpers_block_imports.append(line)
            if ")" in line:
                in_helpers_block = False
    block_src = "\n".join(helpers_block_imports)
    assert not re.search(r"\breshape\b", block_src), (
        "src/candle/_backends/npu/ops/reduce.py still imports `reshape` "
        "from ._helpers but never calls it — drop the dead listing."
    )


def test_npu_ops_modules_do_not_import_unused_int32_dtype():
    """`int32_dtype` is re-exported by `_helpers` so reduce ops (which
    legitimately use it 13 times) can pick it up. Every other ops
    module lists it in the `from ._helpers import (...)` block but
    never references it — drop the dead listings so the import surface
    stays in sync with what is used.
    """
    consumer_modules = (
        "src/candle/_backends/npu/ops/__init__.py",
        "src/candle/_backends/npu/ops/activation.py",
        "src/candle/_backends/npu/ops/conv.py",
        "src/candle/_backends/npu/ops/elementwise.py",
        "src/candle/_backends/npu/ops/linalg.py",
        "src/candle/_backends/npu/ops/math.py",
        "src/candle/_backends/npu/ops/norm.py",
        "src/candle/_backends/npu/ops/optim.py",
        "src/candle/_backends/npu/ops/random.py",
        "src/candle/_backends/npu/ops/shape.py",
        "src/candle/_backends/npu/ops/special.py",
    )
    offenders = []
    for path in consumer_modules:
        src = _source(path)
        if re.search(r"\bint32_dtype\b", src):
            offenders.append(path)
    assert not offenders, (
        "These NPU ops modules still reference `int32_dtype` but never "
        "use it — drop the unused import:\n  " + "\n  ".join(offenders)
    )


def test_npu_ops_modules_do_not_import_unused_numel_or_arange_helpers():
    """The `_numel` and `_npu_arange_1d` helpers are re-exported from
    `_helpers` but only some ops modules use them. The rest list them
    in `from ._helpers import (...)` blocks without ever calling them
    — drop the dead listings so the helper surface stays honest.
    """
    cases = (
        (
            "_numel",
            (
                "src/candle/_backends/npu/ops/__init__.py",
                "src/candle/_backends/npu/ops/elementwise.py",
                "src/candle/_backends/npu/ops/math.py",
                "src/candle/_backends/npu/ops/optim.py",
                "src/candle/_backends/npu/ops/special.py",
            ),
        ),
        (
            "_npu_arange_1d",
            (
                "src/candle/_backends/npu/ops/__init__.py",
                "src/candle/_backends/npu/ops/conv.py",
                "src/candle/_backends/npu/ops/elementwise.py",
                "src/candle/_backends/npu/ops/norm.py",
                "src/candle/_backends/npu/ops/optim.py",
                "src/candle/_backends/npu/ops/special.py",
            ),
        ),
    )
    offenders = []
    for name, consumer_modules in cases:
        for path in consumer_modules:
            src = _source(path)
            if re.search(rf"\b{name}\b", src):
                offenders.append(f"{path}: {name}")
    assert not offenders, (
        "These NPU ops modules still reference unused helpers — "
        "drop the unused imports:\n  " + "\n  ".join(offenders)
    )


def test_npu_ops_modules_do_not_import_unused_tensor_plumbing_helpers():
    """`_dtype_itemsize`, `_wrap_tensor`, and `_unwrap_storage` are core
    tensor-plumbing helpers re-exported from `_helpers`. Several ops modules
    list them in `from ._helpers import (...)` blocks without ever calling
    them — drop the dead listings so the helper surface stays honest.

    The package `__init__.py` re-export is excluded because `_backends/autograd.py`
    consumes these helpers via `from .npu.ops import ...`.
    """
    cases = (
        (
            "_dtype_itemsize",
            (
                "src/candle/_backends/npu/ops/activation.py",
                "src/candle/_backends/npu/ops/math.py",
                "src/candle/_backends/npu/ops/optim.py",
                "src/candle/_backends/npu/ops/special.py",
            ),
        ),
        (
            "_wrap_tensor",
            (
                "src/candle/_backends/npu/ops/activation.py",
                "src/candle/_backends/npu/ops/math.py",
                "src/candle/_backends/npu/ops/optim.py",
            ),
        ),
        (
            "_unwrap_storage",
            (
                "src/candle/_backends/npu/ops/activation.py",
                "src/candle/_backends/npu/ops/math.py",
                "src/candle/_backends/npu/ops/optim.py",
                "src/candle/_backends/npu/ops/special.py",
            ),
        ),
    )
    offenders = []
    for name, consumer_modules in cases:
        for path in consumer_modules:
            src = _source(path)
            if re.search(rf"\b{name}\b", src):
                offenders.append(f"{path}: {name}")
    assert not offenders, (
        "These NPU ops modules still reference unused tensor-plumbing helpers — "
        "drop the unused imports:\n  " + "\n  ".join(offenders)
    )


def test_npu_ops_modules_do_not_import_unused_dtype_constants():
    """`bool_dtype`, `int64_dtype`, and `float_dtype` are re-exported from
    `_helpers` for code that needs the candle dtype singletons. Several ops
    modules — and the package `__init__.py` re-export itself — list them
    without ever using them. No external code in `src/` or `tests/` consumes
    these via `from .npu.ops import` or `npu_ops.X` either, so the
    `__init__.py` re-export is also dead.

    Drop the unused listings so the dtype surface stays honest.
    """
    cases = (
        (
            "bool_dtype",
            (
                "src/candle/_backends/npu/ops/__init__.py",
                "src/candle/_backends/npu/ops/conv.py",
                "src/candle/_backends/npu/ops/linalg.py",
                "src/candle/_backends/npu/ops/math.py",
                "src/candle/_backends/npu/ops/norm.py",
                "src/candle/_backends/npu/ops/optim.py",
                "src/candle/_backends/npu/ops/random.py",
                "src/candle/_backends/npu/ops/special.py",
            ),
        ),
        (
            "int64_dtype",
            (
                "src/candle/_backends/npu/ops/__init__.py",
                "src/candle/_backends/npu/ops/activation.py",
                "src/candle/_backends/npu/ops/math.py",
                "src/candle/_backends/npu/ops/norm.py",
                "src/candle/_backends/npu/ops/optim.py",
                "src/candle/_backends/npu/ops/random.py",
            ),
        ),
        (
            "float_dtype",
            (
                "src/candle/_backends/npu/ops/__init__.py",
                "src/candle/_backends/npu/ops/conv.py",
                "src/candle/_backends/npu/ops/math.py",
                "src/candle/_backends/npu/ops/norm.py",
                "src/candle/_backends/npu/ops/optim.py",
                "src/candle/_backends/npu/ops/shape.py",
            ),
        ),
    )
    offenders = []
    for name, consumer_modules in cases:
        for path in consumer_modules:
            src = _source(path)
            if re.search(rf"\b{name}\b", src):
                offenders.append(f"{path}: {name}")
    assert not offenders, (
        "These NPU ops modules still reference unused dtype constants — "
        "drop the unused imports:\n  " + "\n  ".join(offenders)
    )


def test_npu_ops_modules_do_not_carry_dead_cross_module_imports():
    """A handful of NPU ops modules still list cross-module helpers in their
    top-level (or function-local) `from X import Y` blocks that they never
    actually call. Drop them so the per-file import surface is honest.

    Each pair below is `(file, name)` — if `name` appears in `file`, it must
    be referenced more than once (i.e. used somewhere beyond the import line).
    A single occurrence means the import is dead.
    """
    cases = (
        ("src/candle/_backends/npu/ops/activation.py", "sub"),
        ("src/candle/_backends/npu/ops/activation.py", "maximum"),
        ("src/candle/_backends/npu/ops/activation.py", "minimum"),
        ("src/candle/_backends/npu/ops/conv.py", "floor"),
        ("src/candle/_backends/npu/ops/elementwise.py", "sqrt"),
        ("src/candle/_backends/npu/ops/elementwise.py", "index_put_"),
        ("src/candle/_backends/npu/ops/elementwise.py", "masked_select"),
        ("src/candle/_backends/npu/ops/elementwise.py", "nonzero"),
        ("src/candle/_backends/npu/ops/norm.py", "float16_dtype"),
        ("src/candle/_backends/npu/ops/random.py", "div"),
        ("src/candle/_backends/npu/ops/reduce.py", "Tensor"),
    )
    offenders = []
    for path, name in cases:
        src = _source(path)
        if re.search(rf"\b{name}\b", src):
            offenders.append(f"{path}: {name}")
    assert not offenders, (
        "These NPU ops modules still carry dead cross-module imports — "
        "drop them:\n  " + "\n  ".join(offenders)
    )


def test_npu_shape_does_not_duplicate_storage_meta_helper():
    """`shape.py` defined its own `_npu_storage_meta` whose body is byte-identical
    to `_storage_meta` in `_helpers.py`. Both pull through `_unwrap_storage` and
    `_dtype_itemsize` and return `(data_ptr, offset, (storage_numel,))`. Drop the
    shape-local copy and route through the shared helper so there is one source
    of truth for storage metadata extraction.
    """
    shape_src = _source("src/candle/_backends/npu/ops/shape.py")
    assert "_npu_storage_meta" not in shape_src, (
        "src/candle/_backends/npu/ops/shape.py still defines or calls "
        "`_npu_storage_meta` — replace with `_storage_meta` from `._helpers` "
        "so the duplicate function body is removed."
    )


def test_npu_ops_package_init_does_not_reexport_storage_plumbing_helpers():
    """`npu/ops/__init__.py` historically re-exported a small set of
    private plumbing helpers from `_helpers.py` (and
    `npu_typed_storage_from_ptr` from the Cython storage module).
    None of those re-exports are used outside the `npu/ops/` package —
    no source in `src/` or `tests/` imports them via
    `from candle._backends.npu.ops import ...` nor accesses them as
    `npu_ops._unwrap_storage` etc. Drop every dead re-export so the
    package's public surface reflects only what consumers actually need.
    """
    init_src = _source("src/candle/_backends/npu/ops/__init__.py")
    forbidden = [
        "_unwrap_storage",
        "_wrap_tensor",
        "_dtype_itemsize",
        "npu_typed_storage_from_ptr",
    ]
    for name in forbidden:
        assert name not in init_src, (
            f"`src/candle/_backends/npu/ops/__init__.py` still re-exports "
            f"`{name}` — drop it; no external module consumes it via "
            f"`from candle._backends.npu.ops import {name}`."
        )

    # Cross-check: no consumer in src/ or tests/ should import these names
    # via the `candle._backends.npu.ops` package path. If a new consumer
    # appears, it must import from the actual source module (`_helpers.py`
    # or the Cython storage module) instead of re-introducing the dead
    # re-export.
    consumer_roots = ["src/candle", "tests"]
    import_patterns = [
        # `from candle._backends.npu.ops import <name>` or
        # `from .npu.ops import <name>` (relative form used inside _backends/)
        re.compile(
            r"from\s+[\w.]*\.?npu\.ops\s+import\s+[^\n]*\b("
            + "|".join(re.escape(n) for n in forbidden)
            + r")\b"
        ),
    ]
    offenders = []
    for root in consumer_roots:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for pattern in import_patterns:
                if pattern.search(text):
                    rel = path.relative_to(_REPO_ROOT)
                    offenders.append(str(rel))
                    break
    assert not offenders, (
        "The following modules re-import dropped plumbing helpers via "
        "`candle._backends.npu.ops`: "
        f"{offenders}. Import from the source module "
        "(`candle._backends.npu.ops._helpers` for `_unwrap_storage`, "
        "`_wrap_tensor`, `_dtype_itemsize`; `candle._C` for "
        "`npu_typed_storage_from_ptr`) instead."
    )


def test_npu_ops_modules_lift_fast_helper_imports_to_module_level():
    """Ops-module functions must load their Cython `fast_*` helper through a
    module-level try/except block, not via a function-body
    `from candle._C._npu_ops import ...` line. The body-level form pays a
    Python import lookup on every call and obscures the module's dependency
    surface — keeping the pattern consistent across the package.
    """
    targets = {
        "src/candle/_backends/npu/ops/special.py": [
            "special_sinc",
        ],
    }
    for rel, names in targets.items():
        src = _source(rel)
        for name in names:
            body = _function_source(src, name)
            assert "from candle._C._npu_ops import fast_" not in body, (
                f"{rel}::{name} still has a function-body "
                f"`from candle._C._npu_ops import fast_*` — lift to module level."
            )


def test_npu_ops_modules_do_not_carry_wholly_dead_helper_defs():
    """Several private helpers defined inside `npu/ops/*.py` modules are
    referenced only at their own def site — no other source file in `src/`,
    `tests/`, or `tools/` mentions them. Each represents either a workaround
    that was later replaced by a Cython fast-helper, or a composite that was
    inlined into its caller. Drop the dead defs so the modules stop carrying
    code that the runtime cannot reach.
    """
    targets = {
        "src/candle/_backends/npu/ops/linalg.py": [
            "_einsum_output_shape",
        ],
        "src/candle/_backends/npu/ops/shape.py": [
            "_strided_copy",
            "_nonzero_mask_float",
            "_positive_mask_int64",
            "_negative_mask_int64",
            "_below_negative_lower_bound_mask_int64",
            "_mask_has_any",
            "_move_dim_to_last",
            "_is_advanced_index",
        ],
    }
    for rel, names in targets.items():
        src = _source(rel)
        for name in names:
            assert f"def {name}(" not in src, (
                f"{rel} still defines `{name}` — this helper is referenced "
                f"only at its own def site and should be deleted."
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
        "abs_",
        "acos_",
        "acosh_",
        "allclose",
        "arange",
        "argmax",
        "argmin",
        "argsort",
        "argwhere",
        "as_strided_copy",
        "asin_",
        "asinh_",
        "atan_",
        "atanh_",
        "bincount",
        "bitwise_and",
        "bitwise_not",
        "bitwise_or",
        "bitwise_xor",
        "bucketize",
        "cartesian_prod",
        "ceil_",
        "cos_",
        "cosh_",
        "empty",
        "empty_like",
        "equal",
        "erf_",
        "erfc_",
        "erfinv_",
        "exp2_",
        "exp_",
        "expand_copy",
        "expm1_",
        "eye",
        "flatten",
        "floor_",
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
        "log10_",
        "log1p_",
        "log2_",
        "log_",
        "logical_and",
        "logical_not",
        "logical_or",
        "logical_xor",
        "logspace",
        "movedim",
        "narrow",
        "neg_",
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
        "round_",
        "rsqrt_",
        "searchsorted",
        "sigmoid_",
        "sin_",
        "sinh_",
        "slice_copy",
        "sqrt_",
        "square_",
        "squeeze",
        "tan_",
        "tanh_",
        "tensor",
        "tril_indices",
        "triu_indices",
        "trunc_",
        "unflatten",
        "zeros",
        "zeros_like",
    }

    assert missing_autograd == expected_missing
    assert len(forward_ops) == 432
    assert len(autograd_ops & forward_ops) == 329


def test_npu_activation_module_consolidates_fast_helper_try_blocks():
    """`activation.py` historically grew one `try/except ImportError` block
    per `fast_*` helper — currently 15 separate blocks (14 individual
    primaries + 1 composites). Every other ops module that mirrors this
    pattern (`math.py`, `comparison.py`, `reduce.py`, `elementwise.py`,
    `linalg.py`, `random.py`, `optim.py`) uses a *single* consolidated
    try/except block at module top, importing every `fast_*` helper inside
    a tuple and setting each `_HAS_FAST_X` flag in the success branch /
    `None` + `False` in the failure branch.

    All `fast_*` helpers come from the same Cython extension module
    (`candle._C._npu_ops`) — they either all exist or none do — so
    splitting the import into 15 blocks adds clutter without changing
    runtime semantics. Consolidate them so `activation.py` matches the
    rest of the package and pays one ImportError check instead of 15.
    """
    activation_src = _source("src/candle/_backends/npu/ops/activation.py")
    try_count = len(
        [line for line in activation_src.splitlines() if line.startswith("try:")]
    )
    assert try_count <= 1, (
        "`src/candle/_backends/npu/ops/activation.py` still has "
        f"{try_count} top-level `try:` blocks for importing `fast_*` "
        "helpers. Consolidate them into a single try/except block so the "
        "module matches `math.py`, `comparison.py`, and the rest of the "
        "ops package."
    )


def test_npu_linalg_module_has_no_dead_cross_module_imports():
    """`linalg.py` imports `cos`, `exp`, `sin` from `.math` and `diag`
    from `.shape`, but none of those names is called anywhere in the
    file body — each only appears in code comments (DFT W matrix
    derivation, Taylor expansion explanation, SVD pseudo-inverse
    explanation). The imports were left behind after the corresponding
    composites moved to Cython `fast_*` helpers and are now dead.

    Drop them so the module's dependency surface reflects only the
    cross-module helpers it actually calls.
    """
    linalg_src = _source("src/candle/_backends/npu/ops/linalg.py")
    forbidden_from_math = ["cos", "exp", "sin"]
    forbidden_from_shape = ["diag"]

    # Locate the `from .math import ...` line and the `from .shape import ...`
    # line and assert the forbidden names are not in them.
    for line in linalg_src.splitlines():
        stripped = line.strip()
        if stripped.startswith("from .math import "):
            imported = stripped[len("from .math import "):]
            for name in forbidden_from_math:
                assert not re.search(rf"\b{name}\b", imported), (
                    f"`src/candle/_backends/npu/ops/linalg.py` still imports "
                    f"`{name}` from `.math`, but no function body calls it — "
                    "drop the dead import."
                )
        if stripped.startswith("from .shape import "):
            imported = stripped[len("from .shape import "):]
            for name in forbidden_from_shape:
                assert not re.search(rf"\b{name}\b", imported), (
                    f"`src/candle/_backends/npu/ops/linalg.py` still imports "
                    f"`{name}` from `.shape`, but no function body calls it — "
                    "drop the dead import."
                )


def test_npu_special_module_consolidates_fast_helper_try_blocks():
    """`special.py` carries 2 top-level `try/except ImportError` blocks
    that import distinct `fast_*` helpers from `candle._C._npu_ops` —
    one for `_HAS_FAST_SPECIAL_COMPOSITES` (sinc / xlogy / ndtr etc.)
    and one for `_HAS_FAST_SPECIAL_UNARY` (digamma / erfinv / lgamma).
    The two blocks load from the same Cython module, so they either
    both succeed or both fail at import time; splitting them adds an
    extra `ImportError` check without changing runtime semantics.

    Merge them into one consolidated block — matching the pattern in
    `math.py`, `comparison.py`, `activation.py`, and the rest of the
    ops package — so the module pays a single ImportError check.
    """
    special_src = _source("src/candle/_backends/npu/ops/special.py")
    try_count = len(
        [line for line in special_src.splitlines() if line.startswith("try:")]
    )
    assert try_count <= 1, (
        "`src/candle/_backends/npu/ops/special.py` still has "
        f"{try_count} top-level `try:` blocks for importing `fast_*` "
        "helpers. Consolidate them into a single try/except block so the "
        "module matches `math.py`, `comparison.py`, `activation.py`, and "
        "the rest of the ops package."
    )


def test_npu_binary_op_inlines_native_fast_ops_guard():
    """`_helpers.py::_require_native_fast_ops` is a 3-line wrapper used
    exactly once — by `_binary_op` immediately before calling
    `_fast_binary_op`. The indirection adds no value: the helper just
    raises a `RuntimeError` when `_HAS_FAST_OPS` is false, which
    `_binary_op` could check inline at the same call site.

    Inline the guard into `_binary_op` and delete the helper, so the
    runtime check is co-located with the only call site that uses it.
    Also drop any cross-codebase consumers — there should be none, but
    the audit should verify that claim instead of trusting inspection
    of `_helpers.py` alone.
    """
    helpers_src = _source("src/candle/_backends/npu/ops/_helpers.py")

    # The helper definition should be gone.
    assert "def _require_native_fast_ops(" not in helpers_src, (
        "`_helpers.py` still defines `_require_native_fast_ops`; inline "
        "the `if not _HAS_FAST_OPS:` check into `_binary_op` and drop "
        "the helper."
    )

    # `_binary_op` body must inline the `_HAS_FAST_OPS` guard.
    binary_src = _function_source(helpers_src, "_binary_op")
    assert "_HAS_FAST_OPS" in binary_src, (
        "`_binary_op` no longer guards the call to `_fast_binary_op` with "
        "an inline `_HAS_FAST_OPS` check after dropping the helper — the "
        "runtime safety guard must remain in place."
    )
    assert "_require_native_fast_ops" not in binary_src, (
        "`_binary_op` still calls the helper `_require_native_fast_ops`; "
        "inline the check instead."
    )

    # No other module in src/ or tests/ should reference the helper.
    consumer_roots = ["src/candle", "tests"]
    offenders = []
    pattern = re.compile(r"\b_require_native_fast_ops\b")
    audit_path = Path(__file__).resolve()
    for root in consumer_roots:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            if path.resolve() == audit_path:
                continue
            text = path.read_text(encoding="utf-8")
            if pattern.search(text):
                offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, (
        f"`_require_native_fast_ops` still referenced from: {offenders}. "
        "Remove all consumers before dropping the helper definition."
    )


def test_npu_helpers_imports_reshape_directly_without_view_backend_alias():
    """`_helpers.py` carries two adjacent lines whose only purpose is to
    expose `reshape` as a public name for re-export to consumer modules:

        from ...common import view as view_backend
        reshape = view_backend.reshape

    Inside `_helpers.py`, `view_backend` is never referenced anywhere
    else — it exists solely to define the `reshape` alias on the next
    line. No consumer module imports `view_backend` from `_helpers`
    either; the few NPU files that need `view_backend.permute` etc.
    import `view as view_backend` directly from `..._backends.common`
    in their own function bodies.

    Collapse the two lines into a single direct import:

        from ...common.view import reshape

    so `_helpers.py` does not carry an unused module-level alias.
    """
    helpers_src = _source("src/candle/_backends/npu/ops/_helpers.py")

    # The module-level alias setup line must be gone.
    assert "view as view_backend" not in helpers_src, (
        "`_helpers.py` still imports `view as view_backend` at module "
        "level. The alias has no other use in this file — collapse to "
        "`from ...common.view import reshape`."
    )
    assert "reshape = view_backend.reshape" not in helpers_src, (
        "`_helpers.py` still defines `reshape = view_backend.reshape`. "
        "Replace with a direct `from ...common.view import reshape` to "
        "drop the indirection through `view_backend`."
    )

    # The direct import must be present (so re-export of `reshape`
    # continues to work for consumer modules).
    assert "from ...common.view import reshape" in helpers_src, (
        "`_helpers.py` must import `reshape` directly from "
        "`...common.view` so its existing re-exports (activation.py / "
        "random.py / special.py) keep working."
    )

    # No `view_backend` name at module-level scope of `_helpers.py`.
    for line in helpers_src.splitlines():
        if line.startswith((" ", "\t", "#")) or not line.strip():
            continue
        assert "view_backend" not in line, (
            f"`_helpers.py` still references `view_backend` at module "
            f"level: {line!r}. The alias should be dropped entirely."
        )


def test_npu_linear_index_routed_directly_in_functionalize_not_via_ops_package():
    """`_npu_linear_index` is a private helper that lives in
    `_backends/npu/ops/_helpers.py`. Its only consumer is
    `_dispatch/functionalize.py`, which historically accessed it via the
    `npu_ops` package namespace because `_backends/npu/ops/__init__.py`
    re-exports it at its very first line:

        from ._helpers import _npu_linear_index

    Re-exporting a private `_*` helper through the package's public
    namespace is the same kind of API-surface leak that batch 64 cleaned
    up for the other `_helpers` plumbing helpers. The sole call site can
    import the helper directly from `_helpers`:

        from .._backends.npu.ops._helpers import _npu_linear_index
        linear = _npu_linear_index(...)

    so the package's `__init__.py` no longer needs to expose a private
    name.
    """
    init_src = _source("src/candle/_backends/npu/ops/__init__.py")
    func_src = _source("src/candle/_dispatch/functionalize.py")

    # Package __init__.py must not re-export the private helper any more.
    assert "from ._helpers import _npu_linear_index" not in init_src, (
        "`src/candle/_backends/npu/ops/__init__.py` still re-exports "
        "`_npu_linear_index` from `_helpers`. Drop the re-export and "
        "have `_dispatch/functionalize.py` import the helper directly "
        "from `_helpers`."
    )
    assert "_npu_linear_index" not in init_src, (
        "`src/candle/_backends/npu/ops/__init__.py` still mentions "
        "`_npu_linear_index`; the private helper should not be on the "
        "package's public surface at all."
    )

    # functionalize.py must call the helper directly, not via the
    # package namespace.
    assert "npu_ops._npu_linear_index" not in func_src, (
        "`_dispatch/functionalize.py` still accesses `_npu_linear_index` "
        "via the `npu_ops` package namespace. Import it directly from "
        "`_backends.npu.ops._helpers` and call it locally."
    )
    assert (
        "from .._backends.npu.ops._helpers import _npu_linear_index"
        in func_src
    ), (
        "`_dispatch/functionalize.py` must import `_npu_linear_index` "
        "directly from `_helpers` so the package `__init__.py` no "
        "longer needs to re-export the private helper."
    )

    # Cross-codebase consumer-scan: nothing else should reach the
    # helper via the `npu_ops` namespace, and the only direct importer
    # outside `_helpers.py` itself should be `_dispatch/functionalize.py`.
    consumer_roots = ["src/candle", "tests"]
    namespace_offenders = []
    direct_consumers = []
    namespace_pat = re.compile(r"\bnpu_ops\._npu_linear_index\b")
    direct_pat = re.compile(
        r"from\s+[\w.]*\._helpers\s+import\s+[^\n]*\b_npu_linear_index\b"
    )
    audit_path = Path(__file__).resolve()
    helpers_path = (
        _REPO_ROOT / "src/candle/_backends/npu/ops/_helpers.py"
    ).resolve()
    for root in consumer_roots:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            resolved = path.resolve()
            if resolved == audit_path or resolved == helpers_path:
                continue
            text = path.read_text(encoding="utf-8")
            if namespace_pat.search(text):
                namespace_offenders.append(str(path.relative_to(_REPO_ROOT)))
            if direct_pat.search(text):
                direct_consumers.append(str(path.relative_to(_REPO_ROOT)))
    assert not namespace_offenders, (
        f"`npu_ops._npu_linear_index` still referenced from: "
        f"{namespace_offenders}. Route all consumers through the direct "
        "`_helpers` import."
    )
    assert direct_consumers == ["src/candle/_dispatch/functionalize.py"], (
        f"Unexpected direct importers of `_npu_linear_index` from "
        f"`_helpers`: {direct_consumers}. Only `_dispatch/functionalize.py` "
        "should import it directly."
    )


def test_npu_pow_tensor_scalar_branch_inlined_into_pow_without_wrapper_helper():
    """Audit: `_pow_tensor_scalar_op` was a 4-line single-use helper in
    `_backends/npu/ops/math.py` that wrapped the
    `_fast_pow_tensor_scalar_impl` call for the scalar-exponent path of
    `pow`. Its only caller was `pow` itself.

    Inlining the scalar branch directly into `pow` removes the indirection
    and reduces the function from a two-hop dispatch (`pow` ->
    `_pow_tensor_scalar_op` -> `_fast_pow_tensor_scalar_impl`) to a single
    direct call.
    """
    math_path = _REPO_ROOT / "src/candle/_backends/npu/ops/math.py"
    math_src = math_path.read_text(encoding="utf-8")

    # The wrapper helper itself must be gone.
    assert "def _pow_tensor_scalar_op" not in math_src, (
        "`_pow_tensor_scalar_op` helper still defined in math.py. "
        "Inline its body into `pow` so the scalar-exponent path calls "
        "`_fast_pow_tensor_scalar_impl` directly."
    )

    # `pow` must reference the fast helper directly.
    pow_body = _function_source(math_src, "pow")
    assert "_fast_pow_tensor_scalar_impl" in pow_body, (
        "`pow` body must call `_fast_pow_tensor_scalar_impl` directly "
        "after the wrapper helper is removed."
    )
    assert "_pow_tensor_scalar_op" not in pow_body, (
        "`pow` still references the removed `_pow_tensor_scalar_op` "
        "wrapper helper."
    )

    # Cross-codebase consumer-scan: the wrapper name must not appear
    # anywhere else in `src/candle/` or `tests/` (excluding this audit
    # file's own references to it).
    consumer_roots = ["src/candle", "tests"]
    audit_path = Path(__file__).resolve()
    offenders = []
    for root in consumer_roots:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            if path.resolve() == audit_path:
                continue
            text = path.read_text(encoding="utf-8")
            if "_pow_tensor_scalar_op" in text:
                offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, (
        f"`_pow_tensor_scalar_op` still referenced from: {offenders}. "
        "Drop all remaining references."
    )


def test_npu_adamw_step_registers_adam_step_op_directly_without_rename_wrapper():
    """Audit: `_adamw_step_op` in `_backends/npu/ops/optim.py` was a
    4-line pure-delegation wrapper that called `_adam_step_op` with the
    exact same argument list. Both ops dispatch to the same Cython
    `_fast_adam_step_impl` kernel, so the wrapper added a hop without
    changing behavior.

    Dropping the wrapper registers `_adamw_step` directly to
    `_adam_step_op`, removing one indirection and one symbol from the
    package `__init__.py` re-export surface.
    """
    optim_src = _source("src/candle/_backends/npu/ops/optim.py")
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")
    ops_init_src = _source("src/candle/_backends/npu/ops/__init__.py")

    # The wrapper function definition must be gone.
    assert "def _adamw_step_op" not in optim_src, (
        "`_adamw_step_op` wrapper still defined in `optim.py`. Remove "
        "it and register `_adamw_step` directly with `_adam_step_op`."
    )

    # `_adamw_step` registration must point at `_adam_step_op` directly.
    assert 'registry.register("_adamw_step", "npu", _adam_step_op)' in backend_init_src, (
        "`_adamw_step` must register `_adam_step_op` directly, not the "
        "removed `_adamw_step_op` rename wrapper."
    )
    assert 'registry.register("_adamw_step", "npu", _adamw_step_op)' not in backend_init_src, (
        "`_adamw_step` still registered against the removed "
        "`_adamw_step_op` wrapper in `_backends/npu/__init__.py`."
    )

    # Neither `__init__.py` should still import or re-export the wrapper.
    assert "_adamw_step_op" not in backend_init_src, (
        "`_backends/npu/__init__.py` still references `_adamw_step_op`."
    )
    assert "_adamw_step_op" not in ops_init_src, (
        "`_backends/npu/ops/__init__.py` still re-exports "
        "`_adamw_step_op`."
    )

    # Cross-codebase consumer-scan: nothing else in `src/candle/` or
    # `tests/` should reference the removed wrapper.
    consumer_roots = ["src/candle", "tests"]
    audit_path = Path(__file__).resolve()
    offenders = []
    for root in consumer_roots:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            if path.resolve() == audit_path:
                continue
            text = path.read_text(encoding="utf-8")
            if "_adamw_step_op" in text:
                offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, (
        f"`_adamw_step_op` still referenced from: {offenders}. Drop all "
        "remaining references."
    )


def test_npu_mm_and_bmm_register_matmul_directly_without_alias_wrappers():
    """Audit: `mm_op` and `bmm_op` in `_backends/npu/ops/linalg.py` were
    2-line pure-delegation wrappers that just returned `matmul(a, b)`.
    The `mm` and `bmm` schemas take exactly two tensors, so the
    intermediate names added a hop without changing behavior.

    Registering `mm` and `bmm` directly to `matmul` removes one
    indirection per op and drops both names from the package
    `__init__.py` re-export surface.
    """
    linalg_src = _source("src/candle/_backends/npu/ops/linalg.py")
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")
    ops_init_src = _source("src/candle/_backends/npu/ops/__init__.py")

    # The wrapper definitions must be gone.
    assert "def mm_op" not in linalg_src, (
        "`mm_op` wrapper still defined in `linalg.py`. Register `mm` "
        "directly with `matmul`."
    )
    assert "def bmm_op" not in linalg_src, (
        "`bmm_op` wrapper still defined in `linalg.py`. Register `bmm` "
        "directly with `matmul`."
    )

    # `mm` / `bmm` must register `matmul` directly.
    assert (
        'registry.register("mm", "npu", matmul, meta=meta_infer.infer_matmul)'
        in backend_init_src
    ), (
        "`mm` must register `matmul` directly, not the removed `mm_op` "
        "wrapper."
    )
    assert (
        'registry.register("bmm", "npu", matmul, meta=meta_infer.infer_matmul)'
        in backend_init_src
    ), (
        "`bmm` must register `matmul` directly, not the removed `bmm_op` "
        "wrapper."
    )

    # Neither `__init__.py` should import or re-export the wrappers.
    for name in ("mm_op", "bmm_op"):
        assert name not in backend_init_src, (
            f"`_backends/npu/__init__.py` still references `{name}`."
        )
        assert name not in ops_init_src, (
            f"`_backends/npu/ops/__init__.py` still re-exports `{name}`."
        )

    # Cross-codebase consumer-scan: nothing else in `src/candle/` or
    # `tests/` should reference the removed wrappers.
    consumer_roots = ["src/candle", "tests"]
    audit_path = Path(__file__).resolve()
    offenders = []
    for root in consumer_roots:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            if path.resolve() == audit_path:
                continue
            text = path.read_text(encoding="utf-8")
            if "mm_op" in text or "bmm_op" in text:
                offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, (
        f"`mm_op` / `bmm_op` still referenced from: {offenders}. Drop "
        "all remaining references."
    )


def test_npu_linalg_det_registers_det_op_directly_without_alias_wrapper():
    """Audit: `linalg_det_op` in `_backends/npu/ops/linalg.py` was a
    3-line pure-delegation wrapper that just returned `det_op(a)`.
    The `linalg_det` schema `(Tensor input) -> Tensor` matches `det_op`'s
    signature exactly, so the intermediate name added a hop without
    changing behavior.

    Registering `linalg_det` directly to `det_op` removes one indirection
    and drops `linalg_det_op` from the package `__init__.py` re-export
    surface.
    """
    linalg_src = _source("src/candle/_backends/npu/ops/linalg.py")
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")
    ops_init_src = _source("src/candle/_backends/npu/ops/__init__.py")

    # The wrapper definition must be gone.
    assert "def linalg_det_op" not in linalg_src, (
        "`linalg_det_op` wrapper still defined in `linalg.py`. Register "
        "`linalg_det` directly with `det_op`."
    )

    # `linalg_det` must register `det_op` directly.
    assert (
        'registry.register("linalg_det", "npu", det_op)' in backend_init_src
    ), (
        "`linalg_det` must register `det_op` directly, not the removed "
        "`linalg_det_op` wrapper."
    )

    # Neither `__init__.py` should import or re-export the wrapper.
    assert "linalg_det_op" not in backend_init_src, (
        "`_backends/npu/__init__.py` still references `linalg_det_op`."
    )
    assert "linalg_det_op" not in ops_init_src, (
        "`_backends/npu/ops/__init__.py` still re-exports `linalg_det_op`."
    )

    # Cross-codebase consumer-scan: nothing else in `src/candle/` or
    # `tests/` should reference the removed wrapper.
    consumer_roots = ["src/candle", "tests"]
    audit_path = Path(__file__).resolve()
    offenders = []
    for root in consumer_roots:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            if path.resolve() == audit_path:
                continue
            text = path.read_text(encoding="utf-8")
            if "linalg_det_op" in text:
                offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, (
        f"`linalg_det_op` still referenced from: {offenders}. Drop all "
        "remaining references."
    )


def test_npu_shape_alias_wrappers_register_target_ops_directly():
    """Audit: three shape wrappers in `_backends/npu/ops/shape.py` were
    pure-delegation aliases that just forwarded their arguments to an
    already-registered NPU op of a different name:

      - `broadcast_to_op(a, shape) -> expand(a, shape)`
      - `concatenate(tensors, dim=0) -> cat(tensors, dim=dim)`
      - `row_stack(tensors) -> vstack(tensors)`

    Each schema (`broadcast_to`, `concatenate`, `row_stack`) matches the
    underlying op's schema exactly, so the intermediate names added a
    hop without changing behavior.

    Registering the dispatch names directly to `expand`, `cat`, and
    `vstack` removes one indirection per op and drops the aliases from
    the package `__init__.py` re-export surface.
    """
    shape_src = _source("src/candle/_backends/npu/ops/shape.py")
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")
    ops_init_src = _source("src/candle/_backends/npu/ops/__init__.py")

    # The wrapper definitions must be gone.
    assert "def broadcast_to_op" not in shape_src, (
        "`broadcast_to_op` wrapper still defined in `shape.py`. Register "
        "`broadcast_to` directly with `expand`."
    )
    assert "def concatenate" not in shape_src, (
        "`concatenate` wrapper still defined in `shape.py`. Register "
        "`concatenate` directly with `cat`."
    )
    assert "def row_stack" not in shape_src, (
        "`row_stack` wrapper still defined in `shape.py`. Register "
        "`row_stack` directly with `vstack`."
    )

    # Each dispatch name must register the underlying op directly.
    assert (
        'registry.register("broadcast_to", "npu", expand, '
        "meta=meta_infer.infer_broadcast_to)" in backend_init_src
    ), (
        "`broadcast_to` must register `expand` directly, not the removed "
        "`broadcast_to_op` wrapper."
    )
    assert (
        'registry.register("concatenate", "npu", cat, '
        "meta=meta_infer.infer_cat)" in backend_init_src
    ), (
        "`concatenate` must register `cat` directly, not the removed "
        "`concatenate` wrapper."
    )
    assert (
        'registry.register("row_stack", "npu", vstack, '
        "meta=meta_infer.infer_vstack)" in backend_init_src
    ), (
        "`row_stack` must register `vstack` directly, not the removed "
        "`row_stack` wrapper."
    )

    # Neither `__init__.py` should import or re-export the wrappers.
    # `broadcast_to_op` is distinctive (no collision with dispatch-name
    # strings); `concatenate` and `row_stack` collide with the public op
    # dispatch keys (`"concatenate"`, `"row_stack"`) so they must be
    # checked using the import-block line pattern instead of a bare
    # substring match.
    assert "broadcast_to_op" not in backend_init_src, (
        "`_backends/npu/__init__.py` still references `broadcast_to_op`."
    )
    assert "broadcast_to_op" not in ops_init_src, (
        "`_backends/npu/ops/__init__.py` still re-exports `broadcast_to_op`."
    )
    for name in ("concatenate", "row_stack"):
        import_line = f"\n    {name},\n"
        assert import_line not in backend_init_src, (
            f"`_backends/npu/__init__.py` still imports `{name}` from the "
            "shape ops module."
        )
        assert import_line not in ops_init_src and f" {name}," not in ops_init_src, (
            f"`_backends/npu/ops/__init__.py` still re-exports `{name}`."
        )

    # Cross-codebase consumer-scan for `broadcast_to_op` — `_op` suffix is
    # distinctive enough to bare-word match across the repo. `concatenate`
    # and `row_stack` collide with the public torch API (`torch.concatenate`,
    # `torch.row_stack`) used throughout tests/, so they cannot be scanned
    # by bare-word match; the file-level checks above cover them.
    consumer_roots = ["src/candle", "tests"]
    audit_path = Path(__file__).resolve()
    offenders = []
    for root in consumer_roots:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            if path.resolve() == audit_path:
                continue
            text = path.read_text(encoding="utf-8")
            if "broadcast_to_op" in text:
                offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, (
        f"`broadcast_to_op` still referenced from: {offenders}. Drop "
        "all remaining references."
    )


def test_npu_erfinv_inplace_collapses_def_wrapper_to_cython_alias():
    """Audit: `erfinv_` in `_backends/npu/ops/random.py` was a 3-line
    `def` wrapper whose body just returned `_fast_erfinv_inplace_impl(a)`.
    The schema `erfinv_(Tensor(a!) self) -> Tensor(a)` takes a single
    tensor — identical to the Cython impl — so the wrapper added a Python
    function-call frame without changing behavior.

    Collapse the wrapper to a module-level alias:

      erfinv_ = _fast_erfinv_inplace_impl

    This keeps the `erfinv_` registration import surface intact while
    eliminating the intermediate frame. The audit asserts the `def` is
    gone and the alias assignment is present.
    """
    random_src = _source("src/candle/_backends/npu/ops/random.py")

    assert "def erfinv_(" not in random_src, (
        "`erfinv_` is still defined as a `def` wrapper in `random.py`. "
        "Replace with the module-level alias "
        "`erfinv_ = _fast_erfinv_inplace_impl`."
    )
    assert "erfinv_ = _fast_erfinv_inplace_impl" in random_src, (
        "`random.py` must bind `erfinv_` to `_fast_erfinv_inplace_impl` "
        "via a module-level alias so the registration import surface in "
        "`_backends/npu/__init__.py` continues to work."
    )


def test_npu_operator_intake_tranche1_registers_required_ops():
    """The first NPU operator intake tranche intentionally exposes only
    existing implementations: common view metadata ops and native constant pad.
    """
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")

    expected = {
        "view_as_real": "view_backend.view_as_real",
        "view_as_complex": "view_backend.view_as_complex",
        "constant_pad_nd": "constant_pad_nd",
    }
    for op_name, target in expected.items():
        needle = f'registry.register("{op_name}", "npu", {target}'
        assert needle in backend_init_src, (
            f"NPU backend must register `{op_name}` directly to `{target}` "
            "as part of operator intake tranche 1."
        )


def test_npu_view_as_ops_route_through_common_view_backend():
    """`view_as_real` and `view_as_complex` are metadata/storage
    reinterpretation ops. NPU should reuse the common view backend rather than
    adding a device copy path or backend-specific wrapper.
    """
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")

    assert 'registry.register("view_as_real", "npu", view_backend.view_as_real' in backend_init_src
    assert 'registry.register("view_as_complex", "npu", view_backend.view_as_complex' in backend_init_src
    assert 'registry.register("view_as_real", "npu", convert_backend' not in backend_init_src
    assert 'registry.register("view_as_complex", "npu", convert_backend' not in backend_init_src


def test_npu_constant_pad_nd_routes_to_existing_constant_pad_path():
    """`constant_pad_nd` should be a schema-compatible adapter over the
    existing NPU `pad` implementation, whose constant branch calls
    `aclnn.constant_pad_nd`.
    """
    conv_src = _source("src/candle/_backends/npu/ops/conv.py")
    ops_init_src = _source("src/candle/_backends/npu/ops/__init__.py")
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")

    body = _function_source(conv_src, "constant_pad_nd")
    assert "return pad(input, padding, mode='constant', value=value)" in body or (
        'return pad(input, padding, mode="constant", value=value)' in body
    ), "`constant_pad_nd` must delegate to the existing NPU constant pad path."
    assert "aclnn.constant_pad_nd(" in _function_source(conv_src, "pad"), (
        "NPU `pad` constant branch must remain backed by aclnn.constant_pad_nd."
    )
    assert "constant_pad_nd" in ops_init_src, "NPU ops package must export `constant_pad_nd`."
    assert 'registry.register("constant_pad_nd", "npu", constant_pad_nd' in backend_init_src


def test_npu_operator_intake_tranche2_registers_inplace_unary_ops():
    """The second NPU operator intake tranche exposes 6 native in-place unary
    ops that already have ACLNN-backed Cython fast helpers
    (`fast_*_inplace`). Each is wired to its helper through a module-level
    alias in `math.py` (the same pattern PR #508 established for `erfinv_`).
    """
    math_src = _source("src/candle/_backends/npu/ops/math.py")
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")

    aliases = {
        "neg_": "_fast_neg_inplace_impl",
        "exp_": "_fast_exp_inplace_impl",
        "log_": "_fast_log_inplace_impl",
        "tan_": "_fast_tan_inplace_impl",
        "floor_": "_fast_floor_inplace_impl",
        "ceil_": "_fast_ceil_inplace_impl",
    }
    for op_name, fast_name in aliases.items():
        assert f"def {op_name}(" not in math_src, (
            f"`{op_name}` must be a module-level alias to `{fast_name}`, "
            "not a `def` wrapper."
        )
        assert f"{op_name} = {fast_name}" in math_src, (
            f"`math.py` must bind `{op_name}` to `{fast_name}` via a "
            "module-level alias so the registration import surface in "
            "`_backends/npu/__init__.py` resolves correctly."
        )
        assert f'registry.register("{op_name}", "npu", {op_name}' in backend_init_src, (
            f"NPU backend must register `{op_name}` directly to the "
            "Cython-aliased function as part of operator intake tranche 2."
        )


def test_npu_operator_intake_tranche3a_registers_functional_masked_scatter():
    """The functional (out-of-place) `masked_scatter` is wired as a composite
    of `clone()` + `masked_scatter_` on NPU. All ops stay on the device — no
    CPU fallback. The op already has a generated autograd derivative
    (`grad.masked_fill(mask, 0)` for self, `masked_scatter_backward` for
    source), so requires_grad flows route through AutogradNPU correctly.
    """
    shape_src = _source("src/candle/_backends/npu/ops/shape.py")
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")
    ops_init_src = _source("src/candle/_backends/npu/ops/__init__.py")

    assert "def masked_scatter(a, mask, source):" in shape_src
    assert "result = a.clone()" in shape_src
    assert "masked_scatter_(result, mask, source)" in shape_src

    assert "masked_scatter," in ops_init_src
    assert "masked_scatter,\n" in backend_init_src
    assert (
        'registry.register("masked_scatter", "npu", masked_scatter'
        in backend_init_src
    )


def test_npu_operator_intake_tranche3b_registers_var_mean_composite():
    """`var_mean` returns `(variance, mean)` and is wired on NPU as a composite
    of the existing `var_` + `mean` kernels. Both reductions stay on the NPU
    device — no CPU fallback. The op already has a generated autograd
    derivative (`var_mean_backward`), so requires_grad flows route through
    AutogradNPU correctly. Meta inference returns a `(spec, spec)` tuple
    matching the schema's `-> (Tensor, Tensor)` return type.
    """
    reduce_src = _source("src/candle/_backends/npu/ops/reduce.py")
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")
    ops_init_src = _source("src/candle/_backends/npu/ops/__init__.py")
    meta_src = _source("src/candle/_backends/meta/infer.py")

    assert "def var_mean(a, dim=None, unbiased=True, keepdim=False):" in reduce_src
    assert "var_(a, dim=dim, unbiased=unbiased, keepdim=keepdim)" in reduce_src
    assert "mean(a, dim=dim, keepdim=keepdim)" in reduce_src

    assert "var_mean," in ops_init_src
    assert "var_mean," in backend_init_src
    assert (
        'registry.register("var_mean", "npu", var_mean, meta=meta_infer.infer_var_mean)'
        in backend_init_src
    )
    assert "def infer_var_mean(a, dim=None, unbiased=True, keepdim=False)" in meta_src


def test_npu_operator_intake_tranche3c_registers_inplace_sin_cos():
    """Operator intake tranche 3c extends the in-place unary surface (PR #508,
    tranche 2) with `sin_` and `cos_`. Both are wired through new
    `fast_sin_inplace` / `fast_cos_inplace` Cython helpers that reuse the
    existing `aclnnSin` / `aclnnCos` bindings with output aliased to input —
    so all computation stays on NPU.
    """
    math_src = _source("src/candle/_backends/npu/ops/math.py")
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")
    pyx_src = _source("src/candle/_C/_npu_ops.pyx")

    aliases = {
        "sin_": "_fast_sin_inplace_impl",
        "cos_": "_fast_cos_inplace_impl",
    }
    for op_name, fast_name in aliases.items():
        assert f"def {op_name}(" not in math_src, (
            f"`{op_name}` must be a module-level alias to `{fast_name}`."
        )
        assert f"{op_name} = {fast_name}" in math_src
        assert f'registry.register("{op_name}", "npu", {op_name}' in backend_init_src

    assert "def fast_sin_inplace(a):" in pyx_src
    assert "def fast_cos_inplace(a):" in pyx_src


def test_npu_operator_intake_tranche3d_registers_inplace_sqrt_sigmoid_tanh():
    """Operator intake tranche 3d extends the in-place unary surface with
    `sqrt_`, `sigmoid_`, and `tanh_`. Each reuses the existing
    `aclnnSqrt` / `aclnnSigmoid` / `aclnnTanh` bindings via a new
    `fast_*_inplace` Cython helper that aliases output to input — fully
    NPU-resident.
    """
    math_src = _source("src/candle/_backends/npu/ops/math.py")
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")
    pyx_src = _source("src/candle/_C/_npu_ops.pyx")

    aliases = {
        "sqrt_": "_fast_sqrt_inplace_impl",
        "sigmoid_": "_fast_sigmoid_inplace_impl",
        "tanh_": "_fast_tanh_inplace_impl",
    }
    for op_name, fast_name in aliases.items():
        assert f"def {op_name}(" not in math_src
        assert f"{op_name} = {fast_name}" in math_src
        assert f'registry.register("{op_name}", "npu", {op_name}' in backend_init_src

    assert "def fast_sqrt_inplace(a):" in pyx_src
    assert "def fast_sigmoid_inplace(a):" in pyx_src
    assert "def fast_tanh_inplace(a):" in pyx_src


def test_npu_operator_intake_tranche3e_registers_inplace_abs_round_trunc_log2_log10():
    """Operator intake tranche 3e extends the in-place unary surface with
    `abs_`, `round_`, `trunc_`, `log2_`, and `log10_`. Each reuses the
    existing `aclnnAbs` / `aclnnRound` / `aclnnTrunc` / `aclnnLog2` /
    `aclnnLog10` bindings via a new `fast_*_inplace` Cython helper that
    aliases output to input — fully NPU-resident.
    """
    math_src = _source("src/candle/_backends/npu/ops/math.py")
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")
    pyx_src = _source("src/candle/_C/_npu_ops.pyx")

    aliases = {
        "abs_": "_fast_abs_inplace_impl",
        "round_": "_fast_round_inplace_impl",
        "trunc_": "_fast_trunc_inplace_impl",
        "log2_": "_fast_log2_inplace_impl",
        "log10_": "_fast_log10_inplace_impl",
    }
    for op_name, fast_name in aliases.items():
        assert f"def {op_name}(" not in math_src
        assert f"{op_name} = {fast_name}" in math_src
        assert f'registry.register("{op_name}", "npu", {op_name}' in backend_init_src

    assert "def fast_abs_inplace(a):" in pyx_src
    assert "def fast_round_inplace(a):" in pyx_src
    assert "def fast_trunc_inplace(a):" in pyx_src
    assert "def fast_log2_inplace(a):" in pyx_src
    assert "def fast_log10_inplace(a):" in pyx_src


def test_npu_operator_intake_tranche3f_registers_inplace_expm1_log1p_exp2_erf_erfc():
    """Operator intake tranche 3f extends the in-place unary surface with
    `expm1_`, `log1p_`, `exp2_`, `erf_`, and `erfc_`. Each reuses the
    existing `aclnnExpm1` / `aclnnLog1p` / `aclnnExp2` / `aclnnErf` /
    `aclnnErfc` bindings via a new `fast_*_inplace` Cython helper that
    aliases output to input — fully NPU-resident. New schemas are
    registered in `_dispatch/schemas.py`.
    """
    math_src = _source("src/candle/_backends/npu/ops/math.py")
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")
    pyx_src = _source("src/candle/_C/_npu_ops.pyx")
    schemas_src = _source("src/candle/_dispatch/schemas.py")

    aliases = {
        "expm1_": "_fast_expm1_inplace_impl",
        "log1p_": "_fast_log1p_inplace_impl",
        "exp2_": "_fast_exp2_inplace_impl",
        "erf_": "_fast_erf_inplace_impl",
        "erfc_": "_fast_erfc_inplace_impl",
    }
    for op_name, fast_name in aliases.items():
        assert f"def {op_name}(" not in math_src
        assert f"{op_name} = {fast_name}" in math_src
        assert f'registry.register("{op_name}", "npu", {op_name}' in backend_init_src
        assert f'register_schema("{op_name}",' in schemas_src

    assert "def fast_expm1_inplace(a):" in pyx_src
    assert "def fast_log1p_inplace(a):" in pyx_src
    assert "def fast_exp2_inplace(a):" in pyx_src
    assert "def fast_erf_inplace(a):" in pyx_src
    assert "def fast_erfc_inplace(a):" in pyx_src


def test_npu_operator_intake_tranche3g_registers_inplace_asin_acos_atan_sinh_cosh():
    """Operator intake tranche 3g extends the in-place unary surface with
    `asin_`, `acos_`, `atan_`, `sinh_`, and `cosh_`. Each reuses the
    existing `aclnnAsin` / `aclnnAcos` / `aclnnAtan` / `aclnnSinh` /
    `aclnnCosh` bindings via a new `fast_*_inplace` Cython helper that
    aliases output to input — fully NPU-resident. New schemas are
    registered in `_dispatch/schemas.py`.
    """
    math_src = _source("src/candle/_backends/npu/ops/math.py")
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")
    pyx_src = _source("src/candle/_C/_npu_ops.pyx")
    schemas_src = _source("src/candle/_dispatch/schemas.py")

    aliases = {
        "asin_": "_fast_asin_inplace_impl",
        "acos_": "_fast_acos_inplace_impl",
        "atan_": "_fast_atan_inplace_impl",
        "sinh_": "_fast_sinh_inplace_impl",
        "cosh_": "_fast_cosh_inplace_impl",
    }
    for op_name, fast_name in aliases.items():
        assert f"def {op_name}(" not in math_src
        assert f"{op_name} = {fast_name}" in math_src
        assert f'registry.register("{op_name}", "npu", {op_name}' in backend_init_src
        assert f'register_schema("{op_name}",' in schemas_src

    assert "def fast_asin_inplace(a):" in pyx_src
    assert "def fast_acos_inplace(a):" in pyx_src
    assert "def fast_atan_inplace(a):" in pyx_src
    assert "def fast_sinh_inplace(a):" in pyx_src
    assert "def fast_cosh_inplace(a):" in pyx_src


def test_npu_operator_intake_tranche3h_registers_inplace_asinh_acosh_atanh_rsqrt_square():
    """Operator intake tranche 3h extends the in-place unary surface with
    `asinh_`, `acosh_`, `atanh_`, `rsqrt_`, and `square_`. Each reuses the
    existing `aclnnAsinh` / `aclnnAcosh` / `aclnnAtanh` / `aclnnRsqrt` /
    `aclnnSquare` bindings via a new `fast_*_inplace` Cython helper that
    aliases output to input — fully NPU-resident. New schemas are
    registered in `_dispatch/schemas.py`.
    """
    math_src = _source("src/candle/_backends/npu/ops/math.py")
    backend_init_src = _source("src/candle/_backends/npu/__init__.py")
    pyx_src = _source("src/candle/_C/_npu_ops.pyx")
    schemas_src = _source("src/candle/_dispatch/schemas.py")

    aliases = {
        "asinh_": "_fast_asinh_inplace_impl",
        "acosh_": "_fast_acosh_inplace_impl",
        "atanh_": "_fast_atanh_inplace_impl",
        "rsqrt_": "_fast_rsqrt_inplace_impl",
        "square_": "_fast_square_inplace_impl",
    }
    for op_name, fast_name in aliases.items():
        assert f"def {op_name}(" not in math_src
        assert f"{op_name} = {fast_name}" in math_src
        assert f'registry.register("{op_name}", "npu", {op_name}' in backend_init_src
        assert f'register_schema("{op_name}",' in schemas_src

    assert "def fast_asinh_inplace(a):" in pyx_src
    assert "def fast_acosh_inplace(a):" in pyx_src
    assert "def fast_atanh_inplace(a):" in pyx_src
    assert "def fast_rsqrt_inplace(a):" in pyx_src
    assert "def fast_square_inplace(a):" in pyx_src
