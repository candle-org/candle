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
