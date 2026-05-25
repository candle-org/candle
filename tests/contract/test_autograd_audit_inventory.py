# pylint: disable=missing-function-docstring
"""Source-reading audit of remaining autograd parity gaps.

The test bodies avoid importing ``candle`` directly and inspect the source files
that define the current autograd inventory.
"""

import ast
import pathlib
import re

_ROOT = pathlib.Path(__file__).parent.parent.parent
_SRC = _ROOT / "src" / "candle"

OPTIMIZER_STEP_OPS = {
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
}

CREATION_RANDOM_OPS = {
    "arange",
    "empty",
    "empty_like",
    "eye",
    "full",
    "full_like",
    "linspace",
    "logspace",
    "ones",
    "ones_like",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
    "range",
    "tensor",
    "tril_indices",
    "triu_indices",
    "zeros",
    "zeros_like",
}

INPLACE_OR_MUTATION_OPS = {
    "abs_",
    "acos_",
    "acosh_",
    "asin_",
    "asinh_",
    "atan_",
    "atanh_",
    "bitwise_and_",
    "bitwise_or_",
    "bitwise_xor_",
    "ceil_",
    "cos_",
    "cosh_",
    "digamma_",
    "elu_",
    "erf_",
    "erfc_",
    "erfinv_",
    "exp2_",
    "exp_",
    "expm1_",
    "floor_",
    "hardtanh_",
    "leaky_relu_",
    "log10_",
    "log1p_",
    "log2_",
    "log_",
    "mish_",
    "neg_",
    "pow_",
    "randint_",
    "reciprocal_",
    "relu6_",
    "round_",
    "rsqrt_",
    "sigmoid_",
    "sign_",
    "silu_",
    "sin_",
    "sinh_",
    "sqrt_",
    "square_",
    "tan_",
    "tanh_",
    "trunc_",
}

COMPARISON_INDEX_OPS = {
    "allclose",
    "argsort",
    "argwhere",
    "bincount",
    "bucketize",
    "cartesian_prod",
    "equal",
    "histogram",
    "isclose",
    "isin",
    "isneginf",
    "isposinf",
    "isreal",
    "searchsorted",
}

SHAPE_VIEW_COPY_OPS = {
    "as_strided_copy",
    "expand_copy",
    "flatten",
    "movedim",
    "narrow",
    "slice_copy",
    "squeeze",
    "unflatten",
}

NONDIFFERENTIABLE_OUTPUT_OPS = {
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "logical_xor",
}

MANUAL_REVIEW_SCHEMA_OPS = set()

LIKELY_NONDIFFERENTIABLE_NOT_IMPLEMENTED = {
    "_unique",
    "_unique2",
    "acosh_",
    "asinh_",
    "atanh_",
    "chunk",
    "embedding_renorm_",
    "geqrf",
    "nextafter",
    "unique_consecutive",
    "unique_dim",
    "unique_dim_consecutive",
}

INTERNAL_NOT_IMPLEMENTED = {
    "_cdist_backward",
    "_embedding_bag_backward",
    "_embedding_bag_dense_backward",
    "_pdist_backward",
    "_standard_gamma_grad",
    "batch_norm_backward",
    "cudnn_batch_norm_backward",
    "miopen_batch_norm_backward",
    "native_batch_norm_backward",
    "native_dropout_backward",
    "native_layer_norm_backward",
}

PYTORCH_ERROR_PARITY_NOT_IMPLEMENTED = {
    "igamma",
    "igammac",
    "special_zeta",
}

MANUAL_REVIEW_NOT_IMPLEMENTED = set()


def _read(path):
    return path.read_text(encoding="utf-8")


def _schema_ops():
    tree = ast.parse(_read(_SRC / "_dispatch" / "schemas.py"))
    ops = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register_schema"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            ops.add(node.args[0].value)
    return ops


def _autograd_registered_ops():
    generated = _read(_SRC / "_generated" / "registration.py")
    manual = _read(_SRC / "_backends" / "autograd.py")
    return (
        set(re.findall(r"register_autograd_kernels\('([^']+)'", generated))
        | set(re.findall(r'\("([A-Za-z0-9_]+)",\s*lambda:', manual))
        | set(re.findall(r'\("([A-Za-z0-9_]+)",\s*lambda\s*:', manual))
        | set(re.findall(r'\("([A-Za-z0-9_]+)",\s*_[A-Za-z0-9_]+\)', manual))
    )


def _not_implemented_entries():
    current = None
    entries = []
    for line in _read(_ROOT / "tools" / "autograd" / "derivatives.yaml").splitlines():
        match = re.match(r"- name: ([A-Za-z0-9_\.]+)\(", line)
        if match:
            current = match.group(1)
        if "not_implemented(" in line and current:
            entries.append((current.split(".", 1)[0], line.strip()))
    return entries


def test_autograd_registration_inventory_counts():
    text = _read(_SRC / "_generated" / "registration.py")
    assert len(re.findall(r"register_autograd_kernels\(", text)) == 304
    assert len(re.findall(r"register_autograd_post_kernels\(", text)) == 304
    assert re.findall(r"_VT_PY\.", text) == []
    assert "_VT = _VT_CY if _VT_CY is not None else _VT_PY" in text


def test_schema_ops_without_autograd_registration_are_categorized():
    expected_missing = (
        OPTIMIZER_STEP_OPS
        | CREATION_RANDOM_OPS
        | INPLACE_OR_MUTATION_OPS
        | COMPARISON_INDEX_OPS
        | SHAPE_VIEW_COPY_OPS
        | NONDIFFERENTIABLE_OUTPUT_OPS
        | MANUAL_REVIEW_SCHEMA_OPS
    )
    actual_missing = _schema_ops() - _autograd_registered_ops()
    assert sorted(actual_missing - expected_missing) == []
    assert sorted(expected_missing - actual_missing) == []
    assert len(actual_missing) == 110


def test_derivatives_not_implemented_inventory_is_classified():
    entries = _not_implemented_entries()
    actual_ops = {name for name, _line in entries}
    expected_ops = (
        LIKELY_NONDIFFERENTIABLE_NOT_IMPLEMENTED
        | INTERNAL_NOT_IMPLEMENTED
        | PYTORCH_ERROR_PARITY_NOT_IMPLEMENTED
        | MANUAL_REVIEW_NOT_IMPLEMENTED
    )
    counts_by_op = {name: 0 for name in actual_ops}
    for name, _line in entries:
        counts_by_op[name] += 1
    assert sorted(actual_ops - expected_ops) == []
    assert sorted(expected_ops - actual_ops) == []
    assert counts_by_op == {
        "_cdist_backward": 4,
        "_embedding_bag_backward": 2,
        "_embedding_bag_dense_backward": 2,
        "_pdist_backward": 3,
        "_standard_gamma_grad": 1,
        "_unique": 1,
        "_unique2": 1,
        "acosh_": 1,
        "asinh_": 1,
        "atanh_": 1,
        "batch_norm_backward": 3,
        "chunk": 1,
        "cudnn_batch_norm_backward": 3,
        "embedding_renorm_": 1,
        "geqrf": 1,
        "igamma": 1,
        "igammac": 1,
        "miopen_batch_norm_backward": 2,
        "native_batch_norm_backward": 2,
        "native_dropout_backward": 1,
        "native_layer_norm_backward": 2,
        "nextafter": 2,
        "special_zeta": 2,
        "unique_consecutive": 1,
        "unique_dim": 1,
        "unique_dim_consecutive": 1,
    }
    assert len(entries) == 42


def test_autograd_functional_has_no_remaining_stubs():
    text = _read(_SRC / "autograd" / "functional.py")
    stubs = set(
        re.findall(
            r"def ([A-Za-z_]+)\([^\n]*\):[^\n]*\n\s+raise NotImplementedError",
            text,
        )
    )
    assert stubs == set()


def test_manual_review_schema_ops_are_currently_unregistered():
    schema_ops = _schema_ops()
    autograd_ops = _autograd_registered_ops()
    assert sorted(MANUAL_REVIEW_SCHEMA_OPS - schema_ops) == []
    assert sorted(MANUAL_REVIEW_SCHEMA_OPS & autograd_ops) == []
