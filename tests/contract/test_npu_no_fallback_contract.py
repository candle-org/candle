from pathlib import Path

import pytest

from candle._backends.npu import aclnn
from candle._backends.npu.ops import _helpers


class _FakeRuntime:
    def __init__(self):
        self.stream = 123
        self.freed = []

    def defer_raw_free(self, ptr):
        self.freed.append(("raw", ptr))

    def defer_free(self, ptr):
        self.freed.append(("device", ptr))

    def synchronize(self):
        pass


class _FakeAcl:
    class rt:
        @staticmethod
        def malloc(size, flag):
            return (0xCAFE, 0)


def _install_native_ffi_env(monkeypatch):
    calls = []

    class _FakeFfi:
        def is_initialized(self):
            return True

        def resolve_op(self, op_name):
            calls.append(("resolve_op", op_name))
            return (f"getws:{op_name}", f"exec:{op_name}")

        def binary_op_no_alpha(self, getws_ptr, exec_ptr, *args):
            calls.append(("binary_op_no_alpha", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_scalar_op_with_alpha(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_scalar_op_with_alpha", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_scalar_op_no_alpha(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_scalar_op_no_alpha", getws_ptr, exec_ptr))
            return (0, 0)

        def unary_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("unary_op", getws_ptr, exec_ptr))
            return (0, 0)

        def unary_out_dtype_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("unary_out_dtype_op", getws_ptr, exec_ptr))
            return (0, 0)

        def reduce_sum_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("reduce_sum_op", getws_ptr, exec_ptr))
            return (0, 0)

        def reduce_dims_dtype_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("reduce_dims_dtype_op", getws_ptr, exec_ptr))
            return (0, 0)

        def arg_reduce_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("arg_reduce_op", getws_ptr, exec_ptr))
            return (0, 0)

        def cast_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("cast_op", getws_ptr, exec_ptr))
            return (0, 0)

        def argsort_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("argsort_op", getws_ptr, exec_ptr))
            return (0, 0)

        def dual_output_with_indices_op(self, variant, getws_ptr, exec_ptr, *args):
            calls.append(("dual_output_with_indices_op", variant, getws_ptr, exec_ptr))
            return (0, 0)

        def axis_unary_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("axis_unary_op", getws_ptr, exec_ptr))
            return (0, 0)

        def axis_dtype_unary_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("axis_dtype_unary_op", getws_ptr, exec_ptr))
            return (0, 0)

        def reduce_all_dtype_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("reduce_all_dtype_op", getws_ptr, exec_ptr))
            return (0, 0)

        def axis_keepdim_dtype_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("axis_keepdim_dtype_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_scalar_dtype_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_scalar_dtype_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_scalar_bool_out_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_scalar_bool_out_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_two_ints_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_two_ints_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_three_ints_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_three_ints_op", getws_ptr, exec_ptr))
            return (0, 0)

        def ternary_two_inputs_with_dims_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("ternary_two_inputs_with_dims_op", getws_ptr, exec_ptr))
            return (0, 0)

        def execute(self, exec_ptr, workspace_ptr, workspace_size, executor, stream):
            calls.append(("execute", exec_ptr, workspace_ptr, workspace_size, executor, stream))
            return 0

        def binary_two_inputs_three_attrs_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("binary_two_inputs_three_attrs_op", getws_ptr, exec_ptr))
            return (0, 0)

        def layer_norm_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("layer_norm_op", getws_ptr, exec_ptr))
            return (0, 0)

        def leaky_relu_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("leaky_relu_op", getws_ptr, exec_ptr))
            return (0, 0)

        def rms_norm_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("rms_norm_op", getws_ptr, exec_ptr))
            return (0, 0)

        def binary_two_inputs_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("binary_two_inputs_op", getws_ptr, exec_ptr))
            return (0, 0)

        def binary_two_inputs_with_int8_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("binary_two_inputs_with_int8_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_two_scalars_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("two_tensor_two_scalars_op", getws_ptr, exec_ptr))
            return (0, 0)

        def clamp_optional_scalars_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("clamp_optional_scalars_op", getws_ptr, exec_ptr))
            return (0, 0)

        def clamp_tensor_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("clamp_tensor_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_three_scalars_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_three_scalars_op", getws_ptr, exec_ptr))
            return (0, 0)

        def where_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("where_op", getws_ptr, exec_ptr))
            return (0, 0)

        def slice_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("slice_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_list_axis_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_list_axis_op", getws_ptr, exec_ptr))
            return (0, 0)

        def binary_two_inputs_with_dim_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("binary_two_inputs_with_dim_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_int_array_scalar_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_int_array_scalar_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_int_array_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_int_array_op", getws_ptr, exec_ptr))
            return (0, 0)

        def optional_tensor_int_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("optional_tensor_int_op", getws_ptr, exec_ptr))
            return (0, 0)

        def two_tensor_ints_bool_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("two_tensor_ints_bool_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_int_array_three_ints_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_int_array_three_ints_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_int_array_bool_two_doubles_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_int_array_bool_two_doubles_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_int_array_bool_double_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_int_array_bool_double_op", getws_ptr, exec_ptr))
            return (0, 0)

        def output_tensor_int_array_double_two_ints_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("output_tensor_int_array_double_two_ints_op", getws_ptr, exec_ptr))
            return (0, 0)

        def two_tensor_one_double_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("two_tensor_one_double_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_int_array_bool_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_int_array_bool_op", getws_ptr, exec_ptr))
            return (0, 0)

        def two_tensor_two_ints_bool_mixed_fmt_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("two_tensor_two_ints_bool_mixed_fmt_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_four_int_arrays_two_ints_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_four_int_arrays_two_ints_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_three_int_arrays_two_bools_int64_int8_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_three_int_arrays_two_bools_int64_int8_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_four_int_arrays_bool_two_outputs_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_four_int_arrays_bool_two_outputs_op", getws_ptr, exec_ptr))
            return (0, 0)

        def four_tensor_three_int_arrays_two_bools_int64_int8_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("four_tensor_three_int_arrays_two_bools_int64_int8_op", getws_ptr, exec_ptr))
            return (0, 0)

        def four_tensor_four_int_arrays_bool_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("four_tensor_four_int_arrays_bool_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_int_array_two_outputs_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_int_array_two_outputs_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_list_string_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_list_string_op", getws_ptr, exec_ptr))
            return (0, 0)

        def six_tensor_string_double_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("six_tensor_string_double_op", getws_ptr, exec_ptr))
            return (0, 0)

        def six_tensor_five_floats_two_bools_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("six_tensor_five_floats_two_bools_op", getws_ptr, exec_ptr))
            return (0, 0)

        def batch_norm_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("batch_norm_op", getws_ptr, exec_ptr))
            return (0, 0)

        def layer_norm_backward_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("layer_norm_backward_op", getws_ptr, exec_ptr))
            return (0, 0)

        def batch_norm_backward_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("batch_norm_backward_op", getws_ptr, exec_ptr))
            return (0, 0)

        def group_norm_backward_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("group_norm_backward_op", getws_ptr, exec_ptr))
            return (0, 0)

        def convolution_backward_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("convolution_backward_op", getws_ptr, exec_ptr))
            return (0, 0)

        def rms_norm_grad_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("rms_norm_grad_op", getws_ptr, exec_ptr))
            return (0, 0)

        def grid_sampler2d_backward_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("grid_sampler2d_backward_op", getws_ptr, exec_ptr))
            return (0, 0)

        def group_norm_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("group_norm_op", getws_ptr, exec_ptr))
            return (0, 0)

        def convolution_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("convolution_op", getws_ptr, exec_ptr))
            return (0, 0)


        def tensor_two_int_arrays_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_two_int_arrays_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_two_int_arrays_bool_two_doubles_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_two_int_arrays_bool_two_doubles_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_two_int_arrays_bool_double_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_two_int_arrays_bool_double_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_two_scalars_dim_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_two_scalars_dim_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_int_array_two_bools_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_int_array_two_bools_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_scalar_int_array_bool_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_scalar_int_array_bool_op", getws_ptr, exec_ptr))
            return (0, 0)

        def four_tensor_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("four_tensor_op", getws_ptr, exec_ptr))
            return (0, 0)

        def three_tensor_scalar_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("three_tensor_scalar_op", getws_ptr, exec_ptr))
            return (0, 0)

        def two_tensor_scalar_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("two_tensor_scalar_op", getws_ptr, exec_ptr))
            return (0, 0)

        def two_tensor_two_scalars_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("two_tensor_two_scalars_op", getws_ptr, exec_ptr))
            return (0, 0)

        def two_tensor_scalar_bool_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("two_tensor_scalar_bool_op", getws_ptr, exec_ptr))
            return (0, 0)

        def two_tensor_three_scalars_bool_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("two_tensor_three_scalars_bool_op", getws_ptr, exec_ptr))
            return (0, 0)

        def three_tensor_two_outputs_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("three_tensor_two_outputs_op", getws_ptr, exec_ptr))
            return (0, 0)

        def two_tensor_two_bools_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("two_tensor_scalar_bool_op", getws_ptr, exec_ptr))
            return (0, 0)

        def unary_two_bools_two_outputs_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("unary_two_bools_two_outputs_op", getws_ptr, exec_ptr))
            return (0, 0)

        def output_tensor_three_ints_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("output_tensor_three_ints_op", getws_ptr, exec_ptr))
            return (0, 0)

        def output_tensor_two_ints_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("output_tensor_two_ints_op", getws_ptr, exec_ptr))
            return (0, 0)

        def three_tensor_one_int_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("three_tensor_one_int_op", getws_ptr, exec_ptr))
            return (0, 0)

        def four_tensor_two_ints_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("four_tensor_two_ints_op", getws_ptr, exec_ptr))
            return (0, 0)

        def three_tensor_two_ints_bool_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("three_tensor_two_ints_bool_op", getws_ptr, exec_ptr))
            return (0, 0)

        def four_tensor_two_scalars_one_int8_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("four_tensor_two_scalars_one_int8_op", getws_ptr, exec_ptr))
            return (0, 0)

        def tensor_int_array_bool_two_outputs_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("tensor_int_array_bool_two_outputs_op", getws_ptr, exec_ptr))
            return (0, 0)

        def inplace_unary_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("inplace_unary_op", getws_ptr, exec_ptr))
            return (0, 0)

        def inplace_normal_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("inplace_normal_op", getws_ptr, exec_ptr))
            return (0, 0)

        def inplace_uniform_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("inplace_uniform_op", getws_ptr, exec_ptr))
            return (0, 0)

        def inplace_fill_scalar_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("inplace_fill_scalar_op", getws_ptr, exec_ptr))
            return (0, 0)

        def inplace_copy_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("inplace_copy_op", getws_ptr, exec_ptr))
            return (0, 0)

        def inplace_masked_fill_scalar_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("inplace_masked_fill_scalar_op", getws_ptr, exec_ptr))
            return (0, 0)

        def inplace_index_fill_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("inplace_index_fill_op", getws_ptr, exec_ptr))
            return (0, 0)

        def inplace_index_copy_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("inplace_index_copy_op", getws_ptr, exec_ptr))
            return (0, 0)

        def index_add_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("index_add_op", getws_ptr, exec_ptr))
            return (0, 0)

        def scatter_add_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("scatter_add_op", getws_ptr, exec_ptr))
            return (0, 0)

        def inplace_masked_scatter_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("inplace_masked_scatter_op", getws_ptr, exec_ptr))
            return (0, 0)

        def index_with_optional_tensor_list_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("index_with_optional_tensor_list_op", getws_ptr, exec_ptr))
            return (0, 0)

        def output_tensor_three_scalars_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("output_tensor_three_scalars_op", getws_ptr, exec_ptr))
            return (0, 0)

        def index_put_impl_op(self, getws_ptr, exec_ptr, *args):
            calls.append(("index_put_impl_op", getws_ptr, exec_ptr))
            return (0, 0)

        def create_scalar(self, scalar_bytes, dtype_code):
            handle = 1000 + len([entry for entry in calls if entry[0] == "create_scalar"])
            calls.append(("create_scalar", len(scalar_bytes), dtype_code, handle))
            return handle

        def destroy_scalar(self, handle):
            calls.append(("destroy_scalar", handle))

    monkeypatch.setattr(aclnn, "_ffi", _FakeFfi())
    monkeypatch.setattr(aclnn, "ensure_acl", lambda: _FakeAcl())
    monkeypatch.setattr(aclnn, "_npu_runtime_alloc_device", lambda size, runtime=None: 0xD000 + int(size))
    monkeypatch.setattr(aclnn, "_create_tensor_list", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy _create_tensor_list should not run")))
    monkeypatch.setattr(aclnn, "acl", None)
    monkeypatch.setattr(
        aclnn,
        "_create_tensor",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy _create_tensor should not run")
        ),
    )
    monkeypatch.setattr(
        aclnn,
        "_create_scalar",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy _create_scalar should not run")
        ),
    )
    return calls


def test_npu_binary_helper_requires_native_fast_path_when_missing(monkeypatch):
    monkeypatch.setattr(_helpers, "_HAS_FAST_OPS", False)

    with pytest.raises(RuntimeError, match="native NPU hot path unavailable for op add"):
        _helpers._require_native_fast_ops("add")


def test_npu_aclnn_requires_native_ffi_when_missing(monkeypatch):
    monkeypatch.setattr(aclnn, "_ffi", None)

    with pytest.raises(RuntimeError, match="native NPU hot path unavailable for op maximum"):
        aclnn._require_native_npu_ffi("maximum")


def test_npu_aclnn_legacy_ctypes_helpers_are_disabled():
    with pytest.raises(RuntimeError, match="python/ctypes fallback is disabled"):
        aclnn._create_tensor(None, (), (), "float16", 0)

    with pytest.raises(RuntimeError, match="python/ctypes fallback is disabled"):
        aclnn._create_scalar(None, 1, "float16")


def test_npu_forward_wrapper_errors_with_op_name_when_native_ffi_missing(monkeypatch):
    class _FakeBindings:
        aclnn_batch_norm_get_workspace = object()
        aclnn_batch_norm = object()

    monkeypatch.setattr(aclnn, "get_bindings", lambda: _FakeBindings())
    monkeypatch.setattr(aclnn, "_ffi", None)

    with pytest.raises(RuntimeError, match="native NPU hot path unavailable for op batch_norm"):
        aclnn.batch_norm(
            0, 0, 0, 0, 0, 0,
            (), (), (), (), (), (), (), (), (), (), (), (),
            False, 0.1, 1e-5, "float16", runtime=_FakeRuntime(),
        )


@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "binding_attrs"),
    [
        ("maximum", "Maximum", ("aclnn_maximum_get_workspace", "aclnn_maximum")),
        ("minimum", "Minimum", ("aclnn_minimum_get_workspace", "aclnn_minimum")),
        ("atan2", "Atan2", ("aclnn_atan2_get_workspace", "aclnn_atan2")),
    ],
)
def test_first_batch_binary_wrappers_use_native_ffi(monkeypatch, wrapper_name, ffi_name, binding_attrs):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(
        1,
        2,
        3,
        (2, 2),
        (2, 1),
        (2, 2),
        (2, 1),
        (2, 2),
        (2, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", ffi_name) in calls
    assert ("binary_op_no_alpha", f"getws:{ffi_name}", f"exec:{ffi_name}") in calls


@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "expected_helper", "binding_attrs"),
    [
        (
            "add_scalar",
            "Adds",
            "tensor_scalar_op_with_alpha",
            ("aclnn_add_scalar_get_workspace", "aclnn_add_scalar"),
        ),
        (
            "sub_scalar",
            "Subs",
            "tensor_scalar_op_no_alpha",
            ("aclnn_sub_scalar_get_workspace", "aclnn_sub_scalar"),
        ),
    ],
)
def test_first_batch_scalar_wrappers_use_native_ffi(
    monkeypatch, wrapper_name, ffi_name, expected_helper, binding_attrs
):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(1, 2, 3, (2, 2), (2, 1), "float16", runtime)

    assert ("resolve_op", ffi_name) in calls
    assert (expected_helper, f"getws:{ffi_name}", f"exec:{ffi_name}") in calls


@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "binding_attrs"),
    [
        ("relu", "Relu", ("aclnn_relu_get_workspace", "aclnn_relu")),
    ],
)
def test_second_batch_unary_wrappers_use_native_ffi(monkeypatch, wrapper_name, ffi_name, binding_attrs):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(1, 2, (2, 2), (2, 1), "float16", runtime)

    assert ("resolve_op", ffi_name) in calls
    assert ("unary_op", f"getws:{ffi_name}", f"exec:{ffi_name}") in calls


@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "binding_attrs", "call_args"),
    [
        (
            "argmax",
            "ArgMax",
            ("aclnn_argmax_get_workspace", "aclnn_argmax"),
            (1, 2, (2, 2), (2, 1), "float16", 1, False, (2,), (1,)),
        ),
        (
            "argmin",
            "ArgMin",
            ("aclnn_argmin_get_workspace", "aclnn_argmin"),
            (1, 2, (2, 2), (2, 1), "float16", 1, False, (2,), (1,)),
        ),
    ],
)
def test_second_batch_arg_reduce_wrappers_use_native_ffi(
    monkeypatch, wrapper_name, ffi_name, binding_attrs, call_args
):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(*call_args, runtime)

    assert ("resolve_op", ffi_name) in calls
    assert ("arg_reduce_op", f"getws:{ffi_name}", f"exec:{ffi_name}") in calls


def test_second_batch_reduce_sum_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_reduce_sum_get_workspace": object(),
            "aclnn_reduce_sum": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    dims = {"dims": (1,), "out_shape": (2,), "out_stride": (1,)}
    aclnn.reduce_sum(1, 2, (2, 2), (2, 1), "float16", dims, False, runtime)

    assert ("resolve_op", "ReduceSum") in calls
    assert ("reduce_sum_op", "getws:ReduceSum", "exec:ReduceSum") in calls



def test_third_batch_cast_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_cast_get_workspace": object(),
            "aclnn_cast": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.cast(1, 2, (2, 2), (2, 1), "float16", "float32", runtime)

    assert ("resolve_op", "Cast") in calls
    assert ("cast_op", "getws:Cast", "exec:Cast") in calls



def test_third_batch_argsort_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_argsort_get_workspace": object(),
            "aclnn_argsort": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.argsort(1, 2, (2, 2), (2, 1), 1, False, "float16", runtime)

    assert ("resolve_op", "Argsort") in calls
    assert ("argsort_op", "getws:Argsort", "exec:Argsort") in calls


@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "binding_attrs", "call_args"),
    [
        (
            "max_dim",
            "MaxDim",
            ("aclnn_max_dim_get_workspace", "aclnn_max_dim"),
            (1, 2, 3, (2, 2), (2, 1), "float16", 1, False, (2,), (1,), (1,)),
        ),
        (
            "min_dim",
            "MinDim",
            ("aclnn_min_dim_get_workspace", "aclnn_min_dim"),
            (1, 2, 3, (2, 2), (2, 1), "float16", 1, False, (2,), (1,), (1,)),
        ),
    ],
)
def test_third_batch_dim_wrappers_use_native_ffi(
    monkeypatch, wrapper_name, ffi_name, binding_attrs, call_args
):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(*call_args, runtime)

    assert ("resolve_op", ffi_name) in calls
    assert (
        "dual_output_with_indices_op",
        "dim_reduce",
        f"getws:{ffi_name}",
        f"exec:{ffi_name}",
    ) in calls


@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "call_args", "variant"),
    [
        (
            "sort",
            "Sort",
            (1, 2, 3, (2, 2), (2, 1), 1, False, False, "float16"),
            "sort",
        ),
        (
            "topk",
            "Topk",
            (1, 2, 3, (2, 2), (2, 1), 1, 1, True, True, "float16"),
            "topk",
        ),
    ],
)
def test_third_batch_sort_family_uses_native_ffi(
    monkeypatch, wrapper_name, ffi_name, call_args, variant
):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            f"aclnn_{wrapper_name}_get_workspace": object(),
            f"aclnn_{wrapper_name}": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(*call_args, runtime)

    assert ("resolve_op", ffi_name) in calls
    assert (
        "dual_output_with_indices_op",
        variant,
        f"getws:{ffi_name}",
        f"exec:{ffi_name}",
    ) in calls


@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "binding_attrs", "call_args", "expected_helper"),
    [
        (
            "softmax",
            "Softmax",
            ("aclnn_softmax_get_workspace", "aclnn_softmax"),
            (1, 2, (2, 2), (2, 1), "float16", 1),
            "axis_unary_op",
        ),
        (
            "log_softmax",
            "LogSoftmax",
            ("aclnn_log_softmax_get_workspace", "aclnn_log_softmax"),
            (1, 2, (2, 2), (2, 1), "float16", 1),
            "axis_unary_op",
        ),
        (
            "gelu",
            "Gelu",
            ("aclnn_gelu_get_workspace", "aclnn_gelu"),
            (1, 2, (2, 2), (2, 1), "float16"),
            "unary_op",
        ),
        (
            "silu",
            "Silu",
            ("aclnn_silu_get_workspace", "aclnn_silu"),
            (1, 2, (2, 2), (2, 1), "float16"),
            "unary_op",
        ),
    ],
)
def test_fourth_batch_activation_wrappers_use_native_ffi(
    monkeypatch, wrapper_name, ffi_name, binding_attrs, call_args, expected_helper
):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(*call_args, runtime)

    assert ("resolve_op", ffi_name) in calls
    assert (expected_helper, f"getws:{ffi_name}", f"exec:{ffi_name}") in calls


def test_fourth_batch_layer_norm_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_layer_norm_get_workspace": object(),
            "aclnn_layer_norm": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.layer_norm(
        1,
        2,
        3,
        4,
        5,
        6,
        (2, 4),
        (4, 1),
        (4,),
        (1,),
        (4,),
        (1,),
        (2, 4),
        (4, 1),
        (2,),
        (1,),
        (4,),
        1e-5,
        "float16",
        runtime,
    )

    assert ("resolve_op", "LayerNorm") in calls
    assert ("layer_norm_op", "getws:LayerNorm", "exec:LayerNorm") in calls



def test_next_batch_mean_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_mean_get_workspace": object(),
            "aclnn_mean": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.mean(1, 2, (2, 2), (2, 1), "float16", (1,), False, (2,), (1,), runtime)

    assert ("resolve_op", "Mean") in calls
    assert ("reduce_sum_op", "getws:Mean", "exec:Mean") in calls



def test_next_batch_leaky_relu_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_leaky_relu_get_workspace": object(),
            "aclnn_leaky_relu": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.leaky_relu(1, 2, (2, 2), (2, 1), "float16", 0.1, runtime)

    assert ("resolve_op", "LeakyRelu") in calls
    assert ("leaky_relu_op", "getws:LeakyRelu", "exec:LeakyRelu") in calls



def test_next_batch_rms_norm_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_rms_norm_get_workspace": object(),
            "aclnn_rms_norm": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.rms_norm(
        1,
        2,
        1e-5,
        3,
        4,
        (2, 4),
        (4, 1),
        (4,),
        (1,),
        (2, 4),
        (4, 1),
        (2,),
        (1,),
        "float16",
        runtime,
    )

    assert ("resolve_op", "RmsNorm") in calls
    assert ("rms_norm_op", "getws:RmsNorm", "exec:RmsNorm") in calls


def test_next_batch_mish_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_mish_get_workspace": object(),
            "aclnn_mish": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.mish(1, 2, (2, 2), (2, 1), "float16", runtime)

    assert ("resolve_op", "Mish") in calls
    assert ("unary_op", "getws:Mish", "exec:Mish") in calls



def test_next_batch_prelu_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_prelu_get_workspace": object(),
            "aclnn_prelu": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.prelu(1, 2, 3, (2, 4), (4, 1), (4,), (1,), "float16", runtime)

    assert ("resolve_op", "Prelu") in calls
    assert ("binary_two_inputs_op", "getws:Prelu", "exec:Prelu") in calls



def test_binary_family_logical_xor_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_logical_xor_get_workspace": object(),
            "aclnn_logical_xor": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.logical_xor(1, 2, 3, (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), "float16", runtime)

    assert ("resolve_op", "LogicalXor") in calls
    assert ("binary_two_inputs_op", "getws:LogicalXor", "exec:LogicalXor") in calls



def test_binary_family_bitwise_and_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_bitwise_and_tensor_get_workspace": object(),
            "aclnn_bitwise_and_tensor": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.bitwise_and(1, 2, 3, (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), "int32", runtime)

    assert ("resolve_op", "BitwiseAndTensor") in calls
    assert ("binary_two_inputs_op", "getws:BitwiseAndTensor", "exec:BitwiseAndTensor") in calls



def test_binary_family_eq_tensor_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_eq_tensor_get_workspace": object(),
            "aclnn_eq_tensor": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.eq_tensor(1, 2, 3, (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), "float16", runtime)

    assert ("resolve_op", "EqTensor") in calls
    assert ("binary_two_inputs_op", "getws:EqTensor", "exec:EqTensor") in calls



def test_binary_family_pow_tensor_tensor_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_pow_tensor_tensor_get_workspace": object(),
            "aclnn_pow_tensor_tensor": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.pow_tensor_tensor(1, 2, 3, (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), "float16", runtime)

    assert ("resolve_op", "PowTensorTensor") in calls
    assert ("binary_two_inputs_op", "getws:PowTensorTensor", "exec:PowTensorTensor") in calls



def test_scalar_family_eq_scalar_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_eq_scalar_get_workspace": object(),
            "aclnn_eq_scalar": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.eq_scalar(1, 2.0, 3, (2, 2), (2, 1), "float16", runtime)

    assert ("resolve_op", "EqScalar") in calls
    assert ("tensor_scalar_bool_out_op", "getws:EqScalar", "exec:EqScalar") in calls



def test_scalar_family_pow_tensor_scalar_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_pow_tensor_scalar_get_workspace": object(),
            "aclnn_pow_tensor_scalar": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.pow_tensor_scalar(1, 2.0, 3, (2, 2), (2, 1), "float16", runtime)

    assert ("resolve_op", "PowTensorScalar") in calls
    assert ("tensor_scalar_op_no_alpha", "getws:PowTensorScalar", "exec:PowTensorScalar") in calls



def test_scalar_family_softplus_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_softplus_get_workspace": object(),
            "aclnn_softplus": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.softplus(1, 2, (2, 2), (2, 1), "float16", 1.0, 20.0, runtime)

    assert ("resolve_op", "Softplus") in calls
    assert ("two_tensor_two_scalars_op", "getws:Softplus", "exec:Softplus") in calls



def test_scalar_family_hardtanh_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_hardtanh_get_workspace": object(),
            "aclnn_hardtanh": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.hardtanh(1, 2, (2, 2), (2, 1), "float16", -1.0, 1.0, runtime)

    assert ("resolve_op", "Hardtanh") in calls
    assert ("two_tensor_two_scalars_op", "getws:Hardtanh", "exec:Hardtanh") in calls



def test_scalar_family_clamp_min_scalar_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_clamp_min_get_workspace": object(),
            "aclnn_clamp_min": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.clamp_min_scalar(1, 2, (2, 2), (2, 1), "float16", -1.0, runtime)

    assert ("resolve_op", "ClampMin") in calls
    assert ("tensor_scalar_op_no_alpha", "getws:ClampMin", "exec:ClampMin") in calls



def test_scalar_family_clamp_max_scalar_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_clamp_max_get_workspace": object(),
            "aclnn_clamp_max": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.clamp_max_scalar(1, 2, (2, 2), (2, 1), "float16", 1.0, runtime)

    assert ("resolve_op", "ClampMax") in calls
    assert ("tensor_scalar_op_no_alpha", "getws:ClampMax", "exec:ClampMax") in calls



@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "helper_name", "binding_attrs", "call_args"),
    [
        (
            "clamp_scalar",
            "Clamp",
            "clamp_optional_scalars_op",
            ("aclnn_clamp_get_workspace", "aclnn_clamp"),
            (1, 2, (2, 2), (2, 1), "float16", -1.0, 1.0),
        ),
        (
            "clamp_min_tensor",
            "ClampMinTensor",
            "binary_two_inputs_op",
            ("aclnn_clamp_min_tensor_get_workspace", "aclnn_clamp_min_tensor"),
            (1, 2, 3, (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), "float16"),
        ),
        (
            "clamp_max_tensor",
            "ClampMaxTensor",
            "binary_two_inputs_op",
            ("aclnn_clamp_max_tensor_get_workspace", "aclnn_clamp_max_tensor"),
            (1, 2, 3, (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), "float16"),
        ),
    ],
)
def test_clamp_family_wrappers_use_native_ffi(
    monkeypatch, wrapper_name, ffi_name, helper_name, binding_attrs, call_args
):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(*call_args, runtime)

    assert ("resolve_op", ffi_name) in calls
    assert (helper_name, f"getws:{ffi_name}", f"exec:{ffi_name}") in calls



def test_clamp_tensor_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_clamp_tensor_get_workspace": object(),
            "aclnn_clamp_tensor": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.clamp_tensor(
        1,
        2,
        3,
        4,
        (2, 2),
        (2, 1),
        (2, 2),
        (2, 1),
        (2, 2),
        (2, 1),
        (2, 2),
        (2, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "ClampTensor") in calls
    assert ("clamp_tensor_op", "getws:ClampTensor", "exec:ClampTensor") in calls



def test_next_family_elu_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_elu_get_workspace": object(),
            "aclnn_elu": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.elu(1, 2, (2, 2), (2, 1), "float16", 1.0, runtime)

    assert ("resolve_op", "Elu") in calls
    assert ("tensor_three_scalars_op", "getws:Elu", "exec:Elu") in calls



def test_next_family_swhere_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_swhere_get_workspace": object(),
            "aclnn_swhere": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.swhere(
        1,
        2,
        3,
        4,
        (2, 2),
        (2, 1),
        (2, 2),
        (2, 1),
        (2, 2),
        (2, 1),
        (2, 2),
        (2, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "SWhere") in calls
    assert ("where_op", "getws:SWhere", "exec:SWhere") in calls



def test_next_family_s_where_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_s_where_get_workspace": object(),
            "aclnn_s_where": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.s_where(
        1,
        2,
        3,
        4,
        (2, 2),
        (2, 1),
        "bool",
        (2, 2),
        (2, 1),
        "float16",
        (2, 2),
        (2, 1),
        "float16",
        (2, 2),
        (2, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "SWhere") in calls
    assert ("where_op", "getws:SWhere", "exec:SWhere") in calls



@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "binding_attrs", "call_args"),
    [
        (
            "dot",
            "Dot",
            ("aclnn_dot_get_workspace", "aclnn_dot"),
            (1, 2, 3, (4,), (1,), (4,), (1,), (), (), "float16"),
        ),
        (
            "mv",
            "Mv",
            ("aclnn_mv_get_workspace", "aclnn_mv"),
            (1, 2, 3, (2, 4), (4, 1), (4,), (1,), (2,), (1,), "float16", 1),
        ),
        (
            "ger",
            "Ger",
            ("aclnn_ger_get_workspace", "aclnn_ger"),
            (1, 2, 3, (2,), (1,), (4,), (1,), (2, 4), (4, 1), "float16"),
        ),
    ],
)
def test_gemm_family_binary_wrappers_use_native_ffi(
    monkeypatch, wrapper_name, ffi_name, binding_attrs, call_args
):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(*call_args, runtime)

    assert ("resolve_op", ffi_name) in calls
    expected_helper = "binary_two_inputs_with_int8_op" if wrapper_name == "mv" else "binary_two_inputs_op"
    assert (expected_helper, f"getws:{ffi_name}", f"exec:{ffi_name}") in calls



def test_gemm_family_matmul_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_matmul_get_workspace": object(),
            "aclnn_matmul": object(),
            "aclnn_batch_matmul_get_workspace": object(),
            "aclnn_batch_matmul": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.matmul(1, 2, 3, (2, 4), (4, 1), (4, 3), (3, 1), (2, 3), (3, 1), "float16", runtime)

    assert ("resolve_op", "Matmul") in calls
    assert ("binary_two_inputs_with_int8_op", "getws:Matmul", "exec:Matmul") in calls



def test_gemm_family_batch_matmul_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_matmul_get_workspace": object(),
            "aclnn_matmul": object(),
            "aclnn_batch_matmul_get_workspace": object(),
            "aclnn_batch_matmul": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.matmul(
        1, 2, 3,
        (2, 3, 4), (12, 4, 1),
        (2, 4, 5), (20, 5, 1),
        (2, 3, 5), (15, 5, 1),
        "float16", runtime,
    )

    assert ("resolve_op", "BatchMatMul") in calls
    assert ("binary_two_inputs_with_int8_op", "getws:BatchMatMul", "exec:BatchMatMul") in calls



def test_indexing_family_slice_op_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_slice_get_workspace": object(),
            "aclnn_slice": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.slice_op(
        1,
        (4, 4),
        (4, 1),
        "float16",
        1,
        0,
        4,
        2,
        2,
        (4, 2),
        (2, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "Slice") in calls
    assert ("slice_op", "getws:Slice", "exec:Slice") in calls



def test_indexing_family_cat_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_cat_get_workspace": object(),
            "aclnn_cat": object(),
            "acl_create_tensor_list": object(),
            "acl_destroy_tensor_list": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.cat(
        (1, 2),
        ((2, 3), (2, 3)),
        ((3, 1), (3, 1)),
        ("float16", "float16"),
        0,
        3,
        (4, 3),
        (3, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "Cat") in calls
    assert ("tensor_list_axis_op", "getws:Cat", "exec:Cat") in calls



def test_indexing_family_stack_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_stack_get_workspace": object(),
            "aclnn_stack": object(),
            "acl_create_tensor_list": object(),
            "acl_destroy_tensor_list": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.stack(
        (1, 2),
        ((2, 3), (2, 3)),
        ((3, 1), (3, 1)),
        ("float16", "float16"),
        0,
        3,
        (2, 2, 3),
        (6, 3, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "Stack") in calls
    assert ("tensor_list_axis_op", "getws:Stack", "exec:Stack") in calls



def test_select_family_gather_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_gather_get_workspace": object(),
            "aclnn_gather": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.gather(
        1, 2, 3,
        (4, 4), (4, 1), "float16",
        (4, 2), (2, 1), "int64",
        (4, 2), (2, 1), "float16",
        1, runtime,
    )

    assert ("resolve_op", "Gather") in calls
    assert ("binary_two_inputs_with_dim_op", "getws:Gather", "exec:Gather") in calls



def test_select_family_masked_select_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_masked_select_get_workspace": object(),
            "aclnn_masked_select": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.masked_select(
        1, 2, 3,
        (4, 4), (4, 1), "float16",
        (4, 4), (4, 1), "bool",
        (8,), (1,), "float16",
        runtime,
    )

    assert ("resolve_op", "MaskedSelect") in calls
    assert ("binary_two_inputs_op", "getws:MaskedSelect", "exec:MaskedSelect") in calls



def test_constant_pad_nd_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_constant_pad_nd_get_workspace": object(),
            "aclnn_constant_pad_nd": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.constant_pad_nd(
        1,
        2,
        (2, 3),
        (3, 1),
        "float16",
        (1, 1, 2, 0),
        0.5,
        (4, 5),
        (5, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "ConstantPadNd") in calls
    assert (
        "tensor_int_array_scalar_op",
        "getws:ConstantPadNd",
        "exec:ConstantPadNd",
    ) in calls



def test_flatten_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_flatten_get_workspace": object(),
            "aclnn_flatten": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.flatten(
        1,
        2,
        (2, 3, 4),
        (12, 4, 1),
        "float16",
        1,
        (2, 12),
        (12, 1),
        runtime,
    )

    assert ("resolve_op", "Flatten") in calls
    assert ("axis_unary_op", "getws:Flatten", "exec:Flatten") in calls


def test_flip_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_flip_get_workspace": object(),
            "aclnn_flip": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.flip(
        1,
        2,
        (2, 3, 4),
        (12, 4, 1),
        "float16",
        (0, 2),
        runtime,
    )

    assert ("resolve_op", "Flip") in calls
    assert ("tensor_int_array_op", "getws:Flip", "exec:Flip") in calls



def test_roll_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_roll_get_workspace": object(),
            "aclnn_roll": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.roll(
        1,
        2,
        (2, 3, 4),
        (12, 4, 1),
        "float16",
        (1, -2),
        (0, 2),
        runtime,
    )

    assert ("resolve_op", "Roll") in calls
    assert ("tensor_two_int_arrays_op", "getws:Roll", "exec:Roll") in calls



def test_cumsum_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_cumsum_get_workspace": object(),
            "aclnn_cumsum": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.cumsum(
        1,
        2,
        (2, 3, 4),
        (12, 4, 1),
        "float16",
        1,
        "float16",
        runtime,
    )

    assert ("resolve_op", "Cumsum") in calls
    assert ("axis_dtype_unary_op", "getws:Cumsum", "exec:Cumsum") in calls



def test_cumprod_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_cumprod_get_workspace": object(),
            "aclnn_cumprod": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.cumprod(
        1,
        2,
        (2, 3, 4),
        (12, 4, 1),
        "float16",
        1,
        "float16",
        runtime,
    )

    assert ("resolve_op", "Cumprod") in calls
    assert ("tensor_scalar_dtype_op", "getws:Cumprod", "exec:Cumprod") in calls



def test_cummax_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_cummax_get_workspace": object(),
            "aclnn_cummax": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.cummax(
        1,
        2,
        3,
        (2, 3, 4),
        (12, 4, 1),
        "float16",
        1,
        runtime,
    )

    assert ("resolve_op", "Cummax") in calls
    assert (
        "dual_output_with_indices_op",
        "cummax",
        "getws:Cummax",
        "exec:Cummax",
    ) in calls



def test_tril_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_tril_get_workspace": object(),
            "aclnn_tril": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.tril(
        1,
        2,
        (4, 4),
        (4, 1),
        -1,
        "float16",
        runtime,
    )

    assert ("resolve_op", "Tril") in calls
    assert ("axis_unary_op", "getws:Tril", "exec:Tril") in calls



def test_triu_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_triu_get_workspace": object(),
            "aclnn_triu": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.triu(
        1,
        2,
        (4, 4),
        (4, 1),
        1,
        "float16",
        runtime,
    )

    assert ("resolve_op", "Triu") in calls
    assert ("axis_unary_op", "getws:Triu", "exec:Triu") in calls



@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "binding_attrs"),
    [
        ("lt_tensor", "LtTensor", ("aclnn_lt_tensor_get_workspace", "aclnn_lt_tensor")),
        ("le_tensor", "LeTensor", ("aclnn_le_tensor_get_workspace", "aclnn_le_tensor")),
        ("gt_tensor", "GtTensor", ("aclnn_gt_tensor_get_workspace", "aclnn_gt_tensor")),
        ("ge_tensor", "GeTensor", ("aclnn_ge_tensor_get_workspace", "aclnn_ge_tensor")),
    ],
)
def test_compare_family_tensor_relations_use_native_ffi(monkeypatch, wrapper_name, ffi_name, binding_attrs):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(
        1,
        2,
        3,
        (2, 2),
        (2, 1),
        (2, 2),
        (2, 1),
        (2, 2),
        (2, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", ffi_name) in calls
    assert ("binary_two_inputs_op", f"getws:{ffi_name}", f"exec:{ffi_name}") in calls



def test_nonzero_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_nonzero_get_workspace": object(),
            "aclnn_nonzero": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.nonzero(
        1,
        2,
        (2, 3),
        (3, 1),
        "float16",
        (4, 2),
        (2, 1),
        runtime,
    )

    assert ("resolve_op", "Nonzero") in calls
    assert ("unary_out_dtype_op", "getws:Nonzero", "exec:Nonzero") in calls



def test_repeat_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_repeat_get_workspace": object(),
            "aclnn_repeat": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.repeat(
        1,
        2,
        (2, 3),
        (3, 1),
        "float16",
        (2, 1),
        (4, 3),
        (3, 1),
        runtime,
    )

    assert ("resolve_op", "Repeat") in calls
    assert ("tensor_int_array_op", "getws:Repeat", "exec:Repeat") in calls



def test_repeat_interleave_int_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_repeat_interleave_int_get_workspace": object(),
            "aclnn_repeat_interleave_int": object(),
            "aclnn_repeat_interleave_int_with_dim_get_workspace": object(),
            "aclnn_repeat_interleave_int_with_dim": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.repeat_interleave_int(
        1,
        2,
        (2, 3),
        (3, 1),
        "float16",
        2,
        None,
        12,
        (12,),
        (1,),
        runtime,
    )

    assert ("resolve_op", "RepeatInterleaveInt") in calls
    assert ("tensor_two_ints_op", "getws:RepeatInterleaveInt", "exec:RepeatInterleaveInt") in calls



def test_repeat_interleave_int_with_dim_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_repeat_interleave_int_get_workspace": object(),
            "aclnn_repeat_interleave_int": object(),
            "aclnn_repeat_interleave_int_with_dim_get_workspace": object(),
            "aclnn_repeat_interleave_int_with_dim": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.repeat_interleave_int(
        1,
        2,
        (2, 3),
        (3, 1),
        "float16",
        2,
        1,
        6,
        (2, 6),
        (6, 1),
        runtime,
    )

    assert ("resolve_op", "RepeatInterleaveIntWithDim") in calls
    assert (
        "tensor_three_ints_op",
        "getws:RepeatInterleaveIntWithDim",
        "exec:RepeatInterleaveIntWithDim",
    ) in calls



def test_scatter_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_scatter_get_workspace": object(),
            "aclnn_scatter": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.scatter(
        1,
        2,
        3,
        4,
        (2, 3),
        (3, 1),
        "float16",
        (2, 3),
        (3, 1),
        "int64",
        (2, 3),
        (3, 1),
        "float16",
        1,
        0,
        runtime,
    )

    assert ("resolve_op", "Scatter") in calls
    assert (
        "ternary_two_inputs_with_dims_op",
        "getws:Scatter",
        "exec:Scatter",
    ) in calls



def test_diag_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_diag_get_workspace": object(),
            "aclnn_diag": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.diag(
        1,
        2,
        (4, 4),
        (4, 1),
        "float16",
        -1,
        (3,),
        (1,),
        runtime,
    )

    assert ("resolve_op", "Diag") in calls
    assert ("axis_unary_op", "getws:Diag", "exec:Diag") in calls



def test_cummin_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_cummin_get_workspace": object(),
            "aclnn_cummin": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.cummin(
        1,
        2,
        3,
        (2, 3, 4),
        (12, 4, 1),
        "float16",
        1,
        (2, 3, 4),
        (12, 4, 1),
        runtime,
    )

    assert ("resolve_op", "Cummin") in calls
    assert (
        "dual_output_with_indices_op",
        "cummin",
        "getws:Cummin",
        "exec:Cummin",
    ) in calls



def test_logsumexp_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_logsumexp_get_workspace": object(),
            "aclnn_logsumexp": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.logsumexp(
        1,
        2,
        (2, 3, 4),
        (12, 4, 1),
        "float16",
        (1, 2),
        True,
        (2, 1, 1),
        (1, 1, 1),
        runtime,
    )

    assert ("resolve_op", "LogSumExp") in calls
    assert ("reduce_sum_op", "getws:LogSumExp", "exec:LogSumExp") in calls



def test_reduce_nansum_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_reduce_nansum_get_workspace": object(),
            "aclnn_reduce_nansum": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.reduce_nansum(
        1,
        2,
        (2, 3, 4),
        (12, 4, 1),
        "float16",
        (1, 2),
        True,
        (2, 1, 1),
        (1, 1, 1),
        runtime,
    )

    assert ("resolve_op", "ReduceNansum") in calls
    assert ("reduce_dims_dtype_op", "getws:ReduceNansum", "exec:ReduceNansum") in calls



def test_prod_all_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_prod_get_workspace": object(),
            "aclnn_prod": object(),
            "aclnn_prod_dim_get_workspace": object(),
            "aclnn_prod_dim": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.prod(
        1,
        2,
        (2, 3, 4),
        (12, 4, 1),
        "float16",
        None,
        False,
        (),
        (),
        runtime,
    )

    assert ("resolve_op", "Prod") in calls
    assert ("reduce_all_dtype_op", "getws:Prod", "exec:Prod") in calls



def test_prod_dim_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_prod_get_workspace": object(),
            "aclnn_prod": object(),
            "aclnn_prod_dim_get_workspace": object(),
            "aclnn_prod_dim": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.prod(
        1,
        2,
        (2, 3, 4),
        (12, 4, 1),
        "float16",
        1,
        True,
        (2, 1, 4),
        (4, 4, 1),
        runtime,
    )

    assert ("resolve_op", "ProdDim") in calls
    assert ("axis_keepdim_dtype_op", "getws:ProdDim", "exec:ProdDim") in calls



def test_sisclose_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_isclose_get_workspace": object(),
            "aclnn_isclose": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.sisclose(
        1,
        2,
        3,
        (2, 3),
        (3, 1),
        (2, 3),
        (3, 1),
        (2, 3),
        (3, 1),
        "float16",
        1e-5,
        1e-8,
        False,
        runtime,
    )

    assert ("resolve_op", "IsClose") in calls
    assert (
        "binary_two_inputs_three_attrs_op",
        "getws:IsClose",
        "exec:IsClose",
    ) in calls



def test_floor_divide_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_floor_divide_get_workspace": object(),
            "aclnn_floor_divide": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.floor_divide(
        1,
        2,
        3,
        (2, 3),
        (3, 1),
        (2, 3),
        (3, 1),
        (2, 3),
        (3, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "FloorDivide") in calls
    assert ("binary_two_inputs_op", "getws:FloorDivide", "exec:FloorDivide") in calls



def test_select_family_embedding_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_embedding_get_workspace": object(),
            "aclnn_embedding": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.embedding(
        1, 2, 3,
        (8, 16), (16, 1),
        (4,), (1,),
        (4, 16), (16, 1),
        "float16", "int64",
        runtime,
    )

    assert ("resolve_op", "Embedding") in calls
    assert ("binary_two_inputs_op", "getws:Embedding", "exec:Embedding") in calls



def test_lerp_tensor_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_lerp_get_workspace": object(),
            "aclnn_lerp": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.lerp_tensor(
        1, 2, 3, 4,
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "Lerp") in calls
    assert ("four_tensor_op", "getws:Lerp", "exec:Lerp") in calls



def test_lerp_scalar_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_lerps_get_workspace": object(),
            "aclnn_lerps": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.lerp_scalar(
        1, 2, 3,
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        "float16",
        0.25,
        runtime,
    )

    assert ("resolve_op", "Lerps") in calls
    assert ("binary_two_inputs_op", "getws:Lerps", "exec:Lerps") not in calls
    assert ("two_tensor_scalar_op", "getws:Lerps", "exec:Lerps") in calls



def test_addcmul_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_addcmul_get_workspace": object(),
            "aclnn_addcmul": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.addcmul(
        1, 2, 3, 4,
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        "float16",
        0.5,
        runtime,
    )

    assert ("resolve_op", "Addcmul") in calls
    assert ("three_tensor_scalar_op", "getws:Addcmul", "exec:Addcmul") in calls



def test_addcdiv_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_addcdiv_get_workspace": object(),
            "aclnn_addcdiv": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.addcdiv(
        1, 2, 3, 4,
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        "float16",
        0.5,
        runtime,
    )

    assert ("resolve_op", "Addcdiv") in calls
    assert ("three_tensor_scalar_op", "getws:Addcdiv", "exec:Addcdiv") in calls



def test_slogaddexp_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_logaddexp_get_workspace": object(),
            "aclnn_logaddexp": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.slogaddexp(
        1, 2, 3,
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "LogAddExp") in calls
    assert ("binary_two_inputs_op", "getws:LogAddExp", "exec:LogAddExp") in calls



def test_slogaddexp2_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_logaddexp2_get_workspace": object(),
            "aclnn_logaddexp2": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.slogaddexp2(
        1, 2, 3,
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "LogAddExp2") in calls
    assert ("binary_two_inputs_op", "getws:LogAddExp2", "exec:LogAddExp2") in calls



def test_sremainder_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_remainder_tt_get_workspace": object(),
            "aclnn_remainder_tt": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.sremainder(
        1, 2, 3,
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "RemainderTensorTensor") in calls
    assert ("binary_two_inputs_op", "getws:RemainderTensorTensor", "exec:RemainderTensorTensor") in calls



def test_sfmod_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_fmod_tensor_get_workspace": object(),
            "aclnn_fmod_tensor": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.sfmod(
        1, 2, 3,
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        (2, 3), (3, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "FmodTensor") in calls
    assert ("binary_two_inputs_op", "getws:FmodTensor", "exec:FmodTensor") in calls


def test_arange_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_arange_get_workspace": object(),
            "aclnn_arange": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.arange(
        0,
        8,
        1,
        1,
        (8,),
        (1,),
        "int64",
        runtime,
    )

    assert ("resolve_op", "Arange") in calls
    assert ("output_tensor_three_scalars_op", "getws:Arange", "exec:Arange") in calls



def test_linspace_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_linspace_get_workspace": object(),
            "aclnn_linspace": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.linspace(
        0.0,
        1.0,
        5,
        1,
        (5,),
        (1,),
        "float16",
        runtime,
    )

    assert ("resolve_op", "Linspace") in calls
    assert ("output_tensor_three_scalars_op", "getws:Linspace", "exec:Linspace") in calls



def test_eye_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_eye_get_workspace": object(),
            "aclnn_eye": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.eye(
        4,
        6,
        1,
        (4, 6),
        (6, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "Eye") in calls
    assert ("output_tensor_two_ints_op", "getws:Eye", "exec:Eye") in calls



def test_range_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_range_get_workspace": object(),
            "aclnn_range": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.range_(
        0,
        8,
        2,
        1,
        (4,),
        (1,),
        "int64",
        runtime,
    )

    assert ("resolve_op", "Range") in calls
    assert ("output_tensor_three_scalars_op", "getws:Range", "exec:Range") in calls



def test_erfinv_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_erfinv_get_workspace": object(),
            "aclnn_erfinv": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.erfinv(
        1,
        2,
        (2, 3),
        (3, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "Erfinv") in calls
    assert ("unary_op", "getws:Erfinv", "exec:Erfinv") in calls



def test_linalg_qr_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_linalg_qr_get_workspace": object(),
            "aclnn_linalg_qr": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.linalg_qr(
        1,
        2,
        3,
        (4, 3),
        (3, 1),
        (4, 3),
        (3, 1),
        (3, 3),
        (3, 1),
        "float16",
        1,
        runtime,
    )

    assert ("resolve_op", "LinalgQr") in calls
    assert ("three_tensor_one_int_op", "getws:LinalgQr", "exec:LinalgQr") in calls



def test_one_hot_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_one_hot_get_workspace": object(),
            "aclnn_one_hot": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.one_hot(
        1,
        2,
        3,
        4,
        (2, 3),
        (3, 1),
        "int64",
        (),
        (),
        "float16",
        (),
        (),
        "float16",
        (2, 3, 5),
        (15, 5, 1),
        "float16",
        5,
        -1,
        runtime,
    )

    assert ("resolve_op", "OneHot") in calls
    assert ("four_tensor_two_ints_op", "getws:OneHot", "exec:OneHot") in calls



def test_median_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_median_get_workspace": object(),
            "aclnn_median": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.median(
        1,
        2,
        (6,),
        (1,),
        "float16",
        (1,),
        (1,),
        runtime,
    )

    assert ("resolve_op", "Median") in calls
    assert ("unary_op", "getws:Median", "exec:Median") in calls



def test_median_dim_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_median_dim_get_workspace": object(),
            "aclnn_median_dim": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.median_dim(
        1,
        2,
        3,
        (2, 3),
        (3, 1),
        "float16",
        (2, 1),
        (1, 1),
        1,
        True,
        runtime,
    )

    assert ("resolve_op", "MedianDim") in calls
    assert ("three_tensor_two_ints_bool_op", "getws:MedianDim", "exec:MedianDim") in calls



def test_kthvalue_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_kthvalue_get_workspace": object(),
            "aclnn_kthvalue": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.kthvalue(
        1,
        2,
        3,
        (2, 3),
        (3, 1),
        "float16",
        (2, 1),
        (1, 1),
        2,
        1,
        False,
        runtime,
    )

    assert ("resolve_op", "Kthvalue") in calls
    assert ("three_tensor_two_ints_bool_op", "getws:Kthvalue", "exec:Kthvalue") in calls



def test_addmm_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_addmm_get_workspace": object(),
            "aclnn_addmm": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.addmm(
        1,
        2,
        3,
        4,
        (4, 5),
        (5, 1),
        "float16",
        (4, 6),
        (6, 1),
        (6, 5),
        (5, 1),
        (4, 5),
        (5, 1),
        1.0,
        1.0,
        runtime,
    )

    assert ("resolve_op", "Addmm") in calls
    assert ("four_tensor_two_scalars_one_int8_op", "getws:Addmm", "exec:Addmm") in calls



def test_baddbmm_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_baddbmm_get_workspace": object(),
            "aclnn_baddbmm": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.baddbmm(
        1,
        2,
        3,
        4,
        (2, 4, 5),
        (20, 5, 1),
        (2, 4, 6),
        (24, 6, 1),
        (2, 6, 5),
        (30, 5, 1),
        (2, 4, 5),
        (20, 5, 1),
        "float16",
        1.0,
        1.0,
        runtime,
    )

    assert ("resolve_op", "Baddbmm") in calls
    assert ("four_tensor_two_scalars_one_int8_op", "getws:Baddbmm", "exec:Baddbmm") in calls



def test_linalg_vector_norm_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_linalg_vector_norm_get_workspace": object(),
            "aclnn_linalg_vector_norm": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.linalg_vector_norm(
        1,
        2,
        (2, 3),
        (3, 1),
        (2, 1),
        (1, 1),
        "float16",
        2.0,
        (1,),
        True,
        runtime,
    )

    assert ("resolve_op", "LinalgVectorNorm") in calls
    assert ("tensor_scalar_int_array_bool_op", "getws:LinalgVectorNorm", "exec:LinalgVectorNorm") in calls



def test_aminmax_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_aminmax_get_workspace": object(),
            "aclnn_aminmax": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.aminmax(
        1,
        2,
        3,
        (2, 3),
        (3, 1),
        (2, 1),
        (1, 1),
        "float16",
        (1,),
        True,
        runtime,
    )

    assert ("resolve_op", "Aminmax") in calls
    assert ("tensor_int_array_bool_two_outputs_op", "getws:Aminmax", "exec:Aminmax") in calls



def test_inplace_one_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_inplace_one_get_workspace": object(),
            "aclnn_inplace_one": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.inplace_one(
        1,
        (2, 3),
        (3, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "InplaceOne") in calls
    assert ("inplace_unary_op", "getws:InplaceOne", "exec:InplaceOne") in calls



def test_inplace_zero_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_inplace_zero_get_workspace": object(),
            "aclnn_inplace_zero": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.inplace_zero(
        1,
        (2, 3),
        (3, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "InplaceZero") in calls
    assert ("inplace_unary_op", "getws:InplaceZero", "exec:InplaceZero") in calls


def test_inplace_normal_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_inplace_normal_get_workspace": object(),
            "aclnn_inplace_normal": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.inplace_normal(
        1,
        (2, 3),
        (3, 1),
        "float16",
        0.0,
        1.0,
        123,
        7,
        runtime,
    )

    assert ("resolve_op", "InplaceNormal") in calls
    assert ("inplace_normal_op", "getws:InplaceNormal", "exec:InplaceNormal") in calls


def test_inplace_uniform_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_inplace_uniform_get_workspace": object(),
            "aclnn_inplace_uniform": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.inplace_uniform(
        1,
        (2, 3),
        (3, 1),
        "float16",
        0.0,
        1.0,
        123,
        7,
        runtime,
    )

    assert ("resolve_op", "InplaceUniform") in calls
    assert ("inplace_uniform_op", "getws:InplaceUniform", "exec:InplaceUniform") in calls


def test_inplace_fill_scalar_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_inplace_fill_scalar_get_workspace": object(),
            "aclnn_inplace_fill_scalar": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.inplace_fill_scalar(
        1,
        (2, 3),
        (3, 1),
        "float16",
        2.0,
        runtime,
    )

    assert ("resolve_op", "InplaceFillScalar") in calls
    assert ("inplace_fill_scalar_op", "getws:InplaceFillScalar", "exec:InplaceFillScalar") in calls


def test_inplace_copy_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_inplace_copy_get_workspace": object(),
            "aclnn_inplace_copy": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.inplace_copy(
        1,
        2,
        (2, 3),
        (3, 1),
        "float16",
        (2, 3),
        (3, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "InplaceCopy") in calls
    assert ("inplace_copy_op", "getws:InplaceCopy", "exec:InplaceCopy") in calls


def test_inplace_masked_fill_scalar_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_inplace_masked_fill_scalar_get_workspace": object(),
            "aclnn_inplace_masked_fill_scalar": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.inplace_masked_fill_scalar(
        1,
        (2, 3),
        (3, 1),
        "float16",
        2,
        (2, 3),
        (3, 1),
        "bool",
        1.5,
        runtime,
    )

    assert ("resolve_op", "InplaceMaskedFillScalar") in calls
    assert ("inplace_masked_fill_scalar_op", "getws:InplaceMaskedFillScalar", "exec:InplaceMaskedFillScalar") in calls


def test_inplace_index_fill_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_inplace_index_fill_get_workspace": object(),
            "aclnn_inplace_index_fill": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.inplace_index_fill(
        1,
        (2, 3),
        (3, 1),
        "float16",
        1,
        2,
        (2,),
        (1,),
        "int64",
        1.5,
        runtime,
    )

    assert ("resolve_op", "InplaceIndexFill") in calls
    assert ("inplace_index_fill_op", "getws:InplaceIndexFill", "exec:InplaceIndexFill") in calls


def test_inplace_index_copy_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_inplace_index_copy_get_workspace": object(),
            "aclnn_inplace_index_copy": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.inplace_index_copy(
        1,
        (2, 3),
        (3, 1),
        "float16",
        1,
        2,
        (2,),
        (1,),
        "int64",
        3,
        (2, 2),
        (2, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "InplaceIndexCopy") in calls
    assert ("inplace_index_copy_op", "getws:InplaceIndexCopy", "exec:InplaceIndexCopy") in calls


def test_index_add_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_index_add_get_workspace": object(),
            "aclnn_index_add": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.index_add(
        1,
        (2, 3),
        (3, 1),
        "float16",
        1,
        2,
        (2,),
        (1,),
        "int64",
        3,
        (2, 2),
        (2, 1),
        "float16",
        1.0,
        4,
        (2, 3),
        (3, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "IndexAdd") in calls
    assert ("index_add_op", "getws:IndexAdd", "exec:IndexAdd") in calls



def test_scatter_add_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_scatter_add_get_workspace": object(),
            "aclnn_scatter_add": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.scatter_add_op(
        1,
        (2, 3),
        (3, 1),
        "float16",
        1,
        2,
        (2, 2),
        (2, 1),
        "int64",
        3,
        (2, 2),
        (2, 1),
        "float16",
        4,
        (2, 3),
        (3, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "ScatterAdd") in calls
    assert ("scatter_add_op", "getws:ScatterAdd", "exec:ScatterAdd") in calls



def test_inplace_masked_scatter_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_inplace_masked_scatter_get_workspace": object(),
            "aclnn_inplace_masked_scatter": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.inplace_masked_scatter(
        1,
        (2, 3),
        (3, 1),
        "float16",
        2,
        (2, 3),
        (3, 1),
        "bool",
        3,
        (3,),
        (1,),
        "float16",
        runtime,
    )

    assert ("resolve_op", "InplaceMaskedScatter") in calls
    assert ("inplace_masked_scatter_op", "getws:InplaceMaskedScatter", "exec:InplaceMaskedScatter") in calls



def test_index_put_impl_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_index_put_impl_get_workspace": object(),
            "aclnn_index_put_impl": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.index_put_impl(
        1,
        (2, 3),
        (3, 1),
        "float16",
        [(2, (2,), (1,), "int64")],
        3,
        (2,),
        (1,),
        "float16",
        False,
        False,
        runtime,
    )

    assert ("resolve_op", "IndexPutImpl") in calls
    assert ("index_put_impl_op", "getws:IndexPutImpl", "exec:IndexPutImpl") in calls


def test_index_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_index_get_workspace": object(),
            "aclnn_index": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.index(
        1,
        (2, 3),
        (3, 1),
        "float16",
        [None, (2, (2,), (1,), "int64")],
        3,
        (2,),
        (1,),
        "float16",
        runtime,
    )

    assert ("resolve_op", "Index") in calls
    assert ("index_with_optional_tensor_list_op", "getws:Index", "exec:Index") in calls


def test_search_sorted_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_search_sorted_get_workspace": object(),
            "aclnn_search_sorted": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.search_sorted(
        1,
        2,
        3,
        (8,),
        (1,),
        (4,),
        (1,),
        (4,),
        (1,),
        "float16",
        True,
        False,
        runtime,
    )

    assert ("resolve_op", "SearchSorted") in calls
    assert ("two_tensor_scalar_bool_op", "getws:SearchSorted", "exec:SearchSorted") in calls



def test_unique_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_unique_get_workspace": object(),
            "aclnn_unique": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.unique(
        1,
        2,
        3,
        (8,),
        (1,),
        "float16",
        (4,),
        (1,),
        (8,),
        (1,),
        True,
        True,
        runtime,
    )

    assert ("resolve_op", "Unique") in calls
    assert ("unary_two_bools_two_outputs_op", "getws:Unique", "exec:Unique") in calls



def test_randperm_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_randperm_get_workspace": object(),
            "aclnn_randperm": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.randperm(
        8,
        1,
        "int64",
        runtime,
        seed=123,
        offset=7,
    )

    assert ("resolve_op", "Randperm") in calls
    assert ("output_tensor_three_ints_op", "getws:Randperm", "exec:Randperm") in calls


def test_strace_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_trace_get_workspace": object(),
            "aclnn_trace": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.strace(
        1,
        2,
        (4, 4),
        (4, 1),
        "float16",
        (),
        (),
        runtime,
    )

    assert ("resolve_op", "Trace") in calls
    assert ("unary_op", "getws:Trace", "exec:Trace") in calls


def test_renorm_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_renorm_get_workspace": object(),
            "aclnn_renorm": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.renorm(
        1,
        2,
        (2, 3),
        (3, 1),
        "float16",
        2.0,
        1,
        1.5,
        runtime,
    )

    assert ("resolve_op", "Renorm") in calls
    assert ("tensor_two_scalars_dim_op", "getws:Renorm", "exec:Renorm") in calls



def test_var_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_var_get_workspace": object(),
            "aclnn_var": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.var(
        1,
        2,
        (2, 3, 4),
        (12, 4, 1),
        "float16",
        (1,),
        True,
        False,
        (2, 4),
        (4, 1),
        runtime,
    )

    assert ("resolve_op", "Var") in calls
    assert ("tensor_int_array_two_bools_op", "getws:Var", "exec:Var") in calls



def test_norm_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_norm_get_workspace": object(),
            "aclnn_norm": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.norm(
        1,
        2,
        (2, 3, 4),
        (12, 4, 1),
        "float16",
        2.0,
        (1, 2),
        True,
        (2, 1, 1),
        (1, 1, 1),
        runtime,
    )

    assert ("resolve_op", "Norm") in calls
    assert ("tensor_scalar_int_array_bool_op", "getws:Norm", "exec:Norm") in calls



def test_npu_aclnn_source_has_no_fallback_import():
    src = Path(aclnn.__file__).read_text()
    assert "_aclnn_ffi_fallback" not in src


def test_remaining_batch_linalg_cross_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_linalg_cross_get_workspace": object(),
            "aclnn_linalg_cross": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.linalg_cross(
        1,
        2,
        3,
        (2, 3),
        (3, 1),
        (2, 3),
        (3, 1),
        (2, 3),
        (3, 1),
        "float16",
        1,
        runtime,
    )

    assert ("resolve_op", "LinalgCross") in calls
    assert ("binary_two_inputs_with_dim_op", "getws:LinalgCross", "exec:LinalgCross") in calls


def test_remaining_batch_upsample_nearest2d_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_upsample_nearest2d_get_workspace": object(),
            "aclnn_upsample_nearest2d": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.upsample_nearest2d(
        1,
        2,
        (1, 3, 8, 8),
        (192, 64, 8, 1),
        "float16",
        (16, 16),
        (1, 3, 16, 16),
        (768, 256, 16, 1),
        runtime,
    )

    assert ("resolve_op", "UpsampleNearest2d") in calls
    assert ("tensor_int_array_op", "getws:UpsampleNearest2d", "exec:UpsampleNearest2d") in calls


def test_remaining_batch_adaptive_avg_pool3d_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_adaptive_avg_pool3d_get_workspace": object(),
            "aclnn_adaptive_avg_pool3d": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.adaptive_avg_pool3d(
        1,
        2,
        (1, 3, 8, 8, 8),
        (1536, 512, 64, 8, 1),
        (1, 3, 4, 4, 4),
        (192, 64, 16, 4, 1),
        "float16",
        (4, 4, 4),
        runtime,
    )

    assert ("resolve_op", "AdaptiveAvgPool3d") in calls
    assert ("tensor_int_array_op", "getws:AdaptiveAvgPool3d", "exec:AdaptiveAvgPool3d") in calls


def test_remaining_batch_bincount_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_bincount_get_workspace": object(),
            "aclnn_bincount": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.bincount(
        1,
        None,
        2,
        (6,),
        (1,),
        (8,),
        (1,),
        "int64",
        "int64",
        8,
        runtime=runtime,
    )

    assert ("resolve_op", "Bincount") in calls
    assert ("optional_tensor_int_op", "getws:Bincount", "exec:Bincount") in calls


def test_remaining_batch_upsample_bilinear2d_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_upsample_bilinear2d_get_workspace": object(),
            "aclnn_upsample_bilinear2d": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.upsample_bilinear2d(
        1,
        2,
        (1, 3, 8, 8),
        (192, 64, 8, 1),
        "float16",
        (16, 16),
        False,
        None,
        None,
        (1, 3, 16, 16),
        (768, 256, 16, 1),
        runtime,
    )

    assert ("resolve_op", "UpsampleBilinear2d") in calls
    assert ("tensor_int_array_bool_two_doubles_op", "getws:UpsampleBilinear2d", "exec:UpsampleBilinear2d") in calls


def test_remaining_batch_adaptive_avg_pool2d_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_adaptive_avg_pool2d_get_workspace": object(),
            "aclnn_adaptive_avg_pool2d": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.adaptive_avg_pool2d(
        1,
        2,
        (1, 3, 8, 8),
        (192, 64, 8, 1),
        "float16",
        (4, 4),
        (1, 3, 4, 4),
        (48, 16, 4, 1),
        runtime,
    )

    assert ("resolve_op", "AdaptiveAvgPool2d") in calls
    assert ("tensor_int_array_op", "getws:AdaptiveAvgPool2d", "exec:AdaptiveAvgPool2d") in calls


def test_remaining_batch_upsample_bicubic2d_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_upsample_bicubic2d_get_workspace": object(),
            "aclnn_upsample_bicubic2d": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.upsample_bicubic2d(
        1,
        2,
        (1, 3, 8, 8),
        (192, 64, 8, 1),
        (1, 3, 16, 16),
        (768, 256, 16, 1),
        "float16",
        (16, 16),
        False,
        None,
        None,
        runtime,
    )

    assert ("resolve_op", "UpsampleBicubic2d") in calls
    assert ("tensor_int_array_bool_two_doubles_op", "getws:UpsampleBicubic2d", "exec:UpsampleBicubic2d") in calls


def test_remaining_batch_upsample_linear1d_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_upsample_linear1d_get_workspace": object(),
            "aclnn_upsample_linear1d": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.upsample_linear1d(
        1,
        2,
        (1, 3, 8),
        (24, 8, 1),
        (1, 3, 16),
        (48, 16, 1),
        "float16",
        (16,),
        False,
        None,
        runtime,
    )

    assert ("resolve_op", "UpsampleLinear1d") in calls
    assert ("tensor_int_array_bool_double_op", "getws:UpsampleLinear1d", "exec:UpsampleLinear1d") in calls


def test_remaining_batch_dropout_gen_mask_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_dropout_gen_mask_get_workspace": object(),
            "aclnn_dropout_gen_mask": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.dropout_gen_mask((2, 3), 0.1, 123, 456, 1, 16, runtime)

    assert ("resolve_op", "DropoutGenMask") in calls
    assert ("output_tensor_int_array_double_two_ints_op", "getws:DropoutGenMask", "exec:DropoutGenMask") in calls


def test_remaining_batch_dropout_do_mask_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_dropout_do_mask_get_workspace": object(),
            "aclnn_dropout_do_mask": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.dropout_do_mask(1, 2, 3, (2, 3), (3, 1), "float16", 16, 0.1, runtime)

    assert ("resolve_op", "DropoutDoMask") in calls
    assert ("two_tensor_one_double_op", "getws:DropoutDoMask", "exec:DropoutDoMask") in calls


def test_remaining_batch_sgrid_sampler2d_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_grid_sampler2d_get_workspace": object(),
            "aclnn_grid_sampler2d": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.sgrid_sampler2d(
        1,
        2,
        3,
        (1, 3, 8, 8),
        (192, 64, 8, 1),
        (1, 8, 8, 2),
        (128, 16, 2, 1),
        (1, 3, 8, 8),
        (192, 64, 8, 1),
        "float16",
        0,
        0,
        False,
        runtime,
    )

    assert ("resolve_op", "GridSampler2D") in calls
    assert ("two_tensor_two_ints_bool_mixed_fmt_op", "getws:GridSampler2D", "exec:GridSampler2D") in calls


def test_remaining_batch_saffine_grid_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_affine_grid_get_workspace": object(),
            "aclnn_affine_grid": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.saffine_grid(
        1,
        2,
        (1, 2, 3),
        (6, 3, 1),
        "float16",
        (1, 3, 8, 8),
        False,
        (1, 8, 8, 2),
        (128, 16, 2, 1),
        runtime,
    )

    assert ("resolve_op", "AffineGrid") in calls
    assert ("tensor_int_array_bool_op", "getws:AffineGrid", "exec:AffineGrid") in calls


def test_remaining_batch_max_pool_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_max_pool_get_workspace": object(),
            "aclnn_max_pool": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.max_pool(
        1,
        2,
        (1, 3, 8, 8),
        (192, 64, 8, 1),
        "float16",
        (2, 2),
        (2, 2),
        (0, 0, 0, 0),
        (1, 1),
        False,
        (1, 3, 4, 4),
        (48, 16, 4, 1),
        runtime,
    )

    assert ("resolve_op", "MaxPool") in calls
    assert ("tensor_four_int_arrays_two_ints_op", "getws:MaxPool", "exec:MaxPool") in calls


def test_remaining_batch_max_pool2d_with_mask_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_max_pool2d_with_mask_get_workspace": object(),
            "aclnn_max_pool2d_with_mask": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.max_pool2d_with_mask(
        1,
        2,
        3,
        (1, 3, 8, 8),
        (192, 64, 8, 1),
        "float16",
        (2, 2),
        (2, 2),
        (0, 0),
        (1, 1),
        False,
        (1, 3, 4, 4),
        (48, 16, 4, 1),
        (1, 3, 4, 4),
        (48, 16, 4, 1),
        runtime,
    )

    assert ("resolve_op", "MaxPool2dWithMask") in calls
    assert ("tensor_four_int_arrays_bool_two_outputs_op", "getws:MaxPool2dWithMask", "exec:MaxPool2dWithMask") in calls


def test_remaining_batch_avg_pool2d_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_avg_pool2d_get_workspace": object(),
            "aclnn_avg_pool2d": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.avg_pool2d(
        1,
        2,
        (1, 3, 8, 8),
        (192, 64, 8, 1),
        "float16",
        (2, 2),
        (2, 2),
        (0, 0),
        False,
        True,
        None,
        (1, 3, 4, 4),
        (48, 16, 4, 1),
        runtime,
    )

    assert ("resolve_op", "AvgPool2d") in calls
    assert ("tensor_three_int_arrays_two_bools_int64_int8_op", "getws:AvgPool2d", "exec:AvgPool2d") in calls


def test_remaining_batch_max_pool3d_with_argmax_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_max_pool3d_with_argmax_get_workspace": object(),
            "aclnn_max_pool3d_with_argmax": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.max_pool3d_with_argmax(
        1,
        2,
        3,
        (1, 3, 8, 8, 8),
        (1536, 512, 64, 8, 1),
        "float16",
        (2, 2, 2),
        (2, 2, 2),
        (0, 0, 0),
        (1, 1, 1),
        False,
        (1, 3, 4, 4, 4),
        (192, 64, 16, 4, 1),
        (1, 3, 4, 4, 4),
        (192, 64, 16, 4, 1),
        runtime,
    )

    assert ("resolve_op", "MaxPool3dWithArgmax") in calls
    assert ("tensor_four_int_arrays_bool_two_outputs_op", "getws:MaxPool3dWithArgmax", "exec:MaxPool3dWithArgmax") in calls


def test_remaining_batch_adaptive_max_pool2d_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_adaptive_max_pool2d_get_workspace": object(),
            "aclnn_adaptive_max_pool2d": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.adaptive_max_pool2d(
        1,
        2,
        3,
        (1, 3, 8, 8),
        (192, 64, 8, 1),
        "float16",
        (4, 4),
        (1, 3, 4, 4),
        (48, 16, 4, 1),
        (1, 3, 4, 4),
        (48, 16, 4, 1),
        runtime,
    )

    assert ("resolve_op", "AdaptiveMaxPool2d") in calls
    assert ("tensor_int_array_two_outputs_op", "getws:AdaptiveMaxPool2d", "exec:AdaptiveMaxPool2d") in calls


def test_remaining_batch_einsum_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_einsum_get_workspace": object(),
            "aclnn_einsum": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.einsum(
        [1, 2],
        [(2, 3), (3, 4)],
        [(3, 1), (4, 1)],
        ["float16", "float16"],
        "ab,bc->ac",
        3,
        (2, 4),
        (4, 1),
        "float16",
        runtime,
    )

    assert ("resolve_op", "Einsum") in calls
    assert ("tensor_list_string_op", "getws:Einsum", "exec:Einsum") in calls


def test_remaining_batch_sinstance_norm_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_instance_norm_get_workspace": object(),
            "aclnn_instance_norm": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.sinstance_norm(
        1, 2, 3, 4, 5, 6,
        (1, 3, 8, 8), (192, 64, 8, 1),
        (3,), (1,),
        (3,), (1,),
        (1, 3, 8, 8), (192, 64, 8, 1),
        (3,), (1,),
        (3,), (1,),
        "float16", 1e-5, runtime,
    )

    assert ("resolve_op", "InstanceNorm") in calls
    assert ("six_tensor_string_double_op", "getws:InstanceNorm", "exec:InstanceNorm") in calls


def test_remaining_batch_apply_adam_w_v2_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_apply_adam_w_v2_get_workspace": object(),
            "aclnn_apply_adam_w_v2": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.apply_adam_w_v2(
        1, 2, 3, None, 4, 5,
        (16,), (1,),
        (1,), (1,),
        "float16",
        1e-3, 0.9, 0.999, 0.01, 1e-8,
        False, False,
        runtime,
    )

    assert ("resolve_op", "ApplyAdamWV2") in calls
    assert ("six_tensor_five_floats_two_bools_op", "getws:ApplyAdamWV2", "exec:ApplyAdamWV2") in calls


def test_remaining_batch_batch_norm_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_batch_norm_get_workspace": object(),
            "aclnn_batch_norm": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.batch_norm(
        1, 2, 3, 4, 5, 6,
        (1, 3, 8, 8), (192, 64, 8, 1),
        (3,), (1,),
        (3,), (1,),
        (3,), (1,),
        (3,), (1,),
        (1, 3, 8, 8), (192, 64, 8, 1),
        False, 0.1, 1e-5, "float16", runtime,
    )

    assert ("resolve_op", "BatchNorm") in calls
    assert ("batch_norm_op", "getws:BatchNorm", "exec:BatchNorm") in calls


def test_remaining_batch_group_norm_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_group_norm_get_workspace": object(),
            "aclnn_group_norm": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.group_norm(
        1, 2, 3, 4,
        (1, 4, 8, 8), (256, 64, 8, 1),
        (4,), (1,),
        (4,), (1,),
        (1, 4, 8, 8), (256, 64, 8, 1),
        2, 1e-5, "float16", runtime,
    )

    assert ("resolve_op", "GroupNorm") in calls
    assert ("group_norm_op", "getws:GroupNorm", "exec:GroupNorm") in calls


def test_remaining_batch_convolution_uses_native_ffi(monkeypatch):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type(
        "_FakeBindings",
        (),
        {
            "aclnn_convolution_get_workspace": object(),
            "aclnn_convolution": object(),
        },
    )()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    aclnn.convolution(
        1, 2, 3,
        (1, 3, 8, 8), (192, 64, 8, 1),
        (4, 3, 3, 3), (27, 9, 3, 1),
        (4,), (1,),
        "float16",
        (1, 1), (1, 1), (1, 1), False, (0, 0), 1,
        4, (1, 4, 8, 8), (256, 64, 8, 1),
        runtime,
    )

    assert ("resolve_op", "Convolution") in calls
    assert ("convolution_op", "getws:Convolution", "exec:Convolution") in calls


@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "binding_attrs", "call_args"),
    [
        (
            "softmax_backward",
            "SoftmaxBackward",
            ("aclnn_softmax_backward_get_workspace", "aclnn_softmax_backward"),
            (1, 2, 3, (2, 3), (3, 1), (3, 1), (3, 1), "float16", 1),
        ),
        (
            "log_softmax_backward",
            "LogSoftmaxBackward",
            ("aclnn_log_softmax_backward_get_workspace", "aclnn_log_softmax_backward"),
            (1, 2, 3, (2, 3), (3, 1), (3, 1), (3, 1), "float16", 1),
        ),
    ],
)
def test_remaining_backward_dim_wrappers_use_native_ffi(
    monkeypatch, wrapper_name, ffi_name, binding_attrs, call_args
):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(*call_args, runtime)

    assert ("resolve_op", ffi_name) in calls
    assert ("binary_two_inputs_with_dim_op", f"getws:{ffi_name}", f"exec:{ffi_name}") in calls


@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "binding_attrs", "call_args"),
    [
        (
            "gelu_backward",
            "GeluBackward",
            ("aclnn_gelu_backward_get_workspace", "aclnn_gelu_backward"),
            (1, 2, 3, (2, 3), (3, 1), (3, 1), (3, 1), "float16"),
        ),
        (
            "sigmoid_backward",
            "SigmoidBackward",
            ("aclnn_sigmoid_backward_get_workspace", "aclnn_sigmoid_backward"),
            (1, 2, 3, (2, 3), (3, 1), (3, 1), (3, 1), "float16"),
        ),
        (
            "tanh_backward",
            "TanhBackward",
            ("aclnn_tanh_backward_get_workspace", "aclnn_tanh_backward"),
            (1, 2, 3, (2, 3), (3, 1), (3, 1), (3, 1), "float16"),
        ),
        (
            "silu_backward",
            "SiluBackward",
            ("aclnn_silu_backward_get_workspace", "aclnn_silu_backward"),
            (1, 2, 3, (2, 3), (3, 1), (3, 1), (3, 1), "float16"),
        ),
        (
            "hardswish_backward",
            "HardswishBackward",
            ("aclnn_hardswish_backward_get_workspace", "aclnn_hardswish_backward"),
            (1, 2, 3, (2, 3), (3, 1), (3, 1), (3, 1), "float16"),
        ),
        (
            "hardsigmoid_backward",
            "HardsigmoidBackward",
            ("aclnn_hardsigmoid_backward_get_workspace", "aclnn_hardsigmoid_backward"),
            (1, 2, 3, (2, 3), (3, 1), (3, 1), (3, 1), "float16"),
        ),
        (
            "mish_backward",
            "MishBackward",
            ("aclnn_mish_backward_get_workspace", "aclnn_mish_backward"),
            (1, 2, 3, (2, 3), (3, 1), (3, 1), (3, 1), "float16"),
        ),
        (
            "selu_backward",
            "SeluBackward",
            ("aclnn_selu_backward_get_workspace", "aclnn_selu_backward"),
            (1, 2, 3, (2, 3), (3, 1), (3, 1), (3, 1), "float16"),
        ),
    ],
)
def test_remaining_backward_binary_wrappers_use_native_ffi(
    monkeypatch, wrapper_name, ffi_name, binding_attrs, call_args
):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(*call_args, runtime)

    assert ("resolve_op", ffi_name) in calls
    assert ("binary_two_inputs_op", f"getws:{ffi_name}", f"exec:{ffi_name}") in calls


@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "binding_attrs", "scalar_value"),
    [
        (
            "threshold_backward",
            "ThresholdBackward",
            ("aclnn_threshold_backward_get_workspace", "aclnn_threshold_backward"),
            0.0,
        ),
        (
            "hardshrink_backward",
            "HardshrinkBackward",
            ("aclnn_hardshrink_backward_get_workspace", "aclnn_hardshrink_backward"),
            0.5,
        ),
        (
            "softshrink_backward",
            "SoftshrinkBackward",
            ("aclnn_softshrink_backward_get_workspace", "aclnn_softshrink_backward"),
            0.5,
        ),
    ],
)
def test_remaining_backward_scalar_wrappers_use_native_ffi(
    monkeypatch, wrapper_name, ffi_name, binding_attrs, scalar_value
):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(
        1,
        2,
        3,
        (2, 3),
        (3, 1),
        (3, 1),
        (3, 1),
        "float16",
        scalar_value,
        runtime,
    )

    assert ("resolve_op", ffi_name) in calls
    assert ("two_tensor_scalar_op", f"getws:{ffi_name}", f"exec:{ffi_name}") in calls


@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "binding_attrs", "call_args", "helper_name"),
    [
        (
            "softplus_backward",
            "SoftplusBackward",
            ("aclnn_softplus_backward_get_workspace", "aclnn_softplus_backward"),
            (1, 2, 3, (2, 3), (3, 1), (3, 1), (3, 1), "float16", 1.0, 20.0),
            "two_tensor_two_scalars_op",
        ),
        (
            "hardtanh_backward",
            "HardtanhBackward",
            ("aclnn_hardtanh_backward_get_workspace", "aclnn_hardtanh_backward"),
            (1, 2, 3, (2, 3), (3, 1), (3, 1), (3, 1), "float16", -1.0, 1.0),
            "two_tensor_two_scalars_op",
        ),
        (
            "leaky_relu_backward",
            "LeakyReluBackward",
            ("aclnn_leaky_relu_backward_get_workspace", "aclnn_leaky_relu_backward"),
            (1, 2, 3, (2, 3), (3, 1), (3, 1), (3, 1), "float16", 0.1),
            "two_tensor_scalar_bool_op",
        ),
        (
            "elu_backward",
            "EluBackward",
            ("aclnn_elu_backward_get_workspace", "aclnn_elu_backward"),
            (1, 2, 3, (2, 3), (3, 1), (3, 1), (3, 1), "float16", 1.0, 1.0, 1.0),
            "two_tensor_three_scalars_bool_op",
        ),
        (
            "prelu_backward",
            "PreluBackward",
            ("aclnn_prelu_backward_get_workspace", "aclnn_prelu_backward"),
            (1, 2, 3, 4, 5, (2, 3), (3, 1), (3, 1), (3,), (1,), (3, 1), (1,), "float16"),
            "three_tensor_two_outputs_op",
        ),
    ],
)
def test_remaining_backward_extra_wrappers_use_native_ffi(
    monkeypatch, wrapper_name, ffi_name, binding_attrs, call_args, helper_name
):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(*call_args, runtime)

    assert ("resolve_op", ffi_name) in calls
    assert (helper_name, f"getws:{ffi_name}", f"exec:{ffi_name}") in calls


@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "binding_attrs", "call_args", "helper_name"),
    [
        (
            "upsample_nearest2d_backward",
            "UpsampleNearest2dBackward",
            (
                "aclnn_upsample_nearest2d_backward_get_workspace",
                "aclnn_upsample_nearest2d_backward",
            ),
            (
                1,
                2,
                (1, 3, 16, 16),
                (768, 256, 16, 1),
                (1, 3, 8, 8),
                (192, 64, 8, 1),
                (16, 16),
                (8, 8),
                None,
                None,
                "float16",
            ),
            "tensor_two_int_arrays_op",
        ),
        (
            "upsample_bilinear2d_backward",
            "UpsampleBilinear2dBackward",
            (
                "aclnn_upsample_bilinear2d_backward_get_workspace",
                "aclnn_upsample_bilinear2d_backward",
            ),
            (
                1,
                2,
                (1, 3, 16, 16),
                (768, 256, 16, 1),
                (1, 3, 8, 8),
                (192, 64, 8, 1),
                (16, 16),
                (8, 8),
                False,
                None,
                None,
                "float16",
            ),
            "tensor_two_int_arrays_bool_two_doubles_op",
        ),
        (
            "upsample_bicubic2d_backward",
            "UpsampleBicubic2dBackward",
            (
                "aclnn_upsample_bicubic2d_backward_get_workspace",
                "aclnn_upsample_bicubic2d_backward",
            ),
            (
                1,
                2,
                (1, 3, 16, 16),
                (768, 256, 16, 1),
                (1, 3, 8, 8),
                (192, 64, 8, 1),
                (16, 16),
                (8, 8),
                False,
                None,
                None,
                "float16",
            ),
            "tensor_two_int_arrays_bool_two_doubles_op",
        ),
        (
            "upsample_nearest1d_backward",
            "UpsampleNearest1dBackward",
            (
                "aclnn_upsample_nearest1d_backward_get_workspace",
                "aclnn_upsample_nearest1d_backward",
            ),
            (
                1,
                2,
                (1, 3, 16),
                (48, 16, 1),
                (1, 3, 8),
                (24, 8, 1),
                (16,),
                (8,),
                None,
                "float16",
            ),
            "tensor_two_int_arrays_op",
        ),
        (
            "upsample_linear1d_backward",
            "UpsampleLinear1dBackward",
            (
                "aclnn_upsample_linear1d_backward_get_workspace",
                "aclnn_upsample_linear1d_backward",
            ),
            (
                1,
                2,
                (1, 3, 16),
                (48, 16, 1),
                (1, 3, 8),
                (24, 8, 1),
                (16,),
                (8,),
                False,
                None,
                "float16",
            ),
            "tensor_two_int_arrays_bool_double_op",
        ),
        (
            "adaptive_avg_pool2d_backward",
            "AdaptiveAvgPool2dBackward",
            (
                "aclnn_adaptive_avg_pool2d_backward_get_workspace",
                "aclnn_adaptive_avg_pool2d_backward",
            ),
            (
                1,
                2,
                3,
                (1, 3, 4, 4),
                (48, 16, 4, 1),
                (1, 3, 8, 8),
                (192, 64, 8, 1),
                (1, 3, 8, 8),
                (192, 64, 8, 1),
                "float16",
            ),
            "four_tensor_op",
        ),
        (
            "adaptive_avg_pool3d_backward",
            "AdaptiveAvgPool3dBackward",
            (
                "aclnn_adaptive_avg_pool3d_backward_get_workspace",
                "aclnn_adaptive_avg_pool3d_backward",
            ),
            (
                1,
                2,
                3,
                (1, 3, 4, 4, 4),
                (192, 64, 16, 4, 1),
                (1, 3, 8, 8, 8),
                (1536, 512, 64, 8, 1),
                (1, 3, 8, 8, 8),
                (1536, 512, 64, 8, 1),
                "float16",
            ),
            "four_tensor_op",
        ),
    ],
)
def test_remaining_backward_upsample_pool_wrappers_use_native_ffi(
    monkeypatch, wrapper_name, ffi_name, binding_attrs, call_args, helper_name
):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(*call_args, runtime)

    assert ("resolve_op", ffi_name) in calls
    assert (helper_name, f"getws:{ffi_name}", f"exec:{ffi_name}") in calls


@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "binding_attrs", "call_args", "helper_name"),
    [
        (
            "max_pool2d_with_mask_backward",
            "MaxPool2dWithMaskBackward",
            (
                "aclnn_max_pool2d_with_mask_backward_get_workspace",
                "aclnn_max_pool2d_with_mask_backward",
            ),
            (
                1,
                2,
                3,
                4,
                (1, 3, 4, 4),
                (48, 16, 4, 1),
                (1, 3, 8, 8),
                (192, 64, 8, 1),
                (1, 3, 4, 4),
                (48, 16, 4, 1),
                (1, 3, 8, 8),
                (192, 64, 8, 1),
                (2, 2),
                (2, 2),
                (0, 0),
                (1, 1),
                False,
                "float16",
            ),
            "four_tensor_four_int_arrays_bool_op",
        ),
        (
            "avg_pool2d_backward",
            "AvgPool2dBackward",
            (
                "aclnn_avg_pool2d_backward_get_workspace",
                "aclnn_avg_pool2d_backward",
            ),
            (
                1,
                2,
                3,
                (1, 3, 4, 4),
                (48, 16, 4, 1),
                (1, 3, 8, 8),
                (192, 64, 8, 1),
                (1, 3, 8, 8),
                (192, 64, 8, 1),
                (2, 2),
                (2, 2),
                (0, 0),
                False,
                True,
                None,
                "float16",
            ),
            "four_tensor_three_int_arrays_two_bools_int64_int8_op",
        ),
        (
            "avg_pool3d_backward",
            "AvgPool3dBackward",
            (
                "aclnn_avg_pool3d_backward_get_workspace",
                "aclnn_avg_pool3d_backward",
            ),
            (
                1,
                2,
                3,
                (1, 3, 4, 4, 4),
                (192, 64, 16, 4, 1),
                (1, 3, 8, 8, 8),
                (1536, 512, 64, 8, 1),
                (1, 3, 8, 8, 8),
                (1536, 512, 64, 8, 1),
                (2, 2, 2),
                (2, 2, 2),
                (0, 0, 0),
                False,
                True,
                None,
                "float16",
            ),
            "four_tensor_three_int_arrays_two_bools_int64_int8_op",
        ),
        (
            "max_pool3d_with_argmax_backward",
            "MaxPool3dWithArgmaxBackward",
            (
                "aclnn_max_pool3d_with_argmax_backward_get_workspace",
                "aclnn_max_pool3d_with_argmax_backward",
            ),
            (
                1,
                2,
                3,
                4,
                (1, 3, 4, 4, 4),
                (192, 64, 16, 4, 1),
                (1, 3, 8, 8, 8),
                (1536, 512, 64, 8, 1),
                (1, 3, 4, 4, 4),
                (192, 64, 16, 4, 1),
                (1, 3, 8, 8, 8),
                (1536, 512, 64, 8, 1),
                "float16",
                (2, 2, 2),
                (2, 2, 2),
                (0, 0, 0),
                (1, 1, 1),
                False,
            ),
            "four_tensor_four_int_arrays_bool_op",
        ),
        (
            "adaptive_max_pool2d_backward",
            "AdaptiveMaxPool2dBackward",
            (
                "aclnn_adaptive_max_pool2d_backward_get_workspace",
                "aclnn_adaptive_max_pool2d_backward",
            ),
            (
                1,
                2,
                3,
                4,
                (1, 3, 4, 4),
                (48, 16, 4, 1),
                (1, 3, 8, 8),
                (192, 64, 8, 1),
                (1, 3, 4, 4),
                (48, 16, 4, 1),
                (1, 3, 8, 8),
                (192, 64, 8, 1),
                "float16",
            ),
            "four_tensor_op",
        ),
    ],
)
def test_remaining_backward_pooling_wrappers_use_native_ffi(
    monkeypatch, wrapper_name, ffi_name, binding_attrs, call_args, helper_name
):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(*call_args, runtime)

    assert ("resolve_op", ffi_name) in calls
    assert (helper_name, f"getws:{ffi_name}", f"exec:{ffi_name}") in calls


@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "binding_attrs", "call_args", "helper_name"),
    [
        (
            "embedding_dense_backward",
            "EmbeddingDenseBackward",
            (
                "aclnn_embedding_dense_backward_get_workspace",
                "aclnn_embedding_dense_backward",
            ),
            (
                1,
                2,
                3,
                (4, 8),
                (8, 1),
                (4,),
                (1,),
                (16, 8),
                (8, 1),
                "float16",
                "int64",
                16,
                None,
                False,
            ),
            "two_tensor_two_ints_bool_mixed_fmt_op",
        ),
        (
            "unfold_grad",
            "UnfoldGrad",
            (
                "aclnn_unfold_grad_get_workspace",
                "aclnn_unfold_grad",
            ),
            (
                1,
                2,
                (4, 3),
                (3, 1),
                (6, 4),
                (4, 1),
                (6,),
                0,
                3,
                1,
                "float16",
            ),
            "tensor_int_array_three_ints_op",
        ),
        (
            "repeat_interleave_grad",
            "RepeatInterleaveGrad",
            (
                "aclnn_repeat_interleave_grad_get_workspace",
                "aclnn_repeat_interleave_grad",
            ),
            (
                1,
                2,
                3,
                (6,),
                (1,),
                "float16",
                (3,),
                (1,),
                (3,),
                (1,),
                "float16",
                0,
            ),
            "two_tensor_ints_bool_op",
        ),
    ],
)
def test_remaining_backward_misc_wrappers_use_native_ffi(
    monkeypatch, wrapper_name, ffi_name, binding_attrs, call_args, helper_name
):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(*call_args, runtime)

    assert ("resolve_op", ffi_name) in calls
    assert (helper_name, f"getws:{ffi_name}", f"exec:{ffi_name}") in calls


@pytest.mark.parametrize(
    ("wrapper_name", "ffi_name", "binding_attrs", "call_args", "helper_name"),
    [
        (
            "layer_norm_backward",
            "LayerNormBackward",
            (
                "aclnn_layer_norm_backward_get_workspace",
                "aclnn_layer_norm_backward",
            ),
            (
                1, 2, 3, 4, 5, 6, 7, 8, 9,
                (2, 3), (3, 1),
                (2, 1), (1, 1),
                (3,), (1,),
                (3,), (1,),
                (3,),
                "float16",
            ),
            "layer_norm_backward_op",
        ),
        (
            "convolution_backward",
            "ConvolutionBackward",
            (
                "aclnn_convolution_backward_get_workspace",
                "aclnn_convolution_backward",
            ),
            (
                1, 2, 3,
                (1, 4, 8, 8), (256, 64, 8, 1),
                (1, 3, 10, 10), (300, 100, 10, 1),
                (4, 3, 3, 3), (27, 9, 3, 1),
                "float16",
                (4,),
                (1, 1), (1, 1), (1, 1), False, (0, 0), 1,
                (True, True, True),
                4, 5, 6,
                (1, 3, 10, 10), (300, 100, 10, 1),
                (4, 3, 3, 3), (27, 9, 3, 1),
                (4,), (1,),
            ),
            "convolution_backward_op",
        ),
        (
            "batch_norm_backward",
            "BatchNormBackward",
            (
                "aclnn_batch_norm_backward_get_workspace",
                "aclnn_batch_norm_backward",
            ),
            (
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                (1, 3, 4, 4), (48, 16, 4, 1),
                (1, 3, 4, 4), (48, 16, 4, 1),
                (3,), (1,),
                (3,), (1,), (3,), (1,),
                (3,), (1,), (3,), (1,),
                (1, 3, 4, 4), (48, 16, 4, 1),
                (3,), (1,), (3,), (1,),
                True, 1e-5, (True, True, True), "float16",
            ),
            "batch_norm_backward_op",
        ),
        (
            "rms_norm_grad",
            "RmsNormGrad",
            (
                "aclnn_rms_norm_grad_get_workspace",
                "aclnn_rms_norm_grad",
            ),
            (
                1, 2, 3, 4, 5, 6,
                (2, 3), (3, 1), (2, 3), (3, 1),
                (2, 1), (1, 1),
                (3,), (1,),
                (2, 3), (3, 1), (3,), (1,),
                "float16",
            ),
            "rms_norm_grad_op",
        ),
        (
            "grid_sampler2d_backward",
            "GridSampler2DBackward",
            (
                "aclnn_grid_sampler2d_backward_get_workspace",
                "aclnn_grid_sampler2d_backward",
            ),
            (
                1, 2, 3, 4, 5,
                (1, 3, 4, 4), (48, 16, 4, 1),
                (1, 3, 8, 8), (192, 64, 8, 1),
                (1, 4, 4, 2), (32, 8, 2, 1),
                (1, 3, 8, 8), (192, 64, 8, 1),
                (1, 4, 4, 2), (32, 8, 2, 1),
                0, 0, False, True, True,
                "float16",
            ),
            "grid_sampler2d_backward_op",
        ),
        (
            "group_norm_backward",
            "GroupNormBackward",
            (
                "aclnn_group_norm_backward_get_workspace",
                "aclnn_group_norm_backward",
            ),
            (
                1, 2, 3, 4, 5, 6, 7, 8,
                (2, 4, 3, 3), (36, 9, 3, 1),
                (2, 4, 3, 3), (36, 9, 3, 1),
                (2, 2), (2, 1), (2, 2), (2, 1),
                (4,), (1,),
                (2, 4, 3, 3), (36, 9, 3, 1),
                (4,), (1,),
                (4,), (1,),
                2, 4, 9, 2,
                (True, True, True),
                "float16",
            ),
            "group_norm_backward_op",
        ),
    ],
)
def test_remaining_backward_complex_wrappers_use_native_ffi(
    monkeypatch, wrapper_name, ffi_name, binding_attrs, call_args, helper_name
):
    calls = _install_native_ffi_env(monkeypatch)
    fake_bindings = type("_FakeBindings", (), {name: object() for name in binding_attrs})()
    monkeypatch.setattr(aclnn, "get_bindings", lambda: fake_bindings)

    runtime = _FakeRuntime()
    getattr(aclnn, wrapper_name)(*call_args, runtime)

    assert ("resolve_op", ffi_name) in calls
    assert (helper_name, f"getws:{ffi_name}", f"exec:{ffi_name}") in calls
