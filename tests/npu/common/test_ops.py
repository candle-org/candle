import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import candle as torch
from candle.nn import functional as F


def test_npu_add():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([1.0, 2.0]).to("npu")
    y = torch.tensor([3.0, 4.0]).to("npu")
    z = torch.add(x, y).to("cpu")
    assert z.storage().data.tolist() == [4.0, 6.0]


def test_npu_tensor_empty_list_creates_zero_numel_tensor():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    x = torch.tensor([], device="npu", dtype=torch.float16)

    assert x.device.type == "npu"
    assert x.dtype == torch.float16
    assert x.shape == (0,)
    assert x.numel() == 0


def test_npu_zero_numel_creation_and_clone():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    for factory in (torch.empty, torch.zeros, torch.ones):
        x = factory((2, 0), device="npu", dtype=torch.float16)
        assert x.device.type == "npu"
        assert x.dtype == torch.float16
        assert x.shape == (2, 0)
        assert x.numel() == 0

        cloned = x.clone()
        assert cloned.device.type == "npu"
        assert cloned.dtype == torch.float16
        assert cloned.shape == (2, 0)
        assert cloned.numel() == 0





def test_npu_matmul_2d():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu", dtype=torch.float16)
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="npu", dtype=torch.float16)
    out = torch.matmul(a, b)
    assert out.device.type == "npu"
    assert np.allclose(out.to("cpu").numpy(), np.matmul(a.to("cpu").numpy(), b.to("cpu").numpy()))


def test_npu_matmul_1d_2d():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([1.0, 2.0, 3.0], device="npu", dtype=torch.float16)
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="npu", dtype=torch.float16)
    out = torch.matmul(a, b)
    assert out.shape == (2,)
    assert np.allclose(out.to("cpu").numpy(), np.matmul(a.to("cpu").numpy(), b.to("cpu").numpy()))


def test_npu_matmul_batched_broadcast():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor(
        np.arange(2 * 1 * 2 * 3, dtype=np.float16).reshape(2, 1, 2, 3),
        device="npu",
        dtype=torch.float16,
    )
    b = torch.tensor(
        np.arange(1 * 4 * 3 * 5, dtype=np.float16).reshape(1, 4, 3, 5),
        device="npu",
        dtype=torch.float16,
    )
    out = torch.matmul(a, b)
    assert out.shape == (2, 4, 2, 5)
    assert np.allclose(out.to("cpu").numpy(), np.matmul(a.to("cpu").numpy(), b.to("cpu").numpy()))


def test_npu_matmul_noncontig_batched_input():
    """aclnnMatmul rejects batched non-row-major inputs with workspace error 561103.

    Repro: ``a.transpose(0, 1)`` produces a contiguous-flag=False (3-D) tensor; the
    raw aclnnMatmul call returns ``GetWorkspaceSize failed: 561103``. The Python
    wrapper must contiguify on-device before dispatch.
    """
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    base = torch.randn(4, 16, 64, device="npu", dtype=torch.float16)
    a = base.transpose(0, 1)
    assert not a.is_contiguous()
    b = torch.randn(64, 64, device="npu", dtype=torch.float16)
    out = torch.matmul(a, b)
    expected = np.matmul(a.contiguous().to("cpu").numpy(), b.to("cpu").numpy())
    assert out.shape == (16, 4, 64)
    assert np.allclose(out.to("cpu").numpy(), expected, atol=1e-2, rtol=1e-2)


def test_npu_matmul_attention_rhs_transpose_skips_contiguous(monkeypatch):
    """Rank-4 attention matmul should avoid materializing K.transpose(-2, -1)."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    import candle._backends.npu.ops.linalg as npu_linalg

    calls = {"contiguous": 0}
    original_contiguous = npu_linalg.contiguous

    def wrapped_contiguous(tensor, *args, **kwargs):
        calls["contiguous"] += 1
        return original_contiguous(tensor, *args, **kwargs)

    monkeypatch.setattr(npu_linalg, "contiguous", wrapped_contiguous)
    q = torch.randn(1, 2, 8, 16, device="npu", dtype=torch.float16)
    k = torch.randn(1, 2, 8, 16, device="npu", dtype=torch.float16)
    kt = k.transpose(-2, -1)

    out = torch.matmul(q, kt)
    torch.npu.synchronize()

    expected = np.matmul(q.to("cpu").numpy(), kt.to("cpu").numpy())
    assert out.shape == (1, 2, 8, 8)
    assert np.allclose(out.to("cpu").numpy(), expected, atol=1e-2, rtol=1e-2)
    assert calls["contiguous"] == 0


def test_npu_rank3_linear_uses_cython_fast_path(monkeypatch):
    """Contiguous rank-3 linear training should avoid Python linalg.matmul wrappers."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    import candle._backends.npu.ops.linalg as npu_linalg
    import candle.nn.functional as nn_functional

    calls = {"matmul": 0, "py_linear": 0}
    original_matmul = npu_linalg.aclnn.matmul
    original_py_linear = nn_functional._py_linear

    def wrapped_matmul(*args, **kwargs):
        calls["matmul"] += 1
        return original_matmul(*args, **kwargs)

    def wrapped_py_linear(*args, **kwargs):
        calls["py_linear"] += 1
        return original_py_linear(*args, **kwargs)

    monkeypatch.setattr(npu_linalg.aclnn, "matmul", wrapped_matmul)
    monkeypatch.setattr(nn_functional, "_py_linear", wrapped_py_linear)
    x = torch.randn(2, 4, 8, device="npu", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(12, 8, device="npu", dtype=torch.float16, requires_grad=True)
    bias = torch.randn(12, device="npu", dtype=torch.float16, requires_grad=True)

    out = F.linear(x, weight, bias)
    out.sum().backward()
    torch.npu.synchronize()

    assert out.shape == (2, 4, 12)
    assert x.grad is not None and x.grad.device.type == "npu"
    assert weight.grad is not None and weight.grad.device.type == "npu"
    assert bias.grad is not None and bias.grad.device.type == "npu"

    x_np = x.to("cpu").numpy()
    weight_np = weight.to("cpu").numpy()
    expected_x_grad = np.broadcast_to(weight_np.sum(axis=0), x.shape)
    expected_weight_grad = np.broadcast_to(x_np.reshape(-1, 8).sum(axis=0), weight.shape)
    expected_bias_grad = np.full(bias.shape, x.shape[0] * x.shape[1], dtype=np.float16)
    np.testing.assert_allclose(x.grad.to("cpu").numpy(), expected_x_grad, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(weight.grad.to("cpu").numpy(), expected_weight_grad, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(bias.grad.to("cpu").numpy(), expected_bias_grad, atol=1e-2, rtol=1e-2)
    assert calls == {"matmul": 0, "py_linear": 0}

@pytest.mark.parametrize(
    "op_name, numpy_fn",
    [
        ("abs", np.abs),
        ("neg", np.negative),
        ("exp", np.exp),
        ("log", np.log),
        ("sqrt", np.sqrt),
        ("rsqrt", lambda x: 1.0 / np.sqrt(x)),
        ("sin", np.sin),
        ("cos", np.cos),
        ("tan", np.tan),
        ("tanh", np.tanh),
        ("sigmoid", lambda x: 1.0 / (1.0 + np.exp(-x))),
        ("ceil", np.ceil),
        ("floor", np.floor),
        ("round", np.round),
        ("trunc", np.trunc),
        ("frac", lambda x: x - np.trunc(x)),
        ("log2", np.log2),
        ("log10", np.log10),
        ("exp2", np.exp2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_unary_ops(op_name, numpy_fn, dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    if op_name in {"log", "log2", "log10", "sqrt", "rsqrt"}:
        data = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    else:
        data = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)

    x = torch.tensor(data, device="npu", dtype=dtype)
    op = getattr(torch, op_name)
    out = op(x)
    expected = numpy_fn(data).astype(np.float32)
    assert out.device.type == "npu"
    assert np.allclose(
        out.to("cpu").numpy().astype(np.float32),
        expected,
        atol=1e-3,
        rtol=1e-3,
    )

@pytest.mark.parametrize(
    "op_name, numpy_fn",
    [
        ("cosh", np.cosh),
        ("sinh", np.sinh),
        ("erf", lambda x: np.vectorize(math.erf)(x)),
        ("erfc", lambda x: np.vectorize(math.erfc)(x)),
        ("softplus", lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_unary_ops_extra(op_name, numpy_fn, dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)
    op = getattr(torch, op_name)
    out = op(x)
    expected = numpy_fn(data).astype(np.float32)
    assert out.device.type == "npu"
    assert np.allclose(
        out.to("cpu").numpy().astype(np.float32),
        expected,
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_clamp_ops(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)
    out = torch.clamp(x, -1.0, 1.0)
    out_min = torch.clamp_min(x, -1.0)
    out_max = torch.clamp_max(x, 1.0)
    assert np.allclose(out.to("cpu").numpy(), np.clip(data, -1.0, 1.0).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(out_min.to("cpu").numpy(), np.clip(data, -1.0, None).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(out_max.to("cpu").numpy(), np.clip(data, None, 1.0).astype(np.float32), atol=1e-3, rtol=1e-3)

    tensor_data = np.array([[-2.0, -0.5], [0.5, 2.0]], dtype=np.float32)
    min_data = np.array([[-1.0], [0.0]], dtype=np.float32)
    max_data = np.array([[0.25, 1.0]], dtype=np.float32)
    tensor_x = torch.tensor(tensor_data, device="npu", dtype=dtype)
    tensor_min = torch.tensor(min_data, device="npu", dtype=dtype)
    tensor_max = torch.tensor(max_data, device="npu", dtype=dtype)
    tensor_out = torch.clamp(tensor_x, tensor_min, tensor_max)
    tensor_out_min = torch.clamp_min(tensor_x, tensor_min)
    tensor_out_max = torch.clamp_max(tensor_x, tensor_max)
    expected = np.clip(tensor_data, min_data, max_data).astype(np.float32)
    expected_min = np.clip(tensor_data, min_data, None).astype(np.float32)
    expected_max = np.clip(tensor_data, None, max_data).astype(np.float32)
    assert np.allclose(tensor_out.to("cpu").numpy(), expected, atol=1e-3, rtol=1e-3)
    assert np.allclose(tensor_out_min.to("cpu").numpy(), expected_min, atol=1e-3, rtol=1e-3)
    assert np.allclose(tensor_out_max.to("cpu").numpy(), expected_max, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_relu6_hardtanh(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([-2.0, -0.5, 0.5, 2.0, 7.0], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)
    relu6 = torch.relu6(x)
    hardtanh = torch.hardtanh(x, -1.0, 1.0)
    assert np.allclose(relu6.to("cpu").numpy(), np.clip(data, 0.0, 6.0).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(hardtanh.to("cpu").numpy(), np.clip(data, -1.0, 1.0).astype(np.float32), atol=1e-3, rtol=1e-3)

def test_npu_isfinite_isinf_isnan_signbit():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([0.0, 1.0, -1.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=torch.float32)
    isfinite = torch.isfinite(x)
    isinf = torch.isinf(x)
    isnan = torch.isnan(x)
    signbit = torch.signbit(x)
    assert isfinite.dtype == torch.bool
    assert isinf.dtype == torch.bool
    assert isnan.dtype == torch.bool
    assert signbit.dtype == torch.bool
    assert np.array_equal(isfinite.to("cpu").numpy(), np.isfinite(data))
    assert np.array_equal(isinf.to("cpu").numpy(), np.isinf(data))
    assert np.array_equal(isnan.to("cpu").numpy(), np.isnan(data))
    assert np.array_equal(signbit.to("cpu").numpy(), np.signbit(data))


def test_npu_amin_amax():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 0.5]], device="npu")
    expected_min = np.amin(x.to("cpu").numpy(), axis=1)
    expected_max = np.amax(x.to("cpu").numpy(), axis=1)
    np.testing.assert_allclose(torch.amin(x, dim=1).to("cpu").numpy(), expected_min)
    np.testing.assert_allclose(torch.amax(x, dim=1).to("cpu").numpy(), expected_max)


def test_npu_argmax_argmin():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], device="npu")
    expected_max = np.argmax(x.to("cpu").numpy(), axis=1)
    expected_min = np.argmin(x.to("cpu").numpy(), axis=1)
    np.testing.assert_array_equal(torch.argmax(x, dim=1).to("cpu").numpy(), expected_max)
    np.testing.assert_array_equal(torch.argmin(x, dim=1).to("cpu").numpy(), expected_min)
    np.testing.assert_array_equal(
        torch.argmax(x, dim=1, keepdim=True).to("cpu").numpy(),
        expected_max.reshape(2, 1),
    )
    np.testing.assert_array_equal(
        torch.argmin(x, dim=1, keepdim=True).to("cpu").numpy(),
        expected_min.reshape(2, 1),
    )


def test_npu_all_any():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[True, False], [True, True]], device="npu", dtype=torch.bool)
    expected_all = np.all(x.to("cpu").numpy(), axis=1)
    expected_any = np.any(x.to("cpu").numpy(), axis=1)
    np.testing.assert_array_equal(torch.all(x, dim=1).to("cpu").numpy(), expected_all)
    np.testing.assert_array_equal(torch.any(x, dim=1).to("cpu").numpy(), expected_any)
    expected_keep = np.all(x.to("cpu").numpy(), axis=1, keepdims=True)
    np.testing.assert_array_equal(torch.all(x, dim=1, keepdim=True).to("cpu").numpy(), expected_keep)


def test_npu_count_nonzero():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 3.0]], device="npu")
    expected = np.count_nonzero(x.to("cpu").numpy(), axis=1)
    np.testing.assert_array_equal(torch.count_nonzero(x, dim=1).to("cpu").numpy(), expected)
    expected_keep = np.count_nonzero(x.to("cpu").numpy(), axis=1, keepdims=True)
    np.testing.assert_array_equal(
        torch.count_nonzero(x, dim=1, keepdim=True).to("cpu").numpy(),
        expected_keep,
    )


def test_npu_split_stack_family():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    a = torch.tensor([1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="npu")
    out = torch.chunk(x, 2)
    assert len(out) == 2
    np.testing.assert_allclose(out[0].to("cpu").numpy(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(out[1].to("cpu").numpy(), np.array([3.0, 4.0]))


def test_npu_view_to_cpu_preserves_offset_window():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="npu")
    view = torch.chunk(x, 2)[1]

    assert view.offset != 0
    np.testing.assert_allclose(view.to("cpu").numpy(), np.array([3.0, 4.0]))


def test_npu_grad_split_uses_native_materializing_copy(monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    import candle._backends.npu.ops.shape as npu_shape

    if not npu_shape.aclnn.split_with_size_symbols_ok():
        pytest.skip("aclnnSplitWithSize symbols not available")

    calls = {"split": 0, "slice": 0}
    original_split = npu_shape.aclnn.split_with_size
    original_slice = npu_shape.aclnn.slice_op

    def wrapped_split(*args, **kwargs):
        calls["split"] += 1
        return original_split(*args, **kwargs)

    def wrapped_slice(*args, **kwargs):
        calls["slice"] += 1
        return original_slice(*args, **kwargs)

    monkeypatch.setattr(npu_shape.aclnn, "split_with_size", wrapped_split)
    monkeypatch.setattr(npu_shape.aclnn, "slice_op", wrapped_slice)

    x = torch.tensor(
        np.arange(2 * 3 * 12, dtype=np.float32).reshape(2, 3, 12),
        device="npu",
        dtype=torch.float16,
    )
    x.requires_grad = True
    q, k, v = torch.split(x, 4, dim=-1)
    torch.npu.synchronize()

    assert calls == {"split": 1, "slice": 0}
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    np.testing.assert_allclose(k.to("cpu").numpy(), x.to("cpu").numpy()[..., 4:8])


def test_npu_hstack():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    a = torch.tensor([1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")
    expected = np.hstack([a.to("cpu").numpy(), b.to("cpu").numpy()])
    np.testing.assert_allclose(torch.hstack([a, b]).to("cpu").numpy(), expected)
    torch.npu.synchronize()


def test_npu_vstack_row_stack():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

def test_npu_column_stack():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

def test_npu_dstack():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

def test_npu_hsplit():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="npu")
    out = torch.hsplit(x, 2)
    assert len(out) == 2
    np.testing.assert_allclose(out[0].to("cpu").numpy(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(out[1].to("cpu").numpy(), np.array([3.0, 4.0]))
    torch.npu.synchronize()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_pow(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    base = torch.tensor([1.0, 2.0, 3.0], device="npu", dtype=dtype)
    exp = torch.tensor([2.0, 3.0, 0.5], device="npu", dtype=dtype)
    out = torch.pow(base, exp)
    expected = np.power(base.to("cpu").numpy(), exp.to("cpu").numpy())
    assert np.allclose(
        out.to("cpu").numpy().astype(np.float32),
        expected.astype(np.float32),
        atol=1e-3,
        rtol=1e-3,
    )

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_fmin_matches_torch_npu_cpu_fallback(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    base = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    other = np.array([2.0, 0.5, -0.5, -2.0], dtype=np.float32)
    x = torch.tensor(base, device="npu", dtype=dtype)
    y = torch.tensor(other, device="npu", dtype=dtype)

    out = torch.fmin(x, y)

    assert out.device.type == "npu"
    assert out.dtype == dtype
    expected = np.fmin(base.astype(np.float32), other.astype(np.float32)).astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_hypot_matches_torch_npu_cpu_fallback(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    base = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    other = np.array([2.0, 0.5, -0.5, -2.0], dtype=np.float32)
    x = torch.tensor(base, device="npu", dtype=dtype)
    y = torch.tensor(other, device="npu", dtype=dtype)

    out = torch.hypot(x, y)

    assert out.device.type == "npu"
    assert out.dtype == dtype
    expected = np.hypot(base.astype(np.float32), other.astype(np.float32)).astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_elementwise_batch2(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    base = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    x = torch.tensor(base, device="npu", dtype=dtype)
    y = torch.tensor(base[::-1], device="npu", dtype=dtype)

    expected_asin = np.arcsin(base).astype(np.float32)
    expected_acos = np.arccos(base).astype(np.float32)
    out_asin = torch.asin(x).to("cpu").numpy().astype(np.float32)
    out_acos = torch.acos(x).to("cpu").numpy().astype(np.float32)
    # Only check values in the valid domain [-1, 1]; out-of-domain behavior
    # (nan vs hardware-defined saturation) varies across NPU SoC models.
    valid = np.abs(base) <= 1.0
    assert np.allclose(out_asin[valid], expected_asin[valid], atol=1e-3, rtol=1e-3)
    assert np.allclose(out_acos[valid], expected_acos[valid], atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.atan(x).to("cpu").numpy(), np.arctan(base).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.atan2(x, y).to("cpu").numpy(), np.arctan2(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.asinh(x).to("cpu").numpy(), np.arcsinh(base).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.acosh(torch.abs(x) + 1.5).to("cpu").numpy(), np.arccosh(np.abs(base) + 1.5).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.atanh(x * 0.25).to("cpu").numpy(), np.arctanh(base * 0.25).astype(np.float32), atol=1e-3, rtol=1e-3)

    assert np.allclose(torch.addcmul(x, x, y).to("cpu").numpy(), (base + base * base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.addcdiv(x, x, y).to("cpu").numpy(), (base + base / base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)

    # CANN 8.5.0 had a repeated 910B native sequence instability around
    # maximum/where/logaddexp, but CANN 9.0.0 torch_npu is stable across the
    # same boundary. Keep these common checks as single-call semantic guards;
    # the 910B watchlist covers repeated stability.
    assert np.allclose(torch.logaddexp(x, y).to("cpu").numpy(), np.logaddexp(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.logaddexp2(x, y).to("cpu").numpy(), np.logaddexp2(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.hypot(x, y).to("cpu").numpy(), np.hypot(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)

    assert np.allclose(torch.remainder(x, y).to("cpu").numpy(), np.remainder(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.fmod(x, y).to("cpu").numpy(), np.fmod(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)

    assert np.allclose(torch.fmin(x, y).to("cpu").numpy(), np.fmin(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.fmax(x, y).to("cpu").numpy(), np.fmax(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.min(x, y).to("cpu").numpy(), np.minimum(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)
    assert np.allclose(torch.max(x, y).to("cpu").numpy(), np.maximum(base, base[::-1]).astype(np.float32), atol=1e-3, rtol=1e-3)

    where_cond = torch.tensor([True, False, True, False], device="npu")
    where_out = torch.where(where_cond, x, y)
    expected_where = np.where(np.array([True, False, True, False]), base, base[::-1]).astype(np.float32)
    assert np.allclose(where_out.to("cpu").numpy().astype(np.float32), expected_where, atol=1e-3, rtol=1e-3)

    lerp_out = torch.lerp(x, y, 0.25).to("cpu").numpy()
    expected_lerp = (base + 0.25 * (base[::-1] - base)).astype(np.float32)
    assert np.allclose(lerp_out, expected_lerp, atol=1e-3, rtol=1e-3)

    assert torch.allclose(x, y) == np.allclose(base, base[::-1])
    isclose_out = torch.isclose(x, y).to("cpu").numpy()
    assert np.all(isclose_out == np.isclose(base, base[::-1]))
    assert torch.equal(x, y) == np.array_equal(base, base[::-1])


    scalar_base = torch.tensor(base, device="npu", dtype=dtype)
    scalar_out = torch.pow(scalar_base, 2.0)
    scalar_expected = np.power(base, 2.0)
    assert np.allclose(
        scalar_out.to("cpu").numpy().astype(np.float32),
        scalar_expected.astype(np.float32),
        atol=1e-3,
        rtol=1e-3,
    )


def test_npu_pow_scalar_two_uses_mul_fast_path(monkeypatch):
    """NPU pow(x, 2.0) should use x*x instead of slow aclnnPowTensorScalar."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    import candle._backends.npu.ops.math as npu_math

    calls = {"pow_scalar": 0, "mul": 0}
    original_pow_scalar = npu_math._fast_pow_tensor_scalar_impl
    original_mul = npu_math._fast_mul_impl

    def counted_pow_scalar(*args, **kwargs):
        calls["pow_scalar"] += 1
        return original_pow_scalar(*args, **kwargs)

    def counted_mul(*args, **kwargs):
        calls["mul"] += 1
        return original_mul(*args, **kwargs)

    monkeypatch.setattr(npu_math, "_fast_pow_tensor_scalar_impl", counted_pow_scalar)
    monkeypatch.setattr(npu_math, "_fast_mul_impl", counted_mul)

    x = torch.tensor([1.0, 2.0, 3.0], device="npu", dtype=torch.float32)
    result = torch.pow(x, 2.0)

    assert result.device.type == "npu"
    assert calls == {"pow_scalar": 0, "mul": 1}
    np.testing.assert_allclose(result.to("cpu").numpy(), np.array([1.0, 4.0, 9.0], dtype=np.float32))


def test_npu_dtype_cast_uses_cython_fast_path(monkeypatch):
    """Hot NPU dtype casts should bypass the Python aclnn.cast wrapper."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    import candle._backends.npu.ops._helpers as npu_helpers

    calls = {"cast": 0}
    original_cast = npu_helpers.aclnn.cast

    def counted_cast(*args, **kwargs):
        calls["cast"] += 1
        return original_cast(*args, **kwargs)

    monkeypatch.setattr(npu_helpers.aclnn, "cast", counted_cast)

    x = torch.tensor([1.25, -2.5, 3.75], device="npu", dtype=torch.float16)
    result = x.to(torch.float32)

    assert result.device.type == "npu"
    assert result.dtype == torch.float32
    assert calls == {"cast": 0}
    np.testing.assert_allclose(
        result.to("cpu").numpy(),
        np.array([1.25, -2.5, 3.75], dtype=np.float32),
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "expr, expected",
    [
        (lambda x: x + 0.5, np.array([1.5, 2.5, 3.5], dtype=np.float32)),
        (lambda x: x * 2.0, np.array([2.0, 4.0, 6.0], dtype=np.float32)),
        (lambda x: x - 0.25, np.array([0.75, 1.75, 2.75], dtype=np.float32)),
        (lambda x: x / 2.0, np.array([0.5, 1.0, 1.5], dtype=np.float32)),
    ],
)
def test_npu_scalar_binary_ops_use_cython_only_path(monkeypatch, expr, expected):
    """NPU tensor/scalar arithmetic must bypass Python backend scalar wrappers."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    import candle._backends.npu.aclnn as npu_aclnn
    import candle._backends.npu.ops.math as npu_math

    calls = {
        "scalar_to_tensor": 0,
        "add_scalar": 0,
        "mul_scalar": 0,
        "sub_scalar": 0,
    }

    def forbidden_scalar_to_tensor(*args, **kwargs):
        calls["scalar_to_tensor"] += 1
        raise AssertionError("NPU scalar binary ops must not materialize scalar tensors on device")

    def forbidden_add_scalar(*args, **kwargs):
        calls["add_scalar"] += 1
        raise AssertionError("NPU add scalar must route through Cython _C, not Python aclnn.add_scalar")

    def forbidden_mul_scalar(*args, **kwargs):
        calls["mul_scalar"] += 1
        raise AssertionError("NPU mul scalar must route through Cython _C, not Python aclnn.mul_scalar")

    def forbidden_sub_scalar(*args, **kwargs):
        calls["sub_scalar"] += 1
        raise AssertionError("NPU sub scalar must route through Cython _C, not Python aclnn.sub_scalar")

    monkeypatch.setattr(npu_math, "_scalar_to_npu_tensor", forbidden_scalar_to_tensor)
    monkeypatch.setattr(npu_aclnn, "add_scalar", forbidden_add_scalar)
    monkeypatch.setattr(npu_aclnn, "mul_scalar", forbidden_mul_scalar)
    monkeypatch.setattr(npu_aclnn, "sub_scalar", forbidden_sub_scalar)

    x = torch.tensor([1.0, 2.0, 3.0], device="npu", dtype=torch.float32)
    result = expr(x)

    assert result.device.type == "npu"
    assert calls == {"scalar_to_tensor": 0, "add_scalar": 0, "mul_scalar": 0, "sub_scalar": 0}
    np.testing.assert_allclose(result.to("cpu").numpy(), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "expr, expected, expected_grad",
    [
        (lambda x: x + 0.5, np.array([1.5, 2.5, 3.5], dtype=np.float32), np.ones(3, dtype=np.float32)),
        (lambda x: x * 2.0, np.array([2.0, 4.0, 6.0], dtype=np.float32), np.full(3, 2.0, dtype=np.float32)),
        (lambda x: x - 0.25, np.array([0.75, 1.75, 2.75], dtype=np.float32), np.ones(3, dtype=np.float32)),
        (lambda x: x / 2.0, np.array([0.5, 1.0, 1.5], dtype=np.float32), np.full(3, 0.5, dtype=np.float32)),
    ],
)
def test_npu_scalar_binary_ops_with_grad_use_cython_only_path(monkeypatch, expr, expected, expected_grad):
    """NPU tensor/scalar autograd must not fall back through Python scalar tensor materialization."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    import candle._backends.npu.aclnn as npu_aclnn
    import candle._backends.npu.ops.math as npu_math

    calls = {
        "scalar_to_tensor": 0,
        "add_scalar": 0,
        "mul_scalar": 0,
        "sub_scalar": 0,
    }

    def forbidden_scalar_to_tensor(*args, **kwargs):
        calls["scalar_to_tensor"] += 1
        raise AssertionError("NPU scalar binary autograd must not materialize scalar tensors on device")

    def forbidden_add_scalar(*args, **kwargs):
        calls["add_scalar"] += 1
        raise AssertionError("NPU add scalar autograd must route through Cython _C, not Python aclnn.add_scalar")

    def forbidden_mul_scalar(*args, **kwargs):
        calls["mul_scalar"] += 1
        raise AssertionError("NPU mul scalar autograd must route through Cython _C, not Python aclnn.mul_scalar")

    def forbidden_sub_scalar(*args, **kwargs):
        calls["sub_scalar"] += 1
        raise AssertionError("NPU sub scalar autograd must route through Cython _C, not Python aclnn.sub_scalar")

    monkeypatch.setattr(npu_math, "_scalar_to_npu_tensor", forbidden_scalar_to_tensor)
    monkeypatch.setattr(npu_aclnn, "add_scalar", forbidden_add_scalar)
    monkeypatch.setattr(npu_aclnn, "mul_scalar", forbidden_mul_scalar)
    monkeypatch.setattr(npu_aclnn, "sub_scalar", forbidden_sub_scalar)

    x = torch.tensor([1.0, 2.0, 3.0], device="npu", dtype=torch.float32, requires_grad=True)
    result = expr(x)
    result.sum().backward()

    assert result.device.type == "npu"
    assert x.grad is not None
    assert x.grad.device.type == "npu"
    assert calls == {"scalar_to_tensor": 0, "add_scalar": 0, "mul_scalar": 0, "sub_scalar": 0}
    np.testing.assert_allclose(result.to("cpu").numpy(), expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(x.grad.to("cpu").numpy(), expected_grad, atol=1e-6, rtol=1e-6)


def test_npu_reflected_scalar_binary_ops_use_direct_cython_path(monkeypatch):
    """NPU scalar-left sub/div should use direct Cython tensor paths, not neg+Python dispatch."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    import candle._backends.npu.ops.math as npu_math

    calls = {"neg": 0, "scalar_to_tensor": 0}

    def forbidden_neg(*args, **kwargs):
        calls["neg"] += 1
        raise AssertionError("NPU reflected scalar subtraction must not lower to neg + add")

    def forbidden_scalar_to_tensor(*args, **kwargs):
        calls["scalar_to_tensor"] += 1
        raise AssertionError("NPU reflected scalar ops must not use Python scalar tensor materialization")

    monkeypatch.setattr(npu_math, "_fast_neg_impl", forbidden_neg)
    monkeypatch.setattr(npu_math, "_scalar_to_npu_tensor", forbidden_scalar_to_tensor)

    x = torch.tensor([1.0, 2.0, 4.0], device="npu", dtype=torch.float32)
    out_sub = 10.0 - x
    out_div = 8.0 / x
    torch.npu.synchronize()

    assert calls == {"neg": 0, "scalar_to_tensor": 0}
    np.testing.assert_allclose(out_sub.to("cpu").numpy(), np.array([9.0, 8.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(out_div.to("cpu").numpy(), np.array([8.0, 4.0, 2.0], dtype=np.float32))


def test_npu_reflected_scalar_binary_ops_with_grad_use_direct_cython_path(monkeypatch):
    """NPU scalar-left sub/div autograd should stay on direct Cython paths."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    import candle._backends.npu.ops.math as npu_math

    calls = {"neg": 0, "scalar_to_tensor": 0}

    def forbidden_neg(*args, **kwargs):
        calls["neg"] += 1
        raise AssertionError("NPU reflected scalar subtraction autograd must not lower to neg + add")

    def forbidden_scalar_to_tensor(*args, **kwargs):
        calls["scalar_to_tensor"] += 1
        raise AssertionError("NPU reflected scalar autograd must not use Python scalar tensor materialization")

    monkeypatch.setattr(npu_math, "_fast_neg_impl", forbidden_neg)
    monkeypatch.setattr(npu_math, "_scalar_to_npu_tensor", forbidden_scalar_to_tensor)

    x = torch.tensor([1.0, 2.0, 4.0], device="npu", dtype=torch.float32, requires_grad=True)
    out_sub = 10.0 - x
    out_div = 8.0 / x
    loss = out_sub.sum() + out_div.sum()
    loss.backward()

    assert calls == {"neg": 0, "scalar_to_tensor": 0}
    assert x.grad is not None
    assert x.grad.device.type == "npu"
    np.testing.assert_allclose(out_sub.to("cpu").numpy(), np.array([9.0, 8.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(out_div.to("cpu").numpy(), np.array([8.0, 4.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(
        x.grad.to("cpu").numpy(),
        np.array([-9.0, -3.0, -1.5], dtype=np.float32),
        atol=1e-6,
        rtol=1e-6,
    )


def test_npu_tensor_sub_operator_uses_direct_cython_sub_without_neg_kernel():
    """NPU tensor subtraction operator must not lower to neg + add."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    env = os.environ.copy()
    repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
    env["PYTHONPATH"] = repo_src + os.pathsep + env.get("PYTHONPATH", "")
    script = """
import candle as torch
import candle._backends.npu.ops.math as npu_math
calls = {"neg": 0}
orig = npu_math._fast_neg_impl

def forbidden_neg(*args, **kwargs):
    calls["neg"] += 1
    raise AssertionError("NPU tensor subtraction must route through direct Cython sub, not neg+add")

npu_math._fast_neg_impl = forbidden_neg
x = torch.tensor([4.0, 5.0, 6.0], device="npu", dtype=torch.float32)
y = torch.tensor([1.0, 2.0, 3.0], device="npu", dtype=torch.float32)
try:
    out = x - y
    torch.npu.synchronize()
finally:
    npu_math._fast_neg_impl = orig
assert calls == {"neg": 0}
assert out.device.type == "npu"
assert out.to("cpu").numpy().tolist() == [3.0, 3.0, 3.0]
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_npu_tensor_binary_operators_bypass_functional_wrappers():
    """Hot NPU Tensor operators should route directly to Cython kernels, not Python functional wrappers."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    env = os.environ.copy()
    repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
    env["PYTHONPATH"] = repo_src + os.pathsep + env.get("PYTHONPATH", "")
    script = """
import numpy as np
import candle as torch
import candle._functional as functional

calls = {"add": 0, "mul": 0, "sub": 0, "div": 0}

def forbid(name):
    def inner(*args, **kwargs):
        calls[name] += 1
        raise AssertionError(f"NPU Tensor operator must bypass candle._functional.{name}")
    return inner

functional.add = forbid("add")
functional.mul = forbid("mul")
functional.sub = forbid("sub")
functional.div = forbid("div")

x = torch.tensor([4.0, 5.0, 6.0], device="npu", dtype=torch.float32)
y = torch.tensor([1.0, 2.0, 3.0], device="npu", dtype=torch.float32)
out_add = x + y
out_mul = x * y
out_sub = x - y
out_div = x / y
torch.npu.synchronize()
assert calls == {"add": 0, "mul": 0, "sub": 0, "div": 0}
np.testing.assert_allclose(out_add.to("cpu").numpy(), np.array([5.0, 7.0, 9.0], dtype=np.float32))
np.testing.assert_allclose(out_mul.to("cpu").numpy(), np.array([4.0, 10.0, 18.0], dtype=np.float32))
np.testing.assert_allclose(out_sub.to("cpu").numpy(), np.array([3.0, 3.0, 3.0], dtype=np.float32))
np.testing.assert_allclose(out_div.to("cpu").numpy(), np.array([4.0, 2.5, 2.0], dtype=np.float32))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_npu_scalar_binary_dispatch_uses_cached_tensor_path_by_default():
    """NPU tensor/scalar eager ops should use tensor-tensor PTA-level dispatch overhead."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    x = torch.randn(1024, device="npu", dtype=torch.float16)
    scalar = torch.tensor(0.5, device="npu", dtype=torch.float16)

    def bench(fn, iters=1200, warmup=120):
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - start) * 1e6 / iters

    scalar_us = min(bench(lambda: x + 0.5) for _ in range(3))
    tensor_us = min(bench(lambda: x + scalar) for _ in range(3))

    assert scalar_us <= max(tensor_us * 1.35, tensor_us + 8.0), (
        f"python scalar path {scalar_us:.2f}us should stay near cached tensor path {tensor_us:.2f}us"
    )


def test_npu_add_scalar_inplace_captures_without_host_scalar_copy(monkeypatch):
    """NPU add_(scalar) should be graph-capturable and avoid H2D scalar materialization."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    import candle._backends.npu.ops.math as npu_math

    calls = {"scalar_to_tensor": 0}
    original_scalar_to_tensor = npu_math._scalar_to_npu_tensor

    def counted_scalar_to_tensor(*args, **kwargs):
        calls["scalar_to_tensor"] += 1
        return original_scalar_to_tensor(*args, **kwargs)

    monkeypatch.setattr(npu_math, "_scalar_to_npu_tensor", counted_scalar_to_tensor)

    x = torch.zeros((4,), device="npu", dtype=torch.int64)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        x.add_(1)
    graph.replay()
    torch.npu.synchronize()

    assert calls == {"scalar_to_tensor": 0}
    np.testing.assert_array_equal(x.to("cpu").numpy(), np.full((4,), 1, dtype=np.int64))



def test_npu_scalar_tensor_creation_captures_without_host_copy(monkeypatch):
    """NPU scalar tensor creation should be graph-capturable without H2D copy."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    import candle._backends.npu.creation as npu_creation

    calls = {"h2d": 0}
    original_copy_cpu_to_npu = npu_creation.npu_runtime._copy_cpu_to_npu

    def forbidden_copy_cpu_to_npu(*args, **kwargs):
        calls["h2d"] += 1
        raise AssertionError("graph-captured scalar tensor creation must not use raw H2D copy")

    monkeypatch.setattr(npu_creation.npu_runtime, "_copy_cpu_to_npu", forbidden_copy_cpu_to_npu)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = torch.tensor(float("-inf"), device="npu", dtype=torch.float32)
    graph.replay()
    torch.npu.synchronize()
    monkeypatch.setattr(npu_creation.npu_runtime, "_copy_cpu_to_npu", original_copy_cpu_to_npu)

    assert calls == {"h2d": 0}
    assert out.device.type == "npu"
    assert out.shape == ()
    assert np.isneginf(out.to("cpu").item())



def test_npu_scalar_to_tensor_helper_captures_without_host_copy(monkeypatch):
    """NPU tensor/scalar comparison should be graph-capturable without H2D scalar expansion."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    import candle._backends.npu.ops._helpers as npu_helpers

    calls = {"h2d": 0}
    original_memcpy_h2d = npu_helpers.npu_runtime.memcpy_h2d

    def forbidden_memcpy_h2d(*args, **kwargs):
        calls["h2d"] += 1
        raise AssertionError("graph-captured scalar expansion must not use raw H2D copy")

    monkeypatch.setattr(npu_helpers.npu_runtime, "memcpy_h2d", forbidden_memcpy_h2d)

    mask = torch.ones((4, 4), device="npu", dtype=torch.bool).tril()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = mask == False  # noqa: E712
    graph.replay()
    torch.npu.synchronize()
    monkeypatch.setattr(npu_helpers.npu_runtime, "memcpy_h2d", original_memcpy_h2d)

    assert calls == {"h2d": 0}
    expected = np.triu(np.ones((4, 4), dtype=bool), k=1)
    np.testing.assert_array_equal(out.to("cpu").numpy(), expected)



def test_npu_contiguous_setitem_captures_without_raw_d2d(monkeypatch):
    """NPU setitem into a contiguous view should be graph-capturable without raw D2D memcpy."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    import candle._backends.npu.ops.shape as npu_shape

    calls = {"memcpy_d2d": 0}
    original_memcpy_d2d = npu_shape.npu_runtime.memcpy_d2d

    def forbidden_memcpy_d2d(*args, **kwargs):
        calls["memcpy_d2d"] += 1
        raise AssertionError("graph-captured setitem view copy must not use raw D2D memcpy")

    x = torch.zeros((4,), device="npu", dtype=torch.float32)
    value = torch.tensor(np.array([3.0, 4.0], dtype=np.float32), device="npu")
    monkeypatch.setattr(npu_shape.npu_runtime, "memcpy_d2d", forbidden_memcpy_d2d)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        x[:2] = value
    graph.replay()
    torch.npu.synchronize()
    monkeypatch.setattr(npu_shape.npu_runtime, "memcpy_d2d", original_memcpy_d2d)

    assert calls == {"memcpy_d2d": 0}
    np.testing.assert_allclose(x.to("cpu").numpy(), np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32))



def test_npu_gather_captures_without_host_index_bounds_read(monkeypatch):
    """NPU gather should be graph-capturable without synchronizing index bounds to host."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    import candle._backends.npu.ops.shape as npu_shape

    calls = {"index_read": 0, "scalar_read": 0}

    def forbidden_index_read(*args, **kwargs):
        calls["index_read"] += 1
        raise AssertionError("gather capture path must not read indices on host")

    def forbidden_scalar_read(*args, **kwargs):
        calls["scalar_read"] += 1
        raise AssertionError("gather capture path must not read validation scalars on host")

    monkeypatch.setattr(npu_shape, "_read_index_tensor_to_cpu", forbidden_index_read)
    monkeypatch.setattr(npu_shape, "_read_bool_scalar", forbidden_scalar_read)

    x = torch.tensor(np.arange(12, dtype=np.float32).reshape(3, 4), device="npu")
    index = torch.tensor([[0, 1], [2, 3], [1, 0]], device="npu", dtype=torch.int64)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = torch.gather(x, dim=1, index=index)
    graph.replay()
    torch.npu.synchronize()

    assert calls == {"index_read": 0, "scalar_read": 0}
    expected = np.take_along_axis(np.arange(12, dtype=np.float32).reshape(3, 4),
                                  np.array([[0, 1], [2, 3], [1, 0]], dtype=np.int64), axis=1)
    np.testing.assert_allclose(out.to("cpu").numpy(), expected)



def test_npu_index_select_known_nonnegative_captures_without_host_index_read(monkeypatch):
    """NPU x[:, idx] should capture without reading known nonnegative indices on host."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    import candle._backends.npu.ops.shape as npu_shape

    calls = {"index_read": 0, "scalar_read": 0}

    def forbidden_index_read(*args, **kwargs):
        calls["index_read"] += 1
        raise AssertionError("index_select capture path must not read indices on host")

    def forbidden_scalar_read(*args, **kwargs):
        calls["scalar_read"] += 1
        raise AssertionError("index_select capture path must not read validation scalars on host")

    monkeypatch.setattr(npu_shape, "_read_index_tensor_to_cpu", forbidden_index_read)
    monkeypatch.setattr(npu_shape, "_read_bool_scalar", forbidden_scalar_read)

    x = torch.tensor(np.arange(12, dtype=np.float32).reshape(3, 4), device="npu")
    idx = torch.tensor([0, 2], device="npu", dtype=torch.int64)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = x[:, idx]
    graph.replay()
    torch.npu.synchronize()

    assert calls == {"index_read": 0, "scalar_read": 0}
    np.testing.assert_allclose(out.to("cpu").numpy(), np.array([[0, 2], [4, 6], [8, 10]], dtype=np.float32))



def test_npu_contiguous_size_one_expanded_view_captures_without_raw_d2d(monkeypatch):
    """NPU contiguous() on size-1 expanded views should use ACLNN during graph capture."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    import candle._backends.npu.ops.shape as npu_shape

    calls = {"memcpy_d2d": 0}
    original_memcpy_d2d = npu_shape.npu_runtime.memcpy_d2d

    def forbidden_memcpy_d2d(*args, **kwargs):
        calls["memcpy_d2d"] += 1
        raise AssertionError("graph-captured contiguous view copy must not use raw aclrt D2D memcpy")

    base = torch.arange(0, 8, device="npu", dtype=torch.float32)
    view = base[None, :, None].expand(1, -1, 1)
    monkeypatch.setattr(npu_shape.npu_runtime, "memcpy_d2d", forbidden_memcpy_d2d)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = view.contiguous()
    graph.replay()
    torch.npu.synchronize()
    monkeypatch.setattr(npu_shape.npu_runtime, "memcpy_d2d", original_memcpy_d2d)

    assert calls == {"memcpy_d2d": 0}
    np.testing.assert_allclose(out.to("cpu").numpy(), np.arange(8, dtype=np.float32).reshape(1, 8, 1))


def test_npu_model_dir_probe():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    ok = torch._C._npu_probe_model_dirs()
    assert ok is True


def test_npu_model_dir_selected():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    import os
    path = torch._C._npu_model_dir()
    assert os.path.isdir(path), f"model dir {path} does not exist"


def test_npu_aclnn_available():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    assert torch._C._npu_aclnn_available() is True


def test_aclnn_symbols_present():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    assert torch._C._npu_aclnn_symbols_ok() is True


def test_aclnn_ones_zero_symbols_present():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    assert torch._C._npu_aclnn_ones_zero_ok() is True


def test_npu_add_execute():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")
    out = a + b
    assert out.device.type == "npu"
    assert out.to("cpu").numpy().tolist() == [4.0, 6.0]


def test_npu_mul_relu():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([-1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")
    prod = a * b
    relu = a.relu()
    assert prod.device.type == "npu"
    assert relu.device.type == "npu"
    assert prod.to("cpu").numpy().tolist() == [-3.0, 8.0]
    assert relu.to("cpu").numpy().tolist() == [0.0, 2.0]


def test_npu_mul_scalar_is_capture_safe():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.ones(2, 3, device="npu", dtype=torch.float16, requires_grad=True)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = x * 0.125
        out.sum().backward()
    torch.npu.synchronize()

    graph.replay()
    torch.npu.synchronize()
    assert out.device.type == "npu"
    assert x.grad is not None
    assert x.grad.device.type == "npu"
    np.testing.assert_allclose(out.to("cpu").numpy(), np.full((2, 3), 0.125, dtype=np.float16))
    np.testing.assert_allclose(x.grad.to("cpu").numpy(), np.full((2, 3), 0.125, dtype=np.float16))


def test_npu_sum():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([[1.0, 2.0]], device="npu")
    total = a.sum()
    kept = a.sum(dim=1, keepdim=True)
    assert total.device.type == "npu"
    assert kept.device.type == "npu"
    assert total.to("cpu").numpy().tolist() == 3.0
    assert kept.to("cpu").numpy().tolist() == [[3.0]]

def test_npu_device_index_preserved():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.ones((1,), device="npu:0")
    assert out.device.type == "npu"
    assert out.device.index == 0


def test_npu_cross_device_copy():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    if torch._C._npu_device_count() < 2:
        pytest.skip("Need 2 NPUs")
    src = torch.ones((2,), device="npu:0")
    dst = src.to("npu:1")
    assert dst.device.index == 1
    assert dst.to("cpu").numpy().tolist() == [1.0, 1.0]


def test_npu_cross_device_copy_synchronizes_source_stream(monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    if torch._C._npu_device_count() < 2:
        pytest.skip("Need 2 NPUs")

    from candle._backends.npu import runtime as npu_runtime

    src_runtime = npu_runtime.get_runtime(0)
    original_synchronize_stream = src_runtime.synchronize_stream
    calls = []

    def wrapped_synchronize_stream(stream):
        calls.append(stream)
        return original_synchronize_stream(stream)

    monkeypatch.setattr(src_runtime, "synchronize_stream", wrapped_synchronize_stream)

    src = torch.ones((2,), device="npu:0")
    dst = src.to("npu:1")

    assert dst.device.index == 1
    assert calls, "cross-device copy must order destination users after the source-stream D2D copy"
    assert dst.to("cpu").numpy().tolist() == [1.0, 1.0]


def test_npu_ones():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.ones((1,), device="npu")
    assert out.device.type == "npu"
    assert out.to("cpu").numpy().tolist() == [1.0]


def test_npu_zeros():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.zeros((2,), device="npu")
    assert out.device.type == "npu"
    assert out.to("cpu").numpy().tolist() == [0.0, 0.0]



def test_npu_arange():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.arange(0, 5, device="npu")
    assert out.device.type == "npu"
    np.testing.assert_allclose(out.to("cpu").numpy(), np.array([0, 1, 2, 3, 4]))


def test_npu_range():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.range(0.0, 2.0, 0.5, device="npu")
    expected = np.arange(0.0, 2.0 + 0.5, 0.5)
    assert out.device.type == "npu"
    np.testing.assert_allclose(out.to("cpu").numpy(), expected, atol=1e-6, rtol=1e-6)


def test_npu_linspace():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    # Regression: a prior square-style mul(a, a) cached executor must not be
    # reused by the 910B linspace small-op composite's non-aliased mul(idx, step).
    probe = torch.arange(0.0, 5.0, device="npu")
    torch.mul(probe, probe)
    out = torch.linspace(0.0, 1.0, 5, device="npu")
    assert out.device.type == "npu"
    np.testing.assert_allclose(out.to("cpu").numpy(), np.linspace(0.0, 1.0, 5), atol=1e-6, rtol=1e-6)


def test_npu_logspace():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.logspace(0.0, 2.0, 3, device="npu")
    assert out.device.type == "npu"
    np.testing.assert_allclose(out.to("cpu").numpy(), np.logspace(0.0, 2.0, 3), atol=1e-6, rtol=1e-6)


def test_npu_full():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.full((2, 3), 1.5, device="npu")
    assert out.device.type == "npu"
    np.testing.assert_allclose(out.to("cpu").numpy(), np.full((2, 3), 1.5), atol=1e-6, rtol=1e-6)


def test_npu_eye():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    out = torch.eye(3, 2, device="npu")
    assert out.device.type == "npu"
    np.testing.assert_allclose(out.to("cpu").numpy(), np.eye(3, 2), atol=1e-6, rtol=1e-6)


def test_npu_linspace_prefers_single_op(monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    from candle._backends.npu import aclnn as npu_aclnn

    if hasattr(npu_aclnn, "linspace_symbols_ok") and not npu_aclnn.linspace_symbols_ok():
        pytest.skip("aclnnLinspace not available")

    def _forbid_arange(*args, **kwargs):
        raise AssertionError("linspace should not call aclnn.arange")

    monkeypatch.setattr(npu_aclnn, "arange", _forbid_arange)
    out = torch.linspace(0.0, 1.0, 5, device="npu")
    np.testing.assert_allclose(out.to("cpu").numpy(), np.linspace(0.0, 1.0, 5), atol=1e-6, rtol=1e-6)


def test_npu_eye_prefers_single_op(monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    from candle._backends.npu import aclnn as npu_aclnn

    if hasattr(npu_aclnn, "eye_symbols_ok") and not npu_aclnn.eye_symbols_ok():
        pytest.skip("aclnnEye not available")

    def _forbid_arange(*args, **kwargs):
        raise AssertionError("eye should not call aclnn.arange")

    monkeypatch.setattr(npu_aclnn, "arange", _forbid_arange)
    out = torch.eye(3, 2, device="npu")
    np.testing.assert_allclose(out.to("cpu").numpy(), np.eye(3, 2), atol=1e-6, rtol=1e-6)


def test_npu_range_prefers_single_op(monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    from candle._backends.npu import aclnn as npu_aclnn

    if hasattr(npu_aclnn, "range_symbols_ok") and not npu_aclnn.range_symbols_ok():
        pytest.skip("aclnnRange not available")

    def _forbid_arange(*args, **kwargs):
        raise AssertionError("range should not call aclnn.arange")

    monkeypatch.setattr(npu_aclnn, "arange", _forbid_arange)
    out = torch.range(0.0, 2.0, 0.5, device="npu")
    np.testing.assert_allclose(out.to("cpu").numpy(), np.arange(0.0, 2.0 + 0.5, 0.5), atol=1e-6, rtol=1e-6)


def test_npu_roll():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1, 2], [3, 4]], device="npu")
    out = torch.roll(x, shifts=1, dims=0)
    np.testing.assert_array_equal(out.to("cpu").numpy(), np.roll(x.to("cpu").numpy(), shift=1, axis=0))


def test_npu_nonzero():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[0, 1], [2, 0]], device="npu")
    out = torch.nonzero(x)
    np.testing.assert_array_equal(out.to("cpu").numpy(), np.array([[0, 1], [1, 0]]))


def test_npu_cumsum():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
    out = torch.cumsum(x, dim=1)
    np.testing.assert_allclose(out.to("cpu").numpy(), np.cumsum(x.to("cpu").numpy(), axis=1), atol=1e-6, rtol=1e-6)


def test_npu_cumprod():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
    out = torch.cumprod(x, dim=1)
    np.testing.assert_allclose(out.to("cpu").numpy(), np.cumprod(x.to("cpu").numpy(), axis=1), atol=1e-6, rtol=1e-6)


def test_npu_cummax():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], device="npu")
    values, indices = torch.cummax(x, dim=1)
    expected_vals = np.maximum.accumulate(x.to("cpu").numpy(), axis=1)
    expected_idx = np.array([[0, 1, 1], [0, 0, 2]], dtype=np.int64)
    np.testing.assert_allclose(values.to("cpu").numpy(), expected_vals, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(indices.to("cpu").numpy(), expected_idx)


def test_npu_tril_triu():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
    tril_out = torch.tril(x)
    triu_out = torch.triu(x)
    np.testing.assert_allclose(tril_out.to("cpu").numpy(), np.tril(x.to("cpu").numpy()), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(triu_out.to("cpu").numpy(), np.triu(x.to("cpu").numpy()), atol=1e-6, rtol=1e-6)


def test_npu_rot90():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1, 2], [3, 4]], device="npu")
    out = torch.rot90(x, k=1, dims=(0, 1))
    np.testing.assert_array_equal(out.to("cpu").numpy(), np.rot90(x.to("cpu").numpy(), k=1, axes=(0, 1)))


def test_npu_repeat():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1, 2], [3, 4]], device="npu")
    out = torch.repeat(x, (2, 1))
    np.testing.assert_array_equal(out.to("cpu").numpy(), np.tile(x.to("cpu").numpy(), (2, 1)))


def test_npu_repeat_interleave():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1, 2], [3, 4]], device="npu")
    out = torch.repeat_interleave(x, repeats=2, dim=1)
    np.testing.assert_array_equal(out.to("cpu").numpy(), np.repeat(x.to("cpu").numpy(), 2, axis=1))


def test_npu_tile():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1, 2], [3, 4]], device="npu")
    out = torch.tile(x, (1, 2))
    np.testing.assert_array_equal(out.to("cpu").numpy(), np.tile(x.to("cpu").numpy(), (1, 2)))


def test_npu_scatter():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device="npu")
    index = torch.tensor([[0, 2, 1], [1, 0, 2]], device="npu", dtype=torch.int64)
    src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="npu")
    out = torch.scatter(a, 1, index, src)
    expected = np.zeros((2, 3), dtype=np.float32)
    np.put_along_axis(expected, index.to("cpu").numpy(), src.to("cpu").numpy(), axis=1)
    np.testing.assert_allclose(out.to("cpu").numpy(), expected, atol=1e-6, rtol=1e-6)


def test_npu_tril_indices_triu_indices():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    tril_out = torch.tril_indices(3, 4, offset=0, device="npu")
    triu_out = torch.triu_indices(3, 4, offset=0, device="npu")
    np.testing.assert_array_equal(tril_out.to("cpu").numpy(), np.array(np.tril_indices(3, k=0, m=4), dtype=np.int64))
    np.testing.assert_array_equal(triu_out.to("cpu").numpy(), np.array(np.triu_indices(3, k=0, m=4), dtype=np.int64))


def test_npu_cartesian_prod():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([1, 2], device="npu")
    b = torch.tensor([3, 4], device="npu")
    out = torch.cartesian_prod(a, b)
    expected = np.array([[1, 3], [1, 4], [2, 3], [2, 4]], dtype=np.int64)
    np.testing.assert_array_equal(out.to("cpu").numpy(), expected)


def test_npu_block_diag():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([[1.0, 2.0]], device="npu")
    b = torch.tensor([[3.0], [4.0]], device="npu")
    out = torch.block_diag(a, b)
    expected = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(out.to("cpu").numpy(), expected, atol=1e-6, rtol=1e-6)







def test_npu_cat_concat_concatenate_stack():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([[1.0, 2.0]], device="npu")
    b = torch.tensor([[3.0, 4.0]], device="npu")
    np.testing.assert_allclose(torch.cat([a, b], dim=0).to("cpu").numpy(), np.concatenate([a.to("cpu").numpy(), b.to("cpu").numpy()], axis=0))
    np.testing.assert_allclose(torch.concat([a, b], dim=1).to("cpu").numpy(), np.concatenate([a.to("cpu").numpy(), b.to("cpu").numpy()], axis=1))
    np.testing.assert_allclose(torch.concatenate([a, b], dim=0).to("cpu").numpy(), np.concatenate([a.to("cpu").numpy(), b.to("cpu").numpy()], axis=0))

    c = torch.tensor([1.0, 2.0], device="npu")
    d = torch.tensor([3.0, 4.0], device="npu")
    np.testing.assert_allclose(torch.stack([c, d], dim=0).to("cpu").numpy(), np.stack([c.to("cpu").numpy(), d.to("cpu").numpy()], axis=0))

def test_npu_cat_concat_across_dims():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    a = torch.tensor([[1.0, 2.0]], device="npu")
    b = torch.tensor([[3.0, 4.0]], device="npu")

    out0 = torch.cat([a, b], dim=0)
    np.testing.assert_allclose(out0.to("cpu").numpy(), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    out1 = torch.concat([a, b], dim=1)
    np.testing.assert_allclose(out1.to("cpu").numpy(), np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))


def test_npu_cat_ignores_rank1_empty_tensors():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    empty = torch.tensor([], device="npu", dtype=torch.float16)
    values = torch.arange(24, device="npu", dtype=torch.float16).reshape((1, 2, 3, 4))

    out_before = torch.cat([empty, values], dim=-2)
    out_after = torch.cat([values, empty], dim=1)

    assert out_before.device.type == "npu"
    assert out_after.device.type == "npu"
    assert out_before.shape == values.shape
    assert out_after.shape == values.shape
    np.testing.assert_allclose(out_before.to("cpu").numpy(), values.to("cpu").numpy())
    np.testing.assert_allclose(out_after.to("cpu").numpy(), values.to("cpu").numpy())


def test_npu_vstack_rowstack_columnstack_dstack_vsplit_dsplit():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    a = torch.tensor([1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")
    np.testing.assert_allclose(torch.vstack([a, b]).to("cpu").numpy(), np.vstack([a.to("cpu").numpy(), b.to("cpu").numpy()]))
    np.testing.assert_allclose(torch.row_stack([a, b]).to("cpu").numpy(), np.vstack([a.to("cpu").numpy(), b.to("cpu").numpy()]))
    np.testing.assert_allclose(torch.column_stack([a, b]).to("cpu").numpy(), np.column_stack([a.to("cpu").numpy(), b.to("cpu").numpy()]))
    np.testing.assert_allclose(torch.dstack([a, b]).to("cpu").numpy(), np.dstack([a.to("cpu").numpy(), b.to("cpu").numpy()]))

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
    out_vsplit = torch.vsplit(x, 2)
    assert len(out_vsplit) == 2
    np.testing.assert_allclose(out_vsplit[0].to("cpu").numpy(), np.array([[1.0, 2.0]]))
    np.testing.assert_allclose(out_vsplit[1].to("cpu").numpy(), np.array([[3.0, 4.0]]))

    z = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], device="npu")
    out_dsplit = torch.dsplit(z, 2)
    assert len(out_dsplit) == 2
    np.testing.assert_allclose(out_dsplit[0].to("cpu").numpy(), np.array([[[1.0], [3.0]]]))
    np.testing.assert_allclose(out_dsplit[1].to("cpu").numpy(), np.array([[[2.0], [4.0]]]))

def test_npu_split_unbind_family():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="npu")
    out_int = torch.split(x, 2)
    assert len(out_int) == 3
    np.testing.assert_allclose(out_int[0].to("cpu").numpy(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(out_int[1].to("cpu").numpy(), np.array([3.0, 4.0]))
    np.testing.assert_allclose(out_int[2].to("cpu").numpy(), np.array([5.0]))

    out_sections = torch.split(x, [2, 1, 2])
    assert len(out_sections) == 3
    np.testing.assert_allclose(out_sections[0].to("cpu").numpy(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(out_sections[1].to("cpu").numpy(), np.array([3.0]))
    np.testing.assert_allclose(out_sections[2].to("cpu").numpy(), np.array([4.0, 5.0]))

    y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="npu")
    out_unbind = torch.unbind(y, dim=1)
    assert len(out_unbind) == 3
    np.testing.assert_allclose(out_unbind[0].to("cpu").numpy(), np.array([1.0, 4.0]))
    np.testing.assert_allclose(out_unbind[1].to("cpu").numpy(), np.array([2.0, 5.0]))
    np.testing.assert_allclose(out_unbind[2].to("cpu").numpy(), np.array([3.0, 6.0]))



def test_npu_empty_mul_relu_sign_sum():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    e = torch.empty((2, 2), device="npu")
    assert e.device.type == "npu"
    assert e.shape == (2, 2)

    a = torch.tensor([-1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")
    np.testing.assert_allclose(torch.mul(a, b).to("cpu").numpy(), np.array([-3.0, 8.0]))
    np.testing.assert_allclose(torch.relu(a).to("cpu").numpy(), np.array([0.0, 2.0]))
    np.testing.assert_allclose(torch.sign(a).to("cpu").numpy(), np.sign(np.array([-1.0, 2.0], dtype=np.float32)))

    m = torch.tensor([[1.0, 2.0]], device="npu")
    total = torch.sum(m)
    kept = torch.sum(m, dim=1, keepdim=True)
    np.testing.assert_allclose(total.to("cpu").numpy(), 3.0)
    np.testing.assert_allclose(kept.to("cpu").numpy(), np.array([[3.0]], dtype=np.float32))


def test_npu_concat_concatenate():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([[1.0, 2.0]], device="npu")
    b = torch.tensor([[3.0, 4.0]], device="npu")
    expected_concat = np.concatenate([a.to("cpu").numpy(), b.to("cpu").numpy()], axis=1)
    expected_concatenate = np.concatenate([a.to("cpu").numpy(), b.to("cpu").numpy()], axis=0)
    np.testing.assert_allclose(torch.concat([a, b], dim=1).to("cpu").numpy(), expected_concat)
    np.testing.assert_allclose(torch.concatenate([a, b], dim=0).to("cpu").numpy(), expected_concatenate)


def test_npu_split_int_sections():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="npu")

    out_int = torch.split(x, 2)
    assert len(out_int) == 3
    np.testing.assert_allclose(out_int[0].to("cpu").numpy(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(out_int[1].to("cpu").numpy(), np.array([3.0, 4.0]))
    np.testing.assert_allclose(out_int[2].to("cpu").numpy(), np.array([5.0]))

    out_sections = torch.split(x, [2, 1, 2])
    assert len(out_sections) == 3
    np.testing.assert_allclose(out_sections[0].to("cpu").numpy(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(out_sections[1].to("cpu").numpy(), np.array([3.0]))
    np.testing.assert_allclose(out_sections[2].to("cpu").numpy(), np.array([4.0, 5.0]))


def test_npu_unbind():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="npu")
    out = torch.unbind(x, dim=1)
    assert len(out) == 3
    np.testing.assert_allclose(out[0].to("cpu").numpy(), np.array([1.0, 4.0]))
    np.testing.assert_allclose(out[1].to("cpu").numpy(), np.array([2.0, 5.0]))
    np.testing.assert_allclose(out[2].to("cpu").numpy(), np.array([3.0, 6.0]))

def test_npu_pad_constant():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
    out = F.pad(x, (1, 2, 0, 1), mode="constant", value=0.5)
    expected = np.pad(x.to("cpu").numpy(), ((0, 1), (1, 2)), mode="constant", constant_values=0.5)
    np.testing.assert_allclose(out.to("cpu").numpy(), expected, atol=1e-6, rtol=1e-6)


def test_npu_pad_sequence_right():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([1.0, 2.0], device="npu")
    b = torch.tensor([3.0], device="npu")
    out = torch.pad_sequence([a, b], batch_first=True, padding_value=0.0, padding_side="right")
    expected = np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(out.to("cpu").numpy(), expected, atol=1e-6, rtol=1e-6)


def test_npu_pad_sequence_left():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([1.0, 2.0], device="npu")
    b = torch.tensor([3.0], device="npu")
    out = torch.pad_sequence([a, b], batch_first=True, padding_value=-1.0, padding_side="left")
    expected = np.array([[1.0, 2.0], [-1.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(out.to("cpu").numpy(), expected, atol=1e-6, rtol=1e-6)

def test_npu_to_cpu_synchronizes(monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    calls = []

    from candle._backends.npu import runtime as npu_runtime
    runtime = npu_runtime.get_runtime(0)

    def fake_sync():
        calls.append("sync")

    monkeypatch.setattr(runtime, "synchronize", fake_sync)

    t = torch.ones((1,), device="npu")
    _ = t.to("cpu")
    assert "sync" in calls


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_div(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a_data = np.array([4.0, 6.0, 9.0, 10.0], dtype=np.float32)
    b_data = np.array([2.0, 3.0, 3.0, 5.0], dtype=np.float32)
    a = torch.tensor(a_data, device="npu", dtype=dtype)
    b = torch.tensor(b_data, device="npu", dtype=dtype)
    out = torch.div(a, b)
    expected = (a_data / b_data).astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_mean(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)

    # Mean along dim 1
    out = torch.mean(x, dim=1)
    expected = np.mean(data, axis=1).astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)

    # Mean along dim 0
    out0 = torch.mean(x, dim=0)
    expected0 = np.mean(data, axis=0).astype(np.float32)
    assert np.allclose(out0.to("cpu").numpy().astype(np.float32), expected0, atol=1e-3, rtol=1e-3)

    # Global mean
    out_all = torch.mean(x)
    expected_all = np.mean(data).astype(np.float32)
    assert np.allclose(out_all.to("cpu").numpy().astype(np.float32), expected_all, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_softmax(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)

    from candle.nn import functional as F
    out = F.softmax(x, dim=-1)

    # Compute expected using numpy
    def numpy_softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    expected = numpy_softmax(data).astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)

    # Check that each row sums to ~1.0
    row_sums = np.sum(out.to("cpu").numpy().astype(np.float32), axis=1)
    assert np.allclose(row_sums, np.ones(2), atol=1e-3)


def test_npu_softmax_backward_stays_on_npu():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([[1.0, 2.0, 3.0], [2.0, 0.0, -1.0]], dtype=np.float32)
    upstream = np.array([[0.5, -1.0, 2.0], [1.5, -0.5, 0.25]], dtype=np.float32)
    x = torch.tensor(data, device="npu", requires_grad=True)
    grad_out = torch.tensor(upstream, device="npu")

    from candle.nn import functional as F
    out = F.softmax(x, dim=-1)
    (out * grad_out).sum().backward()

    e_x = np.exp(data - np.max(data, axis=-1, keepdims=True))
    softmax = e_x / np.sum(e_x, axis=-1, keepdims=True)
    expected = softmax * (upstream - np.sum(upstream * softmax, axis=-1, keepdims=True))

    assert x.grad is not None
    assert x.grad.device.type == "npu"
    assert x.grad.shape == x.shape
    assert np.allclose(x.grad.to("cpu").numpy(), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_log_softmax(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)

    from candle.nn import functional as F
    out = F.log_softmax(x, dim=-1)

    def numpy_log_softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return np.log(e_x / np.sum(e_x, axis=axis, keepdims=True))

    expected = numpy_log_softmax(data).astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)


def test_npu_log_softmax_backward_stays_on_npu():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([[1.0, 2.0, 3.0], [2.0, 0.0, -1.0]], dtype=np.float32)
    upstream = np.array([[0.5, -1.0, 2.0], [1.5, -0.5, 0.25]], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=torch.float16, requires_grad=True)
    grad_out = torch.tensor(upstream, device="npu", dtype=torch.float16)

    from candle.nn import functional as F
    out = F.log_softmax(x, dim=-1)
    (out * grad_out).sum().backward()

    e_x = np.exp(data - np.max(data, axis=-1, keepdims=True))
    softmax = e_x / np.sum(e_x, axis=-1, keepdims=True)
    expected = upstream - softmax * np.sum(upstream, axis=-1, keepdims=True)

    assert x.grad is not None
    assert x.grad.device.type == "npu"
    assert x.grad.shape == x.shape
    assert np.allclose(x.grad.to("cpu").numpy().astype(np.float32), expected, atol=2e-3, rtol=2e-3)


def test_npu_clone_default_handles_contiguous_channel_size_one():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.randn((1, 1, 7, 16), device="npu", dtype=torch.float16)
    assert x.is_contiguous()

    cloned = x.clone()

    assert cloned.device.type == "npu"
    assert cloned.shape == x.shape
    assert cloned.stride == x.stride
    assert np.allclose(cloned.to("cpu").numpy(), x.to("cpu").numpy())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_gelu(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)

    from candle.nn import functional as F
    out = F.gelu(x)

    # GELU formula: x * 0.5 * (1 + erf(x / sqrt(2)))
    from scipy.special import erf as scipy_erf
    expected = (data * 0.5 * (1.0 + scipy_erf(data / np.sqrt(2.0)))).astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)


def test_npu_silu_noncontiguous_input_returns_contiguous_output():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    base = torch.arange(0.0, 6.0, device="npu", dtype=torch.float32)
    x = base[1::2]
    assert x.stride == (2,)

    out = F.silu(x)

    assert out.device.type == "npu"
    assert out.shape == (3,)
    assert out.stride == (1,)
    x_np = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    expected = x_np / (1.0 + np.exp(-x_np))
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_embedding(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    weight_data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    weight = torch.tensor(weight_data, device="npu", dtype=dtype)
    indices = torch.tensor([0, 2, 1], device="npu", dtype=torch.int64)

    from candle.nn import functional as F
    out = F.embedding(indices, weight)

    expected = weight_data[np.array([0, 2, 1])].astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_embedding_2d_indices(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    weight_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    weight = torch.tensor(weight_data, device="npu", dtype=dtype)
    indices = torch.tensor([[0, 2], [1, 3]], device="npu", dtype=torch.int64)

    from candle.nn import functional as F
    out = F.embedding(indices, weight)

    assert out.shape == (2, 2, 2)
    expected = weight_data[np.array([[0, 2], [1, 3]])].astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)




def test_npu_allclose_matches_torch_npu_false_case_after_unary_sequence():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    unary_ops = [
        "abs", "neg", "exp", "log", "sqrt", "rsqrt", "sin", "cos", "tan",
        "tanh", "sigmoid", "ceil", "floor", "round", "trunc", "frac",
        "log2", "log10", "exp2",
    ]
    extra_ops = ["cosh", "sinh", "erf", "erfc", "softplus"]
    base = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)

    for name in unary_ops:
        data = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32) if name in {"log", "sqrt", "rsqrt", "log2", "log10"} else base
        _ = getattr(torch, name)(torch.tensor(data, device="npu", dtype=torch.float16)).to("cpu").numpy()
    for name in extra_ops:
        _ = getattr(torch, name)(torch.tensor(base, device="npu", dtype=torch.float16)).to("cpu").numpy()

    x = torch.tensor(base, device="npu", dtype=torch.float16)
    y = torch.tensor(base[::-1], device="npu", dtype=torch.float16)

    assert torch.allclose(x, y) is False


def test_npu_allclose_soc_fallback_avoids_all_reduction(monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    from candle._backends.npu import ops as npu_ops
    from candle._backends.npu.ops import comparison as comparison_mod

    def fail_all(*args, **kwargs):
        raise AssertionError("allclose fallback must avoid the unstable all_ reduction path")

    monkeypatch.setattr(npu_ops, "all_", fail_all)
    monkeypatch.setattr(comparison_mod, "_use_soc_fallback", lambda name: name == "allclose")

    x = torch.tensor([1.0, 2.0], device="npu")
    y = torch.tensor([1.0, 3.0], device="npu")

    assert torch.allclose(x, y) is False


@pytest.mark.parametrize("dtype", [torch.float16])
def test_npu_floor_matches_after_unary_prefix(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    base = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    prefix_ops = ["abs", "neg", "exp", "log", "sqrt", "rsqrt", "sin", "cos", "tan", "tanh", "sigmoid", "ceil"]

    for name in prefix_ops:
        data = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32) if name in {"log", "sqrt", "rsqrt"} else base
        _ = getattr(torch, name)(torch.tensor(data, device="npu", dtype=dtype)).to("cpu").numpy()

    out = torch.floor(torch.tensor(base, device="npu", dtype=dtype)).to("cpu").numpy().astype(np.float32)
    expected = np.floor(base).astype(np.float32)
    assert np.allclose(out, expected, atol=1e-3, rtol=1e-3)


def test_npu_take():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
    index = torch.tensor([0, 3, 1], dtype=torch.int64, device="npu")
    expected = np.take(
        x.to("cpu").numpy().reshape(-1),
        index.to("cpu").numpy().astype(np.int64),
    )
    np.testing.assert_allclose(torch.take(x, index).to("cpu").numpy(), expected)
    neg_index = torch.tensor([-1, 0], dtype=torch.int64, device="npu")
    expected_neg = np.take(
        x.to("cpu").numpy().reshape(-1),
        neg_index.to("cpu").numpy().astype(np.int64),
    )
    np.testing.assert_allclose(torch.take(x, neg_index).to("cpu").numpy(), expected_neg)
    out_of_range = torch.tensor([4], dtype=torch.int64, device="npu")
    np.testing.assert_allclose(torch.take(x, out_of_range).to("cpu").numpy(), np.array([1.0], dtype=np.float32))


def test_npu_index_select():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="npu")
    index = torch.tensor([2, 0], dtype=torch.int64, device="npu")
    expected = np.take(
        x.to("cpu").numpy(),
        index.to("cpu").numpy().astype(np.int64),
        axis=1,
    )
    np.testing.assert_allclose(
        torch.index_select(x, dim=1, index=index).to("cpu").numpy(),
        expected,
    )
    neg_index = torch.tensor([-1, 0], dtype=torch.int64, device="npu")
    expected_neg = np.take(
        x.to("cpu").numpy(),
        neg_index.to("cpu").numpy().astype(np.int64),
        axis=1,
    )
    np.testing.assert_allclose(
        torch.index_select(x, dim=1, index=neg_index).to("cpu").numpy(),
        expected_neg,
    )
    out_of_range = torch.tensor([3], dtype=torch.int64, device="npu")
    with pytest.raises(IndexError):
        torch.index_select(x, dim=1, index=out_of_range)
    bad_index = torch.tensor([[0, 1]], dtype=torch.int64, device="npu")
    with pytest.raises(ValueError):
        torch.index_select(x, dim=1, index=bad_index)


def test_npu_getitem_slice_with_1d_tensor_column_index():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="npu")
    idx = torch.arange(3, device="npu", dtype=torch.int64)
    idx += 1

    out = x[:, idx]

    np.testing.assert_array_equal(
        out.to("cpu").numpy(),
        np.array([[2, 3, 4], [6, 7, 8]]),
    )


def test_npu_masked_select():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1, 2], [3, 4]], device="npu")
    mask = torch.tensor([[True, False], [False, True]], device="npu")
    out = torch.masked_select(x, mask)
    np.testing.assert_array_equal(out.to("cpu").numpy(), np.array([1, 4]))
