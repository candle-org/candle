"""Test RoPE-shaped fp16 NPU tensor neg/cat CANN workarounds."""
import numpy as np

import candle as torch


ROPE_SHAPE = (1, 32, 2048, 128)
HALF = ROPE_SHAPE[-1] // 2


def _assert_fp16_close(actual, expected):
    np.testing.assert_allclose(
        actual.to("cpu").numpy().astype(np.float32),
        expected.astype(np.float32),
        rtol=1e-3,
        atol=1e-3,
    )


def test_neg_fp16_rope_shape():
    """neg must work on (1, 32, 2048, 64) fp16 NPU tensor without workspace error."""
    x = torch.randn((1, 32, 2048, 64), device="npu", dtype=torch.float16)
    expected = -x.to("cpu").numpy()
    result = -x
    assert result.device.type == "npu"
    assert result.shape == x.shape
    assert result.dtype == torch.float16
    assert result.is_contiguous()
    _assert_fp16_close(result, expected)


def test_neg_offset_rope_view_runs_on_npu():
    """neg of an offset RoPE half view must route through the CANN workaround."""
    x = torch.randn(ROPE_SHAPE, device="npu", dtype=torch.float16)
    expected = -x.to("cpu").numpy()[..., HALF:]
    result = -x[..., HALF:]
    assert result.device.type == "npu"
    assert result.shape == (1, 32, 2048, HALF)
    assert result.dtype == torch.float16
    assert result.is_contiguous()
    _assert_fp16_close(result, expected)


def test_cat_strided_rope_half_views_runs_on_npu():
    """cat of RoPE half views must materialize inputs on-device when CANN rejects strides."""
    x = torch.randn(ROPE_SHAPE, device="npu", dtype=torch.float16)
    x_cpu = x.to("cpu").numpy()
    expected = np.concatenate((x_cpu[..., :HALF], x_cpu[..., HALF:]), axis=-1)
    result = torch.cat((x[..., :HALF], x[..., HALF:]), dim=-1)
    assert result.device.type == "npu"
    assert result.shape == x.shape
    assert result.dtype == torch.float16
    assert result.is_contiguous()
    _assert_fp16_close(result, expected)


def test_rope_composite_runs_on_npu():
    """Full RoPE composite must run without error on NPU."""
    x = torch.randn(ROPE_SHAPE, device="npu", dtype=torch.float16)
    cos = torch.randn_like(x)
    sin = torch.randn_like(x)
    t1 = x[..., :HALF].contiguous()
    t2 = x[..., HALF:].contiguous()
    rotated = torch.cat((-t2, t1), dim=-1)
    result = x * cos + rotated * sin
    assert result.device.type == "npu"
    assert result.shape == x.shape
