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


def test_small_last_dim_cat_uses_direct_device_copy(monkeypatch):
    """Small last-dim NPU cat should bypass aclnnCat setup when D2D copies suffice."""
    import candle._backends.npu.ops.shape as npu_shape

    calls = {"cat": 0}
    original_cat = npu_shape.aclnn.cat

    def counted_cat(*args, **kwargs):
        calls["cat"] += 1
        return original_cat(*args, **kwargs)

    monkeypatch.setattr(npu_shape.aclnn, "cat", counted_cat)

    x = torch.randn((1, 4, 8, 16), device="npu", dtype=torch.float16)
    x_cpu = x.to("cpu").numpy()
    half = x.shape[-1] // 2
    expected = np.concatenate((x_cpu[..., :half], x_cpu[..., half:]), axis=-1)

    result = torch.cat((x[..., :half], x[..., half:]), dim=-1)

    assert result.device.type == "npu"
    assert result.shape == x.shape
    assert result.dtype == torch.float16
    assert result.is_contiguous()
    assert calls == {"cat": 0}
    _assert_fp16_close(result, expected)


def test_small_last_dim_slice_uses_cython_view_path(monkeypatch):
    """Basic last-dim NPU slices should avoid the Python getitem view parser."""
    import candle._backends.npu.ops.shape as npu_shape

    calls = {"basic_view": 0}
    original_basic_view = npu_shape._npu_basic_getitem_view

    def counted_basic_view(*args, **kwargs):
        calls["basic_view"] += 1
        return original_basic_view(*args, **kwargs)

    monkeypatch.setattr(npu_shape, "_npu_basic_getitem_view", counted_basic_view)

    x = torch.randn((1, 4, 8, 16), device="npu", dtype=torch.float16)
    x_cpu = x.to("cpu").numpy()
    half = x.shape[-1] // 2

    first = x[..., :half]
    second = x[..., half:]

    assert first.device.type == "npu"
    assert second.device.type == "npu"
    assert first.shape == (1, 4, 8, half)
    assert second.shape == (1, 4, 8, half)
    assert calls == {"basic_view": 0}
    _assert_fp16_close(first, x_cpu[..., :half])
    _assert_fp16_close(second, x_cpu[..., half:])


def test_small_inner_contiguous_copy_uses_cython_d2d_path(monkeypatch):
    """Small inner-contiguous NPU view copies should avoid Python memcpy loops."""
    import candle._backends.npu.ops.shape as npu_shape

    x = torch.randn((1, 4, 8, 16), device="npu", dtype=torch.float16)
    x_cpu = x.to("cpu").numpy()
    half = x.shape[-1] // 2
    expected = x_cpu[..., half:]

    calls = {"memcpy_d2d": 0}
    original_memcpy_d2d = npu_shape.npu_runtime.memcpy_d2d

    def counted_memcpy_d2d(*args, **kwargs):
        calls["memcpy_d2d"] += 1
        return original_memcpy_d2d(*args, **kwargs)

    monkeypatch.setattr(npu_shape.npu_runtime, "memcpy_d2d", counted_memcpy_d2d)

    result = x[..., half:].contiguous()

    assert result.device.type == "npu"
    assert result.shape == (1, 4, 8, half)
    assert result.dtype == torch.float16
    assert result.is_contiguous()
    assert calls == {"memcpy_d2d": 0}
    _assert_fp16_close(result, expected)


def test_small_neg_view_avoids_contiguous_materialization(monkeypatch):
    """Small inner-contiguous NPU view neg should avoid a temp contiguous tensor."""
    import candle._backends.npu.ops.shape as npu_shape

    calls = {"contiguous": 0}
    original_contiguous = npu_shape.contiguous

    def counted_contiguous(*args, **kwargs):
        calls["contiguous"] += 1
        return original_contiguous(*args, **kwargs)

    monkeypatch.setattr(npu_shape, "contiguous", counted_contiguous)

    x = torch.randn((1, 4, 8, 16), device="npu", dtype=torch.float16)
    x_cpu = x.to("cpu").numpy()
    half = x.shape[-1] // 2
    expected = -x_cpu[..., half:]

    result = -x[..., half:]

    assert result.device.type == "npu"
    assert result.shape == (1, 4, 8, half)
    assert result.dtype == torch.float16
    assert result.is_contiguous()
    assert calls == {"contiguous": 0}
    _assert_fp16_close(result, expected)


def test_large_neg_view_avoids_contiguous_materialization(monkeypatch):
    """Large offset NPU view neg should use a native storage-offset descriptor."""
    import candle._backends.npu.ops.shape as npu_shape

    calls = {"contiguous": 0}
    original_contiguous = npu_shape.contiguous

    def counted_contiguous(*args, **kwargs):
        calls["contiguous"] += 1
        return original_contiguous(*args, **kwargs)

    monkeypatch.setattr(npu_shape, "contiguous", counted_contiguous)

    x = torch.randn(ROPE_SHAPE, device="npu", dtype=torch.float16)
    x_cpu = x.to("cpu").numpy()
    expected = -x_cpu[..., HALF:]

    result = -x[..., HALF:]

    assert result.device.type == "npu"
    assert result.shape == (1, 32, 2048, HALF)
    assert result.dtype == torch.float16
    assert result.is_contiguous()
    assert calls == {"contiguous": 0}
    _assert_fp16_close(result, expected)


def test_large_last_dim_slice_contiguous_avoids_index_select(monkeypatch):
    """Large last-dim slice materialization should use native slice, not gather."""
    import candle._backends.npu.ops.shape as npu_shape

    calls = {"index_select": 0}
    original_index_select = npu_shape.index_select

    def counted_index_select(*args, **kwargs):
        calls["index_select"] += 1
        return original_index_select(*args, **kwargs)

    monkeypatch.setattr(npu_shape, "index_select", counted_index_select)

    x = torch.randn(ROPE_SHAPE, device="npu", dtype=torch.float16)
    x_cpu = x.to("cpu").numpy()
    expected = x_cpu[..., HALF:]

    result = x[..., HALF:].contiguous()

    assert result.device.type == "npu"
    assert result.shape == (1, 32, 2048, HALF)
    assert result.dtype == torch.float16
    assert result.is_contiguous()
    assert calls == {"index_select": 0}
    _assert_fp16_close(result, expected)


def test_small_mixed_last_dim_cat_uses_cython_d2d_path(monkeypatch):
    """Small mixed contiguous + view cat should use the lower-overhead D2D path."""
    import candle._backends.npu.ops.shape as npu_shape

    calls = {"cat": 0, "memcpy_d2d": 0}
    original_cat = npu_shape.aclnn.cat
    original_memcpy_d2d = npu_shape.npu_runtime.memcpy_d2d

    def counted_cat(*args, **kwargs):
        calls["cat"] += 1
        return original_cat(*args, **kwargs)

    def counted_memcpy_d2d(*args, **kwargs):
        calls["memcpy_d2d"] += 1
        return original_memcpy_d2d(*args, **kwargs)

    monkeypatch.setattr(npu_shape.aclnn, "cat", counted_cat)
    monkeypatch.setattr(npu_shape.npu_runtime, "memcpy_d2d", counted_memcpy_d2d)

    x = torch.randn((1, 4, 8, 16), device="npu", dtype=torch.float16)
    x_cpu = x.to("cpu").numpy()
    half = x.shape[-1] // 2
    left = x[..., :half].contiguous()
    expected = np.concatenate((x_cpu[..., :half], x_cpu[..., half:]), axis=-1)

    result = torch.cat((left, x[..., half:]), dim=-1)

    assert result.device.type == "npu"
    assert result.shape == x.shape
    assert result.dtype == torch.float16
    assert result.is_contiguous()
    assert calls == {"cat": 0, "memcpy_d2d": 0}
    _assert_fp16_close(result, expected)


def test_small_last_dim_cat_uses_cython_d2d_path(monkeypatch):
    """Small last-dim NPU cat should avoid Python memcpy loops."""
    import candle._backends.npu.ops.shape as npu_shape

    x = torch.randn((1, 4, 8, 16), device="npu", dtype=torch.float16)
    x_cpu = x.to("cpu").numpy()
    half = x.shape[-1] // 2
    expected = np.concatenate((x_cpu[..., :half], x_cpu[..., half:]), axis=-1)

    calls = {"memcpy_d2d": 0}
    original_memcpy_d2d = npu_shape.npu_runtime.memcpy_d2d

    def counted_memcpy_d2d(*args, **kwargs):
        calls["memcpy_d2d"] += 1
        return original_memcpy_d2d(*args, **kwargs)

    monkeypatch.setattr(npu_shape.npu_runtime, "memcpy_d2d", counted_memcpy_d2d)

    result = torch.cat((x[..., :half], x[..., half:]), dim=-1)

    assert result.device.type == "npu"
    assert result.shape == x.shape
    assert result.dtype == torch.float16
    assert result.is_contiguous()
    assert calls == {"memcpy_d2d": 0}
    _assert_fp16_close(result, expected)


def test_small_rotate_half_avoids_index_select_materialization(monkeypatch):
    """RoPE rotate_half-shaped NPU cat should avoid gather-style materialization."""
    import candle._backends.npu.ops.shape as npu_shape

    calls = {"index_select": 0}
    original_index_select = npu_shape.index_select

    def counted_index_select(*args, **kwargs):
        calls["index_select"] += 1
        return original_index_select(*args, **kwargs)

    monkeypatch.setattr(npu_shape, "index_select", counted_index_select)

    x = torch.randn((1, 4, 8, 16), device="npu", dtype=torch.float16)
    x_cpu = x.to("cpu").numpy()
    half = x.shape[-1] // 2
    expected = np.concatenate((-x_cpu[..., half:], x_cpu[..., :half]), axis=-1)

    result = torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    assert result.device.type == "npu"
    assert result.shape == x.shape
    assert result.dtype == torch.float16
    assert result.is_contiguous()
    assert calls == {"index_select": 0}
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
