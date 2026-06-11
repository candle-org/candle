import pytest
import numpy as np

import candle as torch
from candle._tensor import Tensor


def _requires_npu():
    return pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")


def _make_qkv(shape=(1, 2, 16, 32), dtype=None):
    dtype = dtype or torch.float16
    q = torch.randn(shape, device="npu", dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device="npu", dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device="npu", dtype=dtype, requires_grad=True)
    return q, k, v


def _assert_npu_grad(tensor):
    assert tensor.grad is not None
    assert tensor.grad.device.type == "npu"


@_requires_npu()
def test_npu_sdpa_forward_backward_stays_on_npu():
    q, k, v = _make_qkv()

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert out.device.type == "npu"
    out.sum().backward()
    torch.npu.synchronize()

    _assert_npu_grad(q)
    _assert_npu_grad(k)
    _assert_npu_grad(v)


@_requires_npu()
def test_npu_sdpa_causal_forward_backward_stays_on_npu():
    q, k, v = _make_qkv(shape=(1, 2, 12, 32))

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert out.device.type == "npu"
    out.sum().backward()
    torch.npu.synchronize()

    _assert_npu_grad(q)
    _assert_npu_grad(k)
    _assert_npu_grad(v)


@_requires_npu()
def test_npu_sdpa_float_mask_forward_backward_stays_on_npu():
    q, k, v = _make_qkv(shape=(1, 2, 12, 32))
    attn_mask = torch.randn(12, 12, device="npu", dtype=torch.float16)

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    assert out.device.type == "npu"
    out.sum().backward()
    torch.npu.synchronize()

    _assert_npu_grad(q)
    _assert_npu_grad(k)
    _assert_npu_grad(v)


@_requires_npu()
def test_npu_sdpa_bool_mask_forward_backward_stays_on_npu():
    q, k, v = _make_qkv(shape=(1, 2, 8, 32))
    attn_mask = torch.tensor(
        [
            [True, True, False, True, True, False, True, True],
            [True, False, True, True, False, True, True, True],
            [False, True, True, False, True, True, True, False],
            [True, True, True, True, False, True, False, True],
            [True, False, True, True, True, False, True, True],
            [False, True, True, False, True, True, True, True],
            [True, True, False, True, True, True, False, True],
            [True, True, True, False, True, True, True, True],
        ],
        device="npu",
    )

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    assert out.device.type == "npu"
    out.sum().backward()
    torch.npu.synchronize()

    _assert_npu_grad(q)
    _assert_npu_grad(k)
    _assert_npu_grad(v)


@_requires_npu()
def test_npu_sdpa_uses_native_flash_attention_for_default_training(monkeypatch):
    """Default NPU training SDPA should use native fused FlashAttention forward/backward."""
    import candle._functional as candle_functional

    calls = {"matmul": 0}
    original_matmul = candle_functional.matmul

    def guarded_matmul(*args, **kwargs):
        calls["matmul"] += 1
        return original_matmul(*args, **kwargs)

    monkeypatch.setattr(candle_functional, "matmul", guarded_matmul)

    q, k, v = _make_qkv(shape=(1, 2, 16, 32))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    assert out.grad_fn.name() == "_NpuFlashSdpaBackward"
    out.sum().backward()
    torch.npu.synchronize()

    assert calls == {"matmul": 0}
    _assert_npu_grad(q)
    _assert_npu_grad(k)
    _assert_npu_grad(v)


@_requires_npu()
def test_npu_sdpa_uses_native_flash_attention_for_causal_training(monkeypatch):
    """Causal NPU SDPA should use native FlashAttention instead of eager matmul masking."""
    import candle._functional as candle_functional

    calls = {"matmul": 0}
    original_matmul = candle_functional.matmul

    def guarded_matmul(*args, **kwargs):
        calls["matmul"] += 1
        return original_matmul(*args, **kwargs)

    monkeypatch.setattr(candle_functional, "matmul", guarded_matmul)

    q, k, v = _make_qkv(shape=(1, 4, 8, 16))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, scale=0.25)
    assert out.grad_fn.name() == "_NpuFlashSdpaBackward"
    out.sum().backward()
    torch.npu.synchronize()

    assert calls == {"matmul": 0}
    _assert_npu_grad(q)
    _assert_npu_grad(k)
    _assert_npu_grad(v)


@_requires_npu()
def test_npu_sdpa_reuses_single_flash_autograd_class():
    """Native SDPA should reuse the Cython FlashAttention autograd node."""
    q, k, v = _make_qkv(shape=(1, 2, 16, 32))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    q2, k2, v2 = _make_qkv(shape=(1, 2, 16, 32))
    out2 = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0)

    assert out.grad_fn.name() == out2.grad_fn.name() == "_NpuFlashSdpaBackward"


@_requires_npu()
def test_npu_sdpa_native_backward_leaf_grad_accumulation_reuses_fresh_grads(monkeypatch):
    """Fresh native SDPA backward grads should be stored directly into leaf .grad."""
    q, k, v = _make_qkv(shape=(1, 2, 16, 32))
    clone_calls = 0
    original_clone = Tensor.clone

    def track_clone(self, *args, **kwargs):
        nonlocal clone_calls
        if getattr(getattr(self, "device", None), "type", None) == "npu":
            clone_calls += 1
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "clone", track_clone)

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    out.sum().backward()
    torch.npu.synchronize()

    assert clone_calls == 0
    _assert_npu_grad(q)
    _assert_npu_grad(k)
    _assert_npu_grad(v)


@_requires_npu()
def test_npu_sdpa_backward_accepts_noncontiguous_grad_out_without_copy(monkeypatch):
    """Native SDPA backward should consume safe non-contiguous BNSD grad_out directly."""
    q, k, v = _make_qkv(shape=(1, 2, 8, 16))
    grad_base = torch.randn((1, 8, 2, 16), device="npu", dtype=torch.float16)
    grad_out = grad_base.transpose(1, 2)
    assert not grad_out.is_contiguous()

    calls = 0
    original_contiguous = Tensor.contiguous

    def track_contiguous(self, *args, **kwargs):
        nonlocal calls
        if self is grad_out:
            calls += 1
        return original_contiguous(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "contiguous", track_contiguous)

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    out.backward(grad_out)
    torch.npu.synchronize()

    assert calls == 0
    _assert_npu_grad(q)
    _assert_npu_grad(k)
    _assert_npu_grad(v)


@_requires_npu()
def test_npu_sdpa_backward_accepts_zero_stride_sum_grad_without_copy(monkeypatch):
    """Native SDPA backward should consume sum()'s expanded zero-stride grad directly."""
    q, k, v = _make_qkv(shape=(1, 2, 16, 32))
    calls = 0
    original_contiguous = Tensor.contiguous

    def track_contiguous(self, *args, **kwargs):
        nonlocal calls
        if (
            getattr(getattr(self, "device", None), "type", None) == "npu"
            and tuple(self.shape) == tuple(q.shape)
            and tuple(self.stride()) == (0, 0, 0, 0)
        ):
            calls += 1
        return original_contiguous(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "contiguous", track_contiguous)

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    out.sum().backward()
    torch.npu.synchronize()

    assert calls == 0
    _assert_npu_grad(q)
    _assert_npu_grad(k)
    _assert_npu_grad(v)


@_requires_npu()
def test_npu_sdpa_native_flash_attention_matches_composite_values():
    """Native fused SDPA path should match the generic on-device composite."""
    import candle.nn.functional as functional

    q, k, v = _make_qkv(shape=(1, 2, 12, 32))
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    out.sum().backward()
    torch.npu.synchronize()

    original_fused = functional._try_npu_sdpa_flash_attention
    functional._try_npu_sdpa_flash_attention = lambda *args, **kwargs: None
    try:
        ref = functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, dropout_p=0.0)
        ref.sum().backward()
        torch.npu.synchronize()
    finally:
        functional._try_npu_sdpa_flash_attention = original_fused

    np.testing.assert_allclose(out.to("cpu").numpy(), ref.to("cpu").numpy(), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(q.grad.to("cpu").numpy(), q_ref.grad.to("cpu").numpy(), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(k.grad.to("cpu").numpy(), k_ref.grad.to("cpu").numpy(), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(v.grad.to("cpu").numpy(), v_ref.grad.to("cpu").numpy(), atol=2e-2, rtol=2e-2)


@_requires_npu()
def test_npu_sdpa_native_flash_attention_matches_causal_composite_values():
    """Native causal fused SDPA should match the generic on-device composite."""
    import candle.nn.functional as functional

    q, k, v = _make_qkv(shape=(1, 4, 8, 16))
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, scale=0.25)
    out.sum().backward()
    torch.npu.synchronize()

    original_fused = functional._try_npu_sdpa_flash_attention
    functional._try_npu_sdpa_flash_attention = lambda *args, **kwargs: None
    try:
        ref = functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, dropout_p=0.0, is_causal=True, scale=0.25)
        ref.sum().backward()
        torch.npu.synchronize()
    finally:
        functional._try_npu_sdpa_flash_attention = original_fused

    np.testing.assert_allclose(out.to("cpu").numpy(), ref.to("cpu").numpy(), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(q.grad.to("cpu").numpy(), q_ref.grad.to("cpu").numpy(), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(k.grad.to("cpu").numpy(), k_ref.grad.to("cpu").numpy(), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(v.grad.to("cpu").numpy(), v_ref.grad.to("cpu").numpy(), atol=2e-2, rtol=2e-2)


@_requires_npu()
def test_npu_sdpa_graph_capture_uses_composite_not_native_flash(monkeypatch):
    """FlashAttentionScore backward is graph-hostile; capture should use on-device composite."""
    import candle.nn.functional as functional
    import candle._functional as candle_functional
    import candle._C._npu_ops as npu_ops

    calls = {"flash_fwd": 0, "matmul": 0}
    original_fwd = npu_ops.fast_sdpa_flash_attention
    original_matmul = candle_functional.matmul

    def wrapped_fwd(*args, **kwargs):
        calls["flash_fwd"] += 1
        return original_fwd(*args, **kwargs)

    def wrapped_matmul(*args, **kwargs):
        calls["matmul"] += 1
        return original_matmul(*args, **kwargs)

    monkeypatch.setattr(npu_ops, "fast_sdpa_flash_attention", wrapped_fwd)
    monkeypatch.setattr(candle_functional, "matmul", wrapped_matmul)

    q, k, v = _make_qkv(shape=(1, 2, 16, 32))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out.sum().backward()
    torch.npu.synchronize()

    assert calls["flash_fwd"] == 0
    assert calls["matmul"] >= 2
    graph.replay()
    torch.npu.synchronize()
    _assert_npu_grad(q)
    _assert_npu_grad(k)
    _assert_npu_grad(v)


@_requires_npu()
def test_npu_sdpa_no_grad_graph_capture_uses_native_flash(monkeypatch):
    """No-grad NPU graph capture should keep the native FlashAttention inference path."""
    import candle.nn.functional as functional
    import candle._functional as candle_functional
    import candle._C._npu_ops as npu_ops

    calls = {"flash_fwd": 0, "matmul": 0}
    original_fwd = npu_ops.fast_sdpa_flash_attention
    original_matmul = candle_functional.matmul

    def wrapped_fwd(*args, **kwargs):
        calls["flash_fwd"] += 1
        return original_fwd(*args, **kwargs)

    def wrapped_matmul(*args, **kwargs):
        calls["matmul"] += 1
        return original_matmul(*args, **kwargs)

    monkeypatch.setattr(npu_ops, "fast_sdpa_flash_attention", wrapped_fwd)
    monkeypatch.setattr(candle_functional, "matmul", wrapped_matmul)

    q, k, v = _make_qkv(shape=(1, 4, 8, 16))
    graph = torch.npu.NPUGraph()
    with torch.no_grad():
        with torch.npu.graph(graph):
            out = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, scale=0.25)
    torch.npu.synchronize()

    assert out.device.type == "npu"
    assert calls == {"flash_fwd": 1, "matmul": 0}
    graph.replay()
    torch.npu.synchronize()


@_requires_npu()
def test_npu_mha_need_weights_false_forward_backward_stays_on_npu():
    embed_dim, num_heads = 64, 4
    mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0).to("npu").to(torch.float16)
    x = torch.randn(2, 8, embed_dim, device="npu", dtype=torch.float16, requires_grad=True)

    out, attn_weights = mha(x, x, x, need_weights=False)
    assert attn_weights is None
    assert out.device.type == "npu"
    out.sum().backward()
    torch.npu.synchronize()

    _assert_npu_grad(x)
    for parameter in mha.parameters():
        assert parameter.grad is not None
        assert parameter.grad.device.type == "npu"



@_requires_npu()
def test_npu_packed_qkv_forward_avoids_redundant_contiguous_copies(monkeypatch):
    """Packed QKV split should not copy slices that are already contiguous on NPU."""
    import candle.nn.functional as functional

    calls = 0
    original_contiguous = Tensor.contiguous

    def track_contiguous(self, *args, **kwargs):
        nonlocal calls
        if getattr(getattr(self, "device", None), "type", None) == "npu":
            calls += 1
        return original_contiguous(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "contiguous", track_contiguous)

    embed_dim, num_heads = 64, 4
    packed = torch.randn(2, 8, 3 * embed_dim, device="npu", dtype=torch.float16, requires_grad=True)

    q, k, v = functional._split_packed_qkv_projection(
        packed, embed_dim, num_heads, batch_first=True
    )
    torch.npu.synchronize()

    assert calls == 0
    assert q.shape == k.shape == v.shape == (2, num_heads, 8, embed_dim // num_heads)
    assert q.device.type == k.device.type == v.device.type == "npu"


@_requires_npu()
def test_npu_packed_qkv_split_bypasses_python_dispatch(monkeypatch):
    """Packed QKV no-grad materializing split should avoid Python dispatcher overhead."""
    import candle.nn.functional as functional
    import candle._dispatch.dispatcher as dispatcher_mod

    calls = {"split": 0}
    original_dispatch = dispatcher_mod.dispatch

    def wrapped_dispatch(op_name, *args, **kwargs):
        if op_name == "split":
            calls["split"] += 1
        return original_dispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(dispatcher_mod, "dispatch", wrapped_dispatch)

    embed_dim, num_heads = 64, 4
    packed = torch.randn(2, 8, 3 * embed_dim, device="npu", dtype=torch.float16, requires_grad=True)

    q, k, v = functional._split_packed_qkv_projection(
        packed, embed_dim, num_heads, batch_first=True
    )
    torch.npu.synchronize()

    assert calls == {"split": 0}
    assert q.shape == k.shape == v.shape == (2, num_heads, 8, embed_dim // num_heads)
    assert q.device.type == k.device.type == v.device.type == "npu"


@_requires_npu()
def test_npu_packed_qkv_split_uses_single_native_split_not_slice(monkeypatch):
    """Packed QKV split should materialize Q/K/V with one native split, not Slice per output."""
    import candle.nn.functional as functional
    import candle._backends.npu.ops.shape as npu_shape

    def fail_slice(*args, **kwargs):
        raise AssertionError("packed QKV split should not launch aclnnSlice per output")

    monkeypatch.setattr(npu_shape, "_npu_aclnn_slice", fail_slice)

    embed_dim, num_heads = 64, 4
    packed = torch.randn(2, 8, 3 * embed_dim, device="npu", dtype=torch.float16, requires_grad=True)

    q, k, v = functional._split_packed_qkv_projection(
        packed, embed_dim, num_heads, batch_first=True
    )
    out = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    out.sum().backward()
    torch.npu.synchronize()

    assert q.shape == k.shape == v.shape == (2, num_heads, 8, embed_dim // num_heads)
    assert q.device.type == k.device.type == v.device.type == out.device.type == "npu"
    _assert_npu_grad(packed)


@_requires_npu()
def test_npu_sdpa_preserves_dense_query_layout_for_mha_merge(monkeypatch):
    """Native SDPA output should reuse packed-QKV dense layout so MHA merge is a view."""
    import candle.nn.functional as functional

    embed_dim, num_heads = 64, 4
    packed = torch.randn(2, 8, 3 * embed_dim, device="npu", dtype=torch.float16, requires_grad=True)
    q, k, v = functional._split_packed_qkv_projection(
        packed, embed_dim, num_heads, batch_first=True
    )

    out = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    torch.npu.synchronize()
    assert tuple(out.stride()) == tuple(q.stride())

    calls = 0
    original_contiguous = Tensor.contiguous

    def track_contiguous(self, *args, **kwargs):
        nonlocal calls
        if self is not out and getattr(getattr(self, "device", None), "type", None) == "npu":
            calls += 1
        return original_contiguous(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "contiguous", track_contiguous)
    merged = out.transpose(1, 2).reshape(2, 8, embed_dim)
    torch.npu.synchronize()

    assert calls == 0
    assert merged.shape == (2, 8, embed_dim)


@_requires_npu()
def test_npu_mha_backward_reshape_zero_stride_grad_avoids_contiguous(monkeypatch):
    """Reshape backward should keep expanded zero-stride sum gradients as views."""
    embed_dim, num_heads = 64, 4
    mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0).to("npu").to(torch.float16)
    x = torch.randn(2, 8, embed_dim, device="npu", dtype=torch.float16, requires_grad=True)

    out, attn_weights = mha(x, x, x, need_weights=False)
    assert attn_weights is None

    zero_stride_contiguous_calls = 0
    original_contiguous = Tensor.contiguous

    def track_contiguous(self, *args, **kwargs):
        nonlocal zero_stride_contiguous_calls
        if (
            getattr(getattr(self, "device", None), "type", None) == "npu"
            and tuple(self.shape) == (2, 8, embed_dim)
            and tuple(self.stride()) == (0, 0, 0)
        ):
            zero_stride_contiguous_calls += 1
        return original_contiguous(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "contiguous", track_contiguous)
    out.sum().backward()
    torch.npu.synchronize()

    assert zero_stride_contiguous_calls == 0
    _assert_npu_grad(x)


@_requires_npu()
def test_npu_reshape_backward_preserves_zero_stride_sum_grad_values():
    """Fast NPU reshape backward must preserve expanded zero-stride gradients correctly."""
    x = torch.randn(2, 8, 64, device="npu", dtype=torch.float16, requires_grad=True)

    y = x.reshape(16, 64).sum()
    y.backward()
    torch.npu.synchronize()

    _assert_npu_grad(x)
    np.testing.assert_allclose(x.grad.to("cpu").numpy(), np.ones(x.shape, dtype=np.float16))


@_requires_npu()
def test_npu_residual_add_accumulated_leaf_grad_reuses_fresh_sum(monkeypatch):
    """A fresh accumulated NPU grad sum should be stored into leaf .grad without cloning."""
    x = torch.randn(2, 8, 64, device="npu", dtype=torch.float16, requires_grad=True)

    contiguous_clone_calls = 0
    original_clone = Tensor.clone

    def track_clone(self, *args, **kwargs):
        nonlocal contiguous_clone_calls
        if (
            getattr(getattr(self, "device", None), "type", None) == "npu"
            and tuple(self.shape) == tuple(x.shape)
            and tuple(self.stride()) == tuple(x.stride())
        ):
            contiguous_clone_calls += 1
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "clone", track_clone)

    y = (x * 2.0 + x * 3.0).sum()
    y.backward()
    torch.npu.synchronize()

    assert contiguous_clone_calls == 0
    _assert_npu_grad(x)
    np.testing.assert_allclose(x.grad.to("cpu").numpy(), np.full(x.shape, 5.0, dtype=np.float16))


@_requires_npu()
def test_npu_residual_add_zero_stride_branch_grads_avoid_duplicate_clones(monkeypatch):
    """Shared zero-stride residual grads should not be cloned for read-only backward branches."""
    x = torch.randn(2, 8, 64, device="npu", dtype=torch.float16, requires_grad=True)

    zero_stride_clone_calls = 0
    original_clone = Tensor.clone

    def track_clone(self, *args, **kwargs):
        nonlocal zero_stride_clone_calls
        if (
            getattr(getattr(self, "device", None), "type", None) == "npu"
            and tuple(self.shape) == tuple(x.shape)
            and tuple(self.stride()) == (0, 0, 0)
        ):
            zero_stride_clone_calls += 1
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "clone", track_clone)

    y = (x + x * 2.0).sum()
    y.backward()
    torch.npu.synchronize()

    assert zero_stride_clone_calls == 0
    _assert_npu_grad(x)
    np.testing.assert_allclose(x.grad.to("cpu").numpy(), np.full(x.shape, 3.0, dtype=np.float16))


@_requires_npu()
def test_npu_residual_add_dense_branch_grads_avoid_duplicate_clones(monkeypatch):
    """Shared dense residual grads should not clone before read-only branch backward."""
    x = torch.randn(2, 8, 64, device="npu", dtype=torch.float16, requires_grad=True)

    dense_clone_calls = 0
    original_clone = Tensor.clone

    def track_clone(self, *args, **kwargs):
        nonlocal dense_clone_calls
        if (
            getattr(getattr(self, "device", None), "type", None) == "npu"
            and tuple(self.shape) == tuple(x.shape)
            and tuple(self.stride()) == tuple(x.stride())
        ):
            dense_clone_calls += 1
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "clone", track_clone)

    y = ((x + x * 2.0) * 4.0).sum()
    y.backward()
    torch.npu.synchronize()

    assert dense_clone_calls == 0
    _assert_npu_grad(x)
    np.testing.assert_allclose(x.grad.to("cpu").numpy(), np.full(x.shape, 12.0, dtype=np.float16))


@_requires_npu()
def test_npu_residual_linear_branch_avoids_duplicate_dense_grad_clone(monkeypatch):
    """Dense residual grads should be shared with read-only NPU linear backward branches."""
    layer_norm = torch.nn.LayerNorm(64).to("npu").to(torch.float16)
    linear = torch.nn.Linear(64, 64).to("npu").to(torch.float16)
    x = torch.randn(2, 8, 64, device="npu", dtype=torch.float16, requires_grad=True)

    dense_clone_calls = 0
    original_clone = Tensor.clone

    def track_clone(self, *args, **kwargs):
        nonlocal dense_clone_calls
        if (
            getattr(getattr(self, "device", None), "type", None) == "npu"
            and tuple(self.shape) == tuple(x.shape)
            and tuple(self.stride()) == tuple(x.stride())
        ):
            dense_clone_calls += 1
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "clone", track_clone)

    y = ((x + linear(layer_norm(x))) * 4.0).sum()
    y.backward()
    torch.npu.synchronize()

    assert dense_clone_calls == 0
    _assert_npu_grad(x)


@_requires_npu()
def test_npu_mha_input_grad_reshape_preserves_owned_backward_grad(monkeypatch):
    """MHA input grad from packed linear backward should not clone after reshape views."""
    embed_dim, num_heads = 64, 4
    mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0).to("npu").to(torch.float16)
    x = torch.randn(2, 8, embed_dim, device="npu", dtype=torch.float16, requires_grad=True)

    clone_calls = 0
    original_clone = Tensor.clone

    def track_clone(self, *args, **kwargs):
        nonlocal clone_calls
        if getattr(getattr(self, "device", None), "type", None) == "npu":
            clone_calls += 1
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "clone", track_clone)

    out, attn_weights = mha(x, x, x, need_weights=False)
    assert attn_weights is None
    out.sum().backward()
    torch.npu.synchronize()

    assert clone_calls == 0
    _assert_npu_grad(x)


@_requires_npu()
def test_npu_layer_norm_uses_cython_fast_path(monkeypatch):
    """Contiguous NPU LayerNorm training should bypass Python backend wrappers."""
    import candle._backends.npu.ops.norm as npu_norm

    layer_norm = torch.nn.LayerNorm(64).to("npu").to(torch.float16)
    x = torch.randn(2, 8, 64, device="npu", dtype=torch.float16, requires_grad=True)

    calls = 0
    original_native = npu_norm._layer_norm_native

    def track_native(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_native(*args, **kwargs)

    monkeypatch.setattr(npu_norm, "_layer_norm_native", track_native)

    out = layer_norm(x)
    out.sum().backward()
    torch.npu.synchronize()

    assert calls == 0
    _assert_npu_grad(x)
    _assert_npu_grad(layer_norm.weight)
    _assert_npu_grad(layer_norm.bias)


@_requires_npu()
def test_npu_layer_norm_backward_reuses_fresh_native_grads(monkeypatch):
    """Fresh native layer-norm backward grads should be stored directly into leaf .grad."""
    layer_norm = torch.nn.LayerNorm(64).to("npu").to(torch.float16)
    x = torch.randn(2, 8, 64, device="npu", dtype=torch.float16, requires_grad=True)

    clone_calls = 0
    original_clone = Tensor.clone

    def track_clone(self, *args, **kwargs):
        nonlocal clone_calls
        if getattr(getattr(self, "device", None), "type", None) == "npu":
            clone_calls += 1
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "clone", track_clone)

    out = layer_norm(x)
    out.sum().backward()
    torch.npu.synchronize()

    assert clone_calls == 0
    _assert_npu_grad(x)
    _assert_npu_grad(layer_norm.weight)
    _assert_npu_grad(layer_norm.bias)


@_requires_npu()
def test_npu_layer_norm_backward_profiler_reports_no_copy_for_owned_grad_accumulation():
    """Profiler should not show generic NPU copies for owned layer-norm backward grads."""
    from candle.profiler import ProfilerActivity, profile

    layer_norm = torch.nn.LayerNorm(64).to("npu").to(torch.float16)
    x = torch.randn(2, 8, 64, device="npu", dtype=torch.float16, requires_grad=True)

    with profile(activities=[ProfilerActivity.NPU]) as prof:
        out = layer_norm(x)
        out.sum().backward()
        torch.npu.synchronize()

    rows = {row.key: row for row in prof.key_averages()}
    assert "to" not in rows
    assert "clone" not in rows


@_requires_npu()
def test_npu_mha_packed_qkv_backward_avoids_concat_stack_scatter(monkeypatch):
    """Packed self-attention QKV backward should avoid generic concat/stack scatter on NPU."""
    import candle._functional as candle_functional

    calls = {"cat": 0, "stack": 0}
    original_cat = candle_functional.cat
    original_stack = candle_functional.stack

    def track_cat(*args, **kwargs):
        calls["cat"] += 1
        return original_cat(*args, **kwargs)

    def track_stack(*args, **kwargs):
        calls["stack"] += 1
        return original_stack(*args, **kwargs)

    monkeypatch.setattr(candle_functional, "cat", track_cat)
    monkeypatch.setattr(candle_functional, "stack", track_stack)

    embed_dim, num_heads = 64, 4
    mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0).to("npu").to(torch.float16)
    x = torch.randn(2, 8, embed_dim, device="npu", dtype=torch.float16, requires_grad=True)

    out, attn_weights = mha(x, x, x, need_weights=False)
    assert attn_weights is None
    out.sum().backward()
    torch.npu.synchronize()

    assert calls == {"cat": 0, "stack": 0}
    _assert_npu_grad(x)
    for parameter in mha.parameters():
        assert parameter.grad is not None
        assert parameter.grad.device.type == "npu"


@_requires_npu()
def test_npu_packed_qkv_backward_fast_path_matches_generic_stack(monkeypatch):
    """Packed QKV gradient helper should match the generic on-device stack fallback."""
    import candle.nn.functional as functional
    import candle._C._npu_ops as npu_ops

    embed_dim, num_heads = 64, 4
    packed = torch.randn(2, 8, 3 * embed_dim, device="npu", dtype=torch.float16, requires_grad=True)
    packed_ref = packed.detach().clone().requires_grad_(True)

    q, k, v = functional._split_packed_qkv_projection(
        packed, embed_dim, num_heads, batch_first=True
    )
    out = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    out.sum().backward()
    torch.npu.synchronize()

    def force_generic_stack(*args, **kwargs):
        raise RuntimeError("force generic packed QKV stack fallback")

    monkeypatch.setattr(npu_ops, "fast_packed_qkv_projection_backward", force_generic_stack)
    monkeypatch.setattr(npu_ops, "fast_sdpa_flash_attention_backward_packed_qkv", force_generic_stack)
    q_ref, k_ref, v_ref = functional._split_packed_qkv_projection(
        packed_ref, embed_dim, num_heads, batch_first=True
    )
    ref = functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, dropout_p=0.0)
    ref.sum().backward()
    torch.npu.synchronize()

    np.testing.assert_allclose(
        packed.grad.to("cpu").numpy(),
        packed_ref.grad.to("cpu").numpy(),
        atol=1e-3,
        rtol=1e-3,
    )


@_requires_npu()
@pytest.mark.parametrize("batch_first", [True, False])
def test_npu_packed_qkv_sdpa_backward_matches_pytorch(batch_first):
    """Packed self-attention fused SDPA backward should match PyTorch for both layouts."""
    import candle.nn.functional as functional

    rng = np.random.default_rng(20260607 + int(batch_first))
    batch, seq, embed_dim, num_heads = 2, 8, 64, 4
    head_dim = embed_dim // num_heads
    packed_shape = (batch, seq, 3 * embed_dim) if batch_first else (seq, batch, 3 * embed_dim)
    packed_np = (rng.standard_normal(packed_shape) * 0.2).astype(np.float16)
    upstream_np = (rng.standard_normal((batch, num_heads, seq, head_dim)) * 0.1).astype(np.float16)

    packed = torch.from_numpy(packed_np).to("npu").requires_grad_(True)
    upstream = torch.from_numpy(upstream_np).to("npu")
    q, k, v = functional._split_packed_qkv_projection(
        packed, embed_dim, num_heads, batch_first=batch_first
    )
    out = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    out.backward(upstream)
    torch.npu.synchronize()

    import subprocess
    import sys
    import tempfile
    import textwrap

    reference_script = textwrap.dedent(
        """
        import sys
        import numpy as np
        import torch as pt

        in_path, out_path = sys.argv[1], sys.argv[2]
        batch_first = bool(int(sys.argv[3]))
        batch = int(sys.argv[4])
        seq = int(sys.argv[5])
        embed_dim = int(sys.argv[6])
        num_heads = int(sys.argv[7])
        head_dim = embed_dim // num_heads
        data = np.load(in_path)
        packed_np = data["packed"]
        upstream_np = data["upstream"]

        packed_ref = pt.tensor(packed_np, dtype=pt.float32, requires_grad=True)
        if batch_first:
            q_ref = packed_ref[:, :, :embed_dim].reshape(batch, seq, num_heads, head_dim).transpose(1, 2)
            k_ref = packed_ref[:, :, embed_dim:2 * embed_dim].reshape(batch, seq, num_heads, head_dim).transpose(1, 2)
            v_ref = packed_ref[:, :, 2 * embed_dim:].reshape(batch, seq, num_heads, head_dim).transpose(1, 2)
        else:
            q_ref = packed_ref[:, :, :embed_dim].reshape(seq, batch, num_heads, head_dim).permute(1, 2, 0, 3)
            k_ref = packed_ref[:, :, embed_dim:2 * embed_dim].reshape(seq, batch, num_heads, head_dim).permute(1, 2, 0, 3)
            v_ref = packed_ref[:, :, 2 * embed_dim:].reshape(seq, batch, num_heads, head_dim).permute(1, 2, 0, 3)
        out_ref = pt.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, dropout_p=0.0)
        out_ref.backward(pt.tensor(upstream_np, dtype=pt.float32))
        np.savez(out_path, out=out_ref.detach().numpy(), grad=packed_ref.grad.detach().numpy())
        """
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = f"{tmpdir}/packed_sdpa_ref_input.npz"
        output_path = f"{tmpdir}/packed_sdpa_ref_output.npz"
        np.savez(input_path, packed=packed_np, upstream=upstream_np)
        subprocess.run(
            [
                sys.executable,
                "-c",
                reference_script,
                input_path,
                output_path,
                str(int(batch_first)),
                str(batch),
                str(seq),
                str(embed_dim),
                str(num_heads),
            ],
            check=True,
        )
        ref = np.load(output_path)
        out_ref = ref["out"]
        packed_grad_ref = ref["grad"]

    np.testing.assert_allclose(
        out.to("cpu").numpy(),
        out_ref,
        atol=3e-2,
        rtol=5e-2,
    )
    np.testing.assert_allclose(
        packed.grad.to("cpu").numpy(),
        packed_grad_ref,
        atol=3e-2,
        rtol=5e-2,
    )


@_requires_npu()
def test_npu_sdpa_unsupported_native_layout_falls_back_to_composite(monkeypatch):
    """A rank-4 NPU SDPA layout unsupported by FlashAttention should use generic NPU ops."""
    import candle._functional as candle_functional

    calls = {"matmul": 0}
    original_matmul = candle_functional.matmul

    def wrapped_matmul(*args, **kwargs):
        calls["matmul"] += 1
        return original_matmul(*args, **kwargs)

    monkeypatch.setattr(candle_functional, "matmul", wrapped_matmul)

    q_base, k_base, v_base = _make_qkv(shape=(1, 2, 16, 32))
    q = q_base.transpose(-1, -2)
    k = k_base.transpose(-1, -2)
    v = v_base.transpose(-1, -2)
    assert q.stride()[-1] != 1

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    assert out.device.type == "npu"
    out.sum().backward()
    torch.npu.synchronize()

    assert calls["matmul"] >= 2
    _assert_npu_grad(q_base)
    _assert_npu_grad(k_base)
    _assert_npu_grad(v_base)


@_requires_npu()
def test_npu_sdpa_native_runtime_error_does_not_fall_back_to_composite(monkeypatch):
    """True native FlashAttention runtime failures must surface instead of changing kernels."""
    import candle._functional as candle_functional
    import candle._C._functional_ops as cy_functional  # pylint: disable=import-error,no-name-in-module

    calls = {"matmul": 0}
    original_matmul = candle_functional.matmul

    def wrapped_matmul(*args, **kwargs):
        calls["matmul"] += 1
        return original_matmul(*args, **kwargs)

    def fail_native_sdpa(*args, **kwargs):
        raise RuntimeError("native failed")

    monkeypatch.setattr(candle_functional, "matmul", wrapped_matmul)
    monkeypatch.setattr(cy_functional, "npu_flash_sdpa", fail_native_sdpa)

    q, k, v = _make_qkv(shape=(1, 2, 16, 32))
    with pytest.raises(RuntimeError, match="native failed"):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    assert calls == {"matmul": 0}


@_requires_npu()
def test_npu_sdpa_no_cpu_roundtrip(monkeypatch):
    original_to = Tensor.to

    def guard_to(self, *args, **kwargs):
        device = None
        if args:
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]
        if getattr(self, "device", None) is not None and self.device.type == "npu":
            if device == "cpu" or getattr(device, "type", None) == "cpu":
                raise AssertionError("SDPA should not move NPU tensors to CPU")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "to", guard_to)
    q, k, v = _make_qkv(shape=(1, 2, 8, 32))

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert out.device.type == "npu"
    out.sum().backward()
    torch.npu.synchronize()

    _assert_npu_grad(q)
    _assert_npu_grad(k)
    _assert_npu_grad(v)
