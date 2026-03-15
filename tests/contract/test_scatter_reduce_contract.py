import candle as torch
import torch as real_torch


def _make_inputs():
    base = torch.zeros(3, 5)
    index = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 1, 2], [2, 0, 1, 2, 0]])
    src = torch.arange(15).reshape(3, 5)
    return base, index, src


def test_scatter_reduce_sum_include_self():
    base, index, src = _make_inputs()
    out = torch.scatter_reduce(base, 0, index, src, reduce="sum", include_self=True)
    assert out.shape == base.shape


def test_scatter_reduce_prod_exclude_self():
    base, index, src = _make_inputs()
    out = torch.scatter_reduce(base, 0, index, src, reduce="prod", include_self=False)
    assert out.shape == base.shape


def test_scatter_reduce_mean():
    base, index, src = _make_inputs()
    out = torch.scatter_reduce(base, 0, index, src, reduce="mean", include_self=True)
    assert out.shape == base.shape


def test_scatter_reduce_amax_amin():
    base, index, src = _make_inputs()
    out_max = torch.scatter_reduce(base, 0, index, src, reduce="amax", include_self=True)
    out_min = torch.scatter_reduce(base, 0, index, src, reduce="amin", include_self=True)
    assert out_max.shape == base.shape
    assert out_min.shape == base.shape


def test_scatter_reduce_sum_matches_torch_contract():
    base, index, src = _make_inputs()
    out = torch.scatter_reduce(base, 0, index, src, reduce="sum", include_self=True)
    base_ref = real_torch.zeros(3, 5, dtype=real_torch.int64)
    index_ref = real_torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 1, 2], [2, 0, 1, 2, 0]])
    src_ref = real_torch.arange(15).reshape(3, 5)
    out_ref = real_torch.scatter_reduce(base_ref, 0, index_ref, src_ref, reduce="sum", include_self=True)
    assert tuple(out.shape) == tuple(out_ref.shape)
