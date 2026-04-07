import candle as torch
from candle._tensor import Tensor


def test_view_reshape_transpose_share_storage():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = x.reshape((3, 2))
    z = x.transpose(0, 1)
    assert y.storage() is x.storage()
    assert z.storage() is x.storage()
    assert y.shape == (3, 2)
    assert z.shape == (3, 2)
    assert x.stride != z.stride


def test_tensor_reshape_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = x.reshape((6,))

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    y = x.reshape((6,))

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (6,)
    assert y.stride == (1,)
    assert y.tolist() == [1, 2, 3, 4, 5, 6]
    assert y is not x
    assert y.is_contiguous()
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value


def test_tensor_T_for_2d():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = x.T
    assert y.shape == (3, 2)
    assert y.storage() is x.storage()


def test_tensor_T_bypasses_python_tensor_t(monkeypatch):
    from candle._tensor import Tensor

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = x.T

    calls = {"count": 0}
    original = Tensor.t

    def wrapped(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "t", wrapped)

    y = x.T

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (3, 2)
    assert y.stride == (1, 3)
    assert y.tolist() == [[1, 4], [2, 5], [3, 6]]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value


def test_tensor_t__bypasses_python_tensor_transpose_(monkeypatch):
    from candle._tensor import Tensor

    warmup = torch.tensor([[1, 2, 3], [4, 5, 6]])
    _ = warmup.t_()

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    calls = {"count": 0}
    original = Tensor.transpose_

    def wrapped(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "transpose_", wrapped)

    y = x.t_()

    assert calls["count"] == 0
    assert y is x
    assert y.shape == (3, 2)
    assert y.stride == (1, 3)
    assert y.tolist() == [[1, 4], [2, 5], [3, 6]]


def test_tensor_swapdims__bypasses_python_tensor_transpose_(monkeypatch):
    from candle._tensor import Tensor

    warmup = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    _ = warmup.swapdims_(0, 2)

    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    calls = {"count": 0}
    original = Tensor.transpose_

    def wrapped(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "transpose_", wrapped)

    y = x.swapdims_(0, 2)

    assert calls["count"] == 0
    assert y is x
    assert y.shape == (2, 2, 2)
    assert y.stride == (1, 2, 4)
    assert y.tolist() == [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]


def test_tensor_swapaxes__bypasses_python_tensor_swapdims_(monkeypatch):
    from candle._tensor import Tensor

    warmup = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    _ = warmup.swapaxes_(0, 2)

    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    calls = {"count": 0}
    original = Tensor.swapdims_

    def wrapped(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "swapdims_", wrapped)

    y = x.swapaxes_(0, 2)

    assert calls["count"] == 0
    assert y is x
    assert y.shape == (2, 2, 2)
    assert y.stride == (1, 2, 4)
    assert y.tolist() == [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]


    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    flat_all = x.flatten()
    assert flat_all.shape == (8,)
    assert flat_all.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]

    flat_partial = x.flatten(1)
    assert flat_partial.shape == (2, 4)
    assert flat_partial.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]



def test_tensor_view_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = x.view((6,))

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    y = x.view((6,))

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (6,)
    assert y.stride == (1,)
    assert y.tolist() == [1, 2, 3, 4, 5, 6]
    assert y is not x
    assert y.is_contiguous()
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert y.view((2, 3)).tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_tensor_transpose_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = x.transpose(0, 1)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    y = x.transpose(0, 1)

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (3, 2)
    assert y.stride == (1, 3)
    assert y.tolist() == [[1, 4], [2, 5], [3, 6]]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value


def test_toplevel_as_strided_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = torch.as_strided(x, (3, 2), (1, 3), 0)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    y = torch.as_strided(x, (3, 2), (1, 3), 0)

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (3, 2)
    assert y.stride == (1, 3)
    assert y.tolist() == [[1, 4], [2, 5], [3, 6]]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value









def test_tensor_unsqueeze_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = x.unsqueeze(1)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    y = x.unsqueeze(1)

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 1, 3)
    assert y.stride == (3, 3, 1)
    assert y.tolist() == [[[1, 2, 3]], [[4, 5, 6]]]
    assert y is not x
    assert y.is_contiguous()
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value





def test_tensor_permute_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    _ = x.permute(1, 0, 2)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    y = x.permute(1, 0, 2)

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 2, 2)
    assert y.stride == (2, 4, 1)
    assert y.tolist() == [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_tensor_narrow_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = x.narrow(1, 1, 2)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    y = x.narrow(1, 1, 2)

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 2)
    assert y.stride == (3, 1)
    assert y.tolist() == [[2, 3], [5, 6]]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value


def test_tensor_select_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = x.select(1, 1)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    y = x.select(1, 1)

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2,)
    assert y.stride == (3,)
    assert y.tolist() == [2, 5]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_tensor_expand_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[[1, 2, 3]], [[4, 5, 6]]])

    _ = x.expand(2, 4, 3)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    y = x.expand(2, 4, 3)

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 4, 3)
    assert y.stride == (3, 0, 1)
    assert y.tolist() == [
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6]],
    ]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[[1, 2, 3]], [[4, 5, 6]]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_tensor_movedim_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    _ = x.movedim(0, 1)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    y = x.movedim(0, 1)

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 2, 2)
    assert y.stride == (2, 4, 1)
    assert y.tolist() == [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_tensor_moveaxis_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    _ = x.moveaxis(0, 1)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    y = x.moveaxis(0, 1)

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 2, 2)
    assert y.stride == (2, 4, 1)
    assert y.tolist() == [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_tensor_diagonal_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    _ = x.diagonal()

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    y = x.diagonal()

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (3,)
    assert y.stride == (4,)
    assert y.tolist() == [1, 5, 9]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_toplevel_permute_bypasses_python_functional_dispatch(monkeypatch):
    import candle._functional as functional

    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    _ = torch.permute(x, (1, 0, 2))

    calls = {"count": 0, "ops": []}
    original = functional.dispatch

    def wrapped(op_name, *args, **kwargs):
        calls["count"] += 1
        calls["ops"].append(op_name)
        return original(op_name, *args, **kwargs)

    monkeypatch.setattr(functional, "dispatch", wrapped)

    y = torch.permute(x, (1, 0, 2))

    assert calls["count"] == 0
    assert calls["ops"] == []
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 2, 2)
    assert y.stride == (2, 4, 1)
    assert y.tolist() == [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value


def test_toplevel_broadcast_to_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[[1, 2, 3]], [[4, 5, 6]]])

    _ = torch.broadcast_to(x, (2, 4, 3))

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    y = torch.broadcast_to(x, (2, 4, 3))

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 4, 3)
    assert y.stride == (3, 0, 1)
    assert y.tolist() == [
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6]],
    ]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[[1, 2, 3]], [[4, 5, 6]]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_toplevel_view_as_real_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)

    _ = torch.view_as_real(x)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    y = torch.view_as_real(x)

    assert calls["count"] == 0
    assert y.storage() is not x.storage()
    assert y.storage().data_ptr() == x.storage().data_ptr()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 2)
    assert y.stride == (2, 1)
    assert y.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert y is not x
    assert y.is_contiguous()
    assert x.tolist() == [(1 + 2j), (3 + 4j)]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_tensor_unfold_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.arange(1, 8, dtype=torch.float32)

    _ = x.unfold(0, 3, 2)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    y = x.unfold(0, 3, 2)

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (3, 3)
    assert y.stride == (2, 1)
    assert y.tolist() == [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_tensor_unbind_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = x.unbind(0)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    ys = x.unbind(0)

    assert calls["count"] == 0
    assert len(ys) == 2
    assert [y.storage() is x.storage() for y in ys] == [True, True]
    assert [y._base is x for y in ys] == [True, True]
    assert [y._version_counter is x._version_counter for y in ys] == [True, True]
    assert [y.shape for y in ys] == [(3,), (3,)]
    assert [y.stride for y in ys] == [(1,), (1,)]
    assert [y.tolist() for y in ys] == [[1, 2, 3], [4, 5, 6]]
    assert [y is x for y in ys] == [False, False]
    assert [y.is_contiguous() for y in ys] == [True, True]
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert [y._version_counter.value for y in ys] == [x._version_counter.value, x._version_counter.value]



def test_tensor_split_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = x.split(1, 0)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    ys = x.split(1, 0)

    assert calls["count"] == 0
    assert len(ys) == 2
    assert [y.storage() is x.storage() for y in ys] == [True, True]
    assert [y._base is x for y in ys] == [True, True]
    assert [y._version_counter is x._version_counter for y in ys] == [True, True]
    assert [y.shape for y in ys] == [(1, 3), (1, 3)]
    assert [y.stride for y in ys] == [(3, 1), (3, 1)]
    assert [y.tolist() for y in ys] == [[[1, 2, 3]], [[4, 5, 6]]]
    assert [y is x for y in ys] == [False, False]
    assert [y.is_contiguous() for y in ys] == [True, True]
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert [y._version_counter.value for y in ys] == [x._version_counter.value, x._version_counter.value]


def test_tensor_chunk_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = x.chunk(2, 0)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    ys = x.chunk(2, 0)

    assert calls["count"] == 0
    assert len(ys) == 2
    assert [y.storage() is x.storage() for y in ys] == [True, True]
    assert [y._base is x for y in ys] == [True, True]
    assert [y._version_counter is x._version_counter for y in ys] == [True, True]
    assert [y.shape for y in ys] == [(1, 3), (1, 3)]
    assert [y.stride for y in ys] == [(3, 1), (3, 1)]
    assert [y.tolist() for y in ys] == [[[1, 2, 3]], [[4, 5, 6]]]
    assert [y is x for y in ys] == [False, False]
    assert [y.is_contiguous() for y in ys] == [True, True]
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert [y._version_counter.value for y in ys] == [x._version_counter.value, x._version_counter.value]




def test_tensor_hsplit_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = x.hsplit(3)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    ys = x.hsplit(3)

    assert calls["count"] == 0
    assert len(ys) == 3
    assert [y.storage() is x.storage() for y in ys] == [True, True, True]
    assert [y._base is x for y in ys] == [True, True, True]
    assert [y._version_counter is x._version_counter for y in ys] == [True, True, True]
    assert [y.shape for y in ys] == [(2, 1), (2, 1), (2, 1)]
    assert [y.stride for y in ys] == [(3, 1), (3, 1), (3, 1)]
    assert [y.tolist() for y in ys] == [[[1], [4]], [[2], [5]], [[3], [6]]]
    assert [y is x for y in ys] == [False, False, False]
    assert [y.is_contiguous() for y in ys] == [False, False, False]
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()


def test_tensor_dsplit_bypasses_python_common_view_backend(monkeypatch):
    import candle._backends.common.view as view_backend

    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    _ = x.dsplit(2)

    calls = {"count": 0}
    original = view_backend._make_view

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_backend, "_make_view", wrapped)

    ys = x.dsplit(2)

    assert calls["count"] == 0
    assert len(ys) == 2
    assert [y.storage() is x.storage() for y in ys] == [True, True]
    assert [y._base is x for y in ys] == [True, True]
    assert [y._version_counter is x._version_counter for y in ys] == [True, True]
    assert [y.shape for y in ys] == [(2, 2, 1), (2, 2, 1)]
    assert [y.stride for y in ys] == [(4, 2, 1), (4, 2, 1)]
    assert [y.tolist() for y in ys] == [
        [[[1], [3]], [[5], [7]]],
        [[[2], [4]], [[6], [8]]],
    ]
    assert [y is x for y in ys] == [False, False]
    assert [y.is_contiguous() for y in ys] == [False, False]
    assert x.tolist() == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    x._version_counter.bump()
    assert [y._version_counter.value for y in ys] == [x._version_counter.value, x._version_counter.value]



def test_toplevel_unflatten_bypasses_python_functional_dispatch(monkeypatch):
    import candle._functional as functional

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = torch.unflatten(x, 1, (3, 1))

    calls = {"count": 0, "ops": []}
    original = functional.dispatch

    def wrapped(op_name, *args, **kwargs):
        calls["count"] += 1
        calls["ops"].append(op_name)
        return original(op_name, *args, **kwargs)

    monkeypatch.setattr(functional, "dispatch", wrapped)

    y = torch.unflatten(x, 1, (3, 1))

    assert calls["count"] == 0
    assert calls["ops"] == []
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 3, 1)
    assert y.stride == (3, 1, 1)
    assert y.tolist() == [[[1], [2], [3]], [[4], [5], [6]]]
    assert y is not x
    assert y.is_contiguous()
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_toplevel_slice_bypasses_python_functional_dispatch(monkeypatch):
    import candle._functional as functional

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = torch.slice(x, 1, 0, 2, 1)

    calls = {"count": 0, "ops": []}
    original = functional.dispatch

    def wrapped(op_name, *args, **kwargs):
        calls["count"] += 1
        calls["ops"].append(op_name)
        return original(op_name, *args, **kwargs)

    monkeypatch.setattr(functional, "dispatch", wrapped)

    y = torch.slice(x, 1, 0, 2, 1)

    assert calls["count"] == 0
    assert calls["ops"] == []
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 2)
    assert y.stride == (3, 1)
    assert y.offset == 0
    assert y.tolist() == [[1, 2], [4, 5]]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_toplevel_squeeze_bypasses_python_functional_dispatch(monkeypatch):
    import candle._functional as functional

    x = torch.tensor([[[1, 2, 3]], [[4, 5, 6]]])

    _ = torch.squeeze(x, 1)

    calls = {"count": 0, "ops": []}
    original = functional.dispatch

    def wrapped(op_name, *args, **kwargs):
        calls["count"] += 1
        calls["ops"].append(op_name)
        return original(op_name, *args, **kwargs)

    monkeypatch.setattr(functional, "dispatch", wrapped)

    y = torch.squeeze(x, 1)

    assert calls["count"] == 0
    assert calls["ops"] == []
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 3)
    assert y.stride == (3, 1)
    assert y.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert y is not x
    assert y.is_contiguous()
    assert x.tolist() == [[[1, 2, 3]], [[4, 5, 6]]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_toplevel_transpose_bypasses_python_functional_dispatch(monkeypatch):
    import candle._functional as functional

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = torch.transpose(x, 0, 1)

    calls = {"count": 0, "ops": []}
    original = functional.dispatch

    def wrapped(op_name, *args, **kwargs):
        calls["count"] += 1
        calls["ops"].append(op_name)
        return original(op_name, *args, **kwargs)

    monkeypatch.setattr(functional, "dispatch", wrapped)

    y = torch.transpose(x, 0, 1)

    assert calls["count"] == 0
    assert calls["ops"] == []
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (3, 2)
    assert y.stride == (1, 3)
    assert y.tolist() == [[1, 4], [2, 5], [3, 6]]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_toplevel_t_bypasses_python_top_level_transpose(monkeypatch):
    import candle as torch_module

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = torch.t(x)

    calls = {"count": 0}
    original = torch_module.transpose

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "transpose", wrapped)

    y = torch.t(x)

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (3, 2)
    assert y.stride == (1, 3)
    assert y.tolist() == [[1, 4], [2, 5], [3, 6]]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_toplevel_diagflat_bypasses_python_top_level_flatten(monkeypatch):
    import candle as torch_module

    x = torch.tensor([[1, 2], [3, 4]])

    _ = torch.diagflat(x)

    calls = {"count": 0}
    original = torch_module.flatten

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "flatten", wrapped)

    y = torch.diagflat(x)

    assert calls["count"] == 0
    assert y.shape == (4, 4)
    assert y.tolist() == [
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 4],
    ]



def test_toplevel_fliplr_bypasses_python_top_level_flip(monkeypatch):
    import candle as torch_module

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = torch.fliplr(x)

    calls = {"count": 0}
    original = torch_module.flip

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "flip", wrapped)

    y = torch.fliplr(x)

    assert calls["count"] == 0
    assert y.shape == (2, 3)
    assert y.tolist() == [[3, 2, 1], [6, 5, 4]]



def test_toplevel_flipud_bypasses_python_top_level_flip(monkeypatch):
    import candle as torch_module

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = torch.flipud(x)

    calls = {"count": 0}
    original = torch_module.flip

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "flip", wrapped)

    y = torch.flipud(x)

    assert calls["count"] == 0
    assert y.shape == (2, 3)
    assert y.tolist() == [[4, 5, 6], [1, 2, 3]]



def test_toplevel_std_mean_bypasses_python_top_level_std_and_mean(monkeypatch):
    import math
    import candle as torch_module

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])

    _ = torch.std_mean(x)

    calls = {"std": 0, "mean": 0}
    original_std = torch_module.std
    original_mean = torch_module.mean

    def wrapped_std(*args, **kwargs):
        calls["std"] += 1
        return original_std(*args, **kwargs)

    def wrapped_mean(*args, **kwargs):
        calls["mean"] += 1
        return original_mean(*args, **kwargs)

    monkeypatch.setattr(torch_module, "std", wrapped_std)
    monkeypatch.setattr(torch_module, "mean", wrapped_mean)

    s, m = torch.std_mean(x)

    assert calls["std"] == 0
    assert calls["mean"] == 0
    assert math.isclose(s.item(), 1.2909944487358056, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(m.item(), 2.5, rel_tol=0.0, abs_tol=1e-6)

def test_toplevel_rsub_with_alpha_bypasses_python_top_level_sub(monkeypatch):
    import candle as torch_module

    x = torch.tensor([1, 2, 3])
    other = torch.tensor([10, 10, 10])

    _ = torch.rsub(x, other, alpha=2)

    calls = {"count": 0}
    original = torch_module.sub

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "sub", wrapped)

    y = torch.rsub(x, other, alpha=2)

    assert calls["count"] == 0
    assert y.tolist() == [8, 6, 4]




def test_toplevel_nan_to_num__bypasses_python_top_level_copy_(monkeypatch):
    import candle as torch_module

    x = torch.tensor([float("nan"), float("inf"), float("-inf"), 1.0])

    _ = torch.nan_to_num_(x, nan=0.0, posinf=9.0, neginf=-9.0)

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.nan_to_num_(x, nan=0.0, posinf=9.0, neginf=-9.0)

    assert calls["count"] == 0
    assert y is x
    assert y.tolist() == [0.0, 9.0, -9.0, 1.0]


def test_toplevel_swapaxes_bypasses_python_top_level_transpose(monkeypatch):
    import candle as torch_module

    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    _ = torch.swapaxes(x, 0, 2)

    calls = {"count": 0}
    original = torch_module.transpose

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "transpose", wrapped)

    y = torch.swapaxes(x, 0, 2)

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 2, 2)
    assert y.stride == (1, 2, 4)
    assert y.tolist() == [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value

def test_toplevel_ger_bypasses_python_top_level_outer(monkeypatch):
    import candle as torch_module

    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5])

    _ = torch.ger(x, y)

    calls = {"count": 0}
    original = torch_module.outer

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "outer", wrapped)

    z = torch.ger(x, y)

    assert calls["count"] == 0
    assert z.tolist() == [[4, 5], [8, 10], [12, 15]]



def test_toplevel_clamp_min__bypasses_python_top_level_copy_(monkeypatch):
    import candle as torch_module

    x = torch.tensor([-2.0, -0.5, 1.0])

    _ = torch.clamp_min_(x, 0.0)

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.clamp_min_(x, 0.0)

    assert calls["count"] == 0
    assert y is x
    assert y.tolist() == [0.0, 0.0, 1.0]




def test_toplevel_clamp_max__bypasses_python_top_level_copy_(monkeypatch):
    import candle as torch_module

    x = torch.tensor([-2.0, 0.5, 3.0])

    _ = torch.clamp_max_(x, 1.0)

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.clamp_max_(x, 1.0)

    assert calls["count"] == 0
    assert y is x
    assert y.tolist() == [-2.0, 0.5, 1.0]



def test_toplevel_rsqrt__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([4.0, 16.0, 25.0])
    _ = torch.rsqrt_(warmup)

    x = torch.tensor([4.0, 16.0, 25.0])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.rsqrt_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.25, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.2, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_log1p__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, 3.0])
    _ = torch.log1p_(warmup)

    x = torch.tensor([0.0, 1.0, 3.0])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.log1p_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.6931471805599453, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.3862943611198906, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_expm1__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, 2.0])
    _ = torch.expm1_(warmup)

    x = torch.tensor([0.0, 1.0, 2.0])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.expm1_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.718281828459045, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 6.38905609893065, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_exp2__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, 3.0])
    _ = torch.exp2_(warmup)

    x = torch.tensor([0.0, 1.0, 3.0])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.exp2_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 2.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 8.0, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_frac__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([1.25, -2.75, 3.5])
    _ = torch.frac_(warmup)

    x = torch.tensor([1.25, -2.75, 3.5])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.frac_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.25, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], -0.75, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.5, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_sinh__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, -1.0])
    _ = torch.sinh_(warmup)

    x = torch.tensor([0.0, 1.0, -1.0])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.sinh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.1752011936438014, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -1.1752011936438014, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_cosh__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, -1.0])
    _ = torch.cosh_(warmup)

    x = torch.tensor([0.0, 1.0, -1.0])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.cosh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.5430806348152437, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.5430806348152437, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_erf__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, -1.0])
    _ = torch.erf_(warmup)

    x = torch.tensor([0.0, 1.0, -1.0])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.erf_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.8427007929497149, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -0.8427007929497149, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_erfc__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, -1.0])
    _ = torch.erfc_(warmup)

    x = torch.tensor([0.0, 1.0, -1.0])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.erfc_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.15729920705028513, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.842700792949715, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_asinh__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, -1.0])
    _ = torch.asinh_(warmup)

    x = torch.tensor([0.0, 1.0, -1.0])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.asinh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.881373587019543, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -0.881373587019543, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_atan__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, -1.0])
    _ = torch.atan_(warmup)

    x = torch.tensor([0.0, 1.0, -1.0])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.atan_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.7853981633974483, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -0.7853981633974483, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_atanh__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 0.5, -0.5])
    _ = torch.atanh_(warmup)

    x = torch.tensor([0.0, 0.5, -0.5])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.atanh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.5493061443340548, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -0.5493061443340548, rel_tol=0.0, abs_tol=1e-6)


def test_nn_functional_selu__bypasses_python_selu(monkeypatch):
    import math
    import candle.nn.functional as F

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = F.selu_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = F.selu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "selu", wrapped)

    y = F.selu_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -1.1113307378125625, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.0507009873554805, rel_tol=0.0, abs_tol=1e-6)



def test_nn_functional_celu__bypasses_python_celu(monkeypatch):
    import math
    import candle.nn.functional as F

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = F.celu_(warmup, alpha=1.3)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = F.celu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "celu", wrapped)

    y = F.celu_(x, alpha=1.3)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -0.6976198199994721, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.0, rel_tol=0.0, abs_tol=1e-6)



def test_nn_functional_threshold__bypasses_python_threshold(monkeypatch):
    import candle.nn.functional as F

    warmup = torch.tensor([-1.0, 0.5, 2.0])
    _ = F.threshold_(warmup, 0.3, -7.0)

    x = torch.tensor([-1.0, 0.5, 2.0])

    calls = {"count": 0}
    original = F.threshold

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "threshold", wrapped)

    y = F.threshold_(x, 0.3, -7.0)

    assert calls["count"] == 0
    assert y is x
    assert y.tolist() == [-7.0, 0.5, 2.0]



def test_nn_functional_hardtanh__bypasses_python_hardtanh(monkeypatch):
    import candle.nn.functional as F

    warmup = torch.tensor([-2.0, -0.5, 1.5])
    _ = F.hardtanh_(warmup, min_val=-0.25, max_val=1.0)

    x = torch.tensor([-2.0, -0.5, 1.5])

    calls = {"count": 0}
    original = F.hardtanh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "hardtanh", wrapped)

    y = F.hardtanh_(x, min_val=-0.25, max_val=1.0)

    assert calls["count"] == 0
    assert y is x
    assert y.tolist() == [-0.25, -0.25, 1.0]



def test_nn_functional_hardtanh__bypasses_python__functional_hardtanh(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-2.0, -0.5, 1.5])
    _ = F.hardtanh_(warmup, min_val=-0.25, max_val=1.0)

    x = torch.tensor([-2.0, -0.5, 1.5])

    calls = {"count": 0}
    original = _functional.hardtanh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "hardtanh", wrapped)

    y = F.hardtanh_(x, min_val=-0.25, max_val=1.0)

    assert calls["count"] == 0
    assert y is x
    assert y.tolist() == [-0.25, -0.25, 1.0]



def test_nn_functional_rrelu__bypasses_python_rrelu(monkeypatch):
    import candle.nn.functional as F

    warmup = torch.tensor([0.5, 1.0, 3.0])
    _ = F.rrelu_(warmup, training=False)

    x = torch.tensor([0.5, 1.0, 3.0])

    calls = {"count": 0}
    original = F.rrelu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "rrelu", wrapped)

    y = F.rrelu_(x, training=False)

    assert calls["count"] == 0
    assert y is x
    assert y.tolist() == [0.5, 1.0, 3.0]



def test_nn_functional_elu__bypasses_python_elu(monkeypatch):
    import math
    import candle.nn.functional as F

    warmup = torch.tensor([-2.0, 0.5, 3.0])
    _ = F.elu_(warmup, alpha=1.3)

    x = torch.tensor([-2.0, 0.5, 3.0])

    calls = {"count": 0}
    original = F.elu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "elu", wrapped)

    y = F.elu_(x, alpha=1.3)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -1.1240640878677368, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 3.0, rel_tol=0.0, abs_tol=1e-6)



def test_nn_functional_leaky_relu__bypasses_python_leaky_relu(monkeypatch):
    import math
    import candle.nn.functional as F

    warmup = torch.tensor([-2.0, 0.5, 3.0])
    _ = F.leaky_relu_(warmup, negative_slope=0.2)

    x = torch.tensor([-2.0, 0.5, 3.0])

    calls = {"count": 0}
    original = F.leaky_relu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "leaky_relu", wrapped)

    y = F.leaky_relu_(x, negative_slope=0.2)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -0.4, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 3.0, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_acos__bypasses_python_top_level_acos(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([1.0, 0.5, -0.5])
    _ = torch.acos_(warmup)

    x = torch.tensor([1.0, 0.5, -0.5])

    calls = {"count": 0}
    original = torch_module.acos

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "acos", wrapped)

    y = torch.acos_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.0471975511965979, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 2.0943951023931957, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_acos__bypasses_python__functional_acos(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([1.0, 0.5, -0.5])
    _ = torch.acos_(warmup)

    x = torch.tensor([1.0, 0.5, -0.5])

    calls = {"count": 0}
    original = _functional.acos

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "acos", wrapped)

    y = torch.acos_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.0471975511965979, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 2.0943951023931957, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_acosh__bypasses_python_top_level_acosh(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([1.0, 2.0, 4.0])
    _ = torch.acosh_(warmup)

    x = torch.tensor([1.0, 2.0, 4.0])

    calls = {"count": 0}
    original = torch_module.acosh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "acosh", wrapped)

    y = torch.acosh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.3169578969248166, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 2.0634370688955608, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_acosh__bypasses_python__functional_acosh(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([1.0, 2.0, 4.0])
    _ = torch.acosh_(warmup)

    x = torch.tensor([1.0, 2.0, 4.0])

    calls = {"count": 0}
    original = _functional.acosh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "acosh", wrapped)

    y = torch.acosh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.3169578969248166, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 2.0634370688955608, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_asin__bypasses_python_top_level_asin(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-1.0, 0.5, 1.0])
    _ = torch.asin_(warmup)

    x = torch.tensor([-1.0, 0.5, 1.0])

    calls = {"count": 0}
    original = torch_module.asin

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "asin", wrapped)

    y = torch.asin_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -1.5707963267948966, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.5235987755982989, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.5707963267948966, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_asin__bypasses_python__functional_asin(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([-1.0, 0.5, 1.0])
    _ = torch.asin_(warmup)

    x = torch.tensor([-1.0, 0.5, 1.0])

    calls = {"count": 0}
    original = _functional.asin

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "asin", wrapped)

    y = torch.asin_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -1.5707963267948966, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.5235987755982989, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.5707963267948966, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_asinh__bypasses_python_top_level_asinh(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-1.0, 0.0, 2.0])
    _ = torch.asinh_(warmup)

    x = torch.tensor([-1.0, 0.0, 2.0])

    calls = {"count": 0}
    original = torch_module.asinh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "asinh", wrapped)

    y = torch.asinh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -0.881373587019543, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.4436354751788103, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_asinh__bypasses_python__functional_asinh(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([-1.0, 0.0, 2.0])
    _ = torch.asinh_(warmup)

    x = torch.tensor([-1.0, 0.0, 2.0])

    calls = {"count": 0}
    original = _functional.asinh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "asinh", wrapped)

    y = torch.asinh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -0.881373587019543, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.4436354751788103, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_atan__bypasses_python_top_level_atan(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.atan_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = torch_module.atan

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "atan", wrapped)

    y = torch.atan_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -0.7853981633974483, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.7853981633974483, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_atan__bypasses_python__functional_atan(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.atan_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = _functional.atan

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "atan", wrapped)

    y = torch.atan_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -0.7853981633974483, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.7853981633974483, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_atanh__bypasses_python_top_level_atanh(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-0.5, 0.0, 0.5])
    _ = torch.atanh_(warmup)

    x = torch.tensor([-0.5, 0.0, 0.5])

    calls = {"count": 0}
    original = torch_module.atanh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "atanh", wrapped)

    y = torch.atanh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -0.5493061443340549, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.5493061443340549, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_atanh__bypasses_python__functional_atanh(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([-0.5, 0.0, 0.5])
    _ = torch.atanh_(warmup)

    x = torch.tensor([-0.5, 0.0, 0.5])

    calls = {"count": 0}
    original = _functional.atanh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "atanh", wrapped)

    y = torch.atanh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -0.5493061443340549, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.5493061443340549, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_cosh__bypasses_python_top_level_cosh(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.cosh_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = torch_module.cosh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "cosh", wrapped)

    y = torch.cosh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 1.5430806348152437, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.5430806348152437, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_cosh__bypasses_python__functional_cosh(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.cosh_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = _functional.cosh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "cosh", wrapped)

    y = torch.cosh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 1.5430806348152437, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.5430806348152437, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_erf__bypasses_python_top_level_erf(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.erf_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = torch_module.erf

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "erf", wrapped)

    y = torch.erf_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -0.8427007929497149, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.8427007929497149, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_erf__bypasses_python__functional_erf(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.erf_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = _functional.erf

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "erf", wrapped)

    y = torch.erf_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -0.8427007929497149, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.8427007929497149, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_erfc__bypasses_python_top_level_erfc(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.erfc_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = torch_module.erfc

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "erfc", wrapped)

    y = torch.erfc_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 1.842700792949715, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.15729920705028513, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_erfc__bypasses_python__functional_erfc(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.erfc_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = _functional.erfc

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "erfc", wrapped)

    y = torch.erfc_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 1.842700792949715, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.15729920705028513, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_selu__bypasses_python_nn_functional_selu_(monkeypatch):
    import math
    import candle.nn.functional as F

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.selu_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = F.selu_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "selu_", wrapped)

    y = torch.selu_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -1.1113307378125625, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.0507009873554805, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_celu__bypasses_python_nn_functional_celu_(monkeypatch):
    import math
    import candle.nn.functional as F

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.celu_(warmup, alpha=1.3)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = F.celu_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "celu_", wrapped)

    y = torch.celu_(x, alpha=1.3)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -0.6976198199994721, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.0, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_threshold__bypasses_python_nn_functional_threshold_(monkeypatch):
    import candle.nn.functional as F

    warmup = torch.tensor([-1.0, 0.5, 2.0])
    _ = torch.threshold_(warmup, 0.3, -7.0)

    x = torch.tensor([-1.0, 0.5, 2.0])

    calls = {"count": 0}
    original = F.threshold_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "threshold_", wrapped)

    y = torch.threshold_(x, 0.3, -7.0)

    assert calls["count"] == 0
    assert y is x
    assert y.tolist() == [-7.0, 0.5, 2.0]



def test_toplevel_dropout__bypasses_python_nn_functional_dropout(monkeypatch):
    import candle.nn.functional as F

    warmup = torch.tensor([1.0, 2.0, 3.0])
    _ = torch.dropout_(warmup, p=0.25, training=True)

    x = torch.tensor([1.0, 2.0, 3.0])

    calls = {"count": 0}
    original = F.dropout

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "dropout", wrapped)

    y = torch.dropout_(x, p=0.25, training=True)

    assert calls["count"] == 0
    assert y is x



def test_toplevel_bitwise_left_shift__bypasses_python__functional_dispatch(monkeypatch):
    import candle._functional as _functional

    warmup_a = torch.tensor([1, 2, 3], dtype=torch.int64)
    warmup_b = torch.tensor([1, 0, 2], dtype=torch.int64)
    _ = torch.bitwise_left_shift(warmup_a, warmup_b)

    a = torch.tensor([1, 2, 3], dtype=torch.int64)
    b = torch.tensor([1, 0, 2], dtype=torch.int64)

    calls = {"count": 0}
    original = _functional.dispatch

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "dispatch", wrapped)

    y = torch.bitwise_left_shift(a, b)

    assert calls["count"] == 0
    assert y.tolist() == [2, 2, 12]



def test_toplevel_bitwise_right_shift__bypasses_python__functional_dispatch(monkeypatch):
    import candle._functional as _functional

    warmup_a = torch.tensor([4, 8, 15], dtype=torch.int64)
    warmup_b = torch.tensor([1, 2, 1], dtype=torch.int64)
    _ = torch.bitwise_right_shift(warmup_a, warmup_b)

    a = torch.tensor([4, 8, 15], dtype=torch.int64)
    b = torch.tensor([1, 2, 1], dtype=torch.int64)

    calls = {"count": 0}
    original = _functional.dispatch

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "dispatch", wrapped)

    y = torch.bitwise_right_shift(a, b)

    assert calls["count"] == 0
    assert y.tolist() == [2, 2, 7]



def test_nn_functional_linear__bypasses_python__functional_add(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_x = torch.tensor([[1.0, 2.0]])
    warmup_w = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    warmup_b = torch.tensor([0.5, -1.0])
    _ = F.linear(warmup_x, warmup_w, bias=warmup_b)

    x = torch.tensor([[1.0, 2.0]])
    w = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([0.5, -1.0])

    calls = {"count": 0}
    original = _functional.add

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "add", wrapped)

    y = F.linear(x, w, bias=b)

    assert calls["count"] == 0
    assert y.tolist() == [[1.5, 1.0]]



def test_nn_functional_linear__bypasses_python__functional_matmul(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_x = torch.tensor([[1.0, 2.0]])
    warmup_w = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    _ = F.linear(warmup_x, warmup_w)

    x = torch.tensor([[1.0, 2.0]])
    w = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    calls = {"count": 0}
    original = _functional.matmul

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "matmul", wrapped)

    y = F.linear(x, w)

    assert calls["count"] == 0
    assert y.tolist() == [[1.0, 2.0]]



def test_nn_functional_lp_pool1d__bypasses_python__functional_abs(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[[-1.0, 2.0, -3.0, 4.0]]])
    _ = F.lp_pool1d(warmup, norm_type=2.0, kernel_size=2, stride=2)

    x = torch.tensor([[[-1.0, 2.0, -3.0, 4.0]]])

    calls = {"count": 0}
    original = _functional.abs

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "abs", wrapped)

    y = F.lp_pool1d(x, norm_type=2.0, kernel_size=2, stride=2)

    assert calls["count"] == 0
    assert y.shape == (1, 1, 2)

def test_nn_functional_lp_pool1d__bypasses_python__functional_pow(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[[-1.0, 2.0, -3.0, 4.0]]])
    _ = F.lp_pool1d(warmup, norm_type=2.0, kernel_size=2, stride=2)

    x = torch.tensor([[[-1.0, 2.0, -3.0, 4.0]]])

    calls = {"count": 0}
    original = _functional.pow

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "pow", wrapped)

    y = F.lp_pool1d(x, norm_type=2.0, kernel_size=2, stride=2)

    assert calls["count"] == 0
    assert y.shape == (1, 1, 2)


def test_nn_functional_lp_pool2d__bypasses_python__functional_abs(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[[[-1.0, 2.0], [-3.0, 4.0]]]])
    _ = F.lp_pool2d(warmup, norm_type=2.0, kernel_size=2, stride=1)

    x = torch.tensor([[[[-1.0, 2.0], [-3.0, 4.0]]]])

    calls = {"count": 0}
    original = _functional.abs

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "abs", wrapped)

    y = F.lp_pool2d(x, norm_type=2.0, kernel_size=2, stride=1)

    assert calls["count"] == 0
    assert y.shape == (1, 1, 1, 1)


def test_nn_functional_lp_pool2d__bypasses_python__functional_pow(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[[[-1.0, 2.0], [-3.0, 4.0]]]])
    _ = F.lp_pool2d(warmup, norm_type=2.0, kernel_size=2, stride=1)

    x = torch.tensor([[[[-1.0, 2.0], [-3.0, 4.0]]]])

    calls = {"count": 0}
    original = _functional.pow

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "pow", wrapped)

    y = F.lp_pool2d(x, norm_type=2.0, kernel_size=2, stride=1)

    assert calls["count"] == 0
    assert y.shape == (1, 1, 1, 1)


def test_nn_functional_lp_pool3d__abs_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[[[[-1.0, 2.0], [-3.0, 4.0]], [[5.0, -6.0], [7.0, -8.0]]]]])
    _ = F.lp_pool3d(warmup, norm_type=2.0, kernel_size=2, stride=1)

    x = torch.tensor([[[[[-1.0, 2.0], [-3.0, 4.0]], [[5.0, -6.0], [7.0, -8.0]]]]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "abs":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.lp_pool3d(x, norm_type=2.0, kernel_size=2, stride=1)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 1, 1, 1, 1)


def test_nn_functional_lp_pool3d__first_pow_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[[[[-1.0, 2.0], [-3.0, 4.0]], [[5.0, -6.0], [7.0, -8.0]]]]])
    _ = F.lp_pool3d(warmup, norm_type=2.0, kernel_size=2, stride=1)

    x = torch.tensor([[[[[-1.0, 2.0], [-3.0, 4.0]], [[5.0, -6.0], [7.0, -8.0]]]]])

    seen = {"dispatch_device": object(), "pow_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "pow":
            seen["pow_calls"] += 1
            if seen["pow_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.lp_pool3d(x, norm_type=2.0, kernel_size=2, stride=1)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 1, 1, 1, 1)


def test_nn_functional_lp_pool3d__second_pow_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[[[[-1.0, 2.0], [-3.0, 4.0]], [[5.0, -6.0], [7.0, -8.0]]]]])
    _ = F.lp_pool3d(warmup, norm_type=2.0, kernel_size=2, stride=1)

    x = torch.tensor([[[[[-1.0, 2.0], [-3.0, 4.0]], [[5.0, -6.0], [7.0, -8.0]]]]])

    seen = {"dispatch_device": object(), "pow_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "pow":
            seen["pow_calls"] += 1
            if seen["pow_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.lp_pool3d(x, norm_type=2.0, kernel_size=2, stride=1)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 1, 1, 1, 1)


def test_nn_functional_lp_pool3d__bypasses_python__functional_abs(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[[[[-1.0, 2.0], [-3.0, 4.0]], [[5.0, -6.0], [7.0, -8.0]]]]])
    _ = F.lp_pool3d(warmup, norm_type=2.0, kernel_size=2, stride=1)

    x = torch.tensor([[[[[-1.0, 2.0], [-3.0, 4.0]], [[5.0, -6.0], [7.0, -8.0]]]]])

    calls = {"count": 0}
    original = _functional.abs

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "abs", wrapped)

    y = F.lp_pool3d(x, norm_type=2.0, kernel_size=2, stride=1)

    assert calls["count"] == 0
    assert y.shape == (1, 1, 1, 1, 1)


def test_nn_functional_lp_pool3d__bypasses_python__functional_pow(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[[[[-1.0, 2.0], [-3.0, 4.0]], [[5.0, -6.0], [7.0, -8.0]]]]])
    _ = F.lp_pool3d(warmup, norm_type=2.0, kernel_size=2, stride=1)

    x = torch.tensor([[[[[-1.0, 2.0], [-3.0, 4.0]], [[5.0, -6.0], [7.0, -8.0]]]]])

    calls = {"count": 0}
    original = _functional.pow

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "pow", wrapped)

    y = F.lp_pool3d(x, norm_type=2.0, kernel_size=2, stride=1)

    assert calls["count"] == 0
    assert y.shape == (1, 1, 1, 1, 1)


def test_nn_functional_multi_margin_loss__gather_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[0.2, -0.4, 0.9], [1.1, 0.3, -0.2]], dtype=torch.float32)
    warmup_target = torch.tensor([2, 0], dtype=torch.int64)
    _ = F.multi_margin_loss(warmup_input, warmup_target, reduction='mean')

    x = torch.tensor([[0.2, -0.4, 0.9], [1.1, 0.3, -0.2]], dtype=torch.float32)
    target = torch.tensor([2, 0], dtype=torch.int64)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "gather":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.multi_margin_loss(x, target, reduction='mean')

    assert seen["dispatch_device"] is None
    assert y.ndim == 0


def test_nn_functional_multi_margin_loss__eq_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[0.2, -0.4, 0.9], [1.1, 0.3, -0.2]], dtype=torch.float32)
    warmup_target = torch.tensor([2, 0], dtype=torch.int64)
    _ = F.multi_margin_loss(warmup_input, warmup_target, reduction='mean')

    x = torch.tensor([[0.2, -0.4, 0.9], [1.1, 0.3, -0.2]], dtype=torch.float32)
    target = torch.tensor([2, 0], dtype=torch.int64)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "eq":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.multi_margin_loss(x, target, reduction='mean')

    assert seen["dispatch_device"] is None
    assert y.ndim == 0


def test_nn_functional_multi_margin_loss__sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[0.2, -0.4, 0.9], [1.1, 0.3, -0.2]], dtype=torch.float32)
    warmup_target = torch.tensor([2, 0], dtype=torch.int64)
    _ = F.multi_margin_loss(warmup_input, warmup_target, reduction='mean')

    x = torch.tensor([[0.2, -0.4, 0.9], [1.1, 0.3, -0.2]], dtype=torch.float32)
    target = torch.tensor([2, 0], dtype=torch.int64)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.multi_margin_loss(x, target, reduction='mean')

    assert seen["dispatch_device"] is None
    assert y.ndim == 0


def test_nn_functional_multi_margin_loss__div_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[0.2, -0.4, 0.9], [1.1, 0.3, -0.2]], dtype=torch.float32)
    warmup_target = torch.tensor([2, 0], dtype=torch.int64)
    _ = F.multi_margin_loss(warmup_input, warmup_target, reduction='mean')

    x = torch.tensor([[0.2, -0.4, 0.9], [1.1, 0.3, -0.2]], dtype=torch.float32)
    target = torch.tensor([2, 0], dtype=torch.int64)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "div":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.multi_margin_loss(x, target, reduction='mean')

    assert seen["dispatch_device"] is None
    assert y.ndim == 0


def test_nn_functional_multilabel_margin_loss__ge_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[0.6, 0.2, -0.4]], dtype=torch.float32)
    warmup_target = torch.tensor([[0, 2, -1]], dtype=torch.int64)
    _ = F.multilabel_margin_loss(warmup_input, warmup_target, reduction='mean')

    x = torch.tensor([[0.6, 0.2, -0.4]], dtype=torch.float32)
    target = torch.tensor([[0, 2, -1]], dtype=torch.int64)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "ge":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.multilabel_margin_loss(x, target, reduction='mean')

    assert seen["dispatch_device"] is None
    assert y.ndim == 0


def test_nn_functional_multilabel_margin_loss__first_eq_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[0.6, 0.2, -0.4]], dtype=torch.float32)
    warmup_target = torch.tensor([[0, 2, -1]], dtype=torch.int64)
    _ = F.multilabel_margin_loss(warmup_input, warmup_target, reduction='mean')

    x = torch.tensor([[0.6, 0.2, -0.4]], dtype=torch.float32)
    target = torch.tensor([[0, 2, -1]], dtype=torch.int64)

    seen = {"dispatch_device": object(), "eq_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "eq":
            seen["eq_calls"] += 1
            if seen["eq_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.multilabel_margin_loss(x, target, reduction='mean')

    assert seen["dispatch_device"] is None
    assert y.ndim == 0


def test_nn_functional_multilabel_margin_loss__any_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[0.6, 0.2, -0.4]], dtype=torch.float32)
    warmup_target = torch.tensor([[0, 2, -1]], dtype=torch.int64)
    _ = F.multilabel_margin_loss(warmup_input, warmup_target, reduction='mean')

    x = torch.tensor([[0.6, 0.2, -0.4]], dtype=torch.float32)
    target = torch.tensor([[0, 2, -1]], dtype=torch.int64)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "any":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.multilabel_margin_loss(x, target, reduction='mean')

    assert seen["dispatch_device"] is None
    assert y.ndim == 0


def test_nn_functional_multilabel_margin_loss__second_eq_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[0.6, 0.2, -0.4]], dtype=torch.float32)
    warmup_target = torch.tensor([[0, 2, -1]], dtype=torch.int64)
    _ = F.multilabel_margin_loss(warmup_input, warmup_target, reduction='mean')

    x = torch.tensor([[0.6, 0.2, -0.4]], dtype=torch.float32)
    target = torch.tensor([[0, 2, -1]], dtype=torch.int64)

    seen = {"dispatch_device": object(), "eq_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "eq":
            seen["eq_calls"] += 1
            if seen["eq_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.multilabel_margin_loss(x, target, reduction='mean')

    assert seen["dispatch_device"] is None
    assert y.ndim == 0


def test_nn_functional_multilabel_margin_loss__div_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[0.6, 0.2, -0.4]], dtype=torch.float32)
    warmup_target = torch.tensor([[0, 2, -1]], dtype=torch.int64)
    _ = F.multilabel_margin_loss(warmup_input, warmup_target, reduction='mean')

    x = torch.tensor([[0.6, 0.2, -0.4]], dtype=torch.float32)
    target = torch.tensor([[0, 2, -1]], dtype=torch.int64)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "div":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.multilabel_margin_loss(x, target, reduction='mean')

    assert seen["dispatch_device"] is None
    assert y.ndim == 0


def test_nn_functional_multilabel_margin_loss__stack_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[0.6, 0.2, -0.4]], dtype=torch.float32)
    warmup_target = torch.tensor([[0, 2, -1]], dtype=torch.int64)
    _ = F.multilabel_margin_loss(warmup_input, warmup_target, reduction='mean')

    x = torch.tensor([[0.6, 0.2, -0.4]], dtype=torch.float32)
    target = torch.tensor([[0, 2, -1]], dtype=torch.int64)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "stack":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.multilabel_margin_loss(x, target, reduction='mean')

    assert seen["dispatch_device"] is None
    assert y.ndim == 0


def test_nn_functional_multilabel_soft_margin_loss__sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[0.7, -0.2, 0.1]], dtype=torch.float32)
    warmup_target = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
    _ = F.multilabel_soft_margin_loss(warmup_input, warmup_target, reduction='mean')

    x = torch.tensor([[0.7, -0.2, 0.1]], dtype=torch.float32)
    target = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)

    seen = {"dispatch_device": object(), "sum_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["sum_calls"] += 1
            if seen["sum_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.multilabel_soft_margin_loss(x, target, reduction='mean')

    assert seen["dispatch_device"] is None
    assert y.ndim == 0


def test_nn_functional_multilabel_soft_margin_loss__div_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[0.7, -0.2, 0.1]], dtype=torch.float32)
    warmup_target = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
    _ = F.multilabel_soft_margin_loss(warmup_input, warmup_target, reduction='mean')

    x = torch.tensor([[0.7, -0.2, 0.1]], dtype=torch.float32)
    target = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "div":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.multilabel_soft_margin_loss(x, target, reduction='mean')

    assert seen["dispatch_device"] is None
    assert y.ndim == 0


def test_nn_functional_tanhshrink__tanh_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float32)
    _ = F.tanhshrink(warmup)

    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "tanh":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.tanhshrink(x)

    assert seen["dispatch_device"] is None
    assert y.shape == (4,)


def test_nn_functional_tanhshrink__neg_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float32)
    _ = F.tanhshrink(warmup)

    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "neg":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.tanhshrink(x)

    assert seen["dispatch_device"] is None
    assert y.shape == (4,)


def test_nn_functional_tanhshrink__add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float32)
    _ = F.tanhshrink(warmup)

    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.tanhshrink(x)

    assert seen["dispatch_device"] is None
    assert y.shape == (4,)


def test_nn_functional_pixel_shuffle__first_reshape_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
    _ = F.pixel_shuffle(warmup, 2)

    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)

    seen = {"dispatch_device": object(), "reshape_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "reshape":
            seen["reshape_calls"] += 1
            if seen["reshape_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pixel_shuffle(x, 2)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 1, 4, 4)


def test_nn_functional_pixel_shuffle__permute_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
    _ = F.pixel_shuffle(warmup, 2)

    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "permute":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pixel_shuffle(x, 2)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 1, 4, 4)


def test_nn_functional_pixel_shuffle__second_reshape_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
    _ = F.pixel_shuffle(warmup, 2)

    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)

    seen = {"dispatch_device": object(), "reshape_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "reshape":
            seen["reshape_calls"] += 1
            if seen["reshape_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pixel_shuffle(x, 2)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 1, 4, 4)


def test_nn_functional_pixel_unshuffle__first_reshape_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    _ = F.pixel_unshuffle(warmup, 2)

    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    seen = {"dispatch_device": object(), "reshape_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "reshape":
            seen["reshape_calls"] += 1
            if seen["reshape_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pixel_unshuffle(x, 2)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 4, 2, 2)


def test_nn_functional_pixel_unshuffle__permute_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    _ = F.pixel_unshuffle(warmup, 2)

    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "permute":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pixel_unshuffle(x, 2)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 4, 2, 2)


def test_nn_functional_pixel_unshuffle__second_reshape_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    _ = F.pixel_unshuffle(warmup, 2)

    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    seen = {"dispatch_device": object(), "reshape_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "reshape":
            seen["reshape_calls"] += 1
            if seen["reshape_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pixel_unshuffle(x, 2)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 4, 2, 2)


def test_nn_functional_channel_shuffle__first_reshape_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
    _ = F.channel_shuffle(warmup, 2)

    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)

    seen = {"dispatch_device": object(), "reshape_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "reshape":
            seen["reshape_calls"] += 1
            if seen["reshape_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.channel_shuffle(x, 2)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 4, 2, 2)


def test_nn_functional_channel_shuffle__permute_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
    _ = F.channel_shuffle(warmup, 2)

    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "permute":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.channel_shuffle(x, 2)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 4, 2, 2)


def test_nn_functional_channel_shuffle__second_reshape_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
    _ = F.channel_shuffle(warmup, 2)

    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)

    seen = {"dispatch_device": object(), "reshape_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "reshape":
            seen["reshape_calls"] += 1
            if seen["reshape_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.channel_shuffle(x, 2)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 4, 2, 2)


def test_nn_functional_hardtanh__bypasses_python__functional_hardtanh(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    _ = F.hardtanh(warmup, min_val=-0.75, max_val=0.75)

    x = torch.tensor([-2.0, -0.5, 0.5, 2.0])

    calls = {"count": 0}
    original = _functional.hardtanh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "hardtanh", wrapped)

    y = F.hardtanh(x, min_val=-0.75, max_val=0.75)

    assert calls["count"] == 0
    assert y.tolist() == [-0.75, -0.5, 0.5, 0.75]


def test_nn_functional_relu6__bypasses_python__functional_relu6(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-2.0, 0.5, 6.5, 8.0])
    _ = F.relu6(warmup)

    x = torch.tensor([-2.0, 0.5, 6.5, 8.0])

    calls = {"count": 0}
    original = _functional.relu6

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "relu6", wrapped)

    y = F.relu6(x)

    assert calls["count"] == 0
    assert y.tolist() == [0.0, 0.5, 6.0, 6.0]


def test_nn_functional_softplus__bypasses_python__functional_softplus(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-2.0, 0.0, 2.0])
    _ = F.softplus(warmup)

    x = torch.tensor([-2.0, 0.0, 2.0])

    calls = {"count": 0}
    original = _functional.softplus

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "softplus", wrapped)

    y = F.softplus(x)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 3
    assert abs(vals[0] - 0.1269280110429725) < 1e-6
    assert abs(vals[1] - 0.6931471805599453) < 1e-6
    assert abs(vals[2] - 2.1269280110429727) < 1e-6


def test_nn_functional_softmin__bypasses_python__functional_neg(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.softmin(warmup, dim=1)

    x = torch.tensor([[1.0, 2.0, 3.0]])

    calls = {"count": 0}
    original = _functional.neg

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "neg", wrapped)

    y = F.softmin(x, dim=1)

    assert calls["count"] == 0
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6
    assert vals[0] > vals[1] > vals[2]


def test_nn_functional_tanhshrink__bypasses_python__functional_tanh(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-2.0, 0.0, 2.0])
    _ = F.tanhshrink(warmup)

    x = torch.tensor([-2.0, 0.0, 2.0])

    calls = {"count": 0}
    original = _functional.tanh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "tanh", wrapped)

    y = F.tanhshrink(x)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 3
    assert vals[0] < 0.0
    assert vals[1] == 0.0
    assert vals[2] > 0.0


def test_nn_functional_tanhshrink__bypasses_python__functional_neg(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-2.0, 0.0, 2.0])
    _ = F.tanhshrink(warmup)

    x = torch.tensor([-2.0, 0.0, 2.0])

    calls = {"count": 0}
    original = _functional.neg

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "neg", wrapped)

    y = F.tanhshrink(x)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 3
    assert vals[0] < 0.0
    assert vals[1] == 0.0
    assert vals[2] > 0.0


def test_nn_functional_tanhshrink__bypasses_python__functional_add(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-2.0, 0.0, 2.0])
    _ = F.tanhshrink(warmup)

    x = torch.tensor([-2.0, 0.0, 2.0])

    calls = {"count": 0}
    original = _functional.add

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "add", wrapped)

    y = F.tanhshrink(x)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 3
    assert vals[0] < 0.0
    assert vals[1] == 0.0
    assert vals[2] > 0.0


def test_nn_functional_rms_norm__bypasses_python__functional_rms_norm(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weight = torch.tensor([1.0, 1.0])
    _ = F.rms_norm(warmup, (2,), weight=weight, eps=1e-6)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    calls = {"count": 0}
    original = _functional.rms_norm

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "rms_norm", wrapped)

    y = F.rms_norm(x, (2,), weight=weight, eps=1e-6)

    assert calls["count"] == 0
    assert y.shape == (2, 2)


def test_nn_functional_logsigmoid__bypasses_python__functional_neg(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-2.0, 0.0, 2.0])
    _ = F.logsigmoid(warmup)

    x = torch.tensor([-2.0, 0.0, 2.0])

    calls = {"count": 0}
    original = _functional.neg

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "neg", wrapped)

    y = F.logsigmoid(x)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 3
    assert vals[0] < vals[1] < vals[2]


def test_nn_functional_logsigmoid__bypasses_python__functional_softplus(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-2.0, 0.0, 2.0])
    _ = F.logsigmoid(warmup)

    x = torch.tensor([-2.0, 0.0, 2.0])

    calls = {"count": 0}
    original = _functional.softplus

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "softplus", wrapped)

    y = F.logsigmoid(x)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 3
    assert vals[0] < vals[1] < vals[2]


def test_nn_functional_glu__bypasses_python__functional_sigmoid(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    _ = F.glu(warmup, dim=1)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    calls = {"count": 0}
    original = _functional.sigmoid

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "sigmoid", wrapped)

    y = F.glu(x, dim=1)

    assert calls["count"] == 0
    assert y.shape == (1, 2)


def test_nn_functional_glu__bypasses_python__functional_mul(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    _ = F.glu(warmup, dim=1)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    calls = {"count": 0}
    original = _functional.mul

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mul", wrapped)

    y = F.glu(x, dim=1)

    assert calls["count"] == 0
    assert y.shape == (1, 2)



def test_nn_functional_cosine_similarity__bypasses_python__functional_mul(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    warmup_x2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    _ = F.cosine_similarity(warmup_x1, warmup_x2, dim=1)

    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])

    calls = {"count": 0}
    original = _functional.mul

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mul", wrapped)

    y = F.cosine_similarity(x1, x2, dim=1)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 2
    assert vals[0] > 0.0
    assert vals[1] > 0.0


def test_nn_functional_cosine_similarity__bypasses_python__functional_add(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    warmup_x2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    _ = F.cosine_similarity(warmup_x1, warmup_x2, dim=1)

    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])

    calls = {"count": 0}
    original = _functional.add

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "add", wrapped)

    y = F.cosine_similarity(x1, x2, dim=1)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 2
    assert vals[0] > 0.0
    assert vals[1] > 0.0



def test_nn_functional_cosine_similarity__bypasses_python__functional_div(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    warmup_x2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    _ = F.cosine_similarity(warmup_x1, warmup_x2, dim=1)

    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])

    calls = {"count": 0}
    original = _functional.div

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "div", wrapped)

    y = F.cosine_similarity(x1, x2, dim=1)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 2
    assert vals[0] > 0.0
    assert vals[1] > 0.0



def test_nn_functional_pairwise_distance__bypasses_python__functional_add(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    warmup_x2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    _ = F.pairwise_distance(warmup_x1, warmup_x2)

    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])

    calls = {"count": 0}
    original = _functional.add

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "add", wrapped)

    y = F.pairwise_distance(x1, x2)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 2
    assert vals[0] > 0.0
    assert vals[1] > 0.0


def test_nn_functional_pairwise_distance__bypasses_python__functional_neg(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    warmup_x2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    _ = F.pairwise_distance(warmup_x1, warmup_x2)

    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])

    calls = {"count": 0}
    original = _functional.neg

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "neg", wrapped)

    y = F.pairwise_distance(x1, x2)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 2
    assert vals[0] > 0.0
    assert vals[1] > 0.0



def test_nn_functional_pairwise_distance__bypasses_python__functional_abs(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    warmup_x2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    _ = F.pairwise_distance(warmup_x1, warmup_x2, p=1.0)

    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])

    calls = {"count": 0}
    original = _functional.abs

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "abs", wrapped)

    y = F.pairwise_distance(x1, x2, p=1.0)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 2
    assert vals[0] > 0.0
    assert vals[1] > 0.0



def test_nn_functional_pairwise_distance__bypasses_python__functional_pow(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    warmup_x2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    _ = F.pairwise_distance(warmup_x1, warmup_x2, p=3.0)

    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])

    calls = {"count": 0}
    original = _functional.pow

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "pow", wrapped)

    y = F.pairwise_distance(x1, x2, p=3.0)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 2
    assert vals[0] > 0.0
    assert vals[1] > 0.0



def test_nn_functional_gumbel_softmax__bypasses_python__functional_neg(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=False, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    calls = {"count": 0}
    original = _functional.neg

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "neg", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=1)

    assert calls["count"] == 0
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6



def test_nn_functional_gumbel_softmax__uniform_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=False, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "uniform":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=1)

    assert seen["dispatch_device"] is None
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6


def test_nn_functional_gumbel_softmax__first_add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=False, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    seen = {"dispatch_device": object(), "add_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["add_calls"] += 1
            if seen["add_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=1)

    assert seen["add_calls"] >= 1
    assert seen["dispatch_device"] is None
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6


def test_nn_functional_gumbel_softmax__first_log_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=False, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    seen = {"dispatch_device": object(), "log_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "log":
            seen["log_calls"] += 1
            if seen["log_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=1)

    assert seen["log_calls"] >= 1
    assert seen["dispatch_device"] is None
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6


def test_nn_functional_gumbel_softmax__first_neg_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=False, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    seen = {"dispatch_device": object(), "neg_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "neg":
            seen["neg_calls"] += 1
            if seen["neg_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=1)

    assert seen["neg_calls"] >= 1
    assert seen["dispatch_device"] is None
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6


def test_nn_functional_gumbel_softmax__second_add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=False, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    seen = {"dispatch_device": object(), "add_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["add_calls"] += 1
            if seen["add_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=1)

    assert seen["add_calls"] >= 2
    assert seen["dispatch_device"] is None
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6


def test_nn_functional_gumbel_softmax__second_log_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=False, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    seen = {"dispatch_device": object(), "log_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "log":
            seen["log_calls"] += 1
            if seen["log_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=1)

    assert seen["log_calls"] >= 2
    assert seen["dispatch_device"] is None
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6


def test_nn_functional_gumbel_softmax__second_neg_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=False, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    seen = {"dispatch_device": object(), "neg_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "neg":
            seen["neg_calls"] += 1
            if seen["neg_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=1)

    assert seen["neg_calls"] >= 2
    assert seen["dispatch_device"] is None
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6


def test_nn_functional_gumbel_softmax__third_add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=False, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    seen = {"dispatch_device": object(), "add_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["add_calls"] += 1
            if seen["add_calls"] == 3:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=1)

    assert seen["add_calls"] >= 3
    assert seen["dispatch_device"] is None
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6


def test_nn_functional_gumbel_softmax__div_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=False, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "div":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=1)

    assert seen["dispatch_device"] is None
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6


def test_nn_functional_gumbel_softmax__hard_argmax_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=True, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "argmax":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=1)

    assert seen["dispatch_device"] is None
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6
    assert sum(v == 1.0 for v in vals) == 1


def test_nn_functional_gumbel_softmax__hard_one_hot_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=True, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "one_hot":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=1)

    assert seen["dispatch_device"] is None
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6
    assert sum(v == 1.0 for v in vals) == 1


def test_nn_functional_gumbel_softmax__hard_neg_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=True, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    seen = {"dispatch_device": object(), "neg_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "neg":
            seen["neg_calls"] += 1
            if seen["neg_calls"] == 3:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=1)

    assert seen["neg_calls"] >= 3
    assert seen["dispatch_device"] is None
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6
    assert sum(v == 1.0 for v in vals) == 1


def test_nn_functional_gumbel_softmax__hard_base_add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=True, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    seen = {"dispatch_device": object(), "add_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["add_calls"] += 1
            if seen["add_calls"] == 4:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=1)

    assert seen["add_calls"] >= 4
    assert seen["dispatch_device"] is None
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6
    assert sum(v == 1.0 for v in vals) == 1


def test_nn_functional_gumbel_softmax__hard_final_add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=True, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    seen = {"dispatch_device": object(), "add_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["add_calls"] += 1
            if seen["add_calls"] == 5:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=1)

    assert seen["add_calls"] >= 5
    assert seen["dispatch_device"] is None
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6
    assert sum(v == 1.0 for v in vals) == 1


def test_nn_functional_gumbel_softmax__bypasses_python__functional_div(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=False, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    calls = {"count": 0}
    original = _functional.div

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "div", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=1)

    assert calls["count"] == 0
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6



def test_nn_functional_gumbel_softmax__bypasses_python__functional_log(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=False, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    calls = {"count": 0}
    original = _functional.log

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "log", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=1)

    assert calls["count"] == 0
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6



def test_nn_functional_gumbel_softmax__bypasses_python__functional_add(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=False, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    calls = {"count": 0}
    original = _functional.add

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "add", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=1)

    assert calls["count"] == 0
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6



def test_nn_functional_gumbel_softmax__bypasses_python__functional_neg_hard(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=True, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    calls = {"count": 0}
    original = _functional.neg

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "neg", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=1)

    assert calls["count"] == 0
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6
    assert sum(v == 1.0 for v in vals) == 1



def test_nn_functional_gumbel_softmax__bypasses_python__functional_add_hard(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[1.0, 2.0, 3.0]])
    _ = F.gumbel_softmax(warmup, tau=1.0, hard=True, dim=1)

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    calls = {"count": 0}
    original = _functional.add

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "add", wrapped)

    y = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=1)

    assert calls["count"] == 0
    vals = y.tolist()[0]
    assert len(vals) == 3
    assert abs(sum(vals) - 1.0) < 1e-6
    assert sum(v == 1.0 for v in vals) == 1



def test_nn_functional_dropout1d__bypasses_python__functional_mul(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F
    warmup = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    _ = F.dropout1d(warmup, p=0.25, training=True)

    x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    calls = {"count": 0}
    original = _functional.mul

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mul", wrapped)

    _ = F.dropout1d(x, p=0.25, training=True)

    assert calls["count"] == 0



def test_nn_functional_alpha_dropout__bypasses_python__functional_rand(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([1.0, 2.0, 3.0])
    _ = F.alpha_dropout(warmup, p=0.25, training=True)

    x = torch.tensor([1.0, 2.0, 3.0])

    calls = {"count": 0}
    original = _functional.rand

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "rand", wrapped)

    _ = F.alpha_dropout(x, p=0.25, training=True)

    assert calls["count"] == 0



def test_nn_functional_alpha_dropout__uniform_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([1.0, 2.0, 3.0])
    _ = F.alpha_dropout(warmup, p=0.25, training=True)

    x = torch.tensor([1.0, 2.0, 3.0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "uniform":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.alpha_dropout(x, p=0.25, training=True)

    assert seen["dispatch_device"] is None
    assert y.shape == (3,)


def test_nn_functional_alpha_dropout__ge_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([1.0, 2.0, 3.0])
    _ = F.alpha_dropout(warmup, p=0.25, training=True)

    x = torch.tensor([1.0, 2.0, 3.0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "ge":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.alpha_dropout(x, p=0.25, training=True)

    assert seen["dispatch_device"] is None
    assert y.shape == (3,)


def test_nn_functional_alpha_dropout__where_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([1.0, 2.0, 3.0])
    _ = F.alpha_dropout(warmup, p=0.25, training=True)

    x = torch.tensor([1.0, 2.0, 3.0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "where":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.alpha_dropout(x, p=0.25, training=True)

    assert seen["dispatch_device"] is None
    assert y.shape == (3,)


def test_nn_functional_alpha_dropout__mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([1.0, 2.0, 3.0])
    _ = F.alpha_dropout(warmup, p=0.25, training=True)

    x = torch.tensor([1.0, 2.0, 3.0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.alpha_dropout(x, p=0.25, training=True)

    assert seen["dispatch_device"] is None
    assert y.shape == (3,)


def test_nn_functional_alpha_dropout__add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([1.0, 2.0, 3.0])
    _ = F.alpha_dropout(warmup, p=0.25, training=True)

    x = torch.tensor([1.0, 2.0, 3.0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.alpha_dropout(x, p=0.25, training=True)

    assert seen["dispatch_device"] is None
    assert y.shape == (3,)


def test_nn_functional_alpha_dropout__bypasses_python__functional_mul(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([1.0, 2.0, 3.0])
    _ = F.alpha_dropout(warmup, p=0.25, training=True)

    x = torch.tensor([1.0, 2.0, 3.0])

    calls = {"count": 0}
    original = _functional.mul

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mul", wrapped)

    y = F.alpha_dropout(x, p=0.25, training=True)

    assert calls["count"] == 0
    assert y.shape == (3,)



def test_nn_functional_alpha_dropout__bypasses_python__functional_add(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([1.0, 2.0, 3.0])
    _ = F.alpha_dropout(warmup, p=0.25, training=True)

    x = torch.tensor([1.0, 2.0, 3.0])

    calls = {"count": 0}
    original = _functional.add

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "add", wrapped)

    y = F.alpha_dropout(x, p=0.25, training=True)

    assert calls["count"] == 0
    assert y.shape == (3,)



def test_nn_functional_alpha_dropout__bypasses_python__functional_where(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([1.0, 2.0, 3.0])
    _ = F.alpha_dropout(warmup, p=0.25, training=True)

    x = torch.tensor([1.0, 2.0, 3.0])

    calls = {"count": 0}
    original = _functional.where

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "where", wrapped)

    y = F.alpha_dropout(x, p=0.25, training=True)

    assert calls["count"] == 0
    assert y.shape == (3,)



def test_nn_functional_bilinear__permute_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_x1 = torch.tensor([[1.0, 2.0]])
    warmup_x2 = torch.tensor([[3.0, 4.0]])
    warmup_w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    _ = F.bilinear(warmup_x1, warmup_x2, warmup_w)

    x1 = torch.tensor([[1.0, 2.0]])
    x2 = torch.tensor([[3.0, 4.0]])
    w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "permute":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.bilinear(x1, x2, w)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 1)
    assert y.tolist() == [[11.0]]



def test_nn_functional_bilinear__mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_x1 = torch.tensor([[1.0, 2.0]])
    warmup_x2 = torch.tensor([[3.0, 4.0]])
    warmup_w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    _ = F.bilinear(warmup_x1, warmup_x2, warmup_w)

    x1 = torch.tensor([[1.0, 2.0]])
    x2 = torch.tensor([[3.0, 4.0]])
    w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.bilinear(x1, x2, w)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 1)
    assert y.tolist() == [[11.0]]



def test_nn_functional_bilinear__sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_x1 = torch.tensor([[1.0, 2.0]])
    warmup_x2 = torch.tensor([[3.0, 4.0]])
    warmup_w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    _ = F.bilinear(warmup_x1, warmup_x2, warmup_w)

    x1 = torch.tensor([[1.0, 2.0]])
    x2 = torch.tensor([[3.0, 4.0]])
    w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.bilinear(x1, x2, w)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 1)
    assert y.tolist() == [[11.0]]



def test_nn_functional_bilinear__add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_x1 = torch.tensor([[1.0, 2.0]])
    warmup_x2 = torch.tensor([[3.0, 4.0]])
    warmup_w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    warmup_b = torch.tensor([0.5])
    _ = F.bilinear(warmup_x1, warmup_x2, warmup_w, bias=warmup_b)

    x1 = torch.tensor([[1.0, 2.0]])
    x2 = torch.tensor([[3.0, 4.0]])
    w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    b = torch.tensor([0.5])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.bilinear(x1, x2, w, bias=b)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 1)
    assert y.tolist() == [[11.5]]



def test_nn_functional_bilinear__bypasses_python__functional_mul(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_x1 = torch.tensor([[1.0, 2.0]])
    warmup_x2 = torch.tensor([[3.0, 4.0]])
    warmup_w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    _ = F.bilinear(warmup_x1, warmup_x2, warmup_w)

    x1 = torch.tensor([[1.0, 2.0]])
    x2 = torch.tensor([[3.0, 4.0]])
    w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    calls = {"count": 0}
    original = _functional.mul

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mul", wrapped)

    y = F.bilinear(x1, x2, w)

    assert calls["count"] == 0
    assert y.shape == (1, 1)
    assert y.tolist() == [[11.0]]



def test_nn_functional_bilinear__bypasses_python__functional_matmul(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_x1 = torch.tensor([[1.0, 2.0]])
    warmup_x2 = torch.tensor([[3.0, 4.0]])
    warmup_w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    _ = F.bilinear(warmup_x1, warmup_x2, warmup_w)

    x1 = torch.tensor([[1.0, 2.0]])
    x2 = torch.tensor([[3.0, 4.0]])
    w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    calls = {"count": 0}
    original = _functional.matmul

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "matmul", wrapped)

    y = F.bilinear(x1, x2, w)

    assert calls["count"] == 0
    assert y.shape == (1, 1)
    assert y.tolist() == [[11.0]]



def test_nn_functional_bilinear__bypasses_python__functional_add(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_x1 = torch.tensor([[1.0, 2.0]])
    warmup_x2 = torch.tensor([[3.0, 4.0]])
    warmup_w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    warmup_b = torch.tensor([0.5])
    _ = F.bilinear(warmup_x1, warmup_x2, warmup_w, bias=warmup_b)

    x1 = torch.tensor([[1.0, 2.0]])
    x2 = torch.tensor([[3.0, 4.0]])
    w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    b = torch.tensor([0.5])

    calls = {"count": 0}
    original = _functional.add

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "add", wrapped)

    y = F.bilinear(x1, x2, w, bias=b)

    assert calls["count"] == 0
    assert y.shape == (1, 1)
    assert y.tolist() == [[11.5]]



def test_nn_functional_local_response_norm__mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    _ = F.local_response_norm(warmup, size=3)

    x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])

    seen = {"dispatch_device": object(), "mul_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul":
            seen["mul_calls"] += 1
            if seen["mul_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.local_response_norm(x, size=3)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 3, 1, 1)



def test_nn_functional_local_response_norm__second_mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    _ = F.local_response_norm(warmup, size=3)

    x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])

    seen = {"dispatch_device": object(), "mul_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul":
            seen["mul_calls"] += 1
            if seen["mul_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.local_response_norm(x, size=3)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 3, 1, 1)



def test_nn_functional_local_response_norm__second_add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    _ = F.local_response_norm(warmup, size=3)

    x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])

    seen = {"dispatch_device": object(), "add_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["add_calls"] += 1
            if seen["add_calls"] == 8:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.local_response_norm(x, size=3)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 3, 1, 1)



def test_nn_functional_local_response_norm__add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    _ = F.local_response_norm(warmup, size=3)

    x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])

    seen = {"dispatch_device": object(), "add_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["add_calls"] += 1
            if seen["add_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.local_response_norm(x, size=3)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 3, 1, 1)



def test_nn_functional_local_response_norm__bypasses_python__functional_mul(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    _ = F.local_response_norm(warmup, size=3)

    x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])

    calls = {"count": 0}
    original = _functional.mul

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mul", wrapped)

    y = F.local_response_norm(x, size=3)

    assert calls["count"] == 0
    assert y.shape == (1, 3, 1, 1)



def test_nn_functional_local_response_norm__bypasses_python__functional_div(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    _ = F.local_response_norm(warmup, size=3)

    x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])

    calls = {"count": 0}
    original = _functional.div

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "div", wrapped)

    y = F.local_response_norm(x, size=3)

    assert calls["count"] == 0
    assert y.shape == (1, 3, 1, 1)



def test_nn_functional_local_response_norm__bypasses_python__functional_add(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    _ = F.local_response_norm(warmup, size=3)

    x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])

    calls = {"count": 0}
    original = _functional.add

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "add", wrapped)

    y = F.local_response_norm(x, size=3)

    assert calls["count"] == 0
    assert y.shape == (1, 3, 1, 1)



def test_nn_functional_local_response_norm__pow_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    _ = F.local_response_norm(warmup, size=3)

    x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "pow":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.local_response_norm(x, size=3)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 3, 1, 1)



def test_nn_functional_local_response_norm__div_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    _ = F.local_response_norm(warmup, size=3)

    x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "div":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.local_response_norm(x, size=3)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 3, 1, 1)



def test_nn_functional_local_response_norm__setitem_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    _ = F.local_response_norm(warmup, size=3)

    x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "setitem":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.local_response_norm(x, size=3)

    assert seen["dispatch_device"] is None
    assert y.shape == (1, 3, 1, 1)



def test_nn_functional_local_response_norm__bypasses_python__functional_pow(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    _ = F.local_response_norm(warmup, size=3)

    x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])

    calls = {"count": 0}
    original = _functional.pow

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "pow", wrapped)

    y = F.local_response_norm(x, size=3)

    assert calls["count"] == 0
    assert y.shape == (1, 3, 1, 1)



def test_nn_functional_pdist__neg_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _ = F.pdist(warmup)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "neg":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pdist(x)

    assert seen["dispatch_device"] is None
    vals = y.tolist()
    assert len(vals) == 3
    assert all(v > 0.0 for v in vals)



def test_nn_functional_pdist__add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _ = F.pdist(warmup)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pdist(x)

    assert seen["dispatch_device"] is None
    vals = y.tolist()
    assert len(vals) == 3
    assert all(v > 0.0 for v in vals)



def test_nn_functional_pdist__norm_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _ = F.pdist(warmup)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "norm":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pdist(x)

    assert seen["dispatch_device"] is None
    vals = y.tolist()
    assert len(vals) == 3
    assert all(v > 0.0 for v in vals)



def test_nn_functional_pdist__cat_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _ = F.pdist(warmup)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "cat":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pdist(x)

    assert seen["dispatch_device"] is None
    vals = y.tolist()
    assert len(vals) == 3
    assert all(v > 0.0 for v in vals)



def test_nn_functional_pdist__second_sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _ = F.pdist(warmup, p=3.0)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    seen = {"dispatch_device": object(), "sum_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["sum_calls"] += 1
            if seen["sum_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pdist(x, p=3.0)

    assert seen["dispatch_device"] is None
    vals = y.tolist()
    assert len(vals) == 3
    assert all(v > 0.0 for v in vals)



def test_nn_functional_pdist__second_pow_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _ = F.pdist(warmup, p=3.0)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    seen = {"dispatch_device": object(), "pow_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "pow":
            seen["pow_calls"] += 1
            if seen["pow_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pdist(x, p=3.0)

    assert seen["dispatch_device"] is None
    vals = y.tolist()
    assert len(vals) == 3
    assert all(v > 0.0 for v in vals)



def test_nn_functional_pdist__first_pow_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _ = F.pdist(warmup, p=3.0)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    seen = {"dispatch_device": object(), "pow_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "pow":
            seen["pow_calls"] += 1
            if seen["pow_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pdist(x, p=3.0)

    assert seen["dispatch_device"] is None
    vals = y.tolist()
    assert len(vals) == 3
    assert all(v > 0.0 for v in vals)



def test_nn_functional_pdist__amax_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _ = F.pdist(warmup, p=float('inf'))

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "amax":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pdist(x, p=float('inf'))

    assert seen["dispatch_device"] is None
    vals = y.tolist()
    assert len(vals) == 3
    assert all(v > 0.0 for v in vals)



def test_nn_functional_pdist__sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _ = F.pdist(warmup, p=1.0)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pdist(x, p=1.0)

    assert seen["dispatch_device"] is None
    vals = y.tolist()
    assert len(vals) == 3
    assert all(v > 0.0 for v in vals)



def test_nn_functional_pdist__abs_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _ = F.pdist(warmup, p=1.0)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "abs":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.pdist(x, p=1.0)

    assert seen["dispatch_device"] is None
    vals = y.tolist()
    assert len(vals) == 3
    assert all(v > 0.0 for v in vals)



def test_nn_functional_pdist__bypasses_python__functional_neg(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _ = F.pdist(warmup)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    calls = {"count": 0}
    original = _functional.neg

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "neg", wrapped)

    y = F.pdist(x)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 3
    assert all(v > 0.0 for v in vals)



def test_nn_functional_pdist__bypasses_python__functional_add(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _ = F.pdist(warmup)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    calls = {"count": 0}
    original = _functional.add

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "add", wrapped)

    y = F.pdist(x)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 3
    assert all(v > 0.0 for v in vals)



def test_nn_functional_pdist__bypasses_python__functional_abs(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _ = F.pdist(warmup, p=1.0)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    calls = {"count": 0}
    original = _functional.abs

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "abs", wrapped)

    y = F.pdist(x, p=1.0)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 3
    assert all(v > 0.0 for v in vals)



def test_nn_functional_pdist__bypasses_python__functional_pow(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _ = F.pdist(warmup, p=3.0)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    calls = {"count": 0}
    original = _functional.pow

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "pow", wrapped)

    y = F.pdist(x, p=3.0)

    assert calls["count"] == 0
    vals = y.tolist()
    assert len(vals) == 3
    assert all(v > 0.0 for v in vals)



def test_nn_functional_gaussian_nll_loss__clamp_min_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "clamp_min":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__neg_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "neg":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__first_add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    seen = {"dispatch_device": object(), "add_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["add_calls"] += 1
            if seen["add_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__first_mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    seen = {"dispatch_device": object(), "mul_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul":
            seen["mul_calls"] += 1
            if seen["mul_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__log_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "log":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__div_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "div":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__second_add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    seen = {"dispatch_device": object(), "add_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["add_calls"] += 1
            if seen["add_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__second_mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    seen = {"dispatch_device": object(), "mul_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul":
            seen["mul_calls"] += 1
            if seen["mul_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__mean_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mean":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='sum')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='sum')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__full_add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, full=True, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    seen = {"dispatch_device": object(), "add_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["add_calls"] += 1
            if seen["add_calls"] == 3:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, full=True, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__bypasses_python__functional_neg(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    calls = {"count": 0}
    original = _functional.neg

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "neg", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__bypasses_python__functional_log(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    calls = {"count": 0}
    original = _functional.log

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "log", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__bypasses_python__functional_div(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    calls = {"count": 0}
    original = _functional.div

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "div", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__bypasses_python__functional_mul(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    calls = {"count": 0}
    original = _functional.mul

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mul", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__bypasses_python__functional_add(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    calls = {"count": 0}
    original = _functional.add

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "add", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__bypasses_python__functional_mean(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    calls = {"count": 0}
    original = _functional.mean

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mean", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_gaussian_nll_loss__bypasses_python__functional_sum(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    warmup_var = torch.tensor([0.5, 0.75])
    _ = F.gaussian_nll_loss(warmup_input, warmup_target, warmup_var, reduction='sum')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])
    var_t = torch.tensor([0.5, 0.75])

    calls = {"count": 0}
    original = _functional.sum

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "sum", wrapped)

    y = F.gaussian_nll_loss(input_t, target_t, var_t, reduction='sum')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_mse_loss__bypasses_python__functional_mean(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    _ = F.mse_loss(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])

    calls = {"count": 0}
    original = _functional.mean

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mean", wrapped)

    y = F.mse_loss(input_t, target_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_mse_loss__bypasses_python__functional_sum(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    _ = F.mse_loss(warmup_input, warmup_target, reduction='sum')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])

    calls = {"count": 0}
    original = _functional.sum

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "sum", wrapped)

    y = F.mse_loss(input_t, target_t, reduction='sum')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_l1_loss__bypasses_python__functional_mean(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    _ = F.l1_loss(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])

    calls = {"count": 0}
    original = _functional.mean

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mean", wrapped)

    y = F.l1_loss(input_t, target_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_l1_loss__bypasses_python__functional_sum(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    _ = F.l1_loss(warmup_input, warmup_target, reduction='sum')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])

    calls = {"count": 0}
    original = _functional.sum

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "sum", wrapped)

    y = F.l1_loss(input_t, target_t, reduction='sum')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_mse_loss__mean_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    _ = F.mse_loss(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mean":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.mse_loss(input_t, target_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_mse_loss__sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    _ = F.mse_loss(warmup_input, warmup_target, reduction='sum')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.mse_loss(input_t, target_t, reduction='sum')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_l1_loss__mean_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    _ = F.l1_loss(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mean":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.l1_loss(input_t, target_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_l1_loss__sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    _ = F.l1_loss(warmup_input, warmup_target, reduction='sum')

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.l1_loss(input_t, target_t, reduction='sum')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_smooth_l1_loss__bypasses_python__functional_mean(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    _ = F.smooth_l1_loss(warmup_input, warmup_target, reduction='mean', beta=1.0)

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])

    calls = {"count": 0}
    original = _functional.mean

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mean", wrapped)

    y = F.smooth_l1_loss(input_t, target_t, reduction='mean', beta=1.0)

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_smooth_l1_loss__mean_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    _ = F.smooth_l1_loss(warmup_input, warmup_target, reduction='mean', beta=1.0)

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mean":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.smooth_l1_loss(input_t, target_t, reduction='mean', beta=1.0)

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_smooth_l1_loss__bypasses_python__functional_sum(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    _ = F.smooth_l1_loss(warmup_input, warmup_target, reduction='sum', beta=1.0)

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])

    calls = {"count": 0}
    original = _functional.sum

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "sum", wrapped)

    y = F.smooth_l1_loss(input_t, target_t, reduction='sum', beta=1.0)

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_smooth_l1_loss__sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.5, 2.5])
    warmup_target = torch.tensor([1.0, 2.0])
    _ = F.smooth_l1_loss(warmup_input, warmup_target, reduction='sum', beta=1.0)

    input_t = torch.tensor([1.5, 2.5])
    target_t = torch.tensor([1.0, 2.0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.smooth_l1_loss(input_t, target_t, reduction='sum', beta=1.0)

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_nll_loss__bypasses_python__functional_neg(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    warmup_target = torch.tensor([1, 0])
    _ = F.nll_loss(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    target_t = torch.tensor([1, 0])

    calls = {"count": 0}
    original = _functional.neg

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "neg", wrapped)

    y = F.nll_loss(input_t, target_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_nll_loss__bypasses_python__functional_sum(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    warmup_target = torch.tensor([1, 0])
    _ = F.nll_loss(warmup_input, warmup_target, reduction='sum')

    input_t = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    target_t = torch.tensor([1, 0])

    calls = {"count": 0}
    original = _functional.sum

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "sum", wrapped)

    y = F.nll_loss(input_t, target_t, reduction='sum')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_nll_loss__sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    warmup_target = torch.tensor([1, 0])
    _ = F.nll_loss(warmup_input, warmup_target, reduction='sum')

    input_t = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    target_t = torch.tensor([1, 0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.nll_loss(input_t, target_t, reduction='sum')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_nll_loss__mean_total_weight_sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    warmup_target = torch.tensor([1, 0])
    warmup_weight = torch.tensor([1.0, 2.0, 3.0])
    _ = F.nll_loss(warmup_input, warmup_target, weight=warmup_weight, reduction='mean')

    input_t = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    target_t = torch.tensor([1, 0])
    weight_t = torch.tensor([1.0, 2.0, 3.0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum" and seen["dispatch_device"] is not None and args and getattr(args[0], "shape", None) == (2,):
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.nll_loss(input_t, target_t, weight=weight_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_nll_loss__mean_valid_count_sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    warmup_target = torch.tensor([1, 0])
    _ = F.nll_loss(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    target_t = torch.tensor([1, 0])

    seen = {"dispatch_device": object(), "sum_count": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["sum_count"] += 1
            if seen["sum_count"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.nll_loss(input_t, target_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_nll_loss__mean_losses_sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    warmup_target = torch.tensor([1, 0])
    _ = F.nll_loss(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    target_t = torch.tensor([1, 0])

    seen = {"dispatch_device": object(), "sum_count": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["sum_count"] += 1
            if seen["sum_count"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.nll_loss(input_t, target_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_nll_loss__mean_div_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    warmup_target = torch.tensor([1, 0])
    _ = F.nll_loss(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    target_t = torch.tensor([1, 0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "div":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.nll_loss(input_t, target_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_nll_loss__valid_ne_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    warmup_target = torch.tensor([1, 0])
    _ = F.nll_loss(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    target_t = torch.tensor([1, 0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "ne":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.nll_loss(input_t, target_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_nll_loss__target_clamp_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    warmup_target = torch.tensor([1, 0])
    _ = F.nll_loss(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    target_t = torch.tensor([1, 0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "clamp":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.nll_loss(input_t, target_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_nll_loss__gather_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    warmup_target = torch.tensor([1, 0])
    _ = F.nll_loss(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    target_t = torch.tensor([1, 0])

    seen = {"dispatch_device": object(), "gather_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "gather":
            seen["gather_calls"] += 1
            if seen["gather_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.nll_loss(input_t, target_t, reduction='mean')

    assert seen["gather_calls"] >= 1
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_nll_loss__neg_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    warmup_target = torch.tensor([1, 0])
    _ = F.nll_loss(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    target_t = torch.tensor([1, 0])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "neg":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.nll_loss(input_t, target_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_nll_loss__weighted_gather_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    warmup_target = torch.tensor([1, 0])
    warmup_weight = torch.tensor([1.0, 2.0, 3.0])
    _ = F.nll_loss(warmup_input, warmup_target, weight=warmup_weight, reduction='mean')

    input_t = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    target_t = torch.tensor([1, 0])
    weight_t = torch.tensor([1.0, 2.0, 3.0])

    seen = {"dispatch_device": object(), "gather_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "gather":
            seen["gather_calls"] += 1
            if seen["gather_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.nll_loss(input_t, target_t, weight=weight_t, reduction='mean')

    assert seen["gather_calls"] >= 2
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_nll_loss__weighted_mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    warmup_target = torch.tensor([1, 0])
    warmup_weight = torch.tensor([1.0, 2.0, 3.0])
    _ = F.nll_loss(warmup_input, warmup_target, weight=warmup_weight, reduction='mean')

    input_t = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    target_t = torch.tensor([1, 0])
    weight_t = torch.tensor([1.0, 2.0, 3.0])

    seen = {"dispatch_device": object(), "mul_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul":
            seen["mul_calls"] += 1
            if seen["mul_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.nll_loss(input_t, target_t, weight=weight_t, reduction='mean')

    assert seen["mul_calls"] >= 1
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_nll_loss__valid_mask_mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    warmup_target = torch.tensor([1, 0])
    _ = F.nll_loss(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    target_t = torch.tensor([1, 0])

    seen = {"dispatch_device": object(), "mul_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul":
            seen["mul_calls"] += 1
            if seen["mul_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.nll_loss(input_t, target_t, reduction='mean')

    assert seen["mul_calls"] >= 1
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_nll_loss__mean_total_weight_mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    warmup_target = torch.tensor([1, 0])
    warmup_weight = torch.tensor([1.0, 2.0, 3.0])
    _ = F.nll_loss(warmup_input, warmup_target, weight=warmup_weight, reduction='mean')

    input_t = torch.tensor([[-1.0, -0.5, -2.0], [-0.1, -2.0, -3.0]])
    target_t = torch.tensor([1, 0])
    weight_t = torch.tensor([1.0, 2.0, 3.0])

    seen = {"dispatch_device": object(), "mul_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul":
            seen["mul_calls"] += 1
            if seen["mul_calls"] == 3:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.nll_loss(input_t, target_t, weight=weight_t, reduction='mean')

    assert seen["mul_calls"] >= 3
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_cross_entropy__label_smoothing_sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    warmup_target = torch.tensor([0, 1])
    _ = F.cross_entropy(warmup_input, warmup_target, reduction='mean', label_smoothing=0.1)

    input_t = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    target_t = torch.tensor([0, 1])

    seen = {"dispatch_device": object(), "sum_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["sum_calls"] += 1
            if seen["sum_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cross_entropy(input_t, target_t, reduction='mean', label_smoothing=0.1)

    assert seen["sum_calls"] >= 1
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)

def test_nn_functional_cross_entropy__label_smoothing_div_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    warmup_target = torch.tensor([0, 1])
    _ = F.cross_entropy(warmup_input, warmup_target, reduction='mean', label_smoothing=0.1)

    input_t = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    target_t = torch.tensor([0, 1])

    seen = {"dispatch_device": object(), "div_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "div":
            seen["div_calls"] += 1
            if seen["div_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cross_entropy(input_t, target_t, reduction='mean', label_smoothing=0.1)

    assert seen["div_calls"] >= 1
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_cross_entropy__label_smoothing_valid_ne_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    warmup_target = torch.tensor([0, 1])
    _ = F.cross_entropy(warmup_input, warmup_target, reduction='mean', label_smoothing=0.1)

    input_t = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    target_t = torch.tensor([0, 1])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "ne":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cross_entropy(input_t, target_t, reduction='mean', label_smoothing=0.1)

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_cross_entropy__label_smoothing_valid_mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    warmup_target = torch.tensor([0, 1])
    _ = F.cross_entropy(warmup_input, warmup_target, reduction='mean', label_smoothing=0.1)

    input_t = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    target_t = torch.tensor([0, 1])

    seen = {"dispatch_device": object(), "mul_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul":
            seen["mul_calls"] += 1
            if seen["mul_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cross_entropy(input_t, target_t, reduction='mean', label_smoothing=0.1)

    assert seen["mul_calls"] >= 2
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_cross_entropy__label_smoothing_mean_valid_count_sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    warmup_target = torch.tensor([0, 1])
    _ = F.cross_entropy(warmup_input, warmup_target, reduction='mean', label_smoothing=0.1)

    input_t = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    target_t = torch.tensor([0, 1])

    seen = {"dispatch_device": object(), "sum_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["sum_calls"] += 1
            if seen["sum_calls"] == 4:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cross_entropy(input_t, target_t, reduction='mean', label_smoothing=0.1)

    assert seen["sum_calls"] >= 4
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_cross_entropy__label_smoothing_mean_smooth_loss_sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    warmup_target = torch.tensor([0, 1])
    _ = F.cross_entropy(warmup_input, warmup_target, reduction='mean', label_smoothing=0.1)

    input_t = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    target_t = torch.tensor([0, 1])

    seen = {"dispatch_device": object(), "sum_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["sum_calls"] += 1
            if seen["sum_calls"] == 5:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cross_entropy(input_t, target_t, reduction='mean', label_smoothing=0.1)

    assert seen["sum_calls"] >= 5
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_cross_entropy__label_smoothing_mean_div_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    warmup_target = torch.tensor([0, 1])
    _ = F.cross_entropy(warmup_input, warmup_target, reduction='mean', label_smoothing=0.1)

    input_t = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    target_t = torch.tensor([0, 1])

    seen = {"dispatch_device": object(), "div_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "div":
            seen["div_calls"] += 1
            if seen["div_calls"] == 3:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cross_entropy(input_t, target_t, reduction='mean', label_smoothing=0.1)

    assert seen["div_calls"] >= 3
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_cross_entropy__label_smoothing_sum_branch_sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    warmup_target = torch.tensor([0, 1])
    _ = F.cross_entropy(warmup_input, warmup_target, reduction='sum', label_smoothing=0.1)

    input_t = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    target_t = torch.tensor([0, 1])

    seen = {"dispatch_device": object(), "sum_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["sum_calls"] += 1
            if seen["sum_calls"] == 3:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cross_entropy(input_t, target_t, reduction='sum', label_smoothing=0.1)

    assert seen["sum_calls"] >= 3
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_cross_entropy__label_smoothing_final_add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    warmup_target = torch.tensor([0, 1])
    _ = F.cross_entropy(warmup_input, warmup_target, reduction='mean', label_smoothing=0.1)

    input_t = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    target_t = torch.tensor([0, 1])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cross_entropy(input_t, target_t, reduction='mean', label_smoothing=0.1)

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_cross_entropy__label_smoothing_final_left_mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    warmup_target = torch.tensor([0, 1])
    _ = F.cross_entropy(warmup_input, warmup_target, reduction='mean', label_smoothing=0.1)

    input_t = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    target_t = torch.tensor([0, 1])

    seen = {"dispatch_device": object(), "mul_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul":
            seen["mul_calls"] += 1
            if seen["mul_calls"] == 3:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cross_entropy(input_t, target_t, reduction='mean', label_smoothing=0.1)

    assert seen["mul_calls"] >= 3
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_cross_entropy__label_smoothing_final_right_mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    warmup_target = torch.tensor([0, 1])
    _ = F.cross_entropy(warmup_input, warmup_target, reduction='mean', label_smoothing=0.1)

    input_t = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
    target_t = torch.tensor([0, 1])

    seen = {"dispatch_device": object(), "mul_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul":
            seen["mul_calls"] += 1
            if seen["mul_calls"] == 4:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cross_entropy(input_t, target_t, reduction='mean', label_smoothing=0.1)

    assert seen["mul_calls"] >= 4
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_scaled_dot_product_attention__causal_eq_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    warmup_k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    warmup_v = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    _ = F.scaled_dot_product_attention(warmup_q, warmup_k, warmup_v, is_causal=True)

    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "eq":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert seen["dispatch_device"] is None
    assert tuple(y.shape) == (1, 2, 2)



def test_nn_functional_scaled_dot_product_attention__bool_mask_eq_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    warmup_k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    warmup_v = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    warmup_mask = torch.tensor([[True, False], [True, True]], dtype=torch.bool)
    _ = F.scaled_dot_product_attention(warmup_q, warmup_k, warmup_v, attn_mask=warmup_mask)

    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    mask = torch.tensor([[True, False], [True, True]], dtype=torch.bool)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "eq":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    assert seen["dispatch_device"] is None
    assert tuple(y.shape) == (1, 2, 2)



def test_nn_functional_interpolate__nearest2d_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    _ = F.interpolate(warmup_input, size=(4, 4), mode='nearest')

    input_t = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "upsample_nearest2d":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.interpolate(input_t, size=(4, 4), mode='nearest')

    assert seen["dispatch_device"] is None
    assert tuple(y.shape) == (1, 1, 4, 4)



def test_nn_functional_interpolate__nearest1d_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
    _ = F.interpolate(warmup_input, size=6, mode='nearest')

    input_t = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "upsample_nearest1d":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.interpolate(input_t, size=6, mode='nearest')

    assert seen["dispatch_device"] is None
    assert tuple(y.shape) == (1, 1, 6)



def test_nn_functional_interpolate__linear1d_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
    _ = F.interpolate(warmup_input, size=6, mode='linear', align_corners=False)

    input_t = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "upsample_linear1d":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.interpolate(input_t, size=6, mode='linear', align_corners=False)

    assert seen["dispatch_device"] is None
    assert tuple(y.shape) == (1, 1, 6)



def test_nn_functional_interpolate__bilinear2d_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    _ = F.interpolate(warmup_input, size=(4, 4), mode='bilinear', align_corners=False)

    input_t = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "upsample_bilinear2d":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.interpolate(input_t, size=(4, 4), mode='bilinear', align_corners=False)

    assert seen["dispatch_device"] is None
    assert tuple(y.shape) == (1, 1, 4, 4)



def test_nn_functional_interpolate__bicubic2d_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    _ = F.interpolate(warmup_input, size=(4, 4), mode='bicubic', align_corners=False)

    input_t = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "upsample_bicubic2d":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.interpolate(input_t, size=(4, 4), mode='bicubic', align_corners=False)

    assert seen["dispatch_device"] is None
    assert tuple(y.shape) == (1, 1, 4, 4)



def test_nn_functional_interpolate__trilinear3d_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=torch.float32)
    _ = F.interpolate(warmup_input, size=(4, 4, 4), mode='trilinear', align_corners=False)

    input_t = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "upsample_trilinear3d":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.interpolate(input_t, size=(4, 4, 4), mode='trilinear', align_corners=False)

    assert seen["dispatch_device"] is None
    assert tuple(y.shape) == (1, 1, 4, 4, 4)



def test_nn_functional_binary_cross_entropy__bypasses_python__functional_mean(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([0.8, 0.2, 0.6], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([0.8, 0.2, 0.6], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    calls = {"count": 0}
    original = _functional.mean

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mean", wrapped)

    y = F.binary_cross_entropy(input_t, target_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy__weight_affects_result():
    import candle.nn.functional as F

    input_t = torch.tensor([0.8, 0.2, 0.6], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    weight_t = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)

    unweighted = F.binary_cross_entropy(input_t, target_t, reduction='mean')
    weighted = F.binary_cross_entropy(input_t, target_t, weight=weight_t, reduction='mean')

    assert weighted.item() != unweighted.item()



def test_nn_functional_binary_cross_entropy__bypasses_python__functional_sum(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([0.8, 0.2, 0.6], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy(warmup_input, warmup_target, reduction='sum')

    input_t = torch.tensor([0.8, 0.2, 0.6], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    calls = {"count": 0}
    original = _functional.sum

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "sum", wrapped)

    y = F.binary_cross_entropy(input_t, target_t, reduction='sum')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__bypasses_python__functional_mean(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    calls = {"count": 0}
    original = _functional.mean

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mean", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_cosine_embedding_loss__sum_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    warmup_input2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    warmup_target = torch.tensor([1, -1])
    _ = F.cosine_embedding_loss(warmup_input1, warmup_input2, warmup_target, reduction='mean')

    input1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    input2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([1, -1])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "sum":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cosine_embedding_loss(input1, input2, target, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_cosine_embedding_loss__first_norm_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    warmup_input2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    warmup_target = torch.tensor([1, -1])
    _ = F.cosine_embedding_loss(warmup_input1, warmup_input2, warmup_target, reduction='mean')

    input1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    input2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([1, -1])

    seen = {"dispatch_device": object(), "norm_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "norm":
            seen["norm_calls"] += 1
            if seen["norm_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cosine_embedding_loss(input1, input2, target, reduction='mean')

    assert seen["norm_calls"] >= 1
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_cosine_embedding_loss__second_norm_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    warmup_input2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    warmup_target = torch.tensor([1, -1])
    _ = F.cosine_embedding_loss(warmup_input1, warmup_input2, warmup_target, reduction='mean')

    input1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    input2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([1, -1])

    seen = {"dispatch_device": object(), "norm_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "norm":
            seen["norm_calls"] += 1
            if seen["norm_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cosine_embedding_loss(input1, input2, target, reduction='mean')

    assert seen["norm_calls"] >= 2
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_cosine_embedding_loss__div_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    warmup_input2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    warmup_target = torch.tensor([1, -1])
    _ = F.cosine_embedding_loss(warmup_input1, warmup_input2, warmup_target, reduction='mean')

    input1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    input2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([1, -1])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "div":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cosine_embedding_loss(input1, input2, target, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_cosine_embedding_loss__eq_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    warmup_input2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    warmup_target = torch.tensor([1, -1])
    _ = F.cosine_embedding_loss(warmup_input1, warmup_input2, warmup_target, reduction='mean')

    input1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    input2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([1, -1])

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "eq":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.cosine_embedding_loss(input1, input2, target, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__bypasses_python__functional_sum(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='sum')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    calls = {"count": 0}
    original = _functional.sum

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "sum", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='sum')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__bypasses_python__functional_abs(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    calls = {"count": 0}
    original = _functional.abs

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "abs", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__bypasses_python__functional_exp(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    calls = {"count": 0}
    original = _functional.exp

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "exp", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__bypasses_python__functional_log(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    calls = {"count": 0}
    original = _functional.log

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "log", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__bypasses_python__functional_add(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    calls = {"count": 0}
    original = _functional.add

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "add", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__bypasses_python__functional_mul(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    calls = {"count": 0}
    original = _functional.mul

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mul", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__bypasses_python__functional_neg(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    calls = {"count": 0}
    original = _functional.neg

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "neg", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__exp_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "exp":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__log_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "log":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__log_input_add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add" and len(args) == 2 and getattr(args[0], "shape", None) == () and getattr(args[1], "shape", None) == input_t.shape:
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__clamp_min_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "clamp_min":
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__neg_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "neg" and seen["dispatch_device"] is not None:
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    seen = {"dispatch_device": object()}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul" and len(args) == 2 and args[0] is input_t and args[1] is target_t:
            seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__loss_neg_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    seen = {"dispatch_device": object(), "neg_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "neg":
            seen["neg_calls"] += 1
            if seen["neg_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert seen["neg_calls"] >= 2
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__inner_loss_add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    seen = {"dispatch_device": object(), "add_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["add_calls"] += 1
            if seen["add_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert seen["add_calls"] >= 2
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__outer_loss_add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    seen = {"dispatch_device": object(), "add_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["add_calls"] += 1
            if seen["add_calls"] == 3:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, reduction='mean')

    assert seen["add_calls"] >= 3
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_binary_cross_entropy_with_logits__weight_mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    warmup_weight = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, weight=warmup_weight, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    weight_t = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)

    seen = {"dispatch_device": object(), "mul_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul":
            seen["mul_calls"] += 1
            if seen["mul_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, weight=weight_t, reduction='mean')

    assert seen["mul_calls"] >= 2
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__pos_weight_neg_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    warmup_pos_weight = torch.tensor([1.5, 0.5, 2.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, pos_weight=warmup_pos_weight, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    pos_weight_t = torch.tensor([1.5, 0.5, 2.0], dtype=torch.float32)

    seen = {"dispatch_device": object(), "neg_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "neg":
            seen["neg_calls"] += 1
            if seen["neg_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, pos_weight=pos_weight_t, reduction='mean')

    assert seen["neg_calls"] >= 2
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__pos_weight_add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    warmup_pos_weight = torch.tensor([1.5, 0.5, 2.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, pos_weight=warmup_pos_weight, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    pos_weight_t = torch.tensor([1.5, 0.5, 2.0], dtype=torch.float32)

    seen = {"dispatch_device": object(), "add_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["add_calls"] += 1
            if seen["add_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, pos_weight=pos_weight_t, reduction='mean')

    assert seen["add_calls"] >= 2
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__pos_weight_mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    warmup_pos_weight = torch.tensor([1.5, 0.5, 2.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, pos_weight=warmup_pos_weight, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    pos_weight_t = torch.tensor([1.5, 0.5, 2.0], dtype=torch.float32)

    seen = {"dispatch_device": object(), "mul_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul":
            seen["mul_calls"] += 1
            if seen["mul_calls"] == 1:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, pos_weight=pos_weight_t, reduction='mean')

    assert seen["mul_calls"] >= 1
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)



def test_nn_functional_binary_cross_entropy_with_logits__pos_weight_factor_add_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    warmup_pos_weight = torch.tensor([1.5, 0.5, 2.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, pos_weight=warmup_pos_weight, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    pos_weight_t = torch.tensor([1.5, 0.5, 2.0], dtype=torch.float32)

    seen = {"dispatch_device": object(), "add_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "add":
            seen["add_calls"] += 1
            if seen["add_calls"] == 3:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, pos_weight=pos_weight_t, reduction='mean')

    assert seen["add_calls"] >= 3
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)


def test_nn_functional_binary_cross_entropy_with_logits__pos_weight_log_term_mul_dispatch_infers_device(monkeypatch):
    import candle.nn.functional as F
    import candle._dispatch as dispatch_pkg

    warmup_input = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    warmup_pos_weight = torch.tensor([1.5, 0.5, 2.0], dtype=torch.float32)
    _ = F.binary_cross_entropy_with_logits(warmup_input, warmup_target, pos_weight=warmup_pos_weight, reduction='mean')

    input_t = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    pos_weight_t = torch.tensor([1.5, 0.5, 2.0], dtype=torch.float32)

    seen = {"dispatch_device": object(), "mul_calls": 0}
    original = dispatch_pkg.dispatch

    def wrapped(op_name, dispatch_device=None, *args, **kwargs):
        if op_name == "mul":
            seen["mul_calls"] += 1
            if seen["mul_calls"] == 2:
                seen["dispatch_device"] = dispatch_device
        return original(op_name, dispatch_device, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped)

    y = F.binary_cross_entropy_with_logits(input_t, target_t, pos_weight=pos_weight_t, reduction='mean')

    assert seen["mul_calls"] >= 2
    assert seen["dispatch_device"] is None
    assert isinstance(y.item(), float)

def test_nn_functional_poisson_nll_loss__bypasses_python__functional_mean(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float32)
    _ = F.poisson_nll_loss(warmup_input, warmup_target, log_input=True, reduction='mean')

    input_t = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float32)

    calls = {"count": 0}
    original = _functional.mean

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mean", wrapped)

    y = F.poisson_nll_loss(input_t, target_t, log_input=True, reduction='mean')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_poisson_nll_loss__bypasses_python__functional_sum(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup_input = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float32)
    warmup_target = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float32)
    _ = F.poisson_nll_loss(warmup_input, warmup_target, log_input=True, reduction='sum')

    input_t = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float32)
    target_t = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float32)

    calls = {"count": 0}
    original = _functional.sum

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "sum", wrapped)

    y = F.poisson_nll_loss(input_t, target_t, log_input=True, reduction='sum')

    assert calls["count"] == 0
    assert isinstance(y.item(), float)



def test_nn_functional_feature_alpha_dropout__bypasses_python__functional_rand(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    _ = F.feature_alpha_dropout(warmup, p=0.25, training=True)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    calls = {"count": 0}
    original = _functional.rand

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "rand", wrapped)

    y = F.feature_alpha_dropout(x, p=0.25, training=True)

    assert calls["count"] == 0
    assert y.shape == (2, 2)



def test_nn_functional_gelu_tanh__bypasses_python__functional_mul(monkeypatch):
    import math
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = F.gelu(warmup, approximate='tanh')

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = _functional.mul

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "mul", wrapped)

    y = F.gelu(x, approximate='tanh')

    assert calls["count"] == 0
    vals = y.tolist()
    assert math.isclose(vals[0], -0.1588080093917233, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.8411919906082768, rel_tol=0.0, abs_tol=1e-6)



def test_nn_functional_gelu_tanh__bypasses_python__functional_add(monkeypatch):
    import math
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = F.gelu(warmup, approximate='tanh')

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = _functional.add

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "add", wrapped)

    y = F.gelu(x, approximate='tanh')

    assert calls["count"] == 0
    vals = y.tolist()
    assert math.isclose(vals[0], -0.1588080093917233, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.8411919906082768, rel_tol=0.0, abs_tol=1e-6)



def test_nn_functional_gelu_tanh__bypasses_python__functional_pow(monkeypatch):
    import math
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = F.gelu(warmup, approximate='tanh')

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = _functional.pow

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "pow", wrapped)

    y = F.gelu(x, approximate='tanh')

    assert calls["count"] == 0
    vals = y.tolist()
    assert math.isclose(vals[0], -0.1588080093917233, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.8411919906082768, rel_tol=0.0, abs_tol=1e-6)



def test_nn_functional_gelu_tanh__bypasses_python__functional_tanh(monkeypatch):
    import math
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = F.gelu(warmup, approximate='tanh')

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = _functional.tanh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "tanh", wrapped)

    y = F.gelu(x, approximate='tanh')

    assert calls["count"] == 0
    vals = y.tolist()
    assert math.isclose(vals[0], -0.1588080093917233, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.8411919906082768, rel_tol=0.0, abs_tol=1e-6)



def test_nn_functional_sigmoid__bypasses_python__functional_sigmoid(monkeypatch):

    import math
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = F.sigmoid(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = _functional.sigmoid

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "sigmoid", wrapped)

    y = F.sigmoid(x)

    assert calls["count"] == 0
    vals = y.tolist()
    assert math.isclose(vals[0], 0.2689414213699951, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.7310585786300049, rel_tol=0.0, abs_tol=1e-6)



def test_nn_functional_tanh__bypasses_python__functional_tanh(monkeypatch):
    import math
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = F.tanh(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = _functional.tanh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "tanh", wrapped)

    y = F.tanh(x)

    assert calls["count"] == 0
    vals = y.tolist()
    assert math.isclose(vals[0], -0.7615941559557649, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.7615941559557649, rel_tol=0.0, abs_tol=1e-6)





def test_nn_functional_relu___bypasses_python__functional_relu_(monkeypatch):
    import candle._functional as _functional
    import candle.nn.functional as F

    warmup = torch.tensor([-2.0, 0.5, 3.0])
    _ = F.relu_(warmup)

    x = torch.tensor([-2.0, 0.5, 3.0])

    calls = {"count": 0}
    original = _functional.relu_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "relu_", wrapped)

    y = F.relu_(x)

    assert calls["count"] == 0
    assert y is x
    assert y.tolist() == [0.0, 0.5, 3.0]



def test_toplevel_masked_scatter__bypasses_python__functional_dispatch(monkeypatch):

    import candle._functional as _functional

    warmup_input = torch.tensor([1.0, 2.0, 3.0, 4.0])
    warmup_mask = torch.tensor([True, False, True, False])
    warmup_source = torch.tensor([9.0, 8.0])
    _ = torch.masked_scatter(warmup_input, warmup_mask, warmup_source)

    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([True, False, True, False])
    source = torch.tensor([9.0, 8.0])

    calls = {"count": 0}
    original = _functional.dispatch

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "dispatch", wrapped)

    y = torch.masked_scatter(input_tensor, mask, source)

    assert calls["count"] == 0
    assert y.tolist() == [9.0, 2.0, 8.0, 4.0]



def test_toplevel_constant_pad_nd__bypasses_python__functional_dispatch(monkeypatch):
    import candle._functional as _functional

    warmup = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    _ = torch.constant_pad_nd(warmup, (1, 0, 0, 1), value=-5.0)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    calls = {"count": 0}
    original = _functional.dispatch

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "dispatch", wrapped)

    y = torch.constant_pad_nd(x, (1, 0, 0, 1), value=-5.0)

    assert calls["count"] == 0
    assert y.tolist() == [[-5.0, 1.0, 2.0], [-5.0, 3.0, 4.0], [-5.0, -5.0, -5.0]]



def test_toplevel_mode__bypasses_python__functional_dispatch(monkeypatch):
    import candle._functional as _functional

    warmup = torch.tensor([[1.0, 3.0, 3.0], [2.0, 2.0, 5.0]])
    _ = torch.mode(warmup, dim=1, keepdim=False)

    x = torch.tensor([[1.0, 3.0, 3.0], [2.0, 2.0, 5.0]])

    calls = {"count": 0}
    original = _functional.dispatch

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "dispatch", wrapped)

    y = torch.mode(x, dim=1, keepdim=False)

    assert calls["count"] == 0
    assert y.values.tolist() == [3.0, 2.0]
    assert y.indices.tolist() == [1, 0]



def test_toplevel_square__bypasses_python__functional_square(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([-2.0, 0.5, 3.0])
    _ = torch.square_(warmup)

    x = torch.tensor([-2.0, 0.5, 3.0])

    calls = {"count": 0}
    original = _functional.square

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "square", wrapped)

    y = torch.square_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 4.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.25, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 9.0, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_square__bypasses_python_top_level_square(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-2.0, 0.5, 3.0])
    _ = torch.square_(warmup)

    x = torch.tensor([-2.0, 0.5, 3.0])

    calls = {"count": 0}
    original = torch_module.square

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "square", wrapped)

    y = torch.square_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 4.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.25, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 9.0, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_sinh__bypasses_python__functional_sinh(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.sinh_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = _functional.sinh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "sinh", wrapped)

    y = torch.sinh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -1.1752011936438014, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.1752011936438014, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_sinh__bypasses_python_top_level_sinh(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, -1.0])
    _ = torch.sinh_(warmup)

    x = torch.tensor([0.0, 1.0, -1.0])

    calls = {"count": 0}
    original = torch_module.sinh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "sinh", wrapped)

    y = torch.sinh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.1752011936438014, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -1.1752011936438014, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_rsqrt__bypasses_python__functional_rsqrt(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([1.0, 4.0, 9.0])
    _ = torch.rsqrt_(warmup)

    x = torch.tensor([1.0, 4.0, 9.0])

    calls = {"count": 0}
    original = _functional.rsqrt

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "rsqrt", wrapped)

    y = torch.rsqrt_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.3333333333333333, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_rsqrt__bypasses_python_top_level_rsqrt(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([1.0, 4.0, 9.0])
    _ = torch.rsqrt_(warmup)

    x = torch.tensor([1.0, 4.0, 9.0])

    calls = {"count": 0}
    original = torch_module.rsqrt

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "rsqrt", wrapped)

    y = torch.rsqrt_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.3333333333333333, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_log1p__bypasses_python__functional_log1p(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([0.0, 1.0, 3.0])
    _ = torch.log1p_(warmup)

    x = torch.tensor([0.0, 1.0, 3.0])

    calls = {"count": 0}
    original = _functional.log1p

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "log1p", wrapped)

    y = torch.log1p_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.6931471805599453, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.3862943611198906, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_log1p__bypasses_python_top_level_log1p(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, 3.0])
    _ = torch.log1p_(warmup)

    x = torch.tensor([0.0, 1.0, 3.0])

    calls = {"count": 0}
    original = torch_module.log1p

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "log1p", wrapped)

    y = torch.log1p_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.6931471805599453, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.3862943611198906, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_frac__bypasses_python__functional_frac(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([1.25, -2.75, 3.5])
    _ = torch.frac_(warmup)

    x = torch.tensor([1.25, -2.75, 3.5])

    calls = {"count": 0}
    original = _functional.frac

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "frac", wrapped)

    y = torch.frac_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.25, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], -0.75, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.5, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_frac__bypasses_python_top_level_frac(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([1.25, -1.75, 0.0])
    _ = torch.frac_(warmup)

    x = torch.tensor([1.25, -1.75, 0.0])

    calls = {"count": 0}
    original = torch_module.frac

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "frac", wrapped)

    y = torch.frac_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.25, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], -0.75, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 0.0, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_expm1__bypasses_python__functional_expm1(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.expm1_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = _functional.expm1

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "expm1", wrapped)

    y = torch.expm1_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -0.6321205588285577, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.718281828459045, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_expm1__bypasses_python_top_level_expm1(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.expm1_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = torch_module.expm1

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "expm1", wrapped)

    y = torch.expm1_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -0.6321205588285577, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.718281828459045, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_exp2__bypasses_python__functional_exp2(monkeypatch):
    import math
    import candle._functional as _functional

    warmup = torch.tensor([-1.0, 0.0, 2.0])
    _ = torch.exp2_(warmup)

    x = torch.tensor([-1.0, 0.0, 2.0])

    calls = {"count": 0}
    original = _functional.exp2

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_functional, "exp2", wrapped)

    y = torch.exp2_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 4.0, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_exp2__bypasses_python_top_level_exp2(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-1.0, 0.0, 2.0])
    _ = torch.exp2_(warmup)

    x = torch.tensor([-1.0, 0.0, 2.0])

    calls = {"count": 0}
    original = torch_module.exp2

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "exp2", wrapped)

    y = torch.exp2_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 4.0, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_erfc__bypasses_python_top_level_erfc(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, -1.0])
    _ = torch.erfc_(warmup)

    x = torch.tensor([0.0, 1.0, -1.0])

    calls = {"count": 0}
    original = torch_module.erfc

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "erfc", wrapped)

    y = torch.erfc_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.15729920705028513, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.842700792949715, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_erf__bypasses_python_top_level_erf(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, -1.0])
    _ = torch.erf_(warmup)

    x = torch.tensor([0.0, 1.0, -1.0])

    calls = {"count": 0}
    original = torch_module.erf

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "erf", wrapped)

    y = torch.erf_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.8427007929497149, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -0.8427007929497149, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_cosh__bypasses_python_top_level_cosh(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, -1.0])
    _ = torch.cosh_(warmup)

    x = torch.tensor([0.0, 1.0, -1.0])

    calls = {"count": 0}
    original = torch_module.cosh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "cosh", wrapped)

    y = torch.cosh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.5430806348152437, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.5430806348152437, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_atanh__bypasses_python_top_level_atanh(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 0.5, -0.5])
    _ = torch.atanh_(warmup)

    x = torch.tensor([0.0, 0.5, -0.5])

    calls = {"count": 0}
    original = torch_module.atanh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "atanh", wrapped)

    y = torch.atanh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.5493061443340548, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -0.5493061443340548, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_atan__bypasses_python_top_level_atan(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, -1.0])
    _ = torch.atan_(warmup)

    x = torch.tensor([0.0, 1.0, -1.0])

    calls = {"count": 0}
    original = torch_module.atan

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "atan", wrapped)

    y = torch.atan_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.7853981633974483, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -0.7853981633974483, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_asinh__bypasses_python_top_level_asinh(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, -1.0])
    _ = torch.asinh_(warmup)

    x = torch.tensor([0.0, 1.0, -1.0])

    calls = {"count": 0}
    original = torch_module.asinh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "asinh", wrapped)

    y = torch.asinh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.881373587019543, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -0.881373587019543, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_asin__bypasses_python_top_level_asin(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 0.5, -0.5])
    _ = torch.asin_(warmup)

    x = torch.tensor([0.0, 0.5, -0.5])

    calls = {"count": 0}
    original = torch_module.asin

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "asin", wrapped)

    y = torch.asin_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.5235987755982989, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -0.5235987755982989, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_acosh__bypasses_python_top_level_acosh(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([1.0, 2.0, 3.0])
    _ = torch.acosh_(warmup)

    x = torch.tensor([1.0, 2.0, 3.0])

    calls = {"count": 0}
    original = torch_module.acosh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "acosh", wrapped)

    y = torch.acosh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.3169578969248166, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.7627471740390859, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_nan_to_num__bypasses_python_top_level_nan_to_num(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([float("nan"), float("inf"), float("-inf")])
    _ = torch.nan_to_num_(warmup, nan=1.5, posinf=2.5, neginf=-3.5)

    x = torch.tensor([float("nan"), float("inf"), float("-inf")])

    calls = {"count": 0}
    original = torch_module.nan_to_num

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "nan_to_num", wrapped)

    y = torch.nan_to_num_(x, nan=1.5, posinf=2.5, neginf=-3.5)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 1.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 2.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -3.5, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_dropout__bypasses_python_top_level_dropout_non_noop(monkeypatch):
    import candle as torch_module

    warmup = torch.tensor([1.0, 2.0, 3.0])
    _ = torch.dropout_(warmup, p=0.2, training=True)

    x = torch.tensor([1.0, 2.0, 3.0])

    calls = {"count": 0}
    original = torch_module.dropout

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "dropout", wrapped)

    y = torch.dropout_(x, p=0.2, training=True)

    assert calls["count"] == 0
    assert y is x
    assert y.shape == (3,)



def test_toplevel_threshold__bypasses_python_top_level_threshold(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-1.0, 0.5, 2.0])
    _ = torch.threshold_(warmup, 0.0, -3.0)

    x = torch.tensor([-1.0, 0.5, 2.0])

    calls = {"count": 0}
    original = torch_module.threshold

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "threshold", wrapped)

    y = torch.threshold_(x, 0.0, -3.0)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -3.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 2.0, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_celu__bypasses_python_top_level_celu(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.celu_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = torch_module.celu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "celu", wrapped)

    y = torch.celu_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -0.6321205588285577, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.0, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_selu__bypasses_python_top_level_selu(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.selu_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = torch_module.selu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "selu", wrapped)

    y = torch.selu_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -1.1113307378125625, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.0507009873554805, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_arctanh__bypasses_python_top_level_atanh_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 0.5, -0.5])
    _ = torch.arctanh_(warmup)

    x = torch.tensor([0.0, 0.5, -0.5])

    calls = {"count": 0}
    original = torch_module.atanh_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "atanh_", wrapped)

    y = torch.arctanh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.5493061443340548, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -0.5493061443340548, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_arctan__bypasses_python_top_level_atan_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, -1.0])
    _ = torch.arctan_(warmup)

    x = torch.tensor([0.0, 1.0, -1.0])

    calls = {"count": 0}
    original = torch_module.atan_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "atan_", wrapped)

    y = torch.arctan_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.7853981633974483, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -0.7853981633974483, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_arcsinh__bypasses_python_top_level_asinh_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 1.0, -1.0])
    _ = torch.arcsinh_(warmup)

    x = torch.tensor([0.0, 1.0, -1.0])

    calls = {"count": 0}
    original = torch_module.asinh_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "asinh_", wrapped)

    y = torch.arcsinh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.881373587019543, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -0.881373587019543, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_arcsin__bypasses_python_top_level_asin_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 0.5, -0.5])
    _ = torch.arcsin_(warmup)

    x = torch.tensor([0.0, 0.5, -0.5])

    calls = {"count": 0}
    original = torch_module.asin_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "asin_", wrapped)

    y = torch.arcsin_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.5235987755982989, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -0.5235987755982989, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_arccosh__bypasses_python_top_level_acosh_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([1.0, 2.0, 3.0])
    _ = torch.arccosh_(warmup)

    x = torch.tensor([1.0, 2.0, 3.0])

    calls = {"count": 0}
    original = torch_module.acosh_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "acosh_", wrapped)

    y = torch.arccosh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.3169578969248166, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.7627471740390859, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_arccos__bypasses_python_top_level_acos_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([1.0, 0.5, -0.5])
    _ = torch.arccos_(warmup)

    x = torch.tensor([1.0, 0.5, -0.5])

    calls = {"count": 0}
    original = torch_module.acos_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "acos_", wrapped)

    y = torch.arccos_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.0471975511965979, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 2.0943951023931957, rel_tol=0.0, abs_tol=1e-6)



def test_toplevel_clamp_max__bypasses_python_top_level_clamp(monkeypatch):
    import candle as torch_module

    warmup = torch.tensor([-2.0, 0.5, 3.0])
    _ = torch.clamp_max_(warmup, 1.0)

    x = torch.tensor([-2.0, 0.5, 3.0])

    calls = {"count": 0}
    original = torch_module.clamp

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "clamp", wrapped)

    y = torch.clamp_max_(x, 1.0)

    assert calls["count"] == 0
    assert y is x
    assert y.tolist() == [-2.0, 0.5, 1.0]



def test_toplevel_clamp_min__bypasses_python_top_level_clamp(monkeypatch):
    import candle as torch_module

    warmup = torch.tensor([-2.0, 0.5, 3.0])
    _ = torch.clamp_min_(warmup, 0.0)

    x = torch.tensor([-2.0, 0.5, 3.0])

    calls = {"count": 0}
    original = torch_module.clamp

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "clamp", wrapped)

    y = torch.clamp_min_(x, 0.0)

    assert calls["count"] == 0
    assert y is x
    assert y.tolist() == [0.0, 0.5, 3.0]



def test_toplevel_dropout__bypasses_python_top_level_dropout(monkeypatch):
    import candle as torch_module

    warmup = torch.tensor([1.0, 2.0, 3.0])
    _ = torch.dropout_(warmup, p=0.0, training=True)

    x = torch.tensor([1.0, 2.0, 3.0])

    calls = {"count": 0}
    original = torch_module.dropout

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "dropout", wrapped)

    y = torch.dropout_(x, p=0.0, training=True)

    assert calls["count"] == 0
    assert y is x
    assert y.tolist() == [1.0, 2.0, 3.0]



def test_toplevel_fix__bypasses_python_top_level_trunc_(monkeypatch):
    import candle as torch_module

    warmup = torch.tensor([1.9, -2.7, 0.0])
    _ = torch.fix_(warmup)

    x = torch.tensor([1.9, -2.7, 0.0])

    calls = {"count": 0}
    original = torch_module.trunc_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "trunc_", wrapped)

    y = torch.fix_(x)

    assert calls["count"] == 0
    assert y is x
    assert y.tolist() == [1.0, -2.0, 0.0]



def test_toplevel_threshold__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-1.0, 0.5, 2.0])
    _ = torch.threshold_(warmup, 0.0, -3.0)

    x = torch.tensor([-1.0, 0.5, 2.0])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.threshold_(x, 0.0, -3.0)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -3.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 2.0, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_celu__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.celu_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.celu_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -0.6321205588285577, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.0, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_selu__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([-1.0, 0.0, 1.0])
    _ = torch.selu_(warmup)

    x = torch.tensor([-1.0, 0.0, 1.0])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.selu_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], -1.1113307378125625, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.0507009873554805, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_acos__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([1.0, 0.5, -0.5])
    _ = torch.acos_(warmup)

    x = torch.tensor([1.0, 0.5, -0.5])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.acos_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.0471975511965979, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 2.0943951023931957, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_acosh__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([1.0, 2.0, 3.0])
    _ = torch.acosh_(warmup)

    x = torch.tensor([1.0, 2.0, 3.0])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.acosh_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 1.3169578969248166, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], 1.7627471740390859, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_asin__bypasses_python_top_level_copy_(monkeypatch):
    import math
    import candle as torch_module

    warmup = torch.tensor([0.0, 0.5, -0.5])
    _ = torch.asin_(warmup)

    x = torch.tensor([0.0, 0.5, -0.5])

    calls = {"count": 0}
    original = torch_module.copy_

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_module, "copy_", wrapped)

    y = torch.asin_(x)

    assert calls["count"] == 0
    assert y is x
    vals = y.tolist()
    assert math.isclose(vals[0], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[1], 0.5235987755982989, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vals[2], -0.5235987755982989, rel_tol=0.0, abs_tol=1e-6)


def test_toplevel_reshape_bypasses_python_top_level_dispatch(monkeypatch):
    import candle._functional as functional

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _ = torch.reshape(x, (6,))

    calls = {"count": 0, "ops": []}
    original = functional.dispatch

    def wrapped(op_name, *args, **kwargs):
        calls["count"] += 1
        calls["ops"].append(op_name)
        return original(op_name, *args, **kwargs)

    monkeypatch.setattr(functional, "dispatch", wrapped)

    y = torch.reshape(x, (6,))

    assert calls["count"] == 0
    assert calls["ops"] == []
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (6,)
    assert y.stride == (1,)
    assert y.tolist() == [1, 2, 3, 4, 5, 6]
    assert y is not x
    assert y.is_contiguous()
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value

def test_tensor_reshape_as_bypasses_python_tensor_reshape(monkeypatch):
    from candle._tensor import Tensor

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    other = torch.tensor([[0, 0, 0], [0, 0, 0]])

    _ = x.reshape_as(other)

    calls = {"count": 0}
    original = Tensor.reshape

    def wrapped(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "reshape", wrapped)

    y = x.reshape_as(other)

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 3)
    assert y.stride == (3, 1)
    assert y.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert y is not x
    assert y.is_contiguous()
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value


def test_tensor_swapdims_bypasses_python_tensor_transpose(monkeypatch):
    from candle._tensor import Tensor

    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    _ = x.swapdims(0, 2)

    calls = {"count": 0}
    original = Tensor.transpose

    def wrapped(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "transpose", wrapped)

    y = x.swapdims(0, 2)

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 2, 2)
    assert y.stride == (1, 2, 4)
    assert y.tolist() == [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value



def test_tensor_swapaxes_bypasses_python_tensor_swapdims(monkeypatch):
    from candle._tensor import Tensor

    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    _ = x.swapaxes(0, 2)

    calls = {"count": 0}
    original = Tensor.swapdims

    def wrapped(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "swapdims", wrapped)

    y = x.swapaxes(0, 2)

    assert calls["count"] == 0
    assert y.storage() is x.storage()
    assert y._base is x
    assert y._version_counter is x._version_counter
    assert y.shape == (2, 2, 2)
    assert y.stride == (1, 2, 4)
    assert y.tolist() == [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]
    assert y is not x
    assert not y.is_contiguous()
    assert x.tolist() == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    x._version_counter.bump()
    assert y._version_counter.value == x._version_counter.value
