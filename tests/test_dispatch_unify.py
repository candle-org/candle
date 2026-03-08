import numpy as np

import candle._creation as creation
import candle._functional as functional
from candle._dtype import float32
from candle._storage import typed_storage_from_numpy
from candle._tensor import Tensor


def test_tensor_creation_dispatch(monkeypatch):
    called = {}

    def fake_dispatch(name, device, *args, **kwargs):
        called["name"] = name
        storage = typed_storage_from_numpy(np.array([1.0], dtype=np.float32), float32)
        return Tensor(storage, (1,), (1,))

    monkeypatch.setattr(functional, "dispatch", fake_dispatch)
    _ = creation.tensor([1.0])
    assert called["name"] == "tensor"


def test_to_uses_dispatch(monkeypatch):
    called = {}

    def fake_dispatch(name, device, *args, **kwargs):
        called["name"] = name
        storage = typed_storage_from_numpy(np.array([1.0], dtype=np.float32), float32)
        return Tensor(storage, (1,), (1,))

    monkeypatch.setattr(functional, "dispatch", fake_dispatch)
    storage = typed_storage_from_numpy(np.array([1.0], dtype=np.float32), float32)
    x = Tensor(storage, (1,), (1,))
    _ = x.to("cpu")
    assert called["name"] == "to"
