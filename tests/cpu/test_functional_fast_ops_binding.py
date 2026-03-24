import candle as torch
from candle import _functional as F



def test_functional_add_is_bound_to_cython_fast_path():
    assert F.add.__module__ == "candle._cython._fast_ops"

    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    out = F.add(a, b)

    torch.testing.assert_close(out, a + b)



def test_functional_mul_is_bound_to_cython_fast_path():
    assert F.mul.__module__ == "candle._cython._fast_ops"

    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    out = F.mul(a, b)

    torch.testing.assert_close(out, a * b)



def test_functional_matmul_is_bound_to_cython_fast_path():
    assert F.matmul.__module__ == "candle._cython._fast_ops"

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[1.0], [2.0]])
    out = F.matmul(a, b)

    torch.testing.assert_close(out, a @ b)
