import numpy as np
import candle as torch


# --- Arithmetic ---

def test_add_mps():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
    b = torch.tensor([[2.0, 0.5], [1.0, -1.0]], device="mps")
    result = torch.add(a, b)
    assert result.device.type == "mps"
    expected = np.array([[3.0, 2.5], [4.0, 3.0]])
    np.testing.assert_allclose(result.cpu().numpy(), expected)


def test_sub_mps():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
    b = torch.tensor([[2.0, 0.5], [1.0, -1.0]], device="mps")
    result = torch.sub(a, b)
    assert result.device.type == "mps"
    expected = np.array([[-1.0, 1.5], [2.0, 5.0]])
    np.testing.assert_allclose(result.cpu().numpy(), expected)


def test_mul_mps():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
    b = torch.tensor([[2.0, 0.5], [1.0, -1.0]], device="mps")
    result = torch.mul(a, b)
    assert result.device.type == "mps"
    expected = np.array([[2.0, 1.0], [3.0, -4.0]])
    np.testing.assert_allclose(result.cpu().numpy(), expected)


def test_div_mps():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
    b = torch.tensor([[2.0, 0.5], [1.0, -1.0]], device="mps")
    result = torch.div(a, b)
    assert result.device.type == "mps"
    expected = np.array([[0.5, 4.0], [3.0, -4.0]])
    np.testing.assert_allclose(result.cpu().numpy(), expected)


def test_matmul_mps():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
    b = torch.tensor([[2.0, 0.5], [1.0, -1.0]], device="mps")
    result = torch.matmul(a, b)
    assert result.device.type == "mps"
    expected = np.matmul(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[2.0, 0.5], [1.0, -1.0]]),
    )
    np.testing.assert_allclose(result.cpu().numpy(), expected)


def test_mm_mps():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
    b = torch.tensor([[2.0, 0.5], [1.0, -1.0]], device="mps")
    result = torch.mm(a, b)
    assert result.device.type == "mps"
    expected = np.matmul(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[2.0, 0.5], [1.0, -1.0]]),
    )
    np.testing.assert_allclose(result.cpu().numpy(), expected)


def test_bmm_mps():
    a = torch.randn((2, 3, 4), device="mps")
    b = torch.randn((2, 4, 5), device="mps")
    result = torch.bmm(a, b)
    assert result.shape == (2, 3, 5)
    assert result.device.type == "mps"
    a_np = a.cpu().numpy()
    b_np = b.cpu().numpy()
    expected = np.matmul(a_np, b_np)
    np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5, atol=1e-5)


# --- Unary ops ---

def test_abs_mps():
    x = torch.tensor([-1.0, 0.5, 2.0], device="mps")
    expected = np.abs(np.array([-1.0, 0.5, 2.0]))
    np.testing.assert_allclose(torch.abs(x).cpu().numpy(), expected)
    np.testing.assert_allclose(x.abs().cpu().numpy(), expected)


def test_neg_mps():
    x = torch.tensor([-1.0, 0.5, 2.0], device="mps")
    expected = -np.array([-1.0, 0.5, 2.0])
    np.testing.assert_allclose(torch.neg(x).cpu().numpy(), expected)
    np.testing.assert_allclose((-x).cpu().numpy(), expected)


def test_exp_mps():
    x = torch.tensor([0.5, 1.0, 2.0], device="mps")
    expected = np.exp(np.array([0.5, 1.0, 2.0]))
    np.testing.assert_allclose(torch.exp(x).cpu().numpy(), expected, rtol=1e-5)
    np.testing.assert_allclose(x.exp().cpu().numpy(), expected, rtol=1e-5)


def test_log_mps():
    x = torch.tensor([0.5, 1.0, 2.0], device="mps")
    expected = np.log(np.array([0.5, 1.0, 2.0]))
    np.testing.assert_allclose(torch.log(x).cpu().numpy(), expected, rtol=1e-5)
    np.testing.assert_allclose(x.log().cpu().numpy(), expected, rtol=1e-5)


def test_sqrt_mps():
    x = torch.tensor([0.5, 1.0, 4.0], device="mps")
    expected = np.sqrt(np.array([0.5, 1.0, 4.0]))
    np.testing.assert_allclose(torch.sqrt(x).cpu().numpy(), expected, rtol=1e-5)
    np.testing.assert_allclose(x.sqrt().cpu().numpy(), expected, rtol=1e-5)


def test_relu_mps():
    x = torch.tensor([-1.0, 0.0, 2.0], device="mps")
    expected = np.maximum(np.array([-1.0, 0.0, 2.0]), 0.0)
    np.testing.assert_allclose(torch.relu(x).cpu().numpy(), expected)


def test_sigmoid_mps():
    x = torch.tensor([-1.0, 0.0, 1.0], device="mps")
    x_np = np.array([-1.0, 0.0, 1.0])
    expected = 1.0 / (1.0 + np.exp(-x_np))
    np.testing.assert_allclose(torch.sigmoid(x).cpu().numpy(), expected, rtol=1e-5)
    np.testing.assert_allclose(x.sigmoid().cpu().numpy(), expected, rtol=1e-5)


def test_tanh_mps():
    x = torch.tensor([0.0, 0.5, 1.0], device="mps")
    expected = np.tanh(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(torch.tanh(x).cpu().numpy(), expected, rtol=1e-5)
    np.testing.assert_allclose(x.tanh().cpu().numpy(), expected, rtol=1e-5)


# --- Reductions ---

def test_sum_mps():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
    result = torch.sum(x, dim=1, keepdim=True)
    assert result.device.type == "mps"
    expected = np.array([[3.0], [7.0]])
    np.testing.assert_allclose(result.cpu().numpy(), expected)


def test_mean_mps():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
    result = torch.mean(x, dim=1)
    assert result.device.type == "mps"
    expected = np.array([1.5, 3.5])
    np.testing.assert_allclose(result.cpu().numpy(), expected)


def test_max_mps():
    x = torch.tensor([1.0, 3.0, 2.0], device="mps")
    y = torch.tensor([3.0, 1.0, 2.0], device="mps")
    expected = np.maximum(np.array([1.0, 3.0, 2.0]), np.array([3.0, 1.0, 2.0]))
    np.testing.assert_allclose(torch.max(x, y).cpu().numpy(), expected)


def test_min_mps():
    x = torch.tensor([1.0, 3.0, 2.0], device="mps")
    y = torch.tensor([3.0, 1.0, 2.0], device="mps")
    expected = np.minimum(np.array([1.0, 3.0, 2.0]), np.array([3.0, 1.0, 2.0]))
    np.testing.assert_allclose(torch.min(x, y).cpu().numpy(), expected)


def test_argmax_mps():
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], device="mps")
    expected = np.argmax(np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]]), axis=1)
    np.testing.assert_array_equal(torch.argmax(x, dim=1).cpu().numpy(), expected)


def test_argmin_mps():
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], device="mps")
    expected = np.argmin(np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]]), axis=1)
    np.testing.assert_array_equal(torch.argmin(x, dim=1).cpu().numpy(), expected)


# --- Comparison ---

def test_eq_mps():
    x = torch.tensor([1.0, 2.0, 3.0], device="mps")
    y = torch.tensor([1.0, 0.0, 3.0], device="mps")
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(torch.eq(x, y).cpu().numpy(), expected)


def test_ne_mps():
    x = torch.tensor([1.0, 2.0, 3.0], device="mps")
    y = torch.tensor([1.0, 0.0, 3.0], device="mps")
    expected = np.array([False, True, False])
    np.testing.assert_array_equal(x.ne(y).cpu().numpy(), expected)


def test_gt_mps():
    x = torch.tensor([1.0, 2.0, 3.0], device="mps")
    y = torch.tensor([1.0, 0.0, 4.0], device="mps")
    expected = np.array([False, True, False])
    np.testing.assert_array_equal(torch.gt(x, y).cpu().numpy(), expected)


def test_lt_mps():
    x = torch.tensor([1.0, 2.0, 3.0], device="mps")
    y = torch.tensor([1.0, 0.0, 4.0], device="mps")
    expected = np.array([False, False, True])
    np.testing.assert_array_equal(torch.lt(x, y).cpu().numpy(), expected)


def test_ge_mps():
    x = torch.tensor([1.0, 2.0, 3.0], device="mps")
    y = torch.tensor([1.0, 0.0, 4.0], device="mps")
    expected = np.array([True, True, False])
    np.testing.assert_array_equal(torch.ge(x, y).cpu().numpy(), expected)


def test_le_mps():
    x = torch.tensor([1.0, 2.0, 3.0], device="mps")
    y = torch.tensor([1.0, 0.0, 4.0], device="mps")
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(torch.le(x, y).cpu().numpy(), expected)


# --- Shape ops ---

def test_reshape_mps():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
    result = x.reshape(3, 2)
    assert result.shape == (3, 2)
    assert result.device.type == "mps"
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).reshape(3, 2)
    np.testing.assert_allclose(result.cpu().numpy(), expected)


def test_transpose_mps():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
    result = x.transpose(0, 1)
    assert result.shape == (3, 2)
    assert result.device.type == "mps"
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).T
    np.testing.assert_allclose(result.cpu().numpy(), expected)


def test_permute_mps():
    x = torch.randn((2, 3, 4), device="mps")
    result = x.permute(2, 0, 1)
    assert result.shape == (4, 2, 3)
    assert result.device.type == "mps"
    np.testing.assert_allclose(
        result.cpu().numpy(),
        np.transpose(x.cpu().numpy(), (2, 0, 1)),
    )


def test_squeeze_mps():
    x = torch.zeros((1, 3, 1), device="mps")
    result = x.squeeze()
    assert result.shape == (3,)
    assert result.device.type == "mps"


def test_unsqueeze_mps():
    x = torch.zeros((3,), device="mps")
    result = x.unsqueeze(0)
    assert result.shape == (1, 3)
    assert result.device.type == "mps"


def test_cat_mps():
    a = torch.tensor([[1.0, 2.0]], device="mps")
    b = torch.tensor([[3.0, 4.0]], device="mps")
    result = torch.cat([a, b], dim=0)
    assert result.shape == (2, 2)
    assert result.device.type == "mps"
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(result.cpu().numpy(), expected)


def test_stack_mps():
    a = torch.tensor([1.0, 2.0], device="mps")
    b = torch.tensor([3.0, 4.0], device="mps")
    result = torch.stack([a, b], dim=0)
    assert result.shape == (2, 2)
    assert result.device.type == "mps"
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(result.cpu().numpy(), expected)
