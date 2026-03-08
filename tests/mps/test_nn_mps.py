import candle as torch
from candle import nn


def test_linear_mps():
    layer = nn.Linear(4, 3).to("mps")
    x = torch.randn((2, 4), device="mps")
    y = layer(x)
    assert y.shape == (2, 3)
    assert y.device.type == "mps"


def test_conv2d_mps():
    layer = nn.Conv2d(3, 16, kernel_size=3, padding=1).to("mps")
    x = torch.randn((1, 3, 8, 8), device="mps")
    y = layer(x)
    assert y.shape == (1, 16, 8, 8)
    assert y.device.type == "mps"


def test_batchnorm2d_mps():
    layer = nn.BatchNorm2d(3).to("mps")
    layer.eval()
    x = torch.randn((2, 3, 4, 4), device="mps")
    y = layer(x)
    assert y.shape == (2, 3, 4, 4)
    assert y.device.type == "mps"


def test_layernorm_mps():
    layer = nn.LayerNorm(4).to("mps")
    x = torch.randn((2, 3, 4), device="mps")
    y = layer(x)
    assert y.shape == (2, 3, 4)
    assert y.device.type == "mps"


def test_embedding_mps():
    layer = nn.Embedding(10, 4).to("mps")
    x = torch.tensor([0, 3, 7], dtype=torch.int64, device="mps")
    y = layer(x)
    assert y.shape == (3, 4)
    assert y.device.type == "mps"


def test_relu_module_mps():
    act = nn.ReLU()
    x = torch.tensor([-1.0, 0.0, 1.0], device="mps")
    y = act(x)
    assert y.device.type == "mps"
    assert y.cpu().tolist() == [0.0, 0.0, 1.0]


def test_gelu_module_mps():
    act = nn.GELU()
    x = torch.tensor([-1.0, 0.0, 1.0], device="mps")
    y = act(x)
    assert y.shape == (3,)
    assert y.device.type == "mps"


def test_sigmoid_module_mps():
    act = nn.Sigmoid()
    x = torch.tensor([-1.0, 0.0, 1.0], device="mps")
    y = act(x)
    assert y.shape == (3,)
    assert y.device.type == "mps"
    assert abs(y.cpu().tolist()[1] - 0.5) < 1e-5


def test_softmax_mps():
    layer = nn.Softmax(dim=-1)
    x = torch.tensor([[1.0, 2.0, 3.0]], device="mps")
    y = layer(x)
    assert y.shape == (1, 3)
    assert y.device.type == "mps"
    total = y.cpu().sum().item()
    assert abs(total - 1.0) < 1e-5


def test_log_softmax_mps():
    layer = nn.LogSoftmax(dim=-1)
    x = torch.tensor([[1.0, 2.0, 3.0]], device="mps")
    y = layer(x)
    assert y.shape == (1, 3)
    assert y.device.type == "mps"
    # log_softmax values should all be <= 0
    vals = y.cpu().tolist()[0]
    assert all(v <= 0.0 for v in vals)
