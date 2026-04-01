import numpy as np
import candle as torch
from candle import nn, optim


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


def test_sgd_updates_matrix_parameter_on_mps():
    param = torch.randn((3, 4), device="mps", requires_grad=True)
    opt = optim.SGD([param], lr=0.1)
    before = param.detach().cpu().numpy().copy()

    loss = (param * param).sum()
    loss.backward()
    opt.step()

    after = param.detach().cpu().numpy()
    assert param.shape == (3, 4)
    assert not (after == before).all()


def test_sgd_momentum_matches_cpu_on_mps():
    cpu_param = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32, requires_grad=True)
    mps_param = cpu_param.detach().clone().to("mps").requires_grad_(True)

    cpu_opt = optim.SGD([cpu_param], lr=0.1, momentum=0.9)
    mps_opt = optim.SGD([mps_param], lr=0.1, momentum=0.9)

    for _ in range(3):
        cpu_loss = (cpu_param * cpu_param).sum()
        cpu_loss.backward()
        cpu_opt.step()
        cpu_opt.zero_grad()

        mps_loss = (mps_param * mps_param).sum()
        mps_loss.backward()
        mps_opt.step()
        mps_opt.zero_grad()

    np.testing.assert_allclose(
        mps_param.detach().cpu().numpy(),
        cpu_param.detach().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )

    cpu_buf = cpu_opt.state[id(cpu_param)]["momentum_buffer"].detach().numpy()
    mps_buf = mps_opt.state[id(mps_param)]["momentum_buffer"].detach().cpu().numpy()
    np.testing.assert_allclose(mps_buf, cpu_buf, rtol=1e-5, atol=1e-6)

    assert not np.allclose(cpu_param.detach().numpy(), np.array([1.0, -2.0, 3.0], dtype=np.float32))
    assert not np.allclose(mps_param.detach().cpu().numpy(), np.array([1.0, -2.0, 3.0], dtype=np.float32))
