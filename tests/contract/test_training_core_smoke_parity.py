import numpy as np

import candle as torch


def test_training_core_smoke_parity_matches_torch_cpu():
    import torch as real_torch

    x_data = [[1.0, 2.0], [3.0, 4.0]]
    target_data = [[0.5], [1.5]]
    weight_data = [[0.1, -0.2]]
    bias_data = [0.3]
    lr = 0.05
    steps = 4

    candle_model = torch.nn.Linear(2, 1, dtype=torch.float32)
    candle_model.weight.data = torch.tensor(weight_data, dtype=torch.float32)
    candle_model.bias.data = torch.tensor(bias_data, dtype=torch.float32)
    candle_opt = torch.optim.SGD(candle_model.parameters(), lr=lr)

    torch_model = real_torch.nn.Linear(2, 1, dtype=real_torch.float32)
    with real_torch.no_grad():
        torch_model.weight.copy_(real_torch.tensor(weight_data, dtype=real_torch.float32))
        torch_model.bias.copy_(real_torch.tensor(bias_data, dtype=real_torch.float32))
    torch_opt = real_torch.optim.SGD(torch_model.parameters(), lr=lr)

    cx = torch.tensor(x_data, dtype=torch.float32)
    cy = torch.tensor(target_data, dtype=torch.float32)
    tx = real_torch.tensor(x_data, dtype=real_torch.float32)
    ty = real_torch.tensor(target_data, dtype=real_torch.float32)

    candle_losses = []
    torch_losses = []

    for _ in range(steps):
        candle_opt.zero_grad()
        cout = candle_model(cx)
        cdiff = cout - cy
        closs = torch.mean(cdiff * cdiff)
        candle_losses.append(float(closs.item()))
        closs.backward()
        assert candle_model.weight.grad is not None
        assert candle_model.bias.grad is not None
        candle_opt.step()

        torch_opt.zero_grad()
        tout = torch_model(tx)
        tdiff = tout - ty
        tloss = real_torch.mean(tdiff * tdiff)
        torch_losses.append(float(tloss.item()))
        tloss.backward()
        assert torch_model.weight.grad is not None
        assert torch_model.bias.grad is not None
        torch_opt.step()

    assert all(np.isfinite(candle_losses))
    assert all(np.isfinite(torch_losses))
    assert candle_losses[-1] < candle_losses[0]
    assert torch_losses[-1] < torch_losses[0]

    np.testing.assert_allclose(candle_losses, torch_losses, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        candle_model.weight.detach().numpy(),
        torch_model.weight.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        candle_model.bias.detach().numpy(),
        torch_model.bias.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
