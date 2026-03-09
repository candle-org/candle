import json

import pytest

import candle as torch


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_golden_training_loop_loss_decreases_and_is_finite(tmp_path):
    """Golden NPU training loop for 0.1 scope.

    This loop intentionally uses a matmul-based path with explicit upstream
    gradients to avoid reduction-backward paths that currently depend on
    unstable NPU mul kernels in this environment.
    """
    torch.manual_seed(1234)

    x = torch.randn((32, 8), device="npu")
    target = torch.randn((32, 4), device="npu")
    weight = torch.randn((8, 4), device="npu")
    weight.requires_grad_(True)

    lr = 1e-3
    losses = []
    steps = 20

    for _step in range(steps):
        if weight.grad is not None:
            weight.grad = None

        out = torch.matmul(x, weight)
        diff = torch.sub(out, target)

        loss = torch.mean(torch.pow(diff, 2.0))
        losses.append(float(loss.to("cpu").item()))

        out.backward(diff)
        grad = weight.grad
        assert grad is not None

        grad = torch.div(grad, float(x.shape[0]))
        step = torch.div(grad, 1.0 / lr)
        weight = torch.sub(weight, step).detach()
        weight.requires_grad_(True)

    assert all(value == value for value in losses)
    assert all(abs(value) != float("inf") for value in losses)
    assert losses[-1] < losses[0]

    payload = {
        "seed": 1234,
        "steps": steps,
        "lr": lr,
        "losses": losses,
    }
    out_file = tmp_path / "golden_loss_trace.json"
    out_file.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert reloaded["seed"] == 1234
    assert reloaded["steps"] == steps
    assert reloaded["lr"] == lr
    assert len(reloaded["losses"]) == steps
