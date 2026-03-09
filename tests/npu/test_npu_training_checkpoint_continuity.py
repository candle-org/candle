import pytest

import candle as torch


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_checkpoint_roundtrip_continues_training(tmp_path):
    torch.manual_seed(2026)

    x = torch.randn((32, 8), device="npu")
    target = torch.randn((32, 4), device="npu")
    weight = torch.randn((8, 4), device="npu")
    weight.requires_grad_(True)

    lr = 1e-3

    def train_step(w):
        if w.grad is not None:
            w.grad = None

        out = torch.matmul(x, w)
        diff = torch.sub(out, target)
        loss = torch.mean(torch.pow(diff, 2.0))

        out.backward(diff)
        grad = w.grad
        assert grad is not None

        grad = torch.div(grad, float(x.shape[0]))
        step = torch.div(grad, 1.0 / lr)
        next_w = torch.sub(w, step).detach()
        next_w.requires_grad_(True)
        return float(loss.to("cpu").item()), next_w

    pre_losses = []
    for _ in range(5):
        loss_value, weight = train_step(weight)
        pre_losses.append(loss_value)

    ckpt_path = tmp_path / "npu_weight.ckpt"
    torch.save({"weight": weight.detach().to("cpu")}, ckpt_path)

    payload = torch.load(ckpt_path)
    reloaded = payload["weight"].to("npu")
    reloaded.requires_grad_(True)

    post_losses = []
    for _ in range(5):
        loss_value, reloaded = train_step(reloaded)
        post_losses.append(loss_value)

    all_losses = pre_losses + post_losses
    assert all(value == value for value in all_losses)
    assert all(abs(value) != float("inf") for value in all_losses)

    # Continuity checks: post-reload training remains active and does not diverge.
    assert len(post_losses) == 5
    assert post_losses[-1] <= post_losses[0]
