"""Cross entropy L0 parity tests for NPU."""
import numpy as np

import candle as torch
import candle.nn.functional as F


def test_cross_entropy_fp16_logits_int64_targets_runs_on_npu():
    """cross_entropy must accept fp16 logits and int64 targets on NPU."""
    logits = torch.randn((2, 8), device="npu", dtype=torch.float16)
    targets = torch.tensor([1, 6], device="npu", dtype=torch.int64)

    loss = F.cross_entropy(logits, targets)

    assert loss.device.type == "npu"
    assert loss.shape == ()
    assert loss.dtype == torch.float16
    assert np.isfinite(loss.to("cpu").numpy()).all()


def test_cross_entropy_label_smoothing_fp16_npu():
    """label_smoothing branch must keep mask/scalars in logits dtype on NPU."""
    logits = torch.randn((2, 8), device="npu", dtype=torch.float16)
    targets = torch.tensor([0, 7], device="npu", dtype=torch.int64)

    loss = F.cross_entropy(logits, targets, label_smoothing=0.1)

    assert loss.device.type == "npu"
    assert loss.shape == ()
    assert loss.dtype == torch.float16
    assert np.isfinite(loss.to("cpu").numpy()).all()
