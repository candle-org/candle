import candle as torch
import torch as pt
from .helpers import assert_torch_error


def _assert_torch_npu_error(fn_mt, fn_torch):
    """Like assert_torch_error but normalizes 'cuda' -> 'npu' in messages."""
    try:
        fn_torch()
    except Exception as exc_torch:
        torch_exc = exc_torch
    else:
        torch_exc = None

    try:
        fn_mt()
    except Exception as exc_mt:
        mt_exc = exc_mt
    else:
        mt_exc = None

    assert type(mt_exc) is type(torch_exc)
    # Normalize PyTorch's "cuda" to "npu" before comparing
    torch_msg = str(torch_exc).replace("cuda", "npu") if torch_exc is not None else None
    mt_msg = str(mt_exc) if mt_exc is not None else None
    assert mt_msg == torch_msg


def test_npu_set_device_invalid_string_matches_torch():
    def mt():
        torch.npu.set_device("cpu")

    def th():
        pt.cuda.set_device("cpu")

    _assert_torch_npu_error(mt, th)


def test_npu_device_ctx_invalid_string_matches_torch():
    def mt():
        torch.npu.device("cpu").__enter__()

    def th():
        pt.cuda.device("cpu").__enter__()

    _assert_torch_npu_error(mt, th)


def test_npu_memory_summary_returns_str():
    assert isinstance(torch.npu.memory_summary(), str)


def test_npu_memory_snapshot_has_keys():
    snap = torch.npu.memory_snapshot()
    assert isinstance(snap, dict)
    assert "segments" in snap
    assert "device" in snap
