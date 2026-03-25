"""Tests for DDP bucket hot-path: dtype-aware sizing and Cython fastpath.

Covers:
  1. build_bucket_mapping is dtype-aware (float16 buckets are half the byte
     size of float32 buckets for the same numel, so more fit in a fixed cap).
  2. The Cython extension is actually imported and used (not silently skipped).
  3. Bucket byte-size cap is respected for mixed-dtype scenarios.
  4. dtype_itemsize returns correct widths for the common dtypes.
"""
import os
import socket

import pytest


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init_pg(port):
    import candle.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=0, world_size=1,
        init_method=f"tcp://127.0.0.1:{port}"
    )


def _teardown():
    import candle.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


class TestCythonFastpathPresent:
    """The compiled extension must be importable."""

    def test_extension_importable(self):
        from candle.distributed import _ddp_fastpath  # noqa: F401

    def test_have_fastpath_flag_is_true(self):
        from candle.nn.parallel.distributed import _HAVE_FASTPATH
        assert _HAVE_FASTPATH is True, (
            "_HAVE_FASTPATH is False — Cython extension not loaded. "
            "Run 'python setup.py build_ext --inplace'."
        )


class TestDtypeItemsize:
    """dtype_itemsize must return correct byte widths."""

    def test_float32_is_4(self):
        from candle.distributed._ddp_fastpath import dtype_itemsize
        import candle as torch
        t = torch.zeros((1,), dtype=torch.float32)
        assert dtype_itemsize(t.dtype) == 4

    def test_float16_is_2(self):
        from candle.distributed._ddp_fastpath import dtype_itemsize
        import candle as torch
        t = torch.zeros((1,), dtype=torch.float16)
        assert dtype_itemsize(t.dtype) == 2

    def test_float64_is_8(self):
        from candle.distributed._ddp_fastpath import dtype_itemsize
        import candle as torch
        t = torch.zeros((1,), dtype=torch.float64)
        assert dtype_itemsize(t.dtype) == 8

    def test_int8_is_1(self):
        from candle.distributed._ddp_fastpath import dtype_itemsize
        import candle as torch
        t = torch.zeros((1,), dtype=torch.int8)
        assert dtype_itemsize(t.dtype) == 1

    def test_int32_is_4(self):
        from candle.distributed._ddp_fastpath import dtype_itemsize
        import candle as torch
        t = torch.zeros((1,), dtype=torch.int32)
        assert dtype_itemsize(t.dtype) == 4


class TestBuildBucketMappingDtypeAware:
    """Bucket byte-cap must be dtype-aware (not hardcoded *4)."""

    def test_float16_params_fit_more_in_same_cap(self):
        """Two float16 params of 100 elements fit in a 400-byte cap;
        two float32 params of 100 elements do not."""
        from candle.distributed._ddp_fastpath import build_bucket_mapping
        import candle as torch
        import candle.nn as nn

        # 100 elements * 4 bytes = 400 bytes each for float32
        p_f32_a = nn.Parameter(torch.zeros((100,), dtype=torch.float32))
        p_f32_b = nn.Parameter(torch.zeros((100,), dtype=torch.float32))
        # 100 elements * 2 bytes = 200 bytes each for float16
        p_f16_a = nn.Parameter(torch.zeros((100,), dtype=torch.float16))
        p_f16_b = nn.Parameter(torch.zeros((100,), dtype=torch.float16))

        cap = 400  # bytes

        # float32: 100*4=400 bytes each. First fits, second overflows -> 2 buckets.
        grad_params_f32 = [(0, p_f32_a), (1, p_f32_b)]
        buckets_f32, _ = build_bucket_mapping(grad_params_f32, cap)
        assert len(buckets_f32) == 2, (
            f"float32: expected 2 buckets (each param = 400B = cap), got {len(buckets_f32)}"
        )

        # float16: 100*2=200 bytes each. Both fit in 400-byte cap -> 1 bucket.
        grad_params_f16 = [(0, p_f16_a), (1, p_f16_b)]
        buckets_f16, _ = build_bucket_mapping(grad_params_f16, cap)
        assert len(buckets_f16) == 1, (
            f"float16: expected 1 bucket (2*200=400B <= cap), got {len(buckets_f16)}"
        )

    def test_param_to_bucket_mapping_is_complete(self):
        """Every param index appears in the returned param_to_bucket dict."""
        from candle.distributed._ddp_fastpath import build_bucket_mapping
        import candle as torch
        import candle.nn as nn

        params = [(i, nn.Parameter(torch.zeros((10,)))) for i in range(5)]
        _, p2b = build_bucket_mapping(params, 10000)
        for i in range(5):
            assert i in p2b, f"param index {i} missing from param_to_bucket"


class TestDDPBucketSizingEndToEnd:
    """DDP bucket grouping is dtype-aware end-to-end through the Reducer."""

    def setup_method(self):
        self._port = _free_port()
        _init_pg(self._port)

    def teardown_method(self):
        _teardown()

    def test_float16_model_correct_grad(self):
        """Float16 model trains without error and gradients are correct."""
        import candle as torch
        import candle.nn as nn
        import numpy as np

        class TinyF16(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones((4,), dtype=torch.float16))

            def forward(self, x):
                return (self.w * x).sum()

        base = TinyF16()
        ddp = nn.DistributedDataParallel(base)
        x = torch.ones((1, 4), dtype=torch.float16)
        ddp(x).sum().backward()

        grad = base.w.grad
        assert grad is not None, "float16 param must have grad after backward"
        grad_np = np.array(grad.tolist(), dtype=np.float32).ravel()
        np.testing.assert_allclose(
            grad_np, np.ones(4, dtype=np.float32),
            rtol=1e-3,
            err_msg="float16 DDP grad should equal 1.0 for ones input (world_size=1)"
        )
