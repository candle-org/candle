"""Common test utilities — TestCase, run_tests, environment detection."""
import os
import sys
import unittest
import functools
import contextlib
import tempfile

import candle as torch
import numpy as np

# ---------------------------------------------------------------------------
# Environment flags (match PyTorch's names)
# ---------------------------------------------------------------------------
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform == "linux"

TEST_CUDA = (
    torch.cuda.is_available()
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available")
    else False
)
TEST_MPS = (
    torch.mps.is_available()
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available")
    else False
)
TEST_NPU = (
    torch.npu.is_available()
    if hasattr(torch, "npu") and hasattr(torch.npu, "is_available")
    else False
)

# Map CUDA tests to NPU when appropriate
TEST_CUDA = TEST_CUDA or TEST_NPU

TEST_MULTIGPU = False
if TEST_CUDA and hasattr(torch, "cuda") and hasattr(torch.cuda, "device_count"):
    TEST_MULTIGPU = torch.cuda.device_count() > 1


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------
class TestCase(unittest.TestCase):
    """PyTorch-compatible test case base class."""

    precision = 1e-5
    rel_tol = 0

    def setUp(self):
        super().setUp()

    def assertTensorsEqual(self, a, b, prec=None):
        if prec is None:
            prec = self.precision
        np.testing.assert_allclose(
            a.detach().cpu().numpy(),
            b.detach().cpu().numpy(),
            atol=prec,
            rtol=self.rel_tol,
        )

    def assertEqual(self, x, y, msg=None, *, atol=None, rtol=None):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            _atol = atol if atol is not None else self.precision
            _rtol = rtol if rtol is not None else self.rel_tol
            torch.testing.assert_close(x, y, atol=_atol, rtol=_rtol, msg=msg)
        else:
            super().assertEqual(x, y, msg=msg)


# ---------------------------------------------------------------------------
# Skip decorators
# ---------------------------------------------------------------------------
def skipIfNoCuda(fn):
    """Skip a test if no CUDA/NPU device is available."""
    return unittest.skipIf(not TEST_CUDA, "No CUDA/NPU device")(fn)


def skipIfNoMPS(fn):
    """Skip a test if no MPS device is available."""
    return unittest.skipIf(not TEST_MPS, "No MPS device")(fn)


def slowTest(fn):
    """Mark a test as slow; skipped when PYTORCH_TEST_SKIP_SLOW=1."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if os.environ.get("PYTORCH_TEST_SKIP_SLOW", "0") == "1":
            raise unittest.SkipTest("slow test")
        return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# run_tests
# ---------------------------------------------------------------------------
def run_tests():
    """Entry point matching PyTorch's run_tests() — delegates to unittest.main()."""
    unittest.main()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def make_tensor(*shape, dtype, device="cpu", low=None, high=None,
                requires_grad=False, **kwargs):
    """Create a random tensor — thin wrapper around torch.testing.make_tensor."""
    return torch.testing.make_tensor(
        *shape, dtype=dtype, device=device, low=low, high=high,
        requires_grad=requires_grad, **kwargs
    )


@contextlib.contextmanager
def freeze_rng_state():
    """Context manager that saves and restores the RNG state."""
    rng_state = (
        torch.random.get_rng_state()
        if hasattr(torch.random, "get_rng_state")
        else None
    )
    try:
        yield
    finally:
        if rng_state is not None:
            torch.random.set_rng_state(rng_state)


def parametrize(arg_name, arg_values):
    """Simple parametrize decorator compatible with unittest.TestCase."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self):
            for val in arg_values:
                with self.subTest(**{arg_name: val}):
                    fn(self, **{arg_name: val})
        return wrapper
    return decorator


def subtest(arg_values):
    """Decorator form of subTest for parametrize-like usage."""
    return parametrize("x", arg_values)


# ---------------------------------------------------------------------------
# Flags that are always False in candle (kept for API compatibility)
# ---------------------------------------------------------------------------
TEST_WITH_ROCM = False
TEST_WITH_ASAN = False
TEST_WITH_TSAN = False
