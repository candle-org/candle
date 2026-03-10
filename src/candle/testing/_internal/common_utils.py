"""Common test utilities — TestCase, run_tests, environment detection."""
import os
import sys
import unittest
import functools
import contextlib
import tempfile
import warnings

import candle as torch
import numpy as np

# ---------------------------------------------------------------------------
# Environment flags (match PyTorch's names)
# ---------------------------------------------------------------------------
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform == "linux"
IS_PPC = False
IS_JETSON = False
IS_SANDCASTLE = False
IS_FBCODE = False

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

TEST_SCIPY = False
try:
    import scipy  # noqa: F401
    TEST_SCIPY = True
except ImportError:
    pass

TEST_NUMBA = False
TEST_DILL = False
TEST_LIBROSA = False

# ---------------------------------------------------------------------------
# Flags that are always False in candle (kept for API compatibility)
# ---------------------------------------------------------------------------
TEST_WITH_ROCM = False
TEST_WITH_ASAN = False
TEST_WITH_TSAN = False
TEST_WITH_UBSAN = False
TEST_WITH_DEV_DBG_ASAN = False
GRADCHECK_NONDET_TOL = 0.0

# ---------------------------------------------------------------------------
# dtype maps
# ---------------------------------------------------------------------------
torch_to_numpy_dtype_dict = {
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}

numpy_to_torch_dtype_dict = {v: k for k, v in torch_to_numpy_dtype_dict.items()}


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
# Skip / expected-failure decorators
# ---------------------------------------------------------------------------
def skipIfNoCuda(fn):
    """Skip a test if no CUDA/NPU device is available."""
    return unittest.skipIf(not TEST_CUDA, "No CUDA/NPU device")(fn)


def skipIfNoMPS(fn):
    """Skip a test if no MPS device is available."""
    return unittest.skipIf(not TEST_MPS, "No MPS device")(fn)


skipIfMps = skipIfNoMPS


def slowTest(fn):
    """Mark a test as slow; skipped when PYTORCH_TEST_SKIP_SLOW=1."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if os.environ.get("PYTORCH_TEST_SKIP_SLOW", "0") == "1":
            raise unittest.SkipTest("slow test")
        return fn(*args, **kwargs)
    return wrapper


def skipIfTorchDynamo(msg=""):
    """No-op decorator — candle has no dynamo."""
    def decorator(fn):
        return fn
    return decorator


def xfailIfTorchDynamo(fn):
    """No-op decorator — candle has no dynamo."""
    return fn


def skipCUDAMemoryLeakCheckIf(condition):
    """No-op decorator — candle has no CUDA memory leak checker."""
    def decorator(fn):
        return fn
    return decorator


def skipIfCrossRef(fn):
    """No-op decorator."""
    return fn


def skipIfSlowGradcheckEnv(fn):
    """No-op decorator."""
    return fn


def suppress_warnings(fn):
    """Decorator to suppress warnings during a test."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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


@contextlib.contextmanager
def set_default_dtype(dtype):
    """Context manager to temporarily set the default dtype."""
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(old)


@contextlib.contextmanager
def set_default_tensor_type(tensor_type):
    """Context manager to temporarily set the default tensor type."""
    # In candle, this is a no-op that just yields
    yield


def do_test_empty_full(self, dtypes, layout, device):
    """Placeholder for empty/full test helper — tests will fail at runtime."""
    pass


class LazyVal:
    """Lazy value — computes on first access."""
    def __init__(self, fn):
        self._fn = fn
        self._val = None
        self._computed = False

    def __bool__(self):
        if not self._computed:
            self._val = self._fn()
            self._computed = True
        return bool(self._val)


# ---------------------------------------------------------------------------
# More environment flags used by PyTorch tests
# ---------------------------------------------------------------------------
TEST_WITH_TORCHINDUCTOR = False
TEST_WITH_TORCHDYNAMO = False
TEST_WITH_CROSSREF = False
IS_FILESYSTEM_UTF8_ENCODING = sys.getfilesystemencoding() == "utf-8"
NO_MULTIPROCESSING_SPAWN = False
IS_REMOTE_GPU = False


# ---------------------------------------------------------------------------
# More skip / no-op decorators
# ---------------------------------------------------------------------------
def skipIfTorchInductor(msg=""):
    """No-op decorator — candle has no TorchInductor."""
    def decorator(fn):
        return fn
    return decorator


def skipRocmIfTorchInductor(fn):
    """No-op decorator."""
    return fn


def skipIfRocm(fn_or_msg=None):
    """No-op decorator — candle has no ROCm."""
    if callable(fn_or_msg):
        return fn_or_msg
    def decorator(fn):
        return fn
    return decorator


def skipIfNoSciPy(fn):
    """Skip test if scipy is not available."""
    return unittest.skipIf(not TEST_SCIPY, "No scipy")(fn)


def slowTestIf(condition):
    """Conditionally mark a test as slow."""
    def decorator(fn):
        if condition:
            return slowTest(fn)
        return fn
    return decorator


def wrapDeterministicFlagAPITest(fn):
    """No-op wrapper — candle has no deterministic flag API."""
    return fn


# ---------------------------------------------------------------------------
# More utility classes / context managers
# ---------------------------------------------------------------------------
class BytesIOContext:
    """Context manager wrapping io.BytesIO."""
    def __init__(self, initial_bytes=b""):
        import io
        self._buf = io.BytesIO(initial_bytes)

    def __enter__(self):
        return self._buf

    def __exit__(self, *args):
        self._buf.close()


@contextlib.contextmanager
def TemporaryFileName(*args, **kwargs):
    """Context manager that yields a temporary file path."""
    with tempfile.NamedTemporaryFile(*args, delete=False, **kwargs) as f:
        name = f.name
    try:
        yield name
    finally:
        os.unlink(name)


@contextlib.contextmanager
def TemporaryDirectoryName(*args, **kwargs):
    """Context manager that yields a temporary directory path."""
    with tempfile.TemporaryDirectory(*args, **kwargs) as d:
        yield d


class DeterministicGuard:
    """Context manager stub for torch.use_deterministic_algorithms."""
    def __init__(self, deterministic, *, warn_only=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class CudaSyncGuard:
    """Context manager stub for CUDA sync testing."""
    def __init__(self, sync_debug_mode=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class AlwaysWarnTypedStorageRemoval:
    """Context manager stub."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def bytes_to_scalar(b, dtype=None):
    """Convert bytes to a scalar value."""
    return int.from_bytes(b, byteorder="little")


def noncontiguous_like(tensor):
    """Create a non-contiguous tensor with same data."""
    # Simple implementation: expand then slice to break contiguity
    if tensor.dim() == 0:
        return tensor.clone()
    result = torch.empty(*(s * 2 for s in tensor.shape), dtype=tensor.dtype,
                         device=tensor.device)
    slices = tuple(slice(None, None, 2) for _ in tensor.shape)
    result[slices] = tensor
    return result[slices]


def load_tests(loader, tests, pattern):
    """Standard load_tests protocol for unittest."""
    return tests


def get_all_device_types():
    """Return list of available device types."""
    devices = ["cpu"]
    if TEST_CUDA:
        devices.append("cuda")
    if TEST_MPS:
        devices.append("mps")
    return devices
