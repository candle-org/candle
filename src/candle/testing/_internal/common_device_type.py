"""Device-type test infrastructure — instantiate_device_type_tests, @dtypes, etc."""
import os
import unittest
import functools
import inspect
from collections import namedtuple

import candle as torch

from .common_utils import TEST_CUDA, TEST_MPS, TEST_NPU

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def _get_available_devices():
    devices = ["cpu"]
    env_devices = os.environ.get("CANDLE_TEST_DEVICES", "")
    if env_devices:
        return [d.strip() for d in env_devices.split(",")]
    if TEST_CUDA:
        devices.append("cuda")
    if TEST_NPU and not TEST_CUDA:
        devices.append("npu")
    if TEST_MPS:
        devices.append("mps")
    return devices

# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
def dtypes(*dtype_args):
    """Parametrize a test over multiple dtypes."""
    def decorator(fn):
        fn._dtypes = dtype_args
        return fn
    return decorator

def onlyCPU(fn):
    fn._only_device = "cpu"
    return fn

def onlyCUDA(fn):
    fn._only_device = "cuda"
    return fn

def onlyNativeDeviceTypes(fn):
    fn._only_native = True
    return fn

def deviceCountAtLeast(count):
    def decorator(fn):
        fn._min_device_count = count
        return fn
    return decorator

def skipCPUIf(condition, reason=""):
    def decorator(fn):
        fn._skip_cpu_if = (condition, reason)
        return fn
    return decorator

def skipCUDAIf(condition, reason=""):
    def decorator(fn):
        fn._skip_cuda_if = (condition, reason)
        return fn
    return decorator


def expectedFailureMeta(fn):
    """No-op decorator — candle has no Meta device."""
    return fn


def largeTensorTest(size, device=None):
    """No-op decorator — candle doesn't skip large tensor tests."""
    def decorator(fn):
        return fn
    return decorator


def precisionOverride(precisions):
    """Decorator to override precision for specific dtypes."""
    def decorator(fn):
        fn._precision_overrides = precisions
        return fn
    return decorator


def dtypesIfCUDA(*dtype_args):
    """Like @dtypes but only for CUDA/NPU devices."""
    def decorator(fn):
        fn._dtypes_cuda = dtype_args
        return fn
    return decorator


def dtypesIfCPU(*dtype_args):
    """Like @dtypes but only for CPU device."""
    def decorator(fn):
        fn._dtypes_cpu = dtype_args
        return fn
    return decorator


def skipMeta(fn):
    """No-op decorator — candle has no Meta backend."""
    return fn


toleranceOverride = namedtuple("toleranceOverride", ["atol", "rtol"],
                               defaults=[None, None])


def tol(dtype, atol=0, rtol=0):
    """Create a tolerance specification for a dtype."""
    return toleranceOverride(atol=atol, rtol=rtol)


def expectedFailureXLA(fn):
    """No-op decorator — candle has no XLA backend."""
    return fn


def skipXLA(fn):
    """No-op decorator — candle has no XLA backend."""
    return fn


PYTORCH_CUDA_MEMCHECK = False


def get_all_device_types():
    """Return list of available device type strings."""
    return _get_available_devices()

# ---------------------------------------------------------------------------
# instantiate_device_type_tests
# ---------------------------------------------------------------------------
def instantiate_device_type_tests(test_class, scope, except_for=None, only_for=None):
    """Generate device-specific test classes from a generic test class.

    For a class TestFoo with test_bar(self, device, dtype), generates:
      - TestFooCPU with test_bar_cpu_float32(self), test_bar_cpu_float64(self), ...
      - TestFooNPU with test_bar_npu_float32(self), ... (if NPU available)
    """
    devices = _get_available_devices()
    if only_for:
        devices = [d for d in devices if d in only_for]
    if except_for:
        devices = [d for d in devices if d not in except_for]

    for device in devices:
        device_suffix = device.upper()
        class_name = f"{test_class.__name__}{device_suffix}"

        # Collect all original test_ method names so we can mask them later
        original_test_names = [
            name for name in dir(test_class)
            if name.startswith("test_") and callable(getattr(test_class, name))
        ]

        # Create a new class inheriting from the test class.
        # Pre-populate with None for all original test_ methods to shadow
        # inherited methods — unittest ignores non-callable attributes.
        class_dict = {"device_type": device}
        for name in original_test_names:
            class_dict[name] = None
        device_class = type(class_name, (test_class,), class_dict)

        # For each test method, generate device+dtype variants
        for attr_name in original_test_names:
            fn = getattr(test_class, attr_name)

            # Check device-only filters
            only_device = getattr(fn, "_only_device", None)
            if only_device:
                # "cuda" matches both cuda and npu
                if only_device == "cuda" and device not in ("cuda", "npu"):
                    continue
                if only_device != "cuda" and only_device != device:
                    continue

            # Check skip conditions
            skip_cpu = getattr(fn, "_skip_cpu_if", None)
            if skip_cpu and device == "cpu" and skip_cpu[0]:
                continue
            skip_cuda = getattr(fn, "_skip_cuda_if", None)
            if skip_cuda and device in ("cuda", "npu") and skip_cuda[0]:
                continue

            # Get dtype list
            dtype_list = getattr(fn, "_dtypes", None)

            if dtype_list:
                # Generate one test per dtype
                for dt in dtype_list:
                    dt_name = str(dt).split(".")[-1]
                    test_name = f"{attr_name}_{device}_{dt_name}"

                    def make_test(f, d, dtype):
                        @functools.wraps(f)
                        def test_fn(self):
                            return f(self, device=d, dtype=dtype)
                        return test_fn

                    setattr(device_class, test_name, make_test(fn, device, dt))
            else:
                # Single test with device arg
                test_name = f"{attr_name}_{device}"

                def make_test_no_dtype(f, d):
                    @functools.wraps(f)
                    def test_fn(self):
                        sig = inspect.signature(f)
                        if "dtype" in sig.parameters:
                            return f(self, device=d, dtype=torch.float32)
                        return f(self, device=d)
                    return test_fn

                setattr(device_class, test_name, make_test_no_dtype(fn, device))

        # Register in caller's scope
        scope[class_name] = device_class

    # Remove original class from scope
    if test_class.__name__ in scope:
        del scope[test_class.__name__]
