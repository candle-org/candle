"""Stub for torch.testing._internal.common_cuda — CUDA-specific test helpers."""
import contextlib

from .common_utils import TEST_CUDA as _TEST_CUDA
from .common_utils import TEST_MULTIGPU as _TEST_MULTIGPU

TEST_CUDA = _TEST_CUDA
TEST_MULTIGPU = _TEST_MULTIGPU
TEST_CUDNN = False


def _get_torch_cuda_version():
    return (0, 0)


SM53OrLater = False
SM60OrLater = False
SM70OrLater = False
SM75OrLater = False
SM80OrLater = False
SM86OrLater = False
SM90OrLater = False

PLATFORM_SUPPORTS_FLASH_ATTENTION = False
PLATFORM_SUPPORTS_MEM_EFF_ATTENTION = False
PLATFORM_SUPPORTS_FUSED_ATTENTION = False
PLATFORM_SUPPORTS_CUDNN_ATTENTION = False
PLATFORM_SUPPORTS_BF16 = False
PLATFORM_SUPPORTS_FP8 = False


@contextlib.contextmanager
def tf32_on_and_off(tf32_val=None):
    """No-op context manager — candle has no TF32 toggle."""
    yield


def tf32_is_not_fp32():
    """Stub — candle has no TF32."""
    return False


def _create_scaling_case(*args, **kwargs):
    """Stub for GradScaler test helper."""
    return [], [], []


def _create_scaling_models_optimizers(*args, **kwargs):
    """Stub for GradScaler test helper."""
    return [], []
