"""Stub for torch.testing._internal.common_cuda — CUDA-specific test helpers."""
import contextlib

from .common_utils import TEST_CUDA, TEST_MULTIGPU

TEST_CUDA = TEST_CUDA
TEST_MULTIGPU = TEST_MULTIGPU
TEST_CUDNN = False

_get_torch_cuda_version = lambda: (0, 0)

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


tf32_is_not_fp32 = lambda: False


def _create_scaling_case(*args, **kwargs):
    """Stub for GradScaler test helper."""
    return [], [], []


def _create_scaling_models_optimizers(*args, **kwargs):
    """Stub for GradScaler test helper."""
    return [], []
