"""candle.testing._internal — PyTorch-compatible test infrastructure."""

from .common_dtype import (
    all_types, all_types_and_complex, floating_types,
    floating_types_and_half, floating_and_complex_types,
    integral_types, get_all_dtypes, get_all_int_dtypes,
    get_all_fp_dtypes, get_all_complex_dtypes,
)

from .common_utils import (
    TestCase, run_tests, make_tensor,
    IS_WINDOWS, IS_MACOS, IS_LINUX,
    TEST_CUDA, TEST_MPS, TEST_NPU, TEST_MULTIGPU,
    skipIfNoCuda, skipIfNoMPS, slowTest,
    freeze_rng_state, parametrize, subtest,
    TEST_WITH_ROCM, TEST_WITH_ASAN, TEST_WITH_TSAN,
)

from .common_device_type import (
    instantiate_device_type_tests, dtypes, onlyCPU, onlyCUDA,
    onlyNativeDeviceTypes, deviceCountAtLeast,
    skipCPUIf, skipCUDAIf,
)
