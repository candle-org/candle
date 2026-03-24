import multiprocessing
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# macOS defaults to "spawn" which cannot pickle locally-defined classes used in
# multi-process DataLoader tests.  Switch to "fork" so that worker processes
# inherit the parent address space (PyTorch's own test suite does the same).
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

_FORCE_CPU_ONLY_ENV = "CANDLE_TEST_FORCE_CPU_ONLY"


def _npu_available() -> bool:
    # Optional override for CI/local debugging of CPU-only behavior.
    if os.environ.get(_FORCE_CPU_ONLY_ENV) == "1":
        return False

    try:
        import candle as torch

        return bool(torch.npu.is_available())
    except Exception:
        return False


def _npu_device_count() -> int:
    # Optional override for CI/local debugging of CPU-only behavior.
    if os.environ.get(_FORCE_CPU_ONLY_ENV) == "1":
        return 0

    try:
        import candle as torch

        return int(torch.npu.device_count())
    except Exception:
        return 0


def _mps_available() -> bool:
    if os.environ.get(_FORCE_CPU_ONLY_ENV) == "1":
        return False
    try:
        import candle as torch

        return bool(torch.mps.is_available())
    except Exception:
        return False


_NPU_DIRS = (os.sep + "npu" + os.sep, os.sep + "distributed" + os.sep)
_MPS_DIR = os.sep + "mps" + os.sep


# Filename patterns for distributed tests that run on CPU (no NPU needed).
# Add a pattern here for any CPU-only distributed test file whose name does
# not already contain "gloo".
_CPU_DISTRIBUTED_PATTERNS = (
    "gloo",                    # explicit Gloo-backend tests
    "work_async",              # Work/Future isolation (no backend)
    "ddp_async_overlap",       # DDP overlap contracts via single-rank Gloo
    "fsdp_public_api",         # public FSDP namespace tests (single-process)
    "distributed_mvp_baseline",  # baseline integration tests (uses Gloo)
    "ddp_bucket_fastpath",     # DDP bucket sizing (single-rank Gloo)
    "ddp_bucket_view",         # DDP gradient_as_bucket_view (single-rank Gloo)
    "ddp_static_graph",        # DDP static_graph (single-rank Gloo)
    "ddp_unused_params",       # DDP find_unused_parameters (single-rank Gloo)
    "ddp_cpu",                 # DDP on CPU (Gloo)
    "ddp_comm_hook",           # DDP communication hooks (single-rank Gloo)
    "ddp_fastpath_import",     # DDP import regression (no dist needed)
    "fsdp_shard_fastpath",     # FSDP shard Cython unit tests (no dist)
    "pipeline_parallel",       # pipeline parallel stubs (no dist)
    "tensor_parallel",         # tensor parallel monkeypatched (single-rank)
    "dtensor_redistribute",    # DTensor monkeypatched (single-rank)
    "dtensor_fastpath",        # DTensor Cython unit tests (no dist)
    "distributed_checkpoint",  # checkpoint save/load (no dist)
    "p2p_batch_work",          # P2P batch Work stubs (no dist)
    "c10d_cython",             # Cython import tests (no dist)
    "cython_work_direct",      # Cython Work direct tests (no dist)
    "hccl_work_event_completion",  # HCCL event mocks (no real hardware)
)


def _is_gloo_test(item: pytest.Item) -> bool:
    """Test file uses Gloo or CPU-only isolation (does NOT require NPU hardware)."""
    name = os.path.basename(str(item.fspath)).lower()
    return any(pat in name for pat in _CPU_DISTRIBUTED_PATTERNS)


def _in_npu_dir(item: pytest.Item) -> bool:
    """Test lives under tests/npu/ or tests/distributed/ (requires NPU).

    Gloo-based tests in tests/distributed/ are excluded because they run
    on CPU without any accelerator hardware.
    """
    fspath = str(item.fspath)
    if not any(d in fspath for d in _NPU_DIRS):
        return False
    # Gloo tests run on CPU -- do not skip them
    if _is_gloo_test(item):
        return False
    return True


def _in_mps_dir(item: pytest.Item) -> bool:
    """Test lives under tests/mps/ (requires Apple MPS)."""
    return _MPS_DIR in str(item.fspath)


def _requires_multicard(item: pytest.Item) -> bool:
    nodeid = item.nodeid.lower()
    return any(token in nodeid for token in ("2card", "multicard"))


def _requires_hccl_hardware(item: pytest.Item) -> bool:
    """Test lives in tests/distributed/ and is NOT CPU-safe.

    On single-card NPU, these tests either require multi-rank HCCL or run
    expensive probes that may hang.  Skip them when device_count < 2.
    """
    fspath = str(item.fspath)
    if os.sep + "distributed" + os.sep not in fspath:
        return False
    return not _is_gloo_test(item)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _npu_available():
        npu_count = _npu_device_count()
        if npu_count >= 2:
            return

        skip_reason = f"Requires >=2 NPUs, found {npu_count}"
        skip_marker = pytest.mark.skip(reason=skip_reason)
        for item in items:
            if _requires_multicard(item) or _requires_hccl_hardware(item):
                item.add_marker(skip_marker)
        return

    skip_npu = pytest.mark.skip(reason="NPU-only test skipped in CPU-only environment")
    for item in items:
        if _in_npu_dir(item):
            item.add_marker(skip_npu)

    if not _mps_available():
        skip_mps = pytest.mark.skip(reason="MPS-only test skipped (no Apple GPU)")
        for item in items:
            if _in_mps_dir(item):
                item.add_marker(skip_mps)
