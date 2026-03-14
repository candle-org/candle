import ctypes
import os
import threading

_HCCL_LIB = None
_HCCL_LOCK = threading.Lock()

# Legacy fallback paths (for environments without cann_discovery)
_FALLBACK_DIRS = (
    "/usr/local/Ascend/ascend-toolkit/latest/lib64",
    "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64",
    "/usr/local/Ascend/latest/lib64",
    "/usr/local/Ascend/latest/aarch64-linux/lib64",
)


def _candidate_dirs():
    """Build search dirs: prefer cann_discovery, fall back to hardcoded paths."""
    try:
        from candle._backends.npu.cann_discovery import get_lib_dirs
        dirs = get_lib_dirs()
        if dirs:
            return dirs
    except Exception:
        pass
    return list(_FALLBACK_DIRS)


def ensure_hccl():
    global _HCCL_LIB
    if _HCCL_LIB is not None:
        return _HCCL_LIB
    with _HCCL_LOCK:
        if _HCCL_LIB is not None:
            return _HCCL_LIB
        dirs = _candidate_dirs()
        for d in dirs:
            path = os.path.join(d, "libhccl.so")
            if os.path.exists(path):
                _HCCL_LIB = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                return _HCCL_LIB
        raise RuntimeError(
            "libhccl.so not found. Searched: " + ", ".join(dirs)
        )
