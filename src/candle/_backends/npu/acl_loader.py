import ctypes
import glob
import os
import sys
import threading

from . import cann_discovery


_ACL_READY = False
_ACL_MODULE = None
_ACL_LOCK = threading.Lock()

_PRELOAD_LIBS = (
    "libascend_protobuf.so",
    "libascendcl.so",
)


def _existing_dirs():
    return cann_discovery.get_lib_dirs()


def _append_python_path(paths):
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def _align_ascend_env():
    # Set environment variables to point to the detected CANN installation
    cann_root = cann_discovery.get_cann_root()
    if cann_root is not None:
        os.environ["ASCEND_TOOLKIT_HOME"] = cann_root
        os.environ["ASCEND_HOME_PATH"] = cann_root
        opp_dir = cann_discovery.get_opp_dir()
        if opp_dir is not None:
            os.environ["ASCEND_OPP_PATH"] = opp_dir


def _sanitize_ld_library_path(value):
    if not value:
        return ""
    keep = []
    for item in value.split(":"):
        if not item:
            continue
        # Remove toolkit stub paths so real GE/runtime libs are always preferred.
        if "/runtime/lib64/stub" in item:
            continue
        keep.append(item)
    return ":".join(keep)


def _prepend_ld_library_path(paths):
    if not paths:
        return
    existing = _sanitize_ld_library_path(os.environ.get("LD_LIBRARY_PATH", ""))
    prefix = ":".join(paths)
    os.environ["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix


def _preload_libs(paths):
    # Ensure protobuf/provider libs are globally visible before loading ACLNN deps.
    for base in paths:
        candidate = os.path.join(base, "libascend_protobuf.so")
        if os.path.exists(candidate):
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            break
        for match in sorted(glob.glob(candidate + "*")):
            if not os.path.isfile(match):
                continue
            ctypes.CDLL(match, mode=ctypes.RTLD_GLOBAL)
            break

    for base in paths:
        candidate = os.path.join(base, "libplatform.so")
        if os.path.exists(candidate):
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            break

    for base in paths:
        for lib in _PRELOAD_LIBS:
            candidate = os.path.join(base, lib)
            if os.path.exists(candidate):
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                continue
            if lib == "libascend_protobuf.so":
                for match in sorted(glob.glob(candidate + "*")):
                    if not os.path.isfile(match):
                        continue
                    ctypes.CDLL(match, mode=ctypes.RTLD_GLOBAL)
                    break


def _import_acl():
    import acl  # pylint: disable=import-error

    return acl


def ensure_acl():
    global _ACL_READY, _ACL_MODULE
    if _ACL_READY:
        return _ACL_MODULE
    with _ACL_LOCK:
        if _ACL_READY:
            return _ACL_MODULE
        _align_ascend_env()
        paths = _existing_dirs()
        _prepend_ld_library_path(paths)
        _preload_libs(paths)
        try:
            _ACL_MODULE = _import_acl()
        except ModuleNotFoundError:
            _append_python_path(cann_discovery.get_python_dirs())
            _ACL_MODULE = _import_acl()
        _ACL_READY = True
        return _ACL_MODULE
