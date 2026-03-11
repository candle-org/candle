"""
CANN Auto-Discovery Module

Automatically detects CANN installation paths and versions.
Supports:
- Old CANN (8.3.RC2): directory structure ascend-toolkit/<version>
- New CANN (>=8.5):   directory structure cann-<version>
"""

import os
import re


# Base directory for all Ascend installations
_ASCEND_BASE = "/usr/local/Ascend"

# Minimum supported version (8.3.RC2)
_MIN_VERSION = (8, 3)

# Version boundary: >= 8.5 uses new-style directory layout and library names
_NEW_STYLE_MIN = (8, 5)

_NEW_STYLE_PATTERN = re.compile(r"^cann-(\d+\.\d+\.\d+)$")
_VERSION_DIR_PATTERN = re.compile(r"^(\d+\.\d+(?:\.\d+)?(?:\.?RC\d+)?)$")


def _parse_version(version_str):
    """Parse version string like '8.3.RC2' or '8.5.0' into a comparable tuple."""
    numeric = re.sub(r"[^0-9.]", "", version_str)
    parts = numeric.split(".")
    try:
        return tuple(int(p) for p in parts if p)
    except ValueError:
        return (0,)


def _scan_installations():
    """Scan /usr/local/Ascend for all CANN installations."""
    if not os.path.isdir(_ASCEND_BASE):
        return []

    found = []  # list of (version_tuple, style, root_path)

    for entry in os.listdir(_ASCEND_BASE):
        full_path = os.path.join(_ASCEND_BASE, entry)
        if not os.path.isdir(full_path):
            continue

        # New style: cann-8.5.0
        m = _NEW_STYLE_PATTERN.match(entry)
        if m:
            ver = _parse_version(m.group(1))
            if ver >= _MIN_VERSION:
                found.append((ver, "new", full_path))
            continue

        # Old style: ascend-toolkit/<version>
        if entry == "ascend-toolkit":
            for sub in os.listdir(full_path):
                if sub in ("latest", "set_env.sh"):
                    continue
                sub_path = os.path.join(full_path, sub)
                if not os.path.isdir(sub_path):
                    continue
                vm = _VERSION_DIR_PATTERN.match(sub)
                if vm:
                    ver = _parse_version(vm.group(1))
                    if ver >= _MIN_VERSION:
                        found.append((ver, "old", sub_path))

    # Sort newest first; for same version prefer new style
    found.sort(key=lambda x: (x[0], 0 if x[1] == "new" else 1), reverse=True)
    return found


# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------

def _get_lib_dirs(cann_root, style):
    dirs = []
    for sub in ("lib64", os.path.join("aarch64-linux", "lib64")):
        d = os.path.join(cann_root, sub)
        if os.path.isdir(d):
            dirs.append(d)
    # Old style also has libs under opp/lib64
    if style == "old":
        d = os.path.join(cann_root, "opp", "lib64")
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


def _get_python_dirs(cann_root):
    d = os.path.join(cann_root, "python", "site-packages")
    return [d] if os.path.isdir(d) else []


def _get_opp_dir(cann_root):
    d = os.path.join(cann_root, "opp")
    return d if os.path.isdir(d) else None


# ---------------------------------------------------------------------------
# Cached detection
# ---------------------------------------------------------------------------

_cann_root = None
_cann_style = None   # "old" | "new"
_cann_version = None  # tuple
_detected = False


def _detect():
    global _cann_root, _cann_style, _cann_version, _detected
    if _detected:
        return
    _detected = True
    installs = _scan_installations()
    if installs:
        _cann_version, _cann_style, _cann_root = installs[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cann_root():
    """Return the root path of the best detected CANN installation, or None."""
    _detect()
    return _cann_root


def is_new_style():
    """True if detected CANN uses the new (>=8.5) directory/library layout."""
    _detect()
    return _cann_style == "new"


def get_lib_dirs():
    """Library directories for the detected CANN, plus the driver fallback."""
    _detect()
    if _cann_root is None:
        return []
    dirs = _get_lib_dirs(_cann_root, _cann_style)
    driver = os.path.join(_ASCEND_BASE, "driver", "lib64", "driver")
    if os.path.isdir(driver):
        dirs.append(driver)
    return dirs


def get_python_dirs():
    _detect()
    if _cann_root is None:
        return []
    return _get_python_dirs(_cann_root)


def get_opp_dir():
    _detect()
    if _cann_root is None:
        return None
    return _get_opp_dir(_cann_root)


def is_available():
    return get_cann_root() is not None


def get_aclnn_lib_names():
    """Return (base_libs, preload_libs, libs) tuples appropriate for the CANN version.

    Old CANN (<=8.3): libaclnn_ops_infer.so, libaclnn_math.so
    New CANN (>=8.5): libopapi_nn.so, libopapi_math.so
    """
    _detect()
    base = ("libnnopbase.so",)
    preload = ("libopapi.so",)
    if _cann_style == "new":
        libs = ("libopapi_nn.so", "libopapi_math.so", "libopapi.so")
    else:
        libs = ("libaclnn_ops_infer.so", "libaclnn_math.so", "libopapi.so")
    return base, preload, libs


def get_cann_info():
    """Debugging helper."""
    _detect()
    if _cann_root is None:
        return {"available": False}
    return {
        "available": True,
        "root": _cann_root,
        "style": _cann_style,
        "version": ".".join(str(v) for v in _cann_version) if _cann_version else None,
        "is_new_style": _cann_style == "new",
        "lib_dirs": _get_lib_dirs(_cann_root, _cann_style),
        "python_dirs": _get_python_dirs(_cann_root),
        "opp_dir": _get_opp_dir(_cann_root),
        "aclnn_lib_names": get_aclnn_lib_names(),
    }
