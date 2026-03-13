"""NPU test conftest — auto-skip SoC-specific subdirectories on wrong hardware."""
import os
import pytest


_SOC_DIRS = ("910a", "910b", "310b", "310p")


def _current_soc_profile():
    """Return the SoC profile string for the current NPU, or None."""
    try:
        from candle._backends.npu import runtime as npu_runtime
        return npu_runtime.soc_profile()
    except Exception:
        return None


def pytest_collection_modifyitems(config, items):
    profile = _current_soc_profile()
    for item in items:
        fspath = str(item.fspath)
        for soc in _SOC_DIRS:
            soc_dir = os.sep + soc + os.sep
            if soc_dir in fspath and profile != soc:
                reason = (
                    f"Skipped: test requires {soc} hardware, "
                    f"current SoC is {profile or 'unknown'}"
                )
                item.add_marker(pytest.mark.skip(reason=reason))
                break
