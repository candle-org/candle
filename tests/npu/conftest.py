"""NPU test conftest — auto-skip SoC-specific subdirectories on wrong hardware."""
import os
import pytest


@pytest.fixture
def npu_device():
    import candle as torch
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    return torch.device("npu:0")


_SOC_DIRS = ("910a", "910b", "310b", "310p")


def _current_soc_profile():
    """Return the SoC profile string for the current NPU, or None."""
    try:
        from candle._backends.npu import runtime as npu_runtime
        return npu_runtime.soc_profile()
    except Exception:
        return None


def _aclgraph_supported():
    from candle._backends.npu import runtime as npu_runtime

    version = npu_runtime.cann_discovery.get_cann_version() or (0,)
    return tuple(version) >= (8, 5)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_aclgraph: test requires live aclgraph support (CANN >= 8.5)",
    )


def pytest_collection_modifyitems(config, items):
    profile = _current_soc_profile()
    aclgraph_supported = _aclgraph_supported()
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
        else:
            if not aclgraph_supported and item.get_closest_marker("requires_aclgraph"):
                item.add_marker(
                    pytest.mark.skip(reason="aclgraph requires CANN >= 8.5")
                )
