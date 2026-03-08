import pytest

from candle._dispatch.registry import registry


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot and restore the global op registry around every contract test."""
    state = registry.snapshot()
    yield
    registry.restore(state)
