import pytest

from tests.npu import conftest as npu_conftest


class _FakeCannDiscovery:
    def __init__(self, version=None, error=None):
        self._version = version
        self._error = error

    def get_cann_version(self):
        if self._error is not None:
            raise self._error
        return self._version


class _FakeConfig:
    def __init__(self):
        self.lines = []

    def addinivalue_line(self, name, value):
        self.lines.append((name, value))


class _FakeMarker:
    def __init__(self, reason):
        self.kwargs = {"reason": reason}


class _FakeItem:
    def __init__(self, fspath, markers=None):
        self.fspath = fspath
        self._markers = list(markers or [])
        self.added_markers = []
        self.nodeid = fspath

    def add_marker(self, marker):
        self.added_markers.append(_FakeMarker(marker.kwargs["reason"]))

    def get_closest_marker(self, name):
        return name if name in self._markers else None


def _skip_reasons(item):
    return [marker.kwargs["reason"] for marker in item.added_markers]


def test_aclgraph_supported_returns_false_for_old_cann(monkeypatch):
    from candle._backends.npu import runtime as npu_runtime

    monkeypatch.setattr(
        npu_runtime,
        "cann_discovery",
        _FakeCannDiscovery(version=(8, 3, 2)),
    )

    assert npu_conftest._aclgraph_supported() is False


def test_aclgraph_supported_returns_false_for_unsupported_soc(monkeypatch):
    from candle._backends.npu import runtime as npu_runtime

    monkeypatch.setattr(
        npu_runtime,
        "cann_discovery",
        _FakeCannDiscovery(version=(8, 5, 0)),
    )
    monkeypatch.setattr("candle._backends.npu.ops_soc.aclgraph_supported", lambda profile=None: False)

    assert npu_conftest._aclgraph_supported() is False


def test_aclgraph_supported_returns_true_for_supported_soc_on_cann_8_5(monkeypatch):
    from candle._backends.npu import runtime as npu_runtime

    monkeypatch.setattr(
        npu_runtime,
        "cann_discovery",
        _FakeCannDiscovery(version=(8, 5, 0)),
    )
    monkeypatch.setattr("candle._backends.npu.ops_soc.aclgraph_supported", lambda profile=None: True)

    assert npu_conftest._aclgraph_supported() is True


def test_aclgraph_supported_returns_false_for_unknown_version(monkeypatch):
    from candle._backends.npu import runtime as npu_runtime

    monkeypatch.setattr(
        npu_runtime,
        "cann_discovery",
        _FakeCannDiscovery(version=None),
    )

    assert npu_conftest._aclgraph_supported() is False


def test_aclgraph_supported_propagates_unexpected_version_errors(monkeypatch):
    from candle._backends.npu import runtime as npu_runtime

    monkeypatch.setattr(
        npu_runtime,
        "cann_discovery",
        _FakeCannDiscovery(error=RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        npu_conftest._aclgraph_supported()


def test_pytest_configure_registers_requires_aclgraph_marker():
    config = _FakeConfig()

    npu_conftest.pytest_configure(config)

    assert config.lines == [
        (
            "markers",
            "requires_aclgraph: test requires live aclgraph support (CANN >= 8.5 on a supported SoC)",
        )
    ]


def test_collection_hook_preserves_soc_directory_skip_behavior(monkeypatch):
    item = _FakeItem("/tmp/tests/npu/910a/test_example.py")

    monkeypatch.setattr(npu_conftest, "_current_soc_profile", lambda: "910b")
    monkeypatch.setattr(npu_conftest, "_aclgraph_supported", lambda: True)

    npu_conftest.pytest_collection_modifyitems(_FakeConfig(), [item])

    assert _skip_reasons(item) == [
        "Skipped: test requires 910a hardware, current SoC is 910b"
    ]


def test_collection_hook_skips_only_marked_items_when_aclgraph_unsupported(monkeypatch):
    marked = _FakeItem(
        "/tmp/tests/npu/test_aclgraph.py::test_marked",
        markers=["requires_aclgraph"],
    )
    unmarked = _FakeItem("/tmp/tests/npu/test_other.py::test_unmarked")

    monkeypatch.setattr(npu_conftest, "_current_soc_profile", lambda: "910b")
    monkeypatch.setattr(npu_conftest, "_aclgraph_supported", lambda: False)

    npu_conftest.pytest_collection_modifyitems(_FakeConfig(), [marked, unmarked])

    assert _skip_reasons(marked) == ["aclgraph requires CANN >= 8.5 and SoC support"]
    assert _skip_reasons(unmarked) == []


def test_collection_hook_does_not_skip_marked_items_when_aclgraph_supported(monkeypatch):
    marked = _FakeItem(
        "/tmp/tests/npu/test_aclgraph.py::test_marked",
        markers=["requires_aclgraph"],
    )

    monkeypatch.setattr(npu_conftest, "_current_soc_profile", lambda: "910b")
    monkeypatch.setattr(npu_conftest, "_aclgraph_supported", lambda: True)

    npu_conftest.pytest_collection_modifyitems(_FakeConfig(), [marked])

    assert _skip_reasons(marked) == []


def test_collection_hook_soc_skip_wins_over_aclgraph_skip(monkeypatch):
    item = _FakeItem(
        "/tmp/tests/npu/910a/test_aclgraph.py::test_marked",
        markers=["requires_aclgraph"],
    )

    monkeypatch.setattr(npu_conftest, "_current_soc_profile", lambda: "910b")
    monkeypatch.setattr(npu_conftest, "_aclgraph_supported", lambda: False)

    npu_conftest.pytest_collection_modifyitems(_FakeConfig(), [item])

    assert _skip_reasons(item) == [
        "Skipped: test requires 910a hardware, current SoC is 910b"
    ]
