import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = PROJECT_ROOT / "compat" / "pytorch" / "run.py"


def load_run_module():
    spec = importlib.util.spec_from_file_location("compat_pytorch_run", RUN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_uses_root_level_shared_clone():
    mod = load_run_module()
    assert mod.PYTORCH_DIR == mod.PROJECT_ROOT / "compat" / "_pytorch"


def test_setup_pytorch_delegates_to_reference_sync(monkeypatch):
    mod = load_run_module()
    calls = []

    monkeypatch.setattr(
        mod,
        "load_manifest",
        lambda path: {"sources": {"pytorch": {"revision": "v2.5.0"}}},
    )

    def fake_ensure(manifest_path, source_name=None, offline=False):
        calls.append((manifest_path, source_name, offline))
        return {
            "pytorch": {
                "status": "reused-offline",
                "path": str(mod.PYTORCH_DIR),
                "revision": "v2.5.0",
            }
        }

    monkeypatch.setattr(mod, "ensure_reference_sources", fake_ensure)

    mod.setup_pytorch("v2.5.0", offline=True)
    assert calls == [(mod.MANIFEST_PATH, "pytorch", True)]


def test_write_bridge_conftest_targets_shared_clone(tmp_path, monkeypatch):
    mod = load_run_module()
    monkeypatch.setattr(mod, "PYTORCH_DIR", tmp_path / "_pytorch")
    (mod.PYTORCH_DIR / "test").mkdir(parents=True)
    bridge = mod.write_bridge_conftest()
    assert bridge == mod.PYTORCH_DIR / "test" / "conftest.py"
    assert bridge.exists()
