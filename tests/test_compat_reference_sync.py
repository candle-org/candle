import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNC_PATH = PROJECT_ROOT / "compat" / "reference" / "sync.py"


def load_sync_module():
    spec = importlib.util.spec_from_file_location("compat_reference_sync", SYNC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_manifest(tmp_path):
    manifest = {
        "sources": {
            "pytorch": {
                "path": "compat/_pytorch",
                "revision": "v2.5.0",
                "mirrors": ["https://github.com/pytorch/pytorch.git"],
                "roles": ["semantic-baseline", "test-baseline"],
            },
            "torch_npu": {
                "path": "compat/_torch_npu",
                "revision": "v2.5.1-7.0.0",
                "mirrors": ["https://github.com/Ascend/pytorch.git"],
                "roles": ["npu-semantic-baseline", "implementation-reference"],
            },
        },
        "policies": {
            "detached_checkout": True,
            "allow_local_dirty": False,
            "offline_reuse": True,
            "reports_dir": "compat/reference/reports",
        },
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path



def test_load_manifest_reads_two_sources(tmp_path):
    mod = load_sync_module()
    manifest_path = write_manifest(tmp_path)
    manifest = mod.load_manifest(manifest_path)
    assert list(manifest["sources"].keys()) == ["pytorch", "torch_npu"]
    assert manifest["sources"]["pytorch"]["path"] == "compat/_pytorch"
    assert manifest["sources"]["torch_npu"]["revision"] == "v2.5.1-7.0.0"



def test_checkout_path_is_project_root_relative():
    mod = load_sync_module()
    assert mod.checkout_path(PROJECT_ROOT, "compat/_torch_npu") == PROJECT_ROOT / "compat" / "_torch_npu"



def test_offline_mode_reuses_existing_checkout(tmp_path, monkeypatch):
    mod = load_sync_module()
    checkout = tmp_path / "compat" / "_pytorch"
    (checkout / ".git").mkdir(parents=True)

    calls = []

    def fail_run(*args, **kwargs):
        calls.append(args[0])
        raise AssertionError("offline reuse should not call git")

    monkeypatch.setattr(subprocess, "run", fail_run)

    source = {
        "path": "compat/_pytorch",
        "revision": "v2.5.0",
        "mirrors": ["https://github.com/pytorch/pytorch.git"],
        "roles": ["semantic-baseline"],
    }
    result = mod.ensure_checkout("pytorch", source, project_root=tmp_path, policy={"offline_reuse": True}, offline=True)
    assert result["status"] == "reused-offline"
    assert result["path"] == str(checkout)
    assert calls == []



def test_online_clone_uses_first_successful_mirror(tmp_path, monkeypatch):
    mod = load_sync_module()
    calls = []

    def fake_run(cmd, cwd=None, check=False, capture_output=False, text=False):
        calls.append((tuple(cmd), cwd))
        joined = " ".join(cmd)
        if "https://bad.example/pytorch.git" in joined:
            raise subprocess.CalledProcessError(128, cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    source = {
        "path": "compat/_pytorch",
        "revision": "v2.5.0",
        "mirrors": [
            "https://bad.example/pytorch.git",
            "https://github.com/pytorch/pytorch.git",
        ],
        "roles": ["semantic-baseline"],
    }
    result = mod.ensure_checkout("pytorch", source, project_root=tmp_path, policy={"detached_checkout": True}, offline=False)

    assert result["status"] == "cloned"
    assert result["mirror"] == "https://github.com/pytorch/pytorch.git"
    assert any("https://github.com/pytorch/pytorch.git" in " ".join(cmd) for cmd, _ in calls)
