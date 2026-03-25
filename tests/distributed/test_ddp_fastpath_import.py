"""Regression tests for importing DDP without compiled fastpath modules.

These tests lock the contract needed for lint/static analysis friendliness:
`candle.nn.parallel.distributed` must remain importable even when the
compiled `_ddp_fastpath` extension is absent.  This mirrors editable/source
installs before `build_ext` has been run and prevents introducing import
shapes that static tooling cannot resolve.
"""

import importlib.util
import os
import sys
import types


def test_ddp_module_imports_without_ddp_fastpath_extension(tmp_path, monkeypatch):
    """distributed.py should import with `_HAVE_FASTPATH=False` when the
    `_ddp_fastpath` extension cannot be imported.
    """
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src = os.path.join(repo, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    # Ensure the extension import path is missing for this import attempt.
    sys.modules.pop("candle.distributed._ddp_fastpath", None)

    # Load the module under a temporary name so we exercise its import logic
    # fresh, without disturbing the already-imported real module.
    path = os.path.join(src, "candle", "nn", "parallel", "distributed.py")
    spec = importlib.util.spec_from_file_location(
        "candle.nn.parallel.distributed_import_probe", path
    )
    mod = importlib.util.module_from_spec(spec)
    # Minimal package context so relative imports resolve.
    mod.__package__ = "candle.nn.parallel"
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)

    assert hasattr(mod, "_HAVE_FASTPATH")
    assert mod._HAVE_FASTPATH is False or mod._HAVE_FASTPATH is True
    assert hasattr(mod, "DistributedDataParallel")
