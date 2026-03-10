"""Compatibility conftest plugin for running transformers tests under candle.

This module is imported by a bridge conftest.py that run.py generates inside
the transformers test directory.  All compatibility patches are concentrated
here so that candle source code is never modified.

Patches applied:
  a) Version spoofing   - makes candle look like torch >= 2.5.0
  b) Module mirroring   - registers candle.* as torch.* in sys.modules,
                          makes stub __getattr__ lenient instead of raising
  c) Module stubs       - torch.backends.cuda, torch._dynamo, etc.
  d) Safetensors patch  - pure-Python safetensors loader (no C extension)
  e) torch_npu shim     - fake torch_npu so transformers NPU checks pass
  f) Dep check bypass   - prevents transformers from rejecting dep versions
  g) xfail injection    - marks known failures from xfail.yaml
"""
import importlib
import os
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_COMPAT_DIR = Path(__file__).resolve().parent          # compat/transformers/
_COMPAT_ROOT = _COMPAT_DIR.parent                      # compat/
_PROJECT_ROOT = _COMPAT_ROOT.parent                    # project root
_SRC_DIR = _PROJECT_ROOT / "src"

# ---------------------------------------------------------------------------
# Import shared utilities from conftest_base
# ---------------------------------------------------------------------------
if str(_COMPAT_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMPAT_ROOT))

from conftest_base import (  # noqa: E402
    apply_version_spoof,
    install_torch_finder,
    apply_safetensors_patch,
    apply_torch_npu_shim,
    load_xfail_config,
    match_xfail,
)


# ---------------------------------------------------------------------------
# c)  Module stubs  (transformers-specific)
# ---------------------------------------------------------------------------

def _apply_module_stubs():
    """Create stub modules for torch submodules that candle doesn't provide."""
    import candle as torch  # noqa: F811

    # --- torch.backends.cuda ---
    if not hasattr(torch, "backends") or not hasattr(torch.backends, "cuda"):
        backends = getattr(torch, "backends", None)
        if backends is None:
            backends = types.ModuleType("torch.backends")
            torch.backends = backends
            sys.modules["torch.backends"] = backends

        cuda_backend = types.ModuleType("torch.backends.cuda")
        cuda_backend.is_flash_attn_available = lambda: False
        cuda_backend.flash_sdp_enabled = lambda: False
        cuda_backend.math_sdp_enabled = lambda: True
        cuda_backend.mem_efficient_sdp_enabled = lambda: False
        cuda_backend.enable_flash_sdp = lambda enabled: None
        cuda_backend.enable_math_sdp = lambda enabled: None
        cuda_backend.enable_mem_efficient_sdp = lambda enabled: None
        torch.backends.cuda = cuda_backend
        sys.modules["torch.backends.cuda"] = cuda_backend

    # --- torch.backends.cudnn ---
    if not hasattr(torch.backends, "cudnn"):
        cudnn_backend = types.ModuleType("torch.backends.cudnn")
        cudnn_backend.enabled = False
        cudnn_backend.deterministic = False
        cudnn_backend.benchmark = False
        cudnn_backend.allow_tf32 = False
        cudnn_backend.is_available = lambda: False
        cudnn_backend.version = lambda: None
        torch.backends.cudnn = cudnn_backend
        sys.modules["torch.backends.cudnn"] = cudnn_backend

    # --- torch.version ---
    version_mod = getattr(torch, "version", None)
    if version_mod is None:
        version_mod = types.ModuleType("torch.version")
        torch.version = version_mod
        sys.modules["torch.version"] = version_mod
    if not hasattr(version_mod, "cuda"):
        version_mod.cuda = None
    if not hasattr(version_mod, "hip"):
        version_mod.hip = None

    # --- torch._dynamo ---
    if not hasattr(torch, "_dynamo"):
        dynamo = types.ModuleType("torch._dynamo")
        dynamo.is_compiling = lambda: False

        def _noop_decorator(fn=None, **kwargs):
            if fn is not None:
                return fn
            return lambda f: f

        dynamo.optimize = _noop_decorator
        dynamo.disable = _noop_decorator
        dynamo.reset = lambda: None
        torch._dynamo = dynamo
        sys.modules["torch._dynamo"] = dynamo

    # --- torch.compiler ---
    if not hasattr(torch, "compiler"):
        compiler = types.ModuleType("torch.compiler")
        compiler.is_compiling = lambda: False
        compiler.disable = lambda fn=None, **kw: fn if fn else (lambda f: f)
        torch.compiler = compiler
        sys.modules["torch.compiler"] = compiler

    # --- torch.hub ---
    hub_mod = sys.modules.get("torch.hub")
    if hub_mod is None:
        try:
            hub_mod = importlib.import_module("candle.hub")
        except ImportError:
            hub_mod = types.ModuleType("torch.hub")
        sys.modules["torch.hub"] = hub_mod
    _torch_home = os.path.expanduser(
        os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
    )
    hub_mod._get_torch_home = lambda: _torch_home
    hub_mod.get_dir = lambda: os.path.join(_torch_home, "hub")
    if not hasattr(hub_mod, "load_state_dict_from_url"):
        hub_mod.load_state_dict_from_url = lambda *a, **kw: {}

    # --- torch.library ---
    # torchvision's _meta_registrations calls Library("torchvision", "IMPL", "Meta")
    library_mod = sys.modules.get("torch.library")
    if library_mod is None:
        library_mod = types.ModuleType("torch.library")
        sys.modules["torch.library"] = library_mod

    class _LibraryStub:
        def __init__(self, *args, **kwargs):
            pass
        def impl(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    if not hasattr(library_mod, "Library"):
        library_mod.Library = _LibraryStub
    if not hasattr(library_mod, "impl"):
        library_mod.impl = lambda *a, **kw: (lambda fn: fn)

    # --- torch.cuda stubs ---
    if not hasattr(torch, "cuda"):
        cuda_mod = types.ModuleType("torch.cuda")
        cuda_mod.is_available = lambda: False
        cuda_mod.device_count = lambda: 0
        torch.cuda = cuda_mod
        sys.modules["torch.cuda"] = cuda_mod

    # --- SDPA availability flag ---
    torch._sdpa_available = False
    if hasattr(torch.nn, "functional") and hasattr(
        torch.nn.functional, "scaled_dot_product_attention"
    ):
        _orig_sdpa = torch.nn.functional.scaled_dot_product_attention

        def _sdpa_wrapper(*args, **kwargs):
            return _orig_sdpa(*args, **kwargs)

        _sdpa_wrapper._is_optimized = False
        torch.nn.functional.scaled_dot_product_attention = _sdpa_wrapper

    # --- torch.utils._pytree ---
    try:
        import candle.utils  # noqa: F401
    except ImportError:
        pass
    utils_mod = getattr(torch, "utils", None)
    if utils_mod is None:
        utils_mod = types.ModuleType("torch.utils")
        torch.utils = utils_mod
        sys.modules["torch.utils"] = utils_mod
    if not hasattr(utils_mod, "_pytree"):
        pytree = types.ModuleType("torch.utils._pytree")
        pytree.Context = type("Context", (), {})

        def _register_pytree_node(cls, flatten_fn, unflatten_fn, **kwargs):
            pass  # no-op stub

        pytree.register_pytree_node = _register_pytree_node
        pytree._register_pytree_node = _register_pytree_node
        utils_mod._pytree = pytree
        sys.modules["torch.utils._pytree"] = pytree


# ---------------------------------------------------------------------------
# f)  Transformers dependency version check bypass
# ---------------------------------------------------------------------------

def _bypass_transformers_dep_check():
    """Prevent transformers from rejecting mismatched dependency versions.

    transformers.__init__ does ``from . import dependency_versions_check``
    which hard-fails if installed packages don't match its pinned ranges.
    We inject a dummy module so the check never runs.
    """
    key = "transformers.dependency_versions_check"
    if key not in sys.modules:
        mod = types.ModuleType(key)
        mod.dep_version_check = lambda pkg, hint=None: None
        sys.modules[key] = mod


# ---------------------------------------------------------------------------
# Top-level application -- called once when conftest is loaded
# ---------------------------------------------------------------------------

def apply_all_patches():
    """Apply all compatibility patches.  Idempotent."""
    # Ensure candle is importable and aliased as torch
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))

    import candle  # noqa: F401
    sys.modules.setdefault("torch", candle)

    apply_version_spoof()
    _bypass_transformers_dep_check()  # must run before any transformers import
    install_torch_finder()            # meta path finder for torch.* -> candle.*
    _apply_module_stubs()
    apply_safetensors_patch()
    apply_torch_npu_shim()


# ---------------------------------------------------------------------------
# pytest hooks  (used when this file is loaded via conftest bridge)
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """Mark known failures as xfail from xfail.yaml."""
    import pytest  # local import to avoid import-time issues

    xfail_cfg = load_xfail_config(_COMPAT_DIR / "xfail.yaml")
    if not xfail_cfg:
        return

    global_patterns = xfail_cfg.get("_global", [])

    for item in items:
        nodeid = item.nodeid

        # Check global patterns
        reason = match_xfail(nodeid, global_patterns)
        if reason:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
            continue

        # Check per-model patterns
        for model_name, entries in xfail_cfg.items():
            if model_name.startswith("_"):
                continue
            if not entries:
                continue
            if f"test_modeling_{model_name}" in nodeid:
                reason = match_xfail(nodeid, entries)
                if reason:
                    item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
                    break
