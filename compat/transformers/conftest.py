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
import fnmatch
import importlib
import importlib.metadata
import importlib.util
import json
import mmap
import os
import sys
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_COMPAT_DIR = Path(__file__).resolve().parent          # compat/transformers/
_PROJECT_ROOT = _COMPAT_DIR.parent.parent              # project root
_SRC_DIR = _PROJECT_ROOT / "src"

# ---------------------------------------------------------------------------
# a)  Version spoofing
# ---------------------------------------------------------------------------
_SPOOFED_TORCH_VERSION = "2.5.0"


def _apply_version_spoof():
    """Make candle appear as torch 2.5.0 to satisfy transformers version gates."""
    import candle  # noqa: F811

    candle.__version__ = _SPOOFED_TORCH_VERSION

    # Patch importlib.metadata so `importlib.metadata.version("torch")` works
    _original_version = importlib.metadata.version

    def _patched_version(name):
        if name == "torch":
            return _SPOOFED_TORCH_VERSION
        return _original_version(name)

    importlib.metadata.version = _patched_version


# ---------------------------------------------------------------------------
# b)  Module mirroring — candle.* ↔ torch.* via meta path finder
# ---------------------------------------------------------------------------

class _CandleTorchFinder:
    """Meta path finder that resolves ``import torch.*`` to candle modules.

    For any ``torch.X.Y`` import:
      1. Try ``import candle.X.Y`` — if it exists, mirror it.
      2. Otherwise create a lenient stub module on-the-fly so that
         ``from torch.X.Y import Z`` never raises ``ImportError``.

    This eliminates whack-a-mole patching of individual submodules.
    """

    def find_module(self, fullname, path=None):
        if fullname == "torch" or fullname.startswith("torch."):
            # Only handle if not already in sys.modules
            if fullname not in sys.modules:
                return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        # torch.X.Y → candle.X.Y
        candle_name = "candle" + fullname[len("torch"):]

        # Try importing the real candle module
        try:
            real_mod = importlib.import_module(candle_name)
            # Make raising __getattr__ lenient so from-imports work
            self._make_lenient(real_mod)
            sys.modules[fullname] = real_mod
            return real_mod
        except (ImportError, AttributeError):
            pass

        # Create a lenient stub module
        stub = types.ModuleType(fullname)
        stub.__path__ = []  # mark as package so sub-imports work
        stub.__loader__ = self
        stub.__package__ = fullname

        def _lenient_getattr(name):
            """Return a callable no-op for any missing attribute.

            The returned object is callable (returns None), iterable (empty),
            and falsy — so it works for most guard patterns like
            ``if torch.X.some_func(): ...`` or ``for x in torch.X.items: ...``
            """
            class _Stub:
                def __init__(self, *a, **kw):
                    pass
                def __call__(self, *a, **kw):
                    return None
                def __bool__(self):
                    return False
                def __iter__(self):
                    return iter([])
                def __repr__(self):
                    return f"<compat stub {fullname}.{name}>"
            _Stub.__name__ = name
            _Stub.__qualname__ = name
            return _Stub()

        stub.__getattr__ = _lenient_getattr
        sys.modules[fullname] = stub

        # Also attach to parent module
        parts = fullname.rsplit(".", 1)
        if len(parts) == 2:
            parent = sys.modules.get(parts[0])
            if parent is not None:
                setattr(parent, parts[1], stub)

        return stub

    @staticmethod
    def _make_lenient(mod):
        """Replace a raising __getattr__ with one that returns a no-op stub."""
        existing = mod.__dict__.get("__getattr__")
        if existing is None:
            return
        # Test if the existing __getattr__ raises unconditionally
        try:
            existing("__nonexistent_probe__")
        except (AttributeError, NotImplementedError):
            def _lenient_getattr(name):
                class _Stub:
                    def __init__(self, *a, **kw):
                        pass
                    def __call__(self, *a, **kw):
                        return None
                    def __bool__(self):
                        return False
                    def __iter__(self):
                        return iter([])
                _Stub.__name__ = name
                _Stub.__qualname__ = name
                return _Stub()
            mod.__getattr__ = _lenient_getattr
        except Exception:
            pass


def _install_torch_finder():
    """Install the meta path finder and mirror already-loaded candle modules."""
    # Install finder (idempotent)
    if not any(isinstance(f, _CandleTorchFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _CandleTorchFinder())

    # Mirror modules already loaded as candle.* to torch.*
    to_add = {}
    for key, mod in list(sys.modules.items()):
        if key.startswith("candle."):
            torch_key = "torch." + key[len("candle."):]
            if torch_key not in sys.modules:
                to_add[torch_key] = mod
    sys.modules.update(to_add)


# ---------------------------------------------------------------------------
# c)  Module stubs
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
# d)  Safetensors patch  (migrated from tests/run_test.py)
# ---------------------------------------------------------------------------
_MAX_HEADER_SIZE = 100_000_000

_NP_TYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.float16,
    "I64": np.int64,
    "U64": np.uint64,
    "I32": np.int32,
    "U32": np.uint32,
    "I16": np.int16,
    "U16": np.uint16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": bool,
}


class _PySafeSlice:
    """Lazy tensor slice from a safetensors file."""

    def __init__(self, info, bufferfile, base_ptr, buffermmap):
        self.info = info
        self.bufferfile = bufferfile
        self.buffermmap = buffermmap
        self.base_ptr = base_ptr

    @property
    def shape(self):
        return self.info["shape"]

    @property
    def dtype(self):
        return _NP_TYPES[self.info["dtype"]]

    @property
    def start_offset(self):
        return self.base_ptr + self.info["data_offsets"][0]

    def get_shape(self):
        return self.info["shape"]

    def get_dtype(self):
        return self.info["dtype"]

    def get(self, slice_arg=None):
        nbytes = int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize
        buffer = bytearray(nbytes)
        self.bufferfile.seek(self.start_offset)
        self.bufferfile.readinto(buffer)
        array = np.frombuffer(buffer, dtype=self.dtype).reshape(self.shape)
        if slice_arg is not None:
            array = array[slice_arg]
        import candle as torch  # noqa: F811
        return torch.from_numpy(array.copy())

    def __getitem__(self, slice_arg):
        return self.get(slice_arg)


def _read_metadata(buffer):
    buffer.seek(0, 2)
    buffer_len = buffer.tell()
    buffer.seek(0)
    if buffer_len < 8:
        raise ValueError("SafeTensorError::HeaderTooSmall")
    n = np.frombuffer(buffer.read(8), dtype=np.uint64).item()
    if n > _MAX_HEADER_SIZE:
        raise ValueError("SafeTensorError::HeaderTooLarge")
    stop = n + 8
    if stop > buffer_len:
        raise ValueError("SafeTensorError::InvalidHeaderLength")
    tensors = json.loads(buffer.read(n), object_pairs_hook=OrderedDict)
    metadata = tensors.pop("__metadata__", None)
    # validate offsets
    end = 0
    for key, info in tensors.items():
        s, e = info["data_offsets"]
        if e < s:
            raise ValueError(f"SafeTensorError::InvalidOffset({key})")
        if e > end:
            end = e
    if end + 8 + n != buffer_len:
        raise ValueError("SafeTensorError::MetadataIncompleteBuffer")
    return stop, tensors, metadata


class _FastSafeOpen:
    """Pure-Python safetensors reader (no C extension dependency on torch)."""

    def __init__(self, filename, framework=None, device="cpu"):
        self.filename = filename
        self.framework = framework
        self.file = open(self.filename, "rb")  # noqa: SIM115
        self.file_mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_COPY)
        self.base, self.tensors_decs, self._metadata = _read_metadata(self.file)
        self.tensors = OrderedDict()
        for key, info in self.tensors_decs.items():
            self.tensors[key] = _PySafeSlice(info, self.file, self.base, self.file_mmap)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def metadata(self):
        meta = self._metadata
        if meta is not None:
            meta["format"] = "pt"
        return meta

    def keys(self):
        return list(self.tensors.keys())

    def get_tensor(self, name):
        return self.tensors[name].get()

    def get_slice(self, name):
        return self.tensors[name]

    def offset_keys(self):
        return self.keys()


def _safe_load_file(filename, device="cpu"):
    result = {}
    with _FastSafeOpen(filename, framework="pt", device=device) as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    return result


def _apply_safetensors_patch():
    """Patch safetensors to use a pure-Python loader."""
    try:
        import safetensors
        import safetensors.torch as st
    except ImportError:
        return  # safetensors not installed; nothing to patch
    safetensors.safe_open = _FastSafeOpen
    st.load_file = _safe_load_file


# ---------------------------------------------------------------------------
# e)  torch_npu shim  (migrated from tests/run_test.py)
# ---------------------------------------------------------------------------

def _apply_torch_npu_shim():
    """Create a fake torch_npu module delegating to candle.npu."""
    if "torch_npu" in sys.modules:
        return  # already set up

    import candle as torch  # noqa: F811

    torch_npu = types.ModuleType("torch_npu")
    torch_npu.__version__ = "2.1.0"
    torch_npu.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
    torch_npu.__file__ = __file__
    torch_npu.__path__ = []

    # Copy public attributes from candle.npu if it exists
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is not None:
        for attr in dir(npu_mod):
            if not attr.startswith("_"):
                setattr(torch_npu, attr, getattr(npu_mod, attr))

    # Stubs for NPU-specific functions that transformers expects
    def _npu_fusion_attention(
        query, key, value, head_num, input_layout,
        pse=None, padding_mask=None, atten_mask=None,
        scale=1.0, keep_prob=1.0, pre_tockens=2147483647,
        next_tockens=0, inner_precise=1, prefix=None,
        sparse_mode=0, actual_seq_qlen=None,
        actual_seq_kvlen=None, gen_mask_parallel=True,
        sync=False,
    ):
        raise NotImplementedError("NPU fusion attention not available in candle")

    torch_npu.npu_fusion_attention = _npu_fusion_attention
    torch_npu.npu_format_cast = lambda x, fmt: x
    torch_npu.get_npu_format = lambda x: 0

    sys.modules["torch_npu"] = torch_npu


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
# g)  xfail injection
# ---------------------------------------------------------------------------

def _load_xfail_config():
    """Load xfail.yaml and return a dict {model: [patterns]}."""
    xfail_path = _COMPAT_DIR / "xfail.yaml"
    if not xfail_path.exists():
        return {}
    with open(xfail_path) as f:
        return yaml.safe_load(f) or {}


def _match_xfail(nodeid, xfail_entries):
    """Check if a test node ID matches any xfail entry.

    Returns the reason string if matched, None otherwise.
    """
    for entry in xfail_entries:
        if isinstance(entry, str):
            # plain glob pattern, no reason
            if fnmatch.fnmatch(nodeid, entry):
                return "known failure"
        elif isinstance(entry, dict):
            pattern = entry.get("pattern", "")
            reason = entry.get("reason", "known failure")
            if fnmatch.fnmatch(nodeid, f"*{pattern}*"):
                return reason
    return None


# ---------------------------------------------------------------------------
# Top-level application — called once when conftest is loaded
# ---------------------------------------------------------------------------

def apply_all_patches():
    """Apply all compatibility patches.  Idempotent."""
    # Ensure candle is importable and aliased as torch
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))

    import candle  # noqa: F401
    sys.modules.setdefault("torch", candle)

    _apply_version_spoof()
    _bypass_transformers_dep_check()  # must run before any transformers import
    _install_torch_finder()           # meta path finder for torch.* → candle.*
    _apply_module_stubs()
    _apply_safetensors_patch()
    _apply_torch_npu_shim()


# ---------------------------------------------------------------------------
# pytest hooks  (used when this file is loaded via conftest bridge)
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """Mark known failures as xfail from xfail.yaml."""
    import pytest  # local import to avoid import-time issues

    xfail_cfg = _load_xfail_config()
    if not xfail_cfg:
        return

    global_patterns = xfail_cfg.get("_global", [])

    for item in items:
        nodeid = item.nodeid

        # Check global patterns
        reason = _match_xfail(nodeid, global_patterns)
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
                reason = _match_xfail(nodeid, entries)
                if reason:
                    item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
                    break
