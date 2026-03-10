"""Shared compatibility patches for running third-party test suites against candle.

Each compat/<library>/conftest.py imports the utilities it needs from here.

Provided utilities
------------------
- Version spoofing : make candle look like a specific torch version
- Meta path finder : redirect ``import torch.*`` to ``candle.*``
- Safetensors patch: pure-Python safetensors loader (no C extension)
- torch_npu shim   : fake ``torch_npu`` so NPU checks pass
- xfail helpers    : load xfail.yaml and match test node IDs
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

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# a)  Version spoofing
# ---------------------------------------------------------------------------
SPOOFED_TORCH_VERSION = "2.5.0"


def apply_version_spoof():
    """Make candle appear as torch 2.5.0 to satisfy version gates."""
    import candle  # noqa: F811

    candle.__version__ = SPOOFED_TORCH_VERSION

    # Patch importlib.metadata so `importlib.metadata.version("torch")` works
    _original_version = importlib.metadata.version

    def _patched_version(name):
        if name == "torch":
            return SPOOFED_TORCH_VERSION
        return _original_version(name)

    importlib.metadata.version = _patched_version


# ---------------------------------------------------------------------------
# b)  Module mirroring -- candle.* <-> torch.* via meta path finder
# ---------------------------------------------------------------------------

class CandleTorchFinder:
    """Meta path finder that resolves ``import torch.*`` to candle modules.

    For any ``torch.X.Y`` import:
      1. Try ``import candle.X.Y`` -- if it exists, mirror it.
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

        # torch.X.Y -> candle.X.Y
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
            and falsy -- so it works for most guard patterns like
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


def install_torch_finder():
    """Install the meta path finder and mirror already-loaded candle modules."""
    # Install finder (idempotent)
    if not any(isinstance(f, CandleTorchFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, CandleTorchFinder())

    # Mirror modules already loaded as candle.* to torch.*
    to_add = {}
    for key, mod in list(sys.modules.items()):
        if key.startswith("candle."):
            torch_key = "torch." + key[len("candle."):]
            if torch_key not in sys.modules:
                to_add[torch_key] = mod
    sys.modules.update(to_add)


# ---------------------------------------------------------------------------
# c)  Safetensors patch  (pure-Python safetensors loader)
# ---------------------------------------------------------------------------
MAX_HEADER_SIZE = 100_000_000

NP_TYPES = {
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


class PySafeSlice:
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
        return NP_TYPES[self.info["dtype"]]

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


def read_metadata(buffer):
    buffer.seek(0, 2)
    buffer_len = buffer.tell()
    buffer.seek(0)
    if buffer_len < 8:
        raise ValueError("SafeTensorError::HeaderTooSmall")
    n = np.frombuffer(buffer.read(8), dtype=np.uint64).item()
    if n > MAX_HEADER_SIZE:
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


class FastSafeOpen:
    """Pure-Python safetensors reader (no C extension dependency on torch)."""

    def __init__(self, filename, framework=None, device="cpu"):
        self.filename = filename
        self.framework = framework
        self.file = open(self.filename, "rb")  # noqa: SIM115
        self.file_mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_COPY)
        self.base, self.tensors_decs, self._metadata = read_metadata(self.file)
        self.tensors = OrderedDict()
        for key, info in self.tensors_decs.items():
            self.tensors[key] = PySafeSlice(info, self.file, self.base, self.file_mmap)

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


def safe_load_file(filename, device="cpu"):
    result = {}
    with FastSafeOpen(filename, framework="pt", device=device) as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    return result


def apply_safetensors_patch():
    """Patch safetensors to use a pure-Python loader."""
    try:
        import safetensors
        import safetensors.torch as st
    except ImportError:
        return  # safetensors not installed; nothing to patch
    safetensors.safe_open = FastSafeOpen
    st.load_file = safe_load_file


# ---------------------------------------------------------------------------
# d)  torch_npu shim
# ---------------------------------------------------------------------------

def apply_torch_npu_shim():
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
# e)  xfail helpers
# ---------------------------------------------------------------------------

def load_xfail_config(xfail_path):
    """Load xfail.yaml from the given path and return a dict {model: [patterns]}."""
    if not xfail_path.exists():
        return {}
    with open(xfail_path) as f:
        return yaml.safe_load(f) or {}


def match_xfail(nodeid, xfail_entries):
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
