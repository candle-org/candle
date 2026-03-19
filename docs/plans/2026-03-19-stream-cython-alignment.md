# Stream Cython Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move candle's stream/archive reader-writer implementation onto a Cython-backed primary path while matching PyTorch's public import surface (`torch._C.PyTorchFileReader` / `PyTorchFileWriter`) and preserving the existing Python fallback.

**Architecture:** Add a new compiled module `src/candle/_cython/_stream.pyx` that contains the primary implementation of `PyTorchStreamReader`, `PyTorchStreamWriter`, `PyTorchFileReader`, and `PyTorchFileWriter`. Keep `src/candle/_stream.py` as a fallback/glue layer, re-exporting the compiled classes when available and retaining the current pure-Python implementation otherwise. Update `src/candle/_C.py` and `src/candle/serialization.py` so the public import path matches PyTorch more closely.

**Tech Stack:** Cython >= 3.0, Python 3.11, `struct`, `io`, `zipfile`, `zlib`, `hashlib`, `threading`, setuptools extensions

---

### Task 1: Add failing interface-alignment tests

**Files:**
- Modify: `tests/cpu/test_serialization.py`
- Test: `tests/cpu/test_serialization.py`

**Step 1: Write the failing tests**

Append these tests to `tests/cpu/test_serialization.py`:

```python
import io
import candle as torch


def test_c_module_exports_pytorch_file_reader_writer():
    assert hasattr(torch._C, "PyTorchFileReader")
    assert hasattr(torch._C, "PyTorchFileWriter")


def test_serialization_uses_c_module_reader_writer_surface(tmp_path):
    path = tmp_path / "sample.pt"
    torch.save({"x": 1}, path)

    reader = torch._C.PyTorchFileReader(str(path))
    assert reader.has_record("data.pkl")
    assert isinstance(reader.get_all_records(), list)


def test_stream_module_reexports_compiled_surface():
    from candle import _stream

    assert hasattr(_stream, "PyTorchStreamReader")
    assert hasattr(_stream, "PyTorchStreamWriter")
    assert hasattr(_stream, "PyTorchFileReader")
    assert hasattr(_stream, "PyTorchFileWriter")
```

**Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/cpu/test_serialization.py::test_c_module_exports_pytorch_file_reader_writer -v --tb=short
```

Expected: FAIL because `candle._C` currently only exposes `PyTorchFileReader` and has no `PyTorchFileWriter`.

**Step 3: Commit**

```bash
git add tests/cpu/test_serialization.py
git commit -m "test: add stream C-module surface alignment checks"
```

---

### Task 2: Add the new Cython extension to `setup.py`

**Files:**
- Modify: `setup.py:19-88`
- Test: `python setup.py build_ext --inplace`

**Step 1: Add the new extension**

In `setup.py`, add this `Extension` entry inside the `cythonize([...])` list after the existing `_fast_ops` entry and before the distributed entries:

```python
                Extension(
                    "candle._cython._stream",
                    ["src/candle/_cython/_stream.pyx"],
                ),
```

The resulting block should include:

```python
                Extension(
                    "candle._cython._fast_ops",
                    ["src/candle/_cython/_fast_ops.pyx"],
                ),
                Extension(
                    "candle._cython._stream",
                    ["src/candle/_cython/_stream.pyx"],
                ),
                Extension(
                    "candle.distributed._c10d",
                    ["src/candle/distributed/_c10d.pyx"],
                ),
```

**Step 2: Run build to verify it sees the new target**

Run:
```bash
python setup.py build_ext --inplace
```

Expected: build output includes `building 'candle._cython._stream' extension`.

**Step 3: Commit**

```bash
git add setup.py
git commit -m "build: add Cython stream extension"
```

---

### Task 3: Create `_cython/_stream.pyx` by porting the current `_stream.py`

**Files:**
- Create: `src/candle/_cython/_stream.pyx`
- Source reference: `src/candle/_stream.py:1-987`
- Test: `python -m cython src/candle/_cython/_stream.pyx`

**Step 1: Copy the full implementation into the new module**

Create `src/candle/_cython/_stream.pyx` with this header:

```cython
# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython primary implementation of PyTorch-style stream/file readers and writers.

This module mirrors the public API currently provided by `candle._stream`, but
serves as the compiled fast path. The Python module remains as a fallback/glue
layer.
"""
```

Then port the entire implementation from `src/candle/_stream.py` into this file, preserving these public classes and functions exactly:

- constants:
  - `kMinSupportedFileFormatVersion`
  - `kMaxSupportedFileFormatVersion`
  - `kMinProducedFileFormatVersion`
  - `kProducedFileFormatVersion`
  - `kFieldAlignment`
  - `kSerializationIdRecordName`
  - `kDebugPklSuffix`
- helper functions:
  - `_read_le16`
  - `_basename`
  - `_parentdir`
  - `_get_padding`
  - `_hash_combine`
- classes:
  - `ChunkRecordIterator`
  - `ReadAdapterInterface`
  - `FileAdapter`
  - `IStreamAdapter`
  - `PyTorchStreamReader`
  - `PyTorchStreamWriter`
  - `PyTorchFileReader`

Add a new `PyTorchFileWriter` class in this module as a pybind-style wrapper around `PyTorchStreamWriter`:

```python
class PyTorchFileWriter:
    """Drop-in replacement for ``torch._C.PyTorchFileWriter``.

    Wraps :class:`PyTorchStreamWriter` and exposes the writer through the
    `_C`-style surface used by `torch.serialization`.
    """

    def __init__(self, file_name_or_buffer):
        self._writer = PyTorchStreamWriter(file_name_or_buffer)

    def write_record(self, name, data, size=None, compress=False):
        return self._writer.writeRecord(name, data, size=size, compress=compress)

    def write_end_of_file(self):
        return self._writer.writeEndOfFile()

    def archive_name(self):
        return self._writer.archiveName()

    def serialization_id(self):
        return self._writer.serializationId()
```

Keep the first implementation mostly as a direct port. Do **not** redesign internals yet. The goal of this task is functional equivalence with a compiled primary path, not aggressive low-level optimization.

**Step 2: Run Cython syntax verification**

Run:
```bash
python -m cython src/candle/_cython/_stream.pyx
```

Expected: no compile errors.

**Step 3: Commit**

```bash
git add src/candle/_cython/_stream.pyx
git commit -m "feat(cython): add compiled stream reader/writer implementation"
```

---

### Task 4: Turn `_stream.py` into fallback/glue

**Files:**
- Modify: `src/candle/_stream.py`
- Test: import checks

**Step 1: Wrap the existing Python implementation with a compiled fast-path import**

At the very top of `src/candle/_stream.py`, add this pattern before the current implementation body:

```python
try:
    from ._cython._stream import (  # pylint: disable=no-name-in-module
        kMinSupportedFileFormatVersion,
        kMaxSupportedFileFormatVersion,
        kMinProducedFileFormatVersion,
        kProducedFileFormatVersion,
        kFieldAlignment,
        kSerializationIdRecordName,
        kDebugPklSuffix,
        _read_le16,
        _basename,
        _parentdir,
        _get_padding,
        _hash_combine,
        ChunkRecordIterator,
        ReadAdapterInterface,
        FileAdapter,
        IStreamAdapter,
        PyTorchStreamReader,
        PyTorchStreamWriter,
        PyTorchFileReader,
        PyTorchFileWriter,
    )
    _STREAM_CYTHON = True
except ImportError:
    _STREAM_CYTHON = False
```

Then guard the current pure-Python implementation so it only defines those symbols when `_STREAM_CYTHON` is `False`.

The simplest structure is:

```python
if not _STREAM_CYTHON:
    # existing current _stream.py body, indented under this block
```

Do not leave duplicate top-level class definitions active when `_STREAM_CYTHON` succeeds.

**Step 2: Run import verification**

Run:
```bash
python -c "from candle import _stream; print(_stream._STREAM_CYTHON); print(_stream.PyTorchFileReader, _stream.PyTorchFileWriter)"
```

Expected: prints `True` after build succeeds.

**Step 3: Commit**

```bash
git add src/candle/_stream.py
git commit -m "feat: route stream module through Cython fast path with Python fallback"
```

---

### Task 5: Align `candle._C` with PyTorch import surface

**Files:**
- Modify: `src/candle/_C.py:1-4`
- Test: `python -c "import candle as torch; print(torch._C.PyTorchFileReader, torch._C.PyTorchFileWriter)"`

**Step 1: Export both file-reader and file-writer symbols from `_C.py`**

Replace the current contents of `src/candle/_C.py` with:

```python
from ._backends.npu import runtime as npu_runtime
from ._backends.npu import aclnn
from ._stream import PyTorchFileReader, PyTorchFileWriter


def _npu_probe_model_dirs():
    return npu_runtime._probe_model_dirs()


def _npu_model_dir():
    return npu_runtime._model_dir()


def _npu_aclnn_available():
    return aclnn.is_available()


def _npu_aclnn_symbols_ok():
    return aclnn.symbols_ok()


def _npu_aclnn_ones_zero_ok():
    return aclnn.ones_zero_symbols_ok()


def _npu_device_count():
    return npu_runtime.device_count()
```

**Step 2: Run import verification**

Run:
```bash
python -c "import candle as torch; print(torch._C.PyTorchFileReader); print(torch._C.PyTorchFileWriter)"
```

Expected: both symbols resolve successfully.

**Step 3: Commit**

```bash
git add src/candle/_C.py
git commit -m "feat: export PyTorchFileWriter from candle._C"
```

---

### Task 6: Update `serialization.py` to prefer `_C` imports

**Files:**
- Modify: `src/candle/serialization.py:17-35`
- Test: serialization round-trip tests

**Step 1: Replace direct `_stream` imports with `_C`-aligned imports**

Change the import block near the top of `src/candle/serialization.py` from:

```python
from ._stream import PyTorchStreamReader, PyTorchStreamWriter
```

to:

```python
from ._C import PyTorchFileReader, PyTorchFileWriter
from ._stream import PyTorchStreamReader, PyTorchStreamWriter
```

Then update the zip open/write call sites so they preferentially construct the file-style wrappers where appropriate:

- For file-path based loading paths, instantiate `PyTorchFileReader`
- For file-path or writer-based saving paths, instantiate `PyTorchFileWriter` when the file-style surface matches
- Keep direct `PyTorchStreamReader` / `PyTorchStreamWriter` for internal helper paths that depend on the stream-style methods not exposed on the file-style wrapper

This keeps the public import surface aligned with PyTorch without forcing a full rewrite of `serialization.py` internals.

**Step 2: Run targeted serialization tests**

Run:
```bash
python -m pytest tests/cpu/test_serialization.py -v --tb=short
```

Expected: PASS.

**Step 3: Commit**

```bash
git add src/candle/serialization.py
git commit -m "feat: align serialization reader/writer imports with candle._C surface"
```

---

### Task 7: Build and verify end-to-end

**Files:**
- Modify if needed: `tests/cpu/test_serialization.py`
- Test: full verification

**Step 1: Build the extension**

Run:
```bash
python setup.py build_ext --inplace
```

Expected: output includes `building 'candle._cython._stream' extension`.

**Step 2: Run targeted tests**

Run:
```bash
python -m pytest tests/cpu/test_serialization.py -v --tb=short
```

Expected: PASS.

**Step 3: Run focused import smoke test**

Run:
```bash
python -c "import candle as torch; import io; print(torch._C.PyTorchFileReader); print(torch._C.PyTorchFileWriter); buf = io.BytesIO(); w = torch._C.PyTorchFileWriter(lambda b, n: n if b is None else n); print(type(w).__name__)"
```

Expected: all symbols resolve, no import errors.

**Step 4: Run pylint on touched files**

Run:
```bash
pylint src/candle/_stream.py src/candle/_C.py src/candle/serialization.py --rcfile=.github/pylint.conf
```

Expected: 10.00/10 or no new pylint errors.

**Step 5: Commit final fixes if any**

```bash
git add src/candle/_stream.py src/candle/_C.py src/candle/serialization.py src/candle/_cython/_stream.pyx setup.py tests/cpu/test_serialization.py
git commit -m "test: verify Cython stream alignment end-to-end"
```

---

### Task 8: Clean up the public API comments and document fallback semantics

**Files:**
- Modify: `src/candle/_stream.py`
- Modify: `src/candle/_C.py`

**Step 1: Update module docstrings/comments**

Make sure `_stream.py` clearly states that:
- compiled implementation lives in `candle._cython._stream`
- `_stream.py` is now the fallback/glue layer
- `_C.py` is the PyTorch-aligned public import surface for `PyTorchFileReader` / `PyTorchFileWriter`

**Step 2: Run import smoke test again**

Run:
```bash
python -c "import candle as torch; from candle import _stream; print(_stream._STREAM_CYTHON); print(torch._C.PyTorchFileReader); print(torch._C.PyTorchFileWriter)"
```

Expected: `True`, then both classes printed.

**Step 3: Commit**

```bash
git add src/candle/_stream.py src/candle/_C.py
git commit -m "docs: clarify Cython stream fallback and _C export surface"
```
