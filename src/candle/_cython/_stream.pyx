# cython: language_level=3, boundscheck=False, wraparound=False
"""Pure-Python reimplementation of caffe2::serialize::PyTorchStreamReader/Writer.

Faithfully replicates the C++ classes in inline_container.{h,cc} so that
candle can read and write PyTorch zip checkpoints without any native
dependency on miniz or libtorch.

All public method signatures match the C++ originals.
"""

import hashlib
import os
import struct
import sys
import threading
import zipfile
import zlib
from typing import (
    BinaryIO,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

# ── Format version constants (from caffe2/serialize/versions.h) ──────
kMinSupportedFileFormatVersion: int = 0x1
kMaxSupportedFileFormatVersion: int = 0xA
kMinProducedFileFormatVersion: int = 0x3
kProducedFileFormatVersion: int = 0xA

# ── Internal constants ────────────────────────────────────────────────
kFieldAlignment: int = 64
kSerializationIdRecordName: str = ".data/serialization_id"
kDebugPklSuffix: str = ".debug_pkl"
kMagicValueLength: int = 8

# ZIP local-file-header layout
_LFH_SIGNATURE = 0x04034B50
_LFH_SIZE = 30
_LFH_FNAME_LEN_OFS = 26
_LFH_EXTRA_LEN_OFS = 28

# ZIP data-descriptor
_DATA_DESC_SIGNATURE = 0x08074B50

# ZIP central-directory
_CD_SIGNATURE = 0x02014B50
_CD_HEADER_SIZE = 46

# ZIP64 end-of-central-directory
_ZIP64_EOCD_SIGNATURE = 0x06064B50
_ZIP64_EOCD_LOCATOR_SIGNATURE = 0x07064B50

# ZIP end-of-central-directory
_EOCD_SIGNATURE = 0x06054B50

# ZIP64 extra-field tag
_ZIP64_EXTRA_TAG = 0x0001


def _read_le16(buf: bytes, offset: int = 0) -> int:
    return buf[offset] | (buf[offset + 1] << 8)


def _basename(name: str) -> str:
    """Return stem of *name* (strip directory and extension)."""
    start = 0
    for i, ch in enumerate(name):
        if ch in ('\\', '/'):
            start = i + 1
    if start >= len(name):
        return ""
    end = len(name)
    for i in range(end, start, -1):
        if name[i - 1] == '.':
            end = i - 1
            break
    return name[start:end]


def _parentdir(name: str) -> str:
    idx = name.rfind('/')
    if idx == -1:
        idx = name.rfind('\\')
    if idx == -1:
        return ""
    return name[:idx]


def _get_padding(cursor: int, filename_size: int, size: int) -> bytes:
    """Compute FB-prefixed padding so that file data is 64-byte aligned.

    Returns the raw extra-field bytes (including the 'FB' + uint16 header).
    """
    start = cursor + _LFH_SIZE + filename_size + 4  # +4 for FB header itself
    # Account for ZIP64 extra field when sizes exceed 32-bit
    if size >= 0xFFFFFFFF or cursor >= 0xFFFFFFFF:
        start += 4  # ZIP64 tag + size (2+2)
        if size >= 0xFFFFFFFF:
            start += 16  # two uint64 (uncomp + comp)
        if cursor >= 0xFFFFFFFF:
            start += 8   # one uint64 (offset)
    mod = start % kFieldAlignment
    padding_size = 0 if mod == 0 else (kFieldAlignment - mod)
    # FB extra field: 'F' 'B' <uint16 padding_size> <padding_size bytes of 'Z'>
    buf = bytearray(4 + padding_size)
    buf[0] = ord('F')
    buf[1] = ord('B')
    buf[2] = padding_size & 0xFF
    buf[3] = (padding_size >> 8) & 0xFF
    for i in range(4, 4 + padding_size):
        buf[i] = ord('Z')
    return bytes(buf)


# ── ChunkRecordIterator ──────────────────────────────────────────────

class ChunkRecordIterator:
    """Iterate over a zip record in fixed-size chunks.

    Mirrors ``caffe2::serialize::ChunkRecordIterator``.
    """

    def __init__(
        self,
        record_size: int,
        chunk_size: int,
        data: bytes,
    ) -> None:
        self._record_size = record_size
        self._chunk_size = chunk_size
        self._offset = 0
        self._data = data

    # -- public API (matches C++) ------------------------------------------

    def next(self, buf: Optional[bytearray] = None) -> Union[bytes, int]:
        """Read at most *chunk_size* bytes.

        If *buf* is ``None`` return a ``bytes`` object.
        If *buf* is a writable buffer, copy into it and return the number
        of bytes written (mirrors the C++ ``void* buf`` overload).
        """
        want = min(self._chunk_size, self._record_size - self._offset)
        if want == 0:
            return b"" if buf is None else 0
        chunk = self._data[self._offset:self._offset + want]
        self._offset += len(chunk)
        if buf is not None:
            buf[:len(chunk)] = chunk
            return len(chunk)
        return chunk

    @property
    def recordSize(self) -> int:  # noqa: N802 – match C++ name
        return self._record_size

    @property
    def offset(self) -> int:
        return self._offset


# ── ReadAdapterInterface ─────────────────────────────────────────────

class ReadAdapterInterface:
    """Abstract base matching ``caffe2::serialize::ReadAdapterInterface``."""

    def size(self) -> int:
        raise NotImplementedError

    def read(self, pos: int, buf: bytearray, n: int, what: str = "") -> int:
        raise NotImplementedError


class FileAdapter(ReadAdapterInterface):
    """Read adapter backed by a file path."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._fp = open(path, "rb")  # noqa: SIM115
        self._fp.seek(0, 2)
        self._size = self._fp.tell()

    def size(self) -> int:
        return self._size

    def read(self, pos: int, buf: bytearray, n: int, what: str = "") -> int:
        self._fp.seek(pos)
        data = self._fp.read(n)
        if buf is not None:
            buf[:len(data)] = data
        return len(data)

    def read_bytes(self, pos: int, n: int) -> bytes:
        self._fp.seek(pos)
        return self._fp.read(n)

    def close(self) -> None:
        self._fp.close()


class IStreamAdapter(ReadAdapterInterface):
    """Read adapter backed by a file-like (``io.RawIOBase`` / ``io.BufferedIOBase``)."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        cur = stream.tell()
        stream.seek(0, 2)
        self._size = stream.tell()
        stream.seek(cur)

    def size(self) -> int:
        return self._size

    def read(self, pos: int, buf: bytearray, n: int, what: str = "") -> int:
        self._stream.seek(pos)
        data = self._stream.read(n)
        if buf is not None:
            buf[:len(data)] = data
        return len(data)

    def read_bytes(self, pos: int, n: int) -> bytes:
        self._stream.seek(pos)
        return self._stream.read(n)

    def close(self) -> None:
        pass  # caller owns the stream


class _ReadAdapterIO:
    """Seekable file-like wrapper for ReadAdapterInterface."""

    def __init__(self, adapter: ReadAdapterInterface) -> None:
        self._adapter = adapter
        self._pos = 0
        self._size = adapter.size()

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self._size - self._pos
        if n <= 0:
            return b""
        buf = bytearray(n)
        read_n = self._adapter.read(self._pos, buf, n)
        self._pos += read_n
        return bytes(buf[:read_n])

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            pos = offset
        elif whence == 1:
            pos = self._pos + offset
        elif whence == 2:
            pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def close(self) -> None:
        if hasattr(self._adapter, "close"):
            self._adapter.close()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True


# ── PyTorchStreamReader ──────────────────────────────────────────────

class PyTorchStreamReader:
    """Pure-Python replica of ``caffe2::serialize::PyTorchStreamReader``.

    Constructor overloads:
      - ``PyTorchStreamReader(file_name: str)``
      - ``PyTorchStreamReader(stream: BinaryIO)``
      - ``PyTorchStreamReader(adapter: ReadAdapterInterface)``
    """

    def __init__(
        self,
        file_name_or_stream_or_adapter: Union[str, BinaryIO, ReadAdapterInterface],
    ) -> None:
        self._lock = threading.Lock()
        self._load_debug_symbol: bool = True
        self._additional_reader_size_threshold: int = 0
        self._version: int = 0
        self._serialization_id: str = ""
        self._archive_name: str = ""
        self._archive_name_plus_slash: str = ""

        # Build the underlying zipfile and adapter
        if isinstance(file_name_or_stream_or_adapter, str):
            self._in: ReadAdapterInterface = FileAdapter(file_name_or_stream_or_adapter)
            self._zip_source: Union[str, BinaryIO, _ReadAdapterIO] = file_name_or_stream_or_adapter
        elif isinstance(file_name_or_stream_or_adapter, ReadAdapterInterface):
            self._in = file_name_or_stream_or_adapter
            self._zip_source = _ReadAdapterIO(self._in)
        else:
            # file-like / stream
            self._in = IStreamAdapter(file_name_or_stream_or_adapter)
            self._zip_source = file_name_or_stream_or_adapter

        # We use Python's zipfile for the heavy lifting (decompression, CD parsing)
        # but also keep raw access for offset calculations.
        self._raw_size = self._in.size()
        self._init()

    # ── private helpers ───────────────────────────────────────────────

    def _read_raw(self, pos: int, n: int) -> bytes:
        """Low-level read from the underlying adapter."""
        if hasattr(self._in, 'read_bytes'):
            return self._in.read_bytes(pos, n)
        buf = bytearray(n)
        self._in.read(pos, buf, n)
        return bytes(buf)

    def _init(self) -> None:
        # Check for old magic number
        if self._raw_size > kMagicValueLength:
            magic = self._read_raw(0, kMagicValueLength)
            if magic == b"PYTORCH1":
                raise RuntimeError(
                    "File is an unsupported archive format from the preview release."
                )

        # Open the zip using a seekable source so we don't load the full archive.
        self._zf = zipfile.ZipFile(self._zip_source, 'r')

        # Figure out archive_name from the first entry
        names = self._zf.namelist()
        if len(names) == 0:
            raise RuntimeError("archive does not contain any files")
        first = names[0]
        pos = first.find('/')
        if pos == -1:
            raise RuntimeError(f"file in archive is not in a subdirectory: {first}")
        self._archive_name = first[:pos]
        self._archive_name_plus_slash = self._archive_name + "/"

        # Build a lookup: short_name -> ZipInfo
        self._records: Dict[str, zipfile.ZipInfo] = {}
        for info in self._zf.infolist():
            fn = info.filename
            if fn.startswith(self._archive_name_plus_slash):
                short = fn[len(self._archive_name_plus_slash):]
                self._records[short] = info

        # Read serialization id
        if self.hasRecord(kSerializationIdRecordName):
            data, _ = self.getRecord(kSerializationIdRecordName)
            self._serialization_id = data.decode('ascii') if isinstance(data, (bytes, bytearray)) else data

        # Version check
        if self.hasRecord(".data/version"):
            vdata, _ = self.getRecord(".data/version")
        elif self.hasRecord("version"):
            vdata, _ = self.getRecord("version")
        else:
            raise RuntimeError("archive does not contain a version entry")

        version_str = vdata.decode('ascii').strip() if isinstance(vdata, (bytes, bytearray)) else str(vdata).strip()
        try:
            self._version = int(version_str)
        except ValueError as e:
            raise RuntimeError(
                f"Couldn't parse the version {version_str!r} as integer."
            ) from e

        if self._version < kMinSupportedFileFormatVersion:
            raise RuntimeError(
                f"Attempted to read a PyTorch file with version {self._version}, "
                f"but the minimum supported version for reading is "
                f"{kMinSupportedFileFormatVersion}. Your PyTorch script module file "
                f"is too old. Please regenerate it with latest version of PyTorch "
                f"to mitigate this issue."
            )
        if self._version > kMaxSupportedFileFormatVersion:
            raise RuntimeError(
                f"Attempted to read a PyTorch file with version {self._version}, "
                f"but the maximum supported version for reading is "
                f"{kMaxSupportedFileFormatVersion}. The version of your PyTorch "
                f"installation may be too old, please upgrade PyTorch to latest "
                f"version to mitigate this issue."
            )

    def _get_record_id(self, name: str) -> zipfile.ZipInfo:
        """Locate a record by short name, raise if missing."""
        if name not in self._records:
            raise RuntimeError(
                f"file not found in archive: {self._archive_name_plus_slash}{name}"
            )
        return self._records[name]

    def _should_skip_debug(self, name: str) -> bool:
        return (not self._load_debug_symbol) and name.endswith(kDebugPklSuffix)

    # ── public API ────────────────────────────────────────────────────

    def getRecord(self, name: str) -> Tuple[bytes, int]:
        """Return ``(data_bytes, size)`` for the named record.

        Matches the C++ signature ``std::tuple<at::DataPtr, size_t> getRecord(name)``.
        """
        with self._lock:
            if self._should_skip_debug(name):
                return (b"", 0)
            info = self._get_record_id(name)
            data = self._zf.read(info)
            return (data, len(data))

    def getRecordToBuffer(self, name: str, dst: bytearray, n: int) -> int:
        """In-place read into *dst*.

        Matches the C++ overload ``size_t getRecord(name, void* dst, size_t n)``.
        """
        with self._lock:
            if self._should_skip_debug(name):
                return 0
            info = self._get_record_id(name)
            data = self._zf.read(info)
            if n != len(data):
                raise RuntimeError(
                    f"record size {len(data)} mismatch with dst size {n}"
                )
            dst[:n] = data
            return len(data)

    def getRecordMultiReaders(
        self,
        name: str,
        additional_readers: List[ReadAdapterInterface],
        dst: bytearray,
        n: int,
    ) -> int:
        """Concurrent read using multiple readers.

        Matches the C++ ``getRecordMultiReaders`` method.
        In this pure-Python version we use threads for I/O parallelism.
        """
        if not additional_readers:
            data, sz = self.getRecord(name)
            dst[:sz] = data
            return sz

        nthread = len(additional_readers) + 1
        record_off = self.getRecordOffset(name)
        per_thread = (n + nthread - 1) // nthread
        read_sizes = [0] * nthread
        errors: List[Optional[Exception]] = [None] * nthread

        def _worker(idx: int) -> None:
            try:
                start_pos = idx * per_thread
                end_pos = min((idx + 1) * per_thread, n)
                if start_pos >= end_pos:
                    return
                thread_read_size = end_pos - start_pos
                chunk_buf = bytearray(thread_read_size)
                if idx == 0:
                    sz = self._in.read(record_off + start_pos, chunk_buf, thread_read_size)
                else:
                    reader = additional_readers[idx - 1]
                    sz = reader.read(record_off + start_pos, chunk_buf, thread_read_size)
                dst[start_pos:start_pos + sz] = chunk_buf[:sz]
                read_sizes[idx] = sz
                if thread_read_size != sz:
                    raise RuntimeError(
                        f"record size {thread_read_size} mismatch with read size {sz}"
                    )
            except Exception as e:
                errors[idx] = e

        threads = []
        with self._lock:
            for i in range(nthread):
                t = threading.Thread(target=_worker, args=(i,))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

        for e in errors:
            if e is not None:
                raise e

        return sum(read_sizes)

    def getRecordSize(self, name: str) -> int:
        """Return the uncompressed size of the named record."""
        info = self._get_record_id(name)
        return info.file_size

    def getRecordOffset(self, name: str) -> int:
        """Return the byte offset of the record's data in the raw archive.

        Matches the C++ ``getRecordOffset`` which reads the local file header
        to compute: local_header_ofs + 30 + filename_len + extra_len.
        """
        with self._lock:
            info = self._get_record_id(name)
            header_offset = info.header_offset
            # Read the local file header to get filename_len and extra_len
            header = self._read_raw(header_offset, _LFH_SIZE)
            filename_len = _read_le16(header, _LFH_FNAME_LEN_OFS)
            extra_len = _read_le16(header, _LFH_EXTRA_LEN_OFS)
            return header_offset + _LFH_SIZE + filename_len + extra_len

    def getRecordHeaderOffset(self, name: str) -> int:
        """Return the local file header offset for the named record."""
        info = self._get_record_id(name)
        return info.header_offset

    def getRecordOffsetNoRead(
        self,
        zipfile_header_offset: int,
        filename: str,
        size: int,
        storage_alignment: int,
    ) -> int:
        """Compute the data offset without reading the local file header.

        Mirrors the C++ ``getRecordOffsetNoRead`` used for mmap fast-path.
        """
        full_name = self._archive_name_plus_slash + filename
        full_name_len = len(full_name.encode('utf-8'))
        # Base offset past the fixed-size local file header and filename
        data_start = zipfile_header_offset + _LFH_SIZE + full_name_len
        # Account for alignment padding in the extra field
        if storage_alignment > 0 and size > 0:
            padding = (storage_alignment - data_start % storage_alignment) % storage_alignment
            # ZIP64 extra field header (4 bytes tag+size) may precede padding
            # but for the no-read fast path we just add the padding
            data_start += padding
        return data_start

    def hasRecord(self, name: str) -> bool:
        """Return whether the archive contains the named record."""
        with self._lock:
            if self._should_skip_debug(name):
                return False
            return name in self._records

    def getAllRecords(self) -> List[str]:
        """Return all record short-names in the archive."""
        with self._lock:
            out = []
            for short_name in self._records:
                if self._load_debug_symbol or not short_name.endswith(kDebugPklSuffix):
                    out.append(short_name)
            return out

    def isRecordCompressed(self, name: str) -> bool:
        """Return whether the named record uses compression (not ZIP_STORED)."""
        info = self._get_record_id(name)
        return info.compress_type != 0

    def createChunkReaderIter(
        self,
        name: str,
        record_size: int,
        chunk_size: int,
    ) -> ChunkRecordIterator:
        """Create a ``ChunkRecordIterator`` for the named record."""
        info = self._get_record_id(name)
        data = self._zf.read(info)
        return ChunkRecordIterator(record_size, chunk_size, data)

    def version(self) -> int:
        """Return the archive format version."""
        return self._version

    def serializationId(self) -> str:
        """Return the serialization id string."""
        return self._serialization_id

    def setShouldLoadDebugSymbol(self, should_load: bool) -> None:
        """Control whether ``.debug_pkl`` records are visible."""
        self._load_debug_symbol = should_load

    def setAdditionalReaderSizeThreshold(self, size: int) -> None:
        """Set the minimum record size for multi-reader parallel reads."""
        self._additional_reader_size_threshold = size

    def __del__(self) -> None:
        try:
            self._zf.close()
        except Exception:
            pass
        if hasattr(self._in, 'close'):
            try:
                self._in.close()
            except Exception:
                pass


# ── PyTorchStreamWriter ──────────────────────────────────────────────

class PyTorchStreamWriter:
    """Pure-Python replica of ``caffe2::serialize::PyTorchStreamWriter``.

    Constructor overloads:
      - ``PyTorchStreamWriter(file_name: str)``
      - ``PyTorchStreamWriter(writer_func: Callable[[bytes, int], int])``

    All files are stored uncompressed and 64-byte aligned (unless *compress*
    is explicitly requested), matching the C++ writer behaviour.
    """

    def __init__(
        self,
        file_name_or_writer: Union[str, Callable[[Optional[bytes], int], int]],
    ) -> None:
        self._current_pos: int = 0
        self._files_written: Set[str] = set()
        self._archive_name: str = ""
        self._archive_name_plus_slash: str = ""
        self._version: int = kMinProducedFileFormatVersion
        self._finalized: bool = False
        self._err_seen: bool = False
        self._combined_uncomp_crc32: int = 0
        self._serialization_id: str = ""
        self._writer_func: Optional[Callable[[Optional[bytes], int], int]] = None
        self._file_stream: Optional[BinaryIO] = None

        # Central directory entries accumulated during writing
        self._cd_entries: List[bytes] = []
        self._cd_offset: int = 0

        if isinstance(file_name_or_writer, str):
            self._archive_name = _basename(file_name_or_writer)
            self._setup(file_name_or_writer)
        else:
            self._archive_name = "archive"
            self._writer_func = file_name_or_writer
            self._setup(self._archive_name)

    # ── private helpers ───────────────────────────────────────────────

    def _setup(self, file_name: str) -> None:
        if len(self._archive_name) == 0:
            raise RuntimeError(f"invalid file name: {file_name}")
        self._archive_name_plus_slash = self._archive_name + "/"

        if self._writer_func is None:
            dir_name = _parentdir(file_name)
            if dir_name:
                if not os.path.isdir(dir_name):
                    raise RuntimeError(f"Parent directory {dir_name} does not exist.")
            self._file_stream = open(file_name, "wb")  # noqa: SIM115
            self._writer_func = self._file_write_func

    def _file_write_func(self, buf: Optional[bytes], nbytes: int) -> int:
        """Writer function for file-backed archives."""
        assert self._file_stream is not None
        if buf is None:
            # Seek forward (see [Note: write_record_metadata] in C++)
            self._file_stream.seek(nbytes, os.SEEK_CUR)
        else:
            self._file_stream.write(buf)
        return nbytes

    def _write_raw(self, data: bytes) -> int:
        """Write raw bytes through the writer_func, updating position and CRC tracking."""
        assert self._writer_func is not None
        n = len(data)
        ret = self._writer_func(data, n)
        if ret != n:
            self._err_seen = True
        self._current_pos += ret

        # Track data-descriptor CRC32 (matches C++ ostream_write_func logic)
        if n >= 8:
            sig = struct.unpack_from('<I', data, 0)[0]
            if sig == _DATA_DESC_SIGNATURE:
                uncomp_crc32 = struct.unpack_from('<I', data, 4)[0]
                self._combined_uncomp_crc32 = _hash_combine(
                    self._combined_uncomp_crc32, uncomp_crc32
                )
        return ret

    def _valid(self, what: str, info: str = "") -> None:
        if self._err_seen:
            raise RuntimeError(f"PytorchStreamWriter failed {what}{info}.")

    # ── public API ────────────────────────────────────────────────────

    def setMinVersion(self, version: int) -> None:
        """Set the minimum produced version (takes the max of current and *version*)."""
        self._version = max(version, self._version)

    def writeRecord(
        self,
        name: str,
        data: Union[bytes, bytearray, memoryview],
        size: Optional[int] = None,
        compress: bool = False,
    ) -> None:
        """Write a single record into the archive.

        Matches the C++ ``writeRecord(name, data, size, compress)`` signature.
        """
        if self._finalized:
            raise RuntimeError("Cannot write to a finalized archive")
        if name in self._files_written:
            raise RuntimeError(f"Tried to serialize file twice: {name}")
        # Skip duplicate serialization_id from copied records
        if name == kSerializationIdRecordName and not self._serialization_id:
            return

        if isinstance(data, (str,)):
            data = data.encode('utf-8')
        if isinstance(data, memoryview):
            data = bytes(data)
        if size is None:
            size = len(data)
        data = data[:size]

        full_name = self._archive_name_plus_slash + name
        full_name_bytes = full_name.encode('utf-8')

        # Compute padding for 64-byte alignment
        padding = _get_padding(self._current_pos, len(full_name_bytes), size)

        # Compute CRC32 of uncompressed data
        uncomp_crc32 = zlib.crc32(data) & 0xFFFFFFFF

        # Compress if requested
        if compress:
            # ZIP deflate requires raw deflate (wbits=-15), not zlib-wrapped
            comp_obj = zlib.compressobj(9, zlib.DEFLATED, -15)
            comp_data = comp_obj.compress(data) + comp_obj.flush()
            comp_method = 8  # deflate
            comp_size = len(comp_data)
        else:
            comp_data = data
            comp_method = 0  # stored
            comp_size = size

        # Determine if ZIP64 is needed
        use_zip64 = (size >= 0xFFFFFFFF or comp_size >= 0xFFFFFFFF
                      or self._current_pos >= 0xFFFFFFFF)

        # Build ZIP64 extra field if needed
        zip64_extra = b""
        if use_zip64:
            parts = []
            if size >= 0xFFFFFFFF or comp_size >= 0xFFFFFFFF:
                parts.append(struct.pack('<Q', size))
                parts.append(struct.pack('<Q', comp_size))
            if self._current_pos >= 0xFFFFFFFF:
                parts.append(struct.pack('<Q', self._current_pos))
            zip64_body = b"".join(parts)
            zip64_extra = struct.pack('<HH', _ZIP64_EXTRA_TAG, len(zip64_body)) + zip64_body

        # Combine extra fields: ZIP64 (if any) + FB padding
        extra_field = zip64_extra + padding

        # ── Local file header ──
        lfh_size_field = 0xFFFFFFFF if use_zip64 and size >= 0xFFFFFFFF else size
        lfh_comp_field = 0xFFFFFFFF if use_zip64 and comp_size >= 0xFFFFFFFF else comp_size
        local_header = struct.pack(
            '<IHHHHHIIIHH',
            _LFH_SIGNATURE,       # signature
            45 if use_zip64 else 20,  # version needed
            0,                     # flags (bit 3 = data descriptor; we write sizes directly)
            comp_method,           # compression method
            0,                     # mod time
            0,                     # mod date
            uncomp_crc32,          # crc32
            lfh_comp_field,        # compressed size
            lfh_size_field,        # uncompressed size
            len(full_name_bytes),  # filename length
            len(extra_field),      # extra field length
        )

        local_header_offset = self._current_pos

        self._write_raw(local_header)
        self._write_raw(full_name_bytes)
        self._write_raw(extra_field)
        self._write_raw(comp_data)
        self._valid("writing file ", name)

        # ── Central directory entry ──
        cd_extra = zip64_extra  # CD doesn't need FB padding
        cd_size_field = 0xFFFFFFFF if use_zip64 and size >= 0xFFFFFFFF else size
        cd_comp_field = 0xFFFFFFFF if use_zip64 and comp_size >= 0xFFFFFFFF else comp_size
        cd_offset_field = 0xFFFFFFFF if use_zip64 and local_header_offset >= 0xFFFFFFFF else local_header_offset

        cd_entry = struct.pack(
            '<IHHHHHHIIIHHHHHII',
            _CD_SIGNATURE,         # signature
            45 if use_zip64 else 20,  # version made by
            45 if use_zip64 else 20,  # version needed
            0,                     # flags
            comp_method,           # compression method
            0,                     # mod time
            0,                     # mod date
            uncomp_crc32,          # crc32
            cd_comp_field,         # compressed size
            cd_size_field,         # uncompressed size
            len(full_name_bytes),  # filename length
            len(cd_extra),         # extra field length
            0,                     # comment length
            0,                     # disk number start
            0,                     # internal file attributes
            0,                     # external file attributes
            cd_offset_field,       # relative offset of local header
        )
        self._cd_entries.append(cd_entry + full_name_bytes + cd_extra)
        self._files_written.add(name)

    def writeEndOfFile(self) -> None:
        """Finalize the archive: write version, byteorder, serialization_id, and central directory."""
        if self._finalized:
            return
        try:
            all_records = self.getAllWrittenRecords()

            # Write version if not already present
            if ".data/version" not in all_records and "version" not in all_records:
                version_str = str(self._version) + '\n'
                version_bytes = version_str.encode('ascii')
                if self._version >= 0x6:
                    self.writeRecord(".data/version", version_bytes)
                else:
                    self.writeRecord("version", version_bytes)

            # Write byteorder if not already present
            if "byteorder" not in all_records:
                byteorder = "little" if sys.byteorder == "little" else "big"
                self.writeRecord("byteorder", byteorder.encode('ascii'))

            # Write serialization id
            self._writeSerializationId()

            # Write central directory
            cd_start = self._current_pos
            for entry in self._cd_entries:
                self._write_raw(entry)
            cd_end = self._current_pos
            cd_size = cd_end - cd_start
            num_entries = len(self._cd_entries)

            need_zip64 = (num_entries >= 0xFFFF or cd_size >= 0xFFFFFFFF
                          or cd_start >= 0xFFFFFFFF)

            if need_zip64:
                # ZIP64 end of central directory record
                zip64_eocd = struct.pack(
                    '<IQHHIIQQQQQ',
                    _ZIP64_EOCD_SIGNATURE,
                    44,  # size of remaining record
                    45,  # version made by
                    45,  # version needed
                    0,   # disk number
                    0,   # disk with CD start
                    num_entries,
                    num_entries,
                    cd_size,
                    cd_start,
                )
                # Extensible data sector (empty)
                self._write_raw(zip64_eocd)

                # ZIP64 end of central directory locator
                zip64_locator = struct.pack(
                    '<IIQI',
                    _ZIP64_EOCD_LOCATOR_SIGNATURE,
                    0,       # disk with ZIP64 EOCD
                    cd_end,  # offset of ZIP64 EOCD
                    1,       # total disks
                )
                self._write_raw(zip64_locator)

            # End of central directory record
            eocd = struct.pack(
                '<IHHHHIIH',
                _EOCD_SIGNATURE,
                0,  # disk number
                0,  # disk with CD start
                min(num_entries, 0xFFFF),
                min(num_entries, 0xFFFF),
                min(cd_size, 0xFFFFFFFF),
                min(cd_start, 0xFFFFFFFF),
                0,  # comment length
            )
            self._write_raw(eocd)
        finally:
            self._finalized = True
            if self._file_stream is not None:
                self._file_stream.close()

    def getAllWrittenRecords(self) -> Set[str]:
        """Return the set of record names written so far."""
        return self._files_written

    def finalized(self) -> bool:
        """Return whether ``writeEndOfFile`` has been called."""
        return self._finalized

    def archiveName(self) -> str:
        """Return the archive base name."""
        return self._archive_name

    def serializationId(self) -> str:
        """Return the serialization id string."""
        return self._serialization_id

    def _writeSerializationId(self) -> None:
        """Compute and write the serialization id record.

        The id is composed of:
        1) a combined hash of record name hashes
        2) the combined CRC32 of uncompressed data
        """
        if kSerializationIdRecordName in self._files_written:
            return
        combined_name_hash = 0
        for record_name in self._files_written:
            h = _stable_hash64(record_name)
            combined_name_hash = _hash_combine(combined_name_hash, h)
        self._serialization_id = (
            f"{combined_name_hash:020d}{self._combined_uncomp_crc32:020d}"
        )
        self.writeRecord(
            kSerializationIdRecordName,
            self._serialization_id.encode('ascii'),
        )

    def __del__(self) -> None:
        if not self._finalized:
            try:
                self.writeEndOfFile()
            except Exception:
                pass


def _hash_combine(seed: int, value: int) -> int:
    """Mimic ``c10::hash_combine`` (boost hash_combine)."""
    value = value & 0xFFFFFFFFFFFFFFFF
    seed = seed & 0xFFFFFFFFFFFFFFFF
    seed ^= (value + 0x9E3779B9 + (seed << 6) + (seed >> 2)) & 0xFFFFFFFFFFFFFFFF
    return seed


def _stable_hash64(value: str) -> int:
    digest = hashlib.sha1(value.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder='little', signed=False)


# ── PyTorchFileReader (pybind-compatible snake_case API) ─────────────

class PyTorchFileReader:
    """Drop-in replacement for the ``torch._C.PyTorchFileReader`` pybind class.

    Wraps :class:`PyTorchStreamReader` and exposes the same snake_case
    methods that the C++ pybind11 bindings register.
    """

    def __init__(self, file_name_or_buffer):
        if isinstance(file_name_or_buffer, str):
            self._reader = PyTorchStreamReader(file_name_or_buffer)
        else:
            self._reader = PyTorchStreamReader(file_name_or_buffer)

    def get_record(self, key: str) -> bytes:
        data, _ = self._reader.getRecord(key)
        return data

    def has_record(self, key: str) -> bool:
        return self._reader.hasRecord(key)

    def get_storage_from_record(self, key: str, numel: int, dtype) -> "Tensor":
        """Read a storage record and return it as a Tensor.

        *dtype* can be a candle DType or any object with a ``.itemsize``
        attribute (e.g. ``numpy.dtype``).
        """
        from ._dtype import to_numpy_dtype, DType
        from ._storage import typed_storage_from_numpy
        from ._tensor import Tensor
        import numpy as np

        data, size = self._reader.getRecord(key)

        if isinstance(dtype, DType):
            np_dtype = np.dtype(to_numpy_dtype(dtype))
        else:
            np_dtype = np.dtype(dtype)

        expected = numel * np_dtype.itemsize
        if size != expected:
            raise RuntimeError(
                f"record size ({size} bytes) does not match expected size "
                f"({expected} bytes = {numel} elements * "
                f"{np_dtype.itemsize} bytes/element) for dtype {np_dtype}"
            )

        arr = np.frombuffer(data, dtype=np_dtype, count=numel).copy()
        storage = typed_storage_from_numpy(arr, dtype, device="cpu")
        return Tensor(storage, (numel,), (1,))

    def serialization_id(self) -> str:
        return self._reader.serializationId()

    def get_all_records(self) -> List[str]:
        return self._reader.getAllRecords()

    def get_record_offset(self, key: str) -> int:
        return self._reader.getRecordOffset(key)

    def get_record_header_offset(self, key: str) -> int:
        return self._reader.getRecordHeaderOffset(key)

    def get_record_offset_no_read(
        self,
        zipfile_header_offset: int,
        filename: str,
        size: int,
        storage_alignment: int,
    ) -> int:
        return self._reader.getRecordOffsetNoRead(
            zipfile_header_offset, filename, size, storage_alignment,
        )


# ── PyTorchFileWriter (pybind-compatible snake_case API) ─────────────

class PyTorchFileWriter:
    """Drop-in replacement for the ``torch._C.PyTorchFileWriter`` pybind class."""

    def __init__(self, file_name_or_buffer):
        if isinstance(file_name_or_buffer, str):
            self._writer = PyTorchStreamWriter(file_name_or_buffer)
        else:
            self._writer = PyTorchStreamWriter(file_name_or_buffer)

    def write_record(
        self,
        name: str,
        data: Union[bytes, bytearray, memoryview],
        size: Optional[int] = None,
        compress: bool = False,
    ) -> None:
        self._writer.writeRecord(name, data, size, compress)

    def write_end_of_file(self) -> None:
        self._writer.writeEndOfFile()

    def archive_name(self) -> str:
        return self._writer.archiveName()

    def serialization_id(self) -> str:
        return self._writer.serializationId()
