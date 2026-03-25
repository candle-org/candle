"""Tests for Cython C-core objects: StorageImpl."""
import numpy as np
import pytest


class TestStorageImpl:
    def test_wrap_numpy_zero_copy(self):
        from candle._cython._storage_impl import StorageImpl
        arr = np.ones(12, dtype=np.float32)
        s = StorageImpl.from_numpy(arr)
        assert s.data_ptr() == arr.ctypes.data
        assert s.nbytes() == 48
        assert s.device_type() == 0

    def test_cpu_alloc(self):
        from candle._cython._storage_impl import StorageImpl
        s = StorageImpl.alloc_cpu(64)
        assert s.nbytes() == 64
        assert s.data_ptr() != 0
        assert s.device_type() == 0

    def test_nbytes_matches(self):
        from candle._cython._storage_impl import StorageImpl
        arr = np.zeros(100, dtype=np.float64)
        s = StorageImpl.from_numpy(arr)
        assert s.nbytes() == 800

    def test_device_type_and_index(self):
        from candle._cython._storage_impl import StorageImpl
        arr = np.zeros(4, dtype=np.float32)
        s = StorageImpl.from_numpy(arr)
        assert s.device_type() == 0
        assert s.device_index() == -1

    def test_from_device_ptr(self):
        from candle._cython._storage_impl import StorageImpl
        arr = np.zeros(8, dtype=np.float32)
        ptr = arr.ctypes.data
        s = StorageImpl.from_device_ptr(ptr, 32, 0, -1, owner=arr)
        assert s.data_ptr() == ptr
        assert s.nbytes() == 32
        assert s.device_type() == 0
        assert s.device_index() == -1

    def test_from_device_ptr_rejects_numpy_owner_for_non_cpu(self):
        from candle._cython._storage_impl import StorageImpl
        arr = np.zeros(8, dtype=np.float32)
        with pytest.raises(TypeError, match="numpy-backed owner is only valid for CPU storage"):
            StorageImpl.from_device_ptr(arr.ctypes.data, arr.nbytes, 1, 0, owner=arr)

    def test_resizable_flag(self):
        from candle._cython._storage_impl import StorageImpl
        s1 = StorageImpl.alloc_cpu(64)
        assert s1.resizable() == True
        arr = np.zeros(4, dtype=np.float32)
        s2 = StorageImpl.from_numpy(arr)
        assert s2.resizable() == False

    def test_owner_pin_prevents_gc(self):
        import gc
        from candle._cython._storage_impl import StorageImpl
        arr = np.zeros(8, dtype=np.float32)
        ptr = arr.ctypes.data
        s = StorageImpl.from_device_ptr(ptr, 32, 0, -1, owner=arr)
        del arr
        gc.collect()
        # owner is held by s, so data_ptr must still be valid
        assert s.data_ptr() == ptr
        assert s.device_index() == -1

    def test_alloc_cpu_zero_bytes(self):
        from candle._cython._storage_impl import StorageImpl
        # malloc(0) is implementation-defined; handle gracefully
        try:
            s = StorageImpl.alloc_cpu(0)
            assert s.nbytes() == 0
        except MemoryError:
            pass  # acceptable on platforms where malloc(0) returns NULL


def _init_tensor_impl(t, shape_tuple, stride_tuple, dev, dtype_obj):
    """Helper: set shape/stride/device/dtype on a raw TensorImpl from Python.

    cdef-inline methods (_set_shape, _set_device_from_obj, etc.) are not
    callable from pure-Python tests, so we write the public C fields directly.
    """
    # shape
    t.shape = shape_tuple           # uses the property setter -> _set_shape
    t.stride = stride_tuple         # uses the property setter

    # device fields
    t._device_obj = dev
    dt = getattr(dev, "type", str(dev))
    _DEVTYPE = {"cpu": 0, "npu": 1, "cuda": 2, "mps": 3, "meta": 4}
    t._device_type = _DEVTYPE.get(dt, -1)
    idx = getattr(dev, "index", None)
    t._device_index = idx if idx is not None else -1

    # dtype fields
    t._dtype_obj = dtype_obj
    t._itemsize = getattr(dtype_obj, "itemsize", 4)
    _DTCODE = {
        "float32": 0, "float16": 1, "float64": 2, "bfloat16": 3,
        "int32": 4, "int64": 5, "int16": 6, "int8": 7, "uint8": 8, "bool": 9,
    }
    t._dtype_code = _DTCODE.get(getattr(dtype_obj, "name", ""), -1)

    # dispatch keys (simplified: just CPU bit for tests)
    _DK_CPU = 1 << 15
    t._dispatch_keys = _DK_CPU
    t.requires_grad = False


class TestCimportDispatch:
    """Verify dispatcher can read TensorImpl C fields directly via cimport."""

    def test_dispatch_keys_from_tensor_impl(self):
        from candle._cython._tensor_impl import TensorImpl
        from candle._device import device
        from candle._dtype import float32

        t = TensorImpl.__new__(TensorImpl)
        _init_tensor_impl(t, (2, 3), (3, 1), device("cpu"), float32)
        t._c_offset = 0
        t._base = None
        t._version_value = 0
        t._vc_proxy = None
        t.grad = None
        t.grad_fn = None
        t._view_meta = None
        t._pending = False
        t._retain_grad = False
        t._backward_hooks = None
        t._storage = None
        # dispatch_keys should have CPU bit set
        assert t._dispatch_keys != 0
        # _device_type should be 0 (cpu)
        assert t._device_type == 0


class TestViewOps:
    """View operations on TensorImpl — must share storage, not copy."""

    def _make_tensor(self, shape, stride=None):
        from candle._cython._tensor_impl import TensorImpl
        from candle._cython._storage_impl import StorageImpl
        from candle._device import device
        from candle._dtype import float32

        numel = 1
        for s in shape:
            numel *= s
        arr = np.arange(numel, dtype=np.float32)
        storage = StorageImpl.from_numpy(arr)

        if stride is None:
            strides = []
            acc = 1
            for d in reversed(shape):
                strides.append(acc)
                acc *= d
            strides.reverse()
            stride = tuple(strides)

        t = TensorImpl.__new__(TensorImpl)
        t._storage = storage
        _init_tensor_impl(t, tuple(shape), tuple(stride), device("cpu"), float32)
        t._c_offset = 0
        t.grad_fn = None
        t.grad = None
        t._base = None
        t._version_value = 0
        t._vc_proxy = None
        t._view_meta = None
        t._pending = False
        t._retain_grad = False
        t._backward_hooks = None
        return t

    def test_view_shares_storage(self):
        t = self._make_tensor([2, 3])
        v = t.cy_view((6,))
        assert v._storage is t._storage
        assert v.shape == (6,)
        assert v.stride == (1,)
        assert v._c_offset == 0

    def test_view_sets_base(self):
        t = self._make_tensor([2, 3])
        v = t.cy_view((6,))
        assert v._base is t

    def test_as_strided_shares_storage(self):
        t = self._make_tensor([2, 3])
        v = t.cy_as_strided((3, 2), (1, 3), 0)
        assert v._storage is t._storage
        assert v.shape == (3, 2)
        assert v.stride == (1, 3)

    def test_transpose_shares_storage(self):
        t = self._make_tensor([2, 3])
        v = t.cy_transpose(0, 1)
        assert v._storage is t._storage
        assert v.shape == (3, 2)
        assert v.stride == (1, 3)

    def test_transpose_is_not_contiguous(self):
        t = self._make_tensor([2, 3])
        v = t.cy_transpose(0, 1)
        # contiguous stride for (3,2) would be (2,1), not (1,3)
        assert v.stride != (2, 1)

    def test_view_version_counter_shared(self):
        t = self._make_tensor([2, 3])
        v = t.cy_view((6,))
        assert t._version_value == 0
        t._bump_version()
        assert t._version_value == 1

    def test_view_of_view_base_is_root(self):
        t = self._make_tensor([2, 3, 4])
        v1 = t.cy_view((6, 4))
        v2 = v1.cy_view((24,))
        assert v2._base is t

    def test_view_wrong_numel_raises(self):
        t = self._make_tensor([2, 3])
        with pytest.raises(RuntimeError, match="invalid for input of size"):
            t.cy_view((5,))

    def test_transpose_negative_dims(self):
        t = self._make_tensor([2, 3, 4])
        v = t.cy_transpose(-1, -2)
        assert v.shape == (2, 4, 3)

    def test_transpose_out_of_range_raises(self):
        t = self._make_tensor([2, 3])
        with pytest.raises(IndexError):
            t.cy_transpose(0, 5)



class TestTensorDTypeCaching:
    """Regression tests for Tensor dtype metadata cached from storage."""

    def test_tensor_init_sets_dtype_code_from_storage_float16(self):
        import candle as torch
        t = torch.ones(4, dtype=torch.float16)
        assert t._dtype_code == 1

    def test_tensor_init_sets_dtype_code_from_storage_int64(self):
        import candle as torch
        t = torch.arange(4, dtype=torch.int64)
        assert t._dtype_code == 5


class TestTensorFactoryInvariants:
    """Regression tests: tensor factory must always set all core metadata fields."""

    def test_tensor_from_python_init_sets_all_core_dtype_fields(self):
        import candle as torch
        t = torch.ones(4, dtype=torch.float16)
        assert t.dtype == torch.float16
        assert t._dtype_code == 1
        assert t._itemsize == 2
        assert t._device_type == 0
        assert t._device_index == -1

    def test_tensor_from_python_init_sets_device_metadata(self):
        import candle as torch
        t = torch.ones(4, dtype=torch.float32)
        assert t.device.type == "cpu"
        assert t._device_obj.type == "cpu"
        assert t._device_type == 0
        assert isinstance(t._dispatch_keys, int)

    def test_view_tensor_keeps_root_base_and_metadata(self):
        import candle as torch
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        v = t.cy_transpose(0, 1)
        assert v._base is not None
        assert v._base is t._base
        assert v.dtype == t.dtype
        assert v._dtype_code == t._dtype_code
        assert v._device_type == t._device_type
        assert v._storage is t._storage

    def test_scalar_created_tensor_matches_reference_dtype_code(self):
        import candle as torch
        if not torch.npu.is_available():
            pytest.skip("NPU not available")
        from candle._backends.npu.ops._helpers import _scalar_to_npu_tensor
        ref = torch.ones((2, 2), dtype=torch.float16, device="npu")
        scalar_tensor = _scalar_to_npu_tensor(1.0, ref)
        assert scalar_tensor.dtype == ref.dtype
        assert scalar_tensor._dtype_code == ref._dtype_code
        assert scalar_tensor._device_type == ref._device_type

    def test_cy_make_tensor_from_storage_initializes_all_core_fields(self):
        import numpy as np
        from candle._cython._storage_impl import StorageImpl
        from candle._cython._tensor_impl import cy_make_tensor_from_storage
        from candle._dtype import float32
        from candle._device import device

        arr = np.arange(6, dtype=np.float32)
        storage_impl = StorageImpl.from_numpy(arr)

        class WrappedUntyped:
            def __init__(self, impl, dev):
                self._impl = impl
                self.device = dev
            def data_ptr(self):
                return self._impl.data_ptr()

        dev = device("cpu")
        typed_storage = type("_TmpStorage", (), {})()
        typed_storage.device = dev
        typed_storage.dtype = float32
        typed_storage._storage_impl = storage_impl
        typed_storage._untyped = WrappedUntyped(storage_impl, dev)

        t = cy_make_tensor_from_storage(typed_storage, (2, 3), (3, 1), 0, False)
        assert t.shape == (2, 3)
        assert t.stride == (3, 1)
        assert t.offset == 0
        assert t.dtype == float32
        assert t._dtype_code == 0
        assert t._itemsize == 4
        assert t._device_type == 0
        assert t._storage is typed_storage
        assert t._base is None
        assert t._version_value == 0


class TestBuildIsolation:
    """Regression tests for editable install / build isolation issues."""

    def test_setup_py_has_no_top_level_numpy_import(self):
        from pathlib import Path

        setup_py = Path(__file__).resolve().parents[2] / "setup.py"
        text = setup_py.read_text(encoding="utf-8")
        assert "import numpy as np" not in text

    def test_setup_py_has_no_np_get_include_for_storage_impl(self):
        from pathlib import Path

        setup_py = Path(__file__).resolve().parents[2] / "setup.py"
        text = setup_py.read_text(encoding="utf-8")
        assert "np.get_include()" not in text
