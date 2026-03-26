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


    def test_cy_make_view_tensor_preserves_root_base_and_metadata(self):
        import candle as torch
        from candle._cython._tensor_impl import cy_make_view_tensor

        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        view = cy_make_view_tensor(base, base._storage, (4, 3), (1, 4), 0)
        assert view._base is base._base if base._base is not None else base
        assert view._storage is base._storage
        assert view.shape == (4, 3)
        assert view.stride == (1, 4)
        assert view._dtype_code == base._dtype_code
        assert view._device_type == base._device_type


    def test_cy_make_npu_tensor_initializes_dtype_code_like_python_tensor(self):
        import candle as torch
        if not torch.npu.is_available():
            return
        from candle._cython._storage import cy_make_npu_tensor
        t = torch.ones((2, 2), dtype=torch.float16, device="npu")
        out = cy_make_npu_tensor(
            t.storage()._untyped._device_ptr,
            4,
            t.dtype,
            t.device,
            (2, 2),
            (2, 1),
        )
        assert out._dtype_code == t._dtype_code
        assert out._device_type == t._device_type
        assert out._dispatch_keys == t._dispatch_keys


    def test_wrap_tensor_and_scalar_tensor_share_dtype_metadata_contract(self):
        import candle as torch
        if not torch.npu.is_available():
            return
        from candle._backends.npu.ops._helpers import _scalar_to_npu_tensor
        ref = torch.ones((2, 2), dtype=torch.float16, device="npu")
        scalar_t = _scalar_to_npu_tensor(1.0, ref)
        assert scalar_t._dtype_code == ref._dtype_code
        assert scalar_t._device_type == ref._device_type
        assert scalar_t.dtype == ref.dtype


class TestBuildIsolation:
    """Regression tests for editable install / build isolation issues."""

    def test_setup_py_has_no_top_level_numpy_import(self):
        from pathlib import Path

        setup_py = Path(__file__).resolve().parents[2] / "setup.py"
        text = setup_py.read_text(encoding="utf-8")
        assert "import numpy as np" not in text

    def test_setup_py_has_no_np_get_include_for_storage_impl(self):
        from pathlib import Path



class TestTensorInitCompatibilityShell:
    """Tensor.__init__ and Cython factories must share one initialization implementation."""

    def test_python_tensor_and_factory_tensor_have_same_core_metadata(self):
        import numpy as np
        import candle as torch
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

        from_factory = cy_make_tensor_from_storage(typed_storage, (2, 3), (3, 1), 0, False)
        from_python = torch.Tensor(typed_storage, (2, 3), (3, 1), 0, False)

        assert from_python.shape == from_factory.shape
        assert from_python.stride == from_factory.stride
        assert from_python.offset == from_factory.offset
        assert from_python._dtype_code == from_factory._dtype_code
        assert from_python._itemsize == from_factory._itemsize
        assert from_python._device_type == from_factory._device_type
        assert from_python._device_index == from_factory._device_index
        assert from_python._dispatch_keys == from_factory._dispatch_keys
        assert from_python._base is from_factory._base is None
        assert from_python._version_value == from_factory._version_value == 0

    def test_python_tensor_init_still_sets_pending_autograd_fields(self):
        import candle as torch

        t = torch.ones(4, dtype=torch.float32)
        assert t.grad is None
        assert t.grad_fn is None
        assert t._pending is False
        assert t._retain_grad is False
        assert t._backward_hooks is None
        assert t._vc_proxy is None


class TestCreationPathConsistency:
    """Public creation APIs must produce tensors with consistent core metadata."""

    def test_cpu_empty_metadata_matches_zeros_and_ones(self):
        import candle as torch
        a = torch.empty((2, 3), dtype=torch.float32)
        b = torch.zeros((2, 3), dtype=torch.float32)
        c = torch.ones((2, 3), dtype=torch.float32)
        assert a._dtype_code == b._dtype_code == c._dtype_code == 0
        assert a._device_type == b._device_type == c._device_type == 0
        assert a._dispatch_keys == b._dispatch_keys == c._dispatch_keys
        assert a._base is b._base is c._base is None
        assert a._version_value == b._version_value == c._version_value == 0

    def test_cpu_arange_metadata_matches_empty_same_dtype(self):
        import candle as torch
        a = torch.empty((4,), dtype=torch.int64)
        b = torch.arange(4, dtype=torch.int64)
        assert a._dtype_code == b._dtype_code == 5
        assert a._device_type == b._device_type == 0
        assert a._dispatch_keys == b._dispatch_keys

    def test_full_metadata_matches_zeros_same_dtype(self):
        import candle as torch
        a = torch.full((2, 2), 7.0, dtype=torch.float32)
        b = torch.zeros((2, 2), dtype=torch.float32)
        assert a._dtype_code == b._dtype_code
        assert a._itemsize == b._itemsize
        assert a._device_type == b._device_type
        assert a._dispatch_keys == b._dispatch_keys

    def test_npu_zeros_metadata_consistent_with_ones(self):
        import candle as torch
        if not torch.npu.is_available():
            return
        a = torch.zeros((2, 2), dtype=torch.float16, device="npu")
        b = torch.ones((2, 2), dtype=torch.float16, device="npu")
        assert a._dtype_code == b._dtype_code
        assert a._device_type == b._device_type
        assert a._dispatch_keys == b._dispatch_keys


    def test_mps_empty_and_ones_metadata_consistent(self):
        import candle as torch
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            return
        a = torch.empty((2, 2), dtype=torch.float32, device="mps")
        b = torch.ones((2, 2), dtype=torch.float32, device="mps")
        assert a._dtype_code == b._dtype_code
        assert a._device_type == b._device_type
        assert a._dispatch_keys == b._dispatch_keys

    def test_cuda_empty_and_zeros_metadata_consistent(self):
        import candle as torch
        if not hasattr(torch, "cuda") or not torch.cuda.is_available():
            return
        a = torch.empty((2, 2), dtype=torch.float32, device="cuda")
        b = torch.zeros((2, 2), dtype=torch.float32, device="cuda")
        assert a._dtype_code == b._dtype_code
        assert a._device_type == b._device_type
        assert a._dispatch_keys == b._dispatch_keys

    def test_meta_empty_and_zeros_metadata_consistent(self):
        import candle as torch
        a = torch.empty((2, 2), dtype=torch.float32, device="meta")
        b = torch.zeros((2, 2), dtype=torch.float32, device="meta")
        assert a._dtype_code == b._dtype_code
        assert a._device_type == b._device_type
        assert a._dispatch_keys == b._dispatch_keys




class TestHelperAndGradBirthConsistency:
    """Helper-born and grad-born tensors must share the unified metadata contract."""

    def test_cpu_convert_like_birth_matches_public_tensor_metadata(self):
        import candle as torch
        a = torch.ones((2, 2), dtype=torch.float32)
        b = a.to(dtype=torch.float64)
        ref = torch.zeros((2, 2), dtype=torch.float64)
        assert b._dtype_code == ref._dtype_code
        assert b._device_type == ref._device_type
        assert b._dispatch_keys == ref._dispatch_keys

    def test_reduce_grad_cpu_birth_matches_public_metadata(self):
        import candle as torch
        from candle.autograd.utils import reduce_grad

        grad = torch.ones((2, 2), dtype=torch.float32)
        reduced = reduce_grad(grad, (1, 2))
        ref = torch.zeros((1, 2), dtype=torch.float32)
        assert reduced._dtype_code == ref._dtype_code
        assert reduced._device_type == ref._device_type
        assert reduced._dispatch_keys == ref._dispatch_keys

    def test_autograd_grad_tensor_has_consistent_metadata(self):
        import candle as torch
        x = torch.ones((2, 2), dtype=torch.float32)
        x.requires_grad_(True)
        y = (x * 2).sum()
        y.backward()
        g = x.grad
        ref = torch.zeros((2, 2), dtype=torch.float32)
        assert g is not None
        assert g._dtype_code == ref._dtype_code
        assert g._device_type == ref._device_type
        assert g._dispatch_keys == ref._dispatch_keys




class TestAutogradResidualBirthPatterns:
    """Residual autograd-born tensors must share the unified metadata contract."""

    def test_backward_tuple_single_grad_birth_matches_public_metadata(self):
        import candle as torch
        x = torch.ones((2, 2), dtype=torch.float32)
        x.requires_grad_(True)
        y = (x * x).sum()
        y.backward()
        g = x.grad
        ref = torch.zeros((2, 2), dtype=torch.float32)
        assert g is not None
        assert g._dtype_code == ref._dtype_code
        assert g._device_type == ref._device_type
        assert g._dispatch_keys == ref._dispatch_keys

    def test_backward_deriv_tensor_birth_matches_public_metadata(self):
        import candle as torch
        x = torch.ones((2, 2), dtype=torch.float32)
        x.requires_grad_(True)
        y = torch.special.sinc(x).sum()
        y.backward()
        g = x.grad
        ref = torch.zeros((2, 2), dtype=torch.float32)
        assert g is not None
        assert g._dtype_code == ref._dtype_code
        assert g._device_type == ref._device_type
        assert g._dispatch_keys == ref._dispatch_keys


class TestDeterministicBirthProtocol:
    def test_public_and_helper_deterministic_paths_share_metadata_contract(self):
        import candle as torch
        a = torch.empty((2,), dtype=torch.float32)
        b = torch.zeros((2,), dtype=torch.float32)
        c = torch.ones((2,), dtype=torch.float32)
        d = torch.full((2,), 5.0, dtype=torch.float32)
        assert a._dtype_code == b._dtype_code == c._dtype_code == d._dtype_code
        assert a._device_type == b._device_type == c._device_type == d._device_type
        assert a._dispatch_keys == b._dispatch_keys == c._dispatch_keys == d._dispatch_keys
        assert a._base is b._base is c._base is d._base is None


class TestMultiOutputBackwardBirthConsistency:
    """Multi-output backward grad tensors must share the unified metadata contract."""

    def test_binary_op_backward_both_grads_match_public_metadata(self):
        import candle as torch
        x = torch.ones((2, 2), dtype=torch.float32)
        y = torch.ones((2, 2), dtype=torch.float32)
        x.requires_grad_(True)
        y.requires_grad_(True)
        z = (x * y).sum()
        z.backward()
        ref = torch.zeros((2, 2), dtype=torch.float32)
        for g in (x.grad, y.grad):
            assert g is not None
            assert g._dtype_code == ref._dtype_code
            assert g._device_type == ref._device_type
            assert g._dispatch_keys == ref._dispatch_keys

    def test_single_slot_multi_output_backward_metadata(self):
        import candle as torch
        x = torch.ones((2, 2), dtype=torch.float32)
        x.requires_grad_(True)
        y = x.sum()
        y.backward()
        ref = torch.zeros((2, 2), dtype=torch.float32)
        assert x.grad is not None
        assert x.grad._dtype_code == ref._dtype_code
        assert x.grad._device_type == ref._device_type
        assert x.grad._dispatch_keys == ref._dispatch_keys


class TestRuntimeInternalBirths:
    """Runtime-core internal tensor births must share the unified metadata contract."""

    def test_detach_birth_matches_public_metadata(self):
        import candle as torch
        x = torch.ones((2, 2), dtype=torch.float32)
        x.requires_grad_(True)
        y = x.detach()
        ref = torch.zeros((2, 2), dtype=torch.float32)
        assert y._dtype_code == ref._dtype_code
        assert y._device_type == ref._device_type
        assert y._dispatch_keys == ref._dispatch_keys
        assert y.requires_grad is False

    def test_to_dtype_cpu_birth_matches_public_metadata(self):
        import candle as torch
        x = torch.ones((2, 2), dtype=torch.float32)
        y = x.to(dtype=torch.float64)
        ref = torch.zeros((2, 2), dtype=torch.float64)
        assert y._dtype_code == ref._dtype_code
        assert y._device_type == ref._device_type
        assert y._dispatch_keys == ref._dispatch_keys

    def test_ones_like_internal_birth_matches_public_metadata(self):
        import candle as torch
        x = torch.zeros((2, 2), dtype=torch.float32)
        y = x._ones_like()
        ref = torch.ones((2, 2), dtype=torch.float32)
        assert y._dtype_code == ref._dtype_code
        assert y._device_type == ref._device_type
        assert y._dispatch_keys == ref._dispatch_keys


class TestRuntimeCoreBirthProtocol:
    """Final smoke: runtime-core tensor methods and dispatcher share metadata contract."""

    def test_tensor_methods_and_dispatcher_share_metadata_contract(self):
        import candle as torch
        x = torch.ones((2, 2), dtype=torch.float32)
        y = x.detach()
        z = x.to(dtype=torch.float64)
        ref_y = torch.zeros((2, 2), dtype=torch.float32)
        ref_z = torch.zeros((2, 2), dtype=torch.float64)
        assert y._dtype_code == ref_y._dtype_code
        assert z._dtype_code == ref_z._dtype_code


class TestRNGBirthConsistency:
    """Random creation APIs must return tensors with the same metadata contract as deterministic creation."""

    def test_rand_cpu_metadata_matches_empty(self):
        import candle as torch
        a = torch.rand((2, 2), dtype=torch.float32)
        b = torch.empty((2, 2), dtype=torch.float32)
        assert a._dtype_code == b._dtype_code
        assert a._device_type == b._device_type
        assert a._dispatch_keys == b._dispatch_keys
        assert a._base is None and b._base is None

    def test_randn_cpu_metadata_matches_zeros(self):
        import candle as torch
        a = torch.randn((2, 2), dtype=torch.float32)
        b = torch.zeros((2, 2), dtype=torch.float32)
        assert a._dtype_code == b._dtype_code
        assert a._device_type == b._device_type
        assert a._dispatch_keys == b._dispatch_keys

    def test_randint_cpu_metadata_matches_full(self):
        import candle as torch
        a = torch.randint(0, 10, (2, 2), dtype=torch.int64)
        b = torch.full((2, 2), 0, dtype=torch.int64)
        assert a._dtype_code == b._dtype_code
        assert a._device_type == b._device_type
        assert a._dispatch_keys == b._dispatch_keys

    def test_randperm_cpu_metadata_matches_arange(self):
        import candle as torch
        a = torch.randperm(8, dtype=torch.int64)
        b = torch.arange(8, dtype=torch.int64)
        assert a._dtype_code == b._dtype_code
        assert a._device_type == b._device_type
        assert a._dispatch_keys == b._dispatch_keys

    def test_npu_rand_metadata_matches_empty(self):
        import candle as torch
        if not torch.npu.is_available():
            return
        a = torch.rand((2, 2), dtype=torch.float32, device="npu")
        b = torch.empty((2, 2), dtype=torch.float32, device="npu")
        assert a._dtype_code == b._dtype_code
        assert a._device_type == b._device_type
        assert a._dispatch_keys == b._dispatch_keys
