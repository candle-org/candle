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


class TestCrossBoundaryBirthConsistency:
    """Tensors reconstructed across multiprocessing/shared-storage boundaries must share the birth contract."""

    def test_shared_storage_like_birth_matches_public_metadata(self):
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

        t = cy_make_tensor_from_storage(typed_storage, (2, 3), (3, 1), 0, False)
        ref = torch.zeros((2, 3), dtype=torch.float32)
        assert t._dtype_code == ref._dtype_code
        assert t._device_type == ref._device_type
        assert t._dispatch_keys == ref._dispatch_keys

    def test_detached_reconstruction_like_birth_matches_public_metadata(self):
        import candle as torch
        a = torch.ones((2, 2), dtype=torch.float32)
        b = a.detach()
        ref = torch.zeros((2, 2), dtype=torch.float32)
        assert b._dtype_code == ref._dtype_code
        assert b._device_type == ref._device_type
        assert b._dispatch_keys == ref._dispatch_keys


class TestSerializationBirthConsistency:
    """Tensors reconstructed from stream/file helpers must share the unified metadata contract."""

    def test_stream_storage_reconstruction_matches_public_metadata(self):
        import numpy as np
        import candle as torch
        from candle._storage import typed_storage_from_numpy
        from candle._dtype import float32

        arr = np.arange(6, dtype=np.float32)
        storage = typed_storage_from_numpy(arr, float32, device='cpu')
        t = torch.Tensor(storage, (6,), (1,))
        ref = torch.zeros((6,), dtype=torch.float32)
        assert t._dtype_code == ref._dtype_code
        assert t._device_type == ref._device_type
        assert t._dispatch_keys == ref._dispatch_keys

    def test_file_reader_storage_tensor_matches_public_metadata(self):
        import candle as torch
        ref = torch.zeros((4,), dtype=torch.float32)
        tmp = torch.arange(4, dtype=torch.float32)
        assert tmp._dtype_code == ref._dtype_code
        assert tmp._device_type == ref._device_type
        assert tmp._dispatch_keys == ref._dispatch_keys


class TestCallableStrideContract:
    """Unified birth path must preserve callable stride() semantics."""

    def test_factory_born_tensor_stride_is_callable(self):
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
        assert t.stride() == (3, 1)
        assert t.stride(0) == 3
        assert t.stride(1) == 1


class TestTensorLayoutMethodProviders:
    """Layout/view Tensor methods should be served from the Cython tensor API layer."""

    def test_remaining_layout_methods_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "is_contiguous",
            "contiguous",
            "flatten",
            "t",
            "as_strided",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_tensor_api_bound_layout_methods_preserve_behavior(self):
        import candle as torch

        base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        transposed = base.transpose(0, 1)

        assert transposed.is_contiguous() is False
        cont = transposed.contiguous()
        assert cont.is_contiguous() is True
        assert cont.shape == (3, 2)

        flat = base.flatten()
        assert flat.shape == (6,)

        tt = base.t()
        assert tt.shape == (3, 2)
        assert tt.stride() == (1, 3)

        view = base.as_strided((3, 2), (1, 3), 0)
        assert view.shape == (3, 2)
        assert view.stride() == (1, 3)
        assert view._storage is base._storage


class TestTensorAutogradMethodProviders:
    """Autograd state Tensor methods should be served from the Cython tensor API layer."""

    def test_remaining_autograd_methods_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "detach_",
            "retain_grad",
            "requires_grad_",
            "register_hook",
            "_is_view",
            "_check_inplace",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_tensor_api_bound_autograd_methods_preserve_behavior(self):
        import candle as torch

        x = torch.ones((2, 2), dtype=torch.float32)
        x.requires_grad_(True)
        assert x.requires_grad is True

        x.retain_grad()
        assert x._retain_grad is True

        handle = x.register_hook(lambda g: g)
        assert handle.id in x._backward_hooks
        handle.remove()
        assert handle.id not in x._backward_hooks

        v = x.view(4)
        assert v._is_view() is True
        assert x._is_view() is False

        y = torch.ones((2, 2), dtype=torch.float32)
        y.requires_grad_(True)
        y.detach_()


class TestTensorConversionMethodProviders:
    """Conversion Tensor methods should be served from the Cython tensor API layer."""

    def test_remaining_conversion_methods_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "_to_dtype",
            "cpu",
            "cuda",
            "mps",
            "npu",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_tensor_api_bound_conversion_methods_preserve_behavior(self):
        import candle as torch

        x = torch.ones((2, 2), dtype=torch.float32)

        y = x._to_dtype(torch.float64)
        assert y.dtype == torch.float64
        assert y.shape == x.shape

        z = x.cpu()
        assert z.device.type == "cpu"
        assert z.dtype == x.dtype

        if torch.npu.is_available():
            n = x.npu()
            assert n.device.type == "npu"
            assert n.dtype == x.dtype

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            c = x.cuda(0)
            assert c.device.type == "cuda"
            assert c.dtype == x.dtype

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            m = x.mps()
            assert m.device.type == "mps"
            assert m.dtype == x.dtype

class TestTensorIndexingMethodProviders:
    """Indexing Tensor methods should be served from the Cython tensor API layer."""

    def test_indexing_methods_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "__getitem__",
            "__setitem__",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_tensor_api_bound_indexing_methods_preserve_behavior(self):
        import candle as torch

        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        assert x[0, 1].item() == 1.0
        assert x[:, 1].shape == (2,)

class TestTensorInplaceMutationMethodProviders:
    """Representative inplace mutation methods should be served from the Cython tensor API layer."""

    def test_representative_inplace_methods_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "add_",
            "mul_",
            "relu_",
            "zero_",
            "fill_",
            "copy_",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_tensor_api_bound_inplace_methods_preserve_behavior(self):
        import candle as torch

        x = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
        x.add_(2.0)
        assert x.tolist() == [3.0, 0.0, 5.0]

        x.mul_(2.0)
        assert x.tolist() == [6.0, 0.0, 10.0]

        x.relu_()
        assert x.tolist() == [6.0, 0.0, 10.0]

        x.fill_(4.0)
        assert x.tolist() == [4.0, 4.0, 4.0]

        y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        x.copy_(y)
        assert x.tolist() == [1.0, 2.0, 3.0]



class TestTensorUnaryInplaceMethodProviders:
    """Unary inplace mutation methods should be served from the Cython tensor API layer."""

    def test_unary_inplace_methods_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "abs_",
            "neg_",
            "exp_",
            "log_",
            "log2_",
            "log10_",
            "sqrt_",
            "sin_",
            "cos_",
            "tan_",
            "tanh_",
            "sigmoid_",
            "floor_",
            "ceil_",
            "round_",
            "trunc_",
            "pow_",
            "reciprocal_",
            "erfinv_",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_unary_inplace_methods_preserve_behavior(self):
        import candle as torch

        a = torch.tensor([-1.5, 2.2], dtype=torch.float32)
        a.abs_()
        assert all(abs(v - ref) < 1e-6 for v, ref in zip(a.tolist(), [1.5, 2.2]))
        a.neg_()
        assert all(abs(v - ref) < 1e-6 for v, ref in zip(a.tolist(), [-1.5, -2.2]))

        b = torch.tensor([1.0, 4.0], dtype=torch.float32)
        b.sqrt_()
        assert all(abs(v - ref) < 1e-6 for v, ref in zip(b.tolist(), [1.0, 2.0]))
        b.pow_(2.0)
        assert all(abs(v - ref) < 1e-6 for v, ref in zip(b.tolist(), [1.0, 4.0]))
        b.reciprocal_()
        assert all(abs(v - ref) < 1e-6 for v, ref in zip(b.tolist(), [1.0, 0.25]))

        c = torch.tensor([1.0, 2.0], dtype=torch.float32)
        c.exp_()
        assert all(v > 0 for v in c.tolist())
        c.log_()
        assert all(abs(v - ref) < 1e-6 for v, ref in zip(c.tolist(), [1.0, 2.0]))

        d = torch.tensor([1.9, -1.2], dtype=torch.float32)
        d.floor_()
        assert all(abs(v - ref) < 1e-6 for v, ref in zip(d.tolist(), [1.0, -2.0]))
        d.ceil_()
        assert all(abs(v - ref) < 1e-6 for v, ref in zip(d.tolist(), [1.0, -2.0]))
        d.round_()
        assert all(abs(v - ref) < 1e-6 for v, ref in zip(d.tolist(), [1.0, -2.0]))
        d.trunc_()
        assert all(abs(v - ref) < 1e-6 for v, ref in zip(d.tolist(), [1.0, -2.0]))

        e = torch.tensor([0.0, 0.5], dtype=torch.float32)
        e.sin_()
        assert len(e.tolist()) == 2
        e.cos_()
        assert len(e.tolist()) == 2
        e.tan_()
        assert len(e.tolist()) == 2
        e.tanh_()
        assert len(e.tolist()) == 2
class TestTensorParameterizedInplaceMethodProviders:
    """Parameterized and random inplace methods should be served from the Cython tensor API layer."""

    def test_parameterized_inplace_methods_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "sub_",
            "clamp_",
            "uniform_",
            "normal_",
            "random_",
            "randint_",
            "bernoulli_",
            "exponential_",
            "log_normal_",
            "cauchy_",
            "geometric_",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_parameterized_inplace_methods_preserve_behavior(self):
        import candle as torch

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        x.sub_(1.0)
        assert x.tolist() == [0.0, 1.0, 2.0]

        y = torch.tensor([-1.0, 0.5, 3.0], dtype=torch.float32)
        y.clamp_(0.0, 1.0)
        assert y.tolist() == [0.0, 0.5, 1.0]

        z = torch.empty(5, dtype=torch.float32)
        z.uniform_(0.0, 1.0)
        assert len(z.tolist()) == 5
        assert all(0.0 <= v <= 1.0 for v in z.tolist())

        a = torch.empty(5, dtype=torch.float32)
        a.normal_(0.0, 1.0)
        assert len(a.tolist()) == 5

        b = torch.empty(5, dtype=torch.int64)
        b.random_(0, 10)
        assert len(b.tolist()) == 5
        assert all(0 <= v < 10 for v in b.tolist())

        c = torch.empty(5, dtype=torch.int64)
        c.randint_(0, 10)
        assert len(c.tolist()) == 5
        assert all(0 <= v < 10 for v in c.tolist())

        d = torch.empty(8, dtype=torch.float32)
        d.bernoulli_(0.5)
        assert all(v in (0.0, 1.0) for v in d.tolist())

        e = torch.empty(5, dtype=torch.float32)
        e.exponential_(1.0)
        assert all(v >= 0.0 for v in e.tolist())

        f = torch.empty(5, dtype=torch.float32)
        f.log_normal_(0.0, 1.0)
        assert all(v > 0.0 for v in f.tolist())

        g = torch.empty(5, dtype=torch.float32)
        g.cauchy_(0.0, 1.0)
        assert len(g.tolist()) == 5



class TestTensorInplaceLayoutViewMethodProviders:
    """Inplace layout/view methods should be served from the Cython tensor API layer."""

    def test_inplace_layout_view_methods_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "transpose_",
            "t_",
            "squeeze_",
            "unsqueeze_",
            "as_strided_",
            "swapdims_",
            "swapaxes_",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_inplace_layout_view_methods_preserve_behavior(self):
        import candle as torch

        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        out = x.transpose_(0, 1)
        assert out is x
        assert x.shape == (3, 2)
        assert x.stride() == (1, 3)

        y = torch.arange(6, dtype=torch.float32).reshape(1, 2, 1, 3)
        y.squeeze_()
        assert y.shape == (2, 3)
        assert y.stride() == (3, 1)
        y.unsqueeze_(1)
        assert y.shape == (2, 1, 3)
        assert y.stride() == (3, 3, 1)

        z = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        z.as_strided_((3, 2), (1, 3), 0)
        assert z.shape == (3, 2)
        assert z.stride() == (1, 3)

        w = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        w.swapdims_(0, 1)
        assert w.shape == (3, 2)
        assert w.stride() == (1, 3)

        u = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        u.swapaxes_(0, 1)
        assert u.shape == (3, 2)
        assert u.stride() == (1, 3)

        v = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        v.t_()


class TestTensorIndexingWritebackMethodProviders:
    """Indexing write-back methods should be served from the Cython tensor API layer."""

    def test_indexing_writeback_methods_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "scatter_",
            "scatter_add_",
            "masked_fill_",
            "masked_scatter_",
            "index_put_",
            "index_copy_",
            "index_fill_",
            "index_add_",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_indexing_writeback_methods_preserve_behavior(self):
        import candle as torch

        index = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int64)
        src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)

        x = torch.zeros((2, 3), dtype=torch.float32)
        x.scatter_(1, index, src)
        assert x.tolist() == [[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]]

        y = torch.zeros((2, 3), dtype=torch.float32)
        y.scatter_add_(1, index, src)
        assert y.tolist() == [[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]]

        mask = torch.tensor([[True, False], [False, True]])
        m = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        m.masked_fill_(mask, 9.0)
        assert m.tolist() == [[9.0, 2.0], [3.0, 9.0]]

        ms = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        ms.masked_scatter_(mask, torch.tensor([7.0, 8.0], dtype=torch.float32))
        assert ms.tolist() == [[7.0, 2.0], [3.0, 8.0]]

        ip = torch.zeros((2, 3), dtype=torch.float32)
        ip.index_put_((torch.tensor([0, 1]), torch.tensor([1, 2])), torch.tensor([5.0, 6.0], dtype=torch.float32))
        assert ip.tolist() == [[0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]

        ic = torch.zeros((2, 3), dtype=torch.float32)
        ic.index_copy_(1, torch.tensor([0, 2]), torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
        assert ic.tolist() == [[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]]

        iff = torch.zeros((2, 3), dtype=torch.float32)
        iff.index_fill_(1, torch.tensor([1]), 7.0)
        assert iff.tolist() == [[0.0, 7.0, 0.0], [0.0, 7.0, 0.0]]

        ia = torch.zeros((2, 3), dtype=torch.float32)
        ia.index_add_(1, torch.tensor([0, 2]), torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32), alpha=1.0)




class TestTensorNumpyAndPinningMethodProviders:
    """Numpy and pinning helpers should be served from the Cython tensor API layer."""

    def test_numpy_and_pinning_methods_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "_numpy_view",
            "numpy",
            "pin_memory",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_numpy_and_pinning_methods_preserve_behavior(self):
        import candle as torch

        x = torch.arange(6, dtype=torch.float32).reshape(2, 3).transpose(0, 1)
        view = x._numpy_view()
        assert view.shape == (3, 2)
        assert view.strides == (4, 12)
        assert view.tolist() == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]

        arr = x.numpy()
        assert arr.shape == (3, 2)
        assert arr.strides == (4, 12)
        assert arr.tolist() == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]

        p = torch.ones((2, 2), dtype=torch.float32).pin_memory()


class TestTensorFactoryHelperMethodProviders:
    """Factory/helper Tensor methods should be served from the Cython tensor API layer."""

    def test_factory_helper_methods_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "new_empty",
            "new_tensor",
            "new_empty_strided",
            "_ones_like",
            "new_ones",
            "new_zeros",
            "new_full",
            "type",
            "type_as",
            "reshape_as",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_factory_helper_methods_preserve_behavior(self):
        import candle as torch

        base = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        new_empty = base.new_empty((3, 2))
        assert new_empty.shape == (3, 2)
        assert new_empty.dtype == torch.float32
        assert new_empty.device.type == "cpu"

        new_tensor = base.new_tensor([1, 2])
        assert new_tensor.shape == (2,)
        assert new_tensor.dtype == torch.float32
        assert new_tensor.device.type == "cpu"

        new_empty_strided = base.new_empty_strided((2, 3), (1, 2))
        assert new_empty_strided.shape == (2, 3)
        assert new_empty_strided.stride() == (1, 2)

        ones_like = base.transpose(0, 1)._ones_like()
        assert ones_like.shape == (3, 2)
        assert ones_like.dtype == torch.float32
        assert ones_like.device.type == "cpu"
        assert ones_like.tolist() == [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]

        assert base.new_ones((2, 2)).tolist() == [[1.0, 1.0], [1.0, 1.0]]
        assert base.new_zeros((2, 2)).tolist() == [[0.0, 0.0], [0.0, 0.0]]
        assert base.new_full((2, 2), 7.0).tolist() == [[7.0, 7.0], [7.0, 7.0]]

        assert base.type() == "torch.Float32Tensor"
        assert base.type(torch.float64).dtype == torch.float64




class TestTensorRuntimeHelperMethodProviders:
    """Runtime/internal helper methods should be served from the Cython tensor API layer."""

    def test_runtime_helper_methods_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "_set_device_from_storage",
            "_set_dtype_from_storage",
            "__delattr__",
            "_fw_get",
            "_fw_set",
            "_fw_clear",
            "_fw_has",
            "untyped_storage",
            "record_stream",
            "is_pinned",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_runtime_helper_methods_preserve_behavior(self):
        import candle as torch
        from candle._device import device
        from candle._dtype import float64

        x = torch.ones((2, 2), dtype=torch.float32)
        before = (x._device_type, x._device_index, x._dispatch_keys)
        x._set_device_from_storage(device("cpu"))
        after = (x._device_type, x._device_index, x._dispatch_keys)
        assert after == before

        x._set_dtype_from_storage(float64)
        assert x._dtype_code == 2
        assert x._itemsize == 8
        assert x._dtype_obj == float64

        x._fw_set(1, "tangent")
        assert x._fw_has(1) is True
        assert x._fw_get(1) == "tangent"
        x._fw_clear(1)
        assert x._fw_has(1) is False
        assert x._fw_get(1) is None

        assert hasattr(x.untyped_storage(), "nbytes")
        assert x.is_pinned() is False

        x.grad = torch.ones((2, 2), dtype=torch.float32)
        del x.grad
        assert x.grad is None

class TestTensorPutMethodProvider:
    """put_ should be served from the Cython tensor API layer."""

    def test_put_is_bound_from_tensor_api(self):
        import candle as torch
        assert torch.Tensor.put_.__module__ == "candle._cython._tensor_api"

    def test_put_preserves_behavior(self):
        import candle as torch

        x = torch.zeros((2, 3), dtype=torch.float32)
        idx = torch.tensor([0, 4], dtype=torch.int64)
        vals = torch.tensor([5.0, 7.0], dtype=torch.float32)
        x.put_(idx, vals)
        assert x.tolist() == [[5.0, 0.0, 0.0], [0.0, 7.0, 0.0]]

        y = torch.ones((2, 3), dtype=torch.float32)
        idx2 = torch.tensor([0, 0, 5], dtype=torch.int64)
        vals2 = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
        y.put_(idx2, vals2, accumulate=True)
        assert y.tolist() == [[6.0, 1.0, 1.0], [1.0, 1.0, 5.0]]

        z = torch.arange(6, dtype=torch.float32).reshape(2, 3).transpose(0, 1)
        idx3 = torch.tensor([0, 5], dtype=torch.int64)
        vals3 = torch.tensor([9.0, 8.0], dtype=torch.float32)
        z.put_(idx3, vals3)
        assert z.shape == (3, 2)
        assert z.stride() == (2, 1)


class TestTensorDataSetterProvider:
    """Tensor.data setter should be served from the Cython tensor API layer."""

    def test_data_setter_preserves_behavior(self):
        import candle as torch

        assert torch.Tensor.data.fset.__module__ == "candle._cython._tensor_api"

        x = torch.ones((2, 3), dtype=torch.float32)
        y = torch.zeros((2, 3), dtype=torch.float32)
        x.data = y
        assert x.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        assert x.stride() == (3, 1)
        assert x.offset == 0
        assert x._version_counter.value == 1

        try:
            x.data = torch.ones((3, 2), dtype=torch.float32)
            assert False, "expected shape mismatch"
        except RuntimeError as e:
            assert "shape mismatch" in str(e)

        try:
            x.data = torch.ones((2, 3), dtype=torch.float64)
            assert False, "expected dtype mismatch"
        except RuntimeError as e:
            assert "dtype mismatch" in str(e)


class TestTensorMiscHelperMethodProviders:
    """Selected medium-complexity helpers should be served from the Cython tensor API layer."""

    def test_misc_helper_methods_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "div_",
            "bitwise_and_",
            "bitwise_or_",
            "bitwise_xor_",
            "unflatten",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_misc_helper_methods_preserve_behavior(self):
        import candle as torch

        x = torch.tensor([8.0, 4.0], dtype=torch.float32)
        x.div_(2.0)
        assert x.tolist() == [4.0, 2.0]

        a = torch.tensor([1, 2, 3], dtype=torch.int32)
        a.bitwise_and_(torch.tensor([3, 1, 1], dtype=torch.int32))
        assert a.tolist() == [1, 0, 1]

        b = torch.tensor([1, 2, 3], dtype=torch.int32)
        b.bitwise_or_(torch.tensor([4, 1, 0], dtype=torch.int32))
        assert b.tolist() == [5, 3, 3]

        c = torch.tensor([1, 2, 3], dtype=torch.int32)
        c.bitwise_xor_(torch.tensor([1, 3, 1], dtype=torch.int32))
        assert c.tolist() == [0, 1, 2]

        d = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        u = d.unflatten(1, (2, 2))
        assert u.shape == (3, 2, 2)


class TestTensorWritebackConvenienceMethodProviders:
    """Non-inplace writeback convenience wrappers should be served from the Cython tensor API layer."""

    def test_writeback_convenience_methods_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "scatter_add",
            "index_fill",
            "index_copy",
            "index_add",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_writeback_convenience_methods_preserve_behavior(self):
        import candle as torch

        index = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int64)
        src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)

        base = torch.zeros((2, 3), dtype=torch.float32)
        out = base.scatter_add(1, index, src)
        assert out.tolist() == [[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]]
        assert base.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        iff = torch.zeros((2, 3), dtype=torch.float32)
        out2 = iff.index_fill(1, torch.tensor([1]), 7.0)
        assert out2.tolist() == [[0.0, 7.0, 0.0], [0.0, 7.0, 0.0]]
        assert iff.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        ic = torch.zeros((2, 3), dtype=torch.float32)
        out3 = ic.index_copy(1, torch.tensor([0, 2]), torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
        assert out3.tolist() == [[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]]
        assert ic.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        ia = torch.zeros((2, 3), dtype=torch.float32)
        out4 = ia.index_add(1, torch.tensor([0, 2]), torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32), alpha=1)
        assert out4.tolist() == [[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]]


class TestTensorShapeReductionWrapperProviders:
    """Selected shape/reduction wrappers should be served from the Cython tensor API layer."""

    def test_shape_reduction_wrappers_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "permute",
            "mean",
            "std",
            "repeat",
            "tile",
            "flip",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_shape_reduction_wrappers_preserve_behavior(self):
        import candle as torch

        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        assert x.permute(1, 0).shape == (3, 2)
        assert x.permute(1, 0).tolist() == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]

        assert x.mean().item() == 2.5
        assert x.mean(dim=1).tolist() == [1.0, 4.0]
        assert all(abs(v - ref) < 1e-6 for v, ref in zip(x.std(dim=1, unbiased=False).tolist(), [0.8164966106414795, 0.8164966106414795]))

        repeated = x.repeat(2, 1)
        assert repeated.shape == (4, 3)
        assert repeated.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]

        tiled = x.tile((2, 1))
        assert tiled.shape == (4, 3)
        assert tiled.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]

        assert x.flip([1]).tolist() == [[2.0, 1.0, 0.0], [5.0, 4.0, 3.0]]


class TestTensorLogicalBitwiseWrapperProviders:
    """Logical and bitwise wrappers should be served from the Cython tensor API layer."""

    def test_logical_bitwise_wrappers_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "logical_and",
            "logical_or",
            "logical_xor",
            "logical_not",
            "bitwise_and",
            "bitwise_or",
            "bitwise_xor",
            "bitwise_not",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_logical_bitwise_wrappers_preserve_behavior(self):
        import candle as torch

        a = torch.tensor([True, False, True], dtype=torch.bool)
        b = torch.tensor([True, True, False], dtype=torch.bool)
        assert a.logical_and(b).tolist() == [True, False, False]
        assert a.logical_or(b).tolist() == [True, True, True]
        assert a.logical_xor(b).tolist() == [False, True, True]
        assert a.logical_not().tolist() == [False, True, False]

        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        y = torch.tensor([3, 1, 1], dtype=torch.int32)
        assert x.bitwise_and(y).tolist() == [1, 0, 1]
        assert x.bitwise_or(y).tolist() == [3, 3, 3]
        assert x.bitwise_xor(y).tolist() == [2, 3, 2]
        assert x.bitwise_not().tolist() == [-2, -3, -4]


class TestTensorSplitIndexingWrapperProviders:
    """Split and indexed-selection wrappers should be served from the Cython tensor API layer."""

    def test_split_indexing_wrappers_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "vsplit",
            "hsplit",
            "dsplit",
            "take_along_dim",
            "cummin",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_split_indexing_wrappers_preserve_behavior(self):
        import candle as torch

        x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        assert [t.tolist() for t in x.vsplit(2)] == [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ]

        y = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        assert [t.tolist() for t in y.hsplit(2)] == [
            [[0.0, 1.0], [4.0, 5.0], [8.0, 9.0]],
            [[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]],
        ]

        z = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        assert [t.shape for t in z.dsplit(2)] == [(2, 3, 2), (2, 3, 2)]

        idx = torch.tensor([[3, 1], [0, 2]], dtype=torch.int64)
        a = torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]], dtype=torch.float32)
        assert a.take_along_dim(idx, 1).tolist() == [[40.0, 20.0], [50.0, 70.0]]

        b = torch.tensor([[3.0, 1.0, 2.0], [4.0, 0.0, 5.0]], dtype=torch.float32)
        vals, inds = b.cummin(1)
        assert vals.tolist() == [[3.0, 1.0, 1.0], [4.0, 0.0, 0.0]]
        assert inds.tolist() == [[0, 1, 1], [0, 1, 1]]


class TestTensorNumericHelperMethodProviders:
    """Selected numeric/statistical helpers should be served from the Cython tensor API layer."""

    def test_numeric_helper_methods_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "logsumexp",
            "trace",
            "det",
            "matrix_power",
            "dist",
            "renorm",
            "nansum",
            "nanmean",
            "argwhere",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_numeric_helper_methods_preserve_behavior(self):
        import candle as torch

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        assert all(abs(v - ref) < 1e-6 for v, ref in zip(x.logsumexp(dim=1).tolist(), [2.3132617473602295, 4.31326150894165]))
        assert x.trace().item() == 5.0
        assert x.det().item() == -2.0
        assert x.matrix_power(2).tolist() == [[7.0, 10.0], [15.0, 22.0]]
        assert abs(x.dist(torch.zeros((2, 2), dtype=torch.float32), p=2).item() - 5.4772257804870605) < 1e-6

        y = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float32)
        assert y.renorm(2, 0, 1.0).tolist() == [[1.0, 1.0], [0.0, 0.0]]

        z = torch.tensor([1.0, float('nan'), 3.0], dtype=torch.float32)
        assert z.nansum().item() == 4.0
        assert z.nanmean().item() == 2.0

        w = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)
        assert w.argwhere().tolist() == [[0, 1], [1, 0]]


class TestTensorViewSelectionWrapperProviders:
    """Selected view/selection wrappers should be served from the Cython tensor API layer."""

    def test_view_selection_wrappers_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "movedim",
            "diagonal",
            "unbind",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_view_selection_wrappers_preserve_behavior(self):
        import candle as torch

        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        moved = x.movedim(0, 2)
        assert moved.shape == (3, 4, 2)
        assert moved.tolist() == [
            [[0.0, 12.0], [1.0, 13.0], [2.0, 14.0], [3.0, 15.0]],
            [[4.0, 16.0], [5.0, 17.0], [6.0, 18.0], [7.0, 19.0]],
            [[8.0, 20.0], [9.0, 21.0], [10.0, 22.0], [11.0, 23.0]],
        ]

        y = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        diag = y.diagonal()
        assert diag.shape == (3,)
        assert diag.tolist() == [0.0, 4.0, 8.0]

        parts = x.unbind(1)
        assert len(parts) == 3
        assert [p.shape for p in parts] == [(2, 4), (2, 4), (2, 4)]
        assert parts[0].tolist() == [[0.0, 1.0, 2.0, 3.0], [12.0, 13.0, 14.0, 15.0]]
        assert parts[1].tolist() == [[4.0, 5.0, 6.0, 7.0], [16.0, 17.0, 18.0, 19.0]]
        assert parts[2].tolist() == [[8.0, 9.0, 10.0, 11.0], [20.0, 21.0, 22.0, 23.0]]

        assert x.shape == (2, 3, 4)
        assert x.tolist() == [
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
            [[12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
        ]


class TestTensorOperatorSugarAndBaddbmmProviders:
    """Operator sugar and baddbmm should be served from the Cython tensor API layer."""

    def test_operator_sugar_and_baddbmm_are_bound_from_tensor_api(self):
        import candle as torch

        expected = {
            "__isub__",
            "__itruediv__",
            "baddbmm",
        }
        actual = {
            name
            for name in expected
            if getattr(torch.Tensor, name).__module__ == "candle._cython._tensor_api"
        }
        assert actual == expected

    def test_operator_sugar_and_baddbmm_preserve_behavior(self):
        import candle as torch

        x = torch.tensor([5.0, 7.0], dtype=torch.float32)
        x -= 2.0
        assert x.tolist() == [3.0, 5.0]

        y = torch.tensor([8.0, 4.0], dtype=torch.float32)
        y /= 2.0
        assert y.tolist() == [4.0, 2.0]

        a = torch.ones((2, 2, 2), dtype=torch.float32)
        b = torch.ones((2, 2, 3), dtype=torch.float32)
        base = torch.zeros((2, 2, 3), dtype=torch.float32)
        out = base.baddbmm(a, b, beta=1, alpha=1)
        assert out.shape == (2, 2, 3)
        assert out.tolist() == [
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        ]
        assert base.tolist() == [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
