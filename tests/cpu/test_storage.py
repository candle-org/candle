import candle as torch


def test_typed_untyped_basic():
    t = torch.tensor([1.0, 2.0])
    st = t.storage()
    ust = t.untyped_storage()
    assert st.dtype.name == "float32"
    assert st.nbytes() == 8
    assert ust.nbytes() == st.nbytes()


def test_typed_storage_len_matches_size():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    assert len(storage) == storage.size() == 2


def test_untyped_storage_len_matches_nbytes():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    assert len(storage) == storage.nbytes() == 8


def test_untyped_storage_iteration_exposes_raw_bytes():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    assert list(storage) == [0, 0, 128, 63, 0, 0, 0, 64]


def test_untyped_storage_indexing_exposes_raw_bytes():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    assert storage[0] == 0
    assert storage[7] == 64


def test_typed_storage_setitem_writes_back_like_tensor():
    tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
    storage = tensor.storage()
    storage[0] = 9.0
    assert tensor.tolist() == [9.0, 2.0]


def test_typed_storage_slice_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[:]
    except RuntimeError as exc:
        assert "slices are only supported in UntypedStorage" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for TypedStorage slice access")


def test_typed_storage_full_slice_setitem_matches_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    storage[:] = 1.0
    assert list(storage) == [1.0, 1.0]


def test_typed_storage_step_slice_setitem_is_noop_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    before = list(storage)
    storage[::2] = 1.0
    assert list(storage) == before


def test_untyped_storage_setitem_writes_back_raw_bytes():
    tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
    storage = tensor.untyped_storage()
    storage[0] = 255
    assert list(storage) == [255, 0, 128, 63, 0, 0, 0, 64]


def test_untyped_storage_oob_setitem_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    try:
        storage[99] = 1
    except RuntimeError as exc:
        assert str(exc) == "out of bounds"
    else:
        raise AssertionError("expected RuntimeError for OOB untyped storage write")


def test_untyped_storage_slice_returns_storage_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    sliced = storage[:]
    assert isinstance(sliced, torch.UntypedStorage)
    assert list(sliced) == [0, 0, 128, 63, 0, 0, 0, 64]




def test_typed_storage_oob_setitem_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage[9] = 1.0
    except IndexError as exc:
        assert str(exc) == "index 9 is out of bounds for dimension 0 with size 2"
    else:
        raise AssertionError("expected IndexError for OOB typed storage write")


def test_typed_storage_negative_oob_setitem_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage[-3] = 1.0
    except IndexError as exc:
        assert str(exc) == "index -3 is out of bounds for dimension 0 with size 2"
    else:
        raise AssertionError("expected IndexError for negative OOB typed storage write")


def test_typed_storage_tensor_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[torch.tensor(0)]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'torch.Tensor'>"
    else:
        raise AssertionError("expected RuntimeError for tensor typed storage index")


def test_typed_storage_list_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[[]]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'list'>"
    else:
        raise AssertionError("expected RuntimeError for list typed storage index")


def test_typed_storage_tensor_setitem_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage[torch.tensor(0)] = 1.0
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'torch.Tensor'>"
    else:
        raise AssertionError("expected RuntimeError for tensor typed storage setitem")


def test_typed_storage_tuple_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[(0,)]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'tuple'>"
    else:
        raise AssertionError("expected RuntimeError for tuple typed storage index")


def test_typed_storage_tuple_setitem_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage[(0,)] = 1.0
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'tuple'>"
    else:
        raise AssertionError("expected RuntimeError for tuple typed storage setitem")


def test_typed_storage_numpy_array_index_raises_like_torch():
    import numpy as np

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[np.array(0)]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'numpy.ndarray'>"
    else:
        raise AssertionError("expected RuntimeError for numpy-array typed storage index")


def test_typed_storage_numpy_array_setitem_raises_like_torch():
    import numpy as np

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage[np.array(0)] = 1.0
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'numpy.ndarray'>"
    else:
        raise AssertionError("expected RuntimeError for numpy-array typed storage setitem")


def test_typed_storage_numpy_int64_index_raises_like_torch():
    import numpy as np

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[np.int64(0)]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'numpy.int64'>"
    else:
        raise AssertionError("expected RuntimeError for numpy-int64 typed storage index")


def test_typed_storage_numpy_int64_setitem_raises_like_torch():
    import numpy as np

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage[np.int64(0)] = 1.0
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'numpy.int64'>"
    else:
        raise AssertionError("expected RuntimeError for numpy-int64 typed storage setitem")


def test_typed_storage_none_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[None]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'NoneType'>"
    else:
        raise AssertionError("expected RuntimeError for None typed storage index")


def test_typed_storage_ellipsis_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[...]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'ellipsis'>"
    else:
        raise AssertionError("expected RuntimeError for ellipsis typed storage index")


def test_typed_storage_none_setitem_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage[None] = 1.0
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'NoneType'>"
    else:
        raise AssertionError("expected RuntimeError for None typed storage setitem")


def test_typed_storage_bool_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[True]
    except TypeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'bool'>"
    else:
        raise AssertionError("expected TypeError for bool typed storage index")


def test_typed_storage_float_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[0.0]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'float'>"
    else:
        raise AssertionError("expected RuntimeError for float typed storage index")


def test_typed_storage_float_setitem_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage[0.0] = 1.0
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'float'>"
    else:
        raise AssertionError("expected RuntimeError for float typed storage setitem")


def test_typed_storage_bool_setitem_matches_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    storage[True] = 9.0
    assert list(storage) == [9.0, 9.0]


def test_typed_storage_complex_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[0j]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'complex'>"
    else:
        raise AssertionError("expected RuntimeError for complex typed storage index")


def test_typed_storage_complex_setitem_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage[0j] = 1.0
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'complex'>"
    else:
        raise AssertionError("expected RuntimeError for complex typed storage setitem")


def test_typed_storage_str_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage["0"]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'str'>"
    else:
        raise AssertionError("expected RuntimeError for str typed storage index")


def test_typed_storage_str_setitem_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage["0"] = 1.0
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'str'>"
    else:
        raise AssertionError("expected RuntimeError for str typed storage setitem")


def test_typed_storage_numpy_float64_index_raises_like_torch():
    import numpy as np

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[np.float64(0.0)]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'numpy.float64'>"
    else:
        raise AssertionError("expected RuntimeError for numpy-float64 typed storage index")


def test_typed_storage_numpy_float64_setitem_raises_like_torch():
    import numpy as np

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage[np.float64(0.0)] = 1.0
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'numpy.float64'>"
    else:
        raise AssertionError("expected RuntimeError for numpy-float64 typed storage setitem")


def test_untyped_storage_bool_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    try:
        _ = storage[True]
    except TypeError as exc:
        assert str(exc) == "can't index a torch.UntypedStorage with bool"
    else:
        raise AssertionError("expected TypeError for bool untyped storage index")


def test_untyped_storage_float_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    try:
        _ = storage[0.0]
    except TypeError as exc:
        assert str(exc) == "can't index a torch.UntypedStorage with float"
    else:
        raise AssertionError("expected TypeError for float untyped storage index")


def test_untyped_storage_str_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    try:
        _ = storage["0"]
    except TypeError as exc:
        assert str(exc) == "can't index a torch.UntypedStorage with str"
    else:
        raise AssertionError("expected TypeError for str untyped storage index")


def test_untyped_storage_none_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    try:
        _ = storage[None]
    except TypeError as exc:
        assert str(exc) == "can't index a torch.UntypedStorage with NoneType"
    else:
        raise AssertionError("expected TypeError for None untyped storage index")


def test_untyped_storage_ellipsis_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    try:
        _ = storage[...]
    except TypeError as exc:
        assert str(exc) == "can't index a torch.UntypedStorage with ellipsis"
    else:
        raise AssertionError("expected TypeError for ellipsis untyped storage index")


def test_untyped_storage_numpy_bool_index_raises_like_torch():
    import numpy as np

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    try:
        _ = storage[np.bool_(True)]
    except TypeError as exc:
        assert str(exc) == "can't index a torch.UntypedStorage with numpy.bool_"
    else:
        raise AssertionError("expected TypeError for numpy.bool_ untyped storage index")


def test_untyped_storage_numpy_float64_index_raises_like_torch():
    import numpy as np

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    try:
        _ = storage[np.float64(0.0)]
    except TypeError as exc:
        assert str(exc) == "can't index a torch.UntypedStorage with numpy.float64"
    else:
        raise AssertionError("expected TypeError for numpy.float64 untyped storage index")


def test_untyped_storage_list_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    try:
        _ = storage[[0]]
    except TypeError as exc:
        assert str(exc) == "can't index a torch.UntypedStorage with list"
    else:
        raise AssertionError("expected TypeError for list untyped storage index")


def test_untyped_storage_tuple_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    try:
        _ = storage[(0,)]
    except TypeError as exc:
        assert str(exc) == "can't index a torch.UntypedStorage with tuple"
    else:
        raise AssertionError("expected TypeError for tuple untyped storage index")


def test_untyped_storage_tensor_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    try:
        _ = storage[torch.tensor(0)]
    except TypeError as exc:
        assert str(exc) == "can't index a torch.UntypedStorage with Tensor"
    else:
        raise AssertionError("expected TypeError for Tensor untyped storage index")


def test_untyped_storage_list_assignment_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    before = list(storage)
    try:
        storage[[0]] = 255
    except SystemError as exc:
        assert str(exc) == "error return without exception set"
        assert list(storage) == before
    else:
        raise AssertionError("expected SystemError for list untyped storage assignment")


def test_untyped_storage_tuple_assignment_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    before = list(storage)
    try:
        storage[(0,)] = 255
    except SystemError as exc:
        assert str(exc) == "error return without exception set"
        assert list(storage) == before
    else:
        raise AssertionError("expected SystemError for tuple untyped storage assignment")


def test_untyped_storage_none_assignment_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    before = list(storage)
    try:
        storage[None] = 255
    except SystemError as exc:
        assert str(exc) == "error return without exception set"
        assert list(storage) == before
    else:
        raise AssertionError("expected SystemError for None untyped storage assignment")


def test_untyped_storage_ellipsis_assignment_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    before = list(storage)
    try:
        storage[...] = 255
    except SystemError as exc:
        assert str(exc) == "error return without exception set"
        assert list(storage) == before
    else:
        raise AssertionError("expected SystemError for ellipsis untyped storage assignment")


def test_untyped_storage_numpy_bool_assignment_raises_like_torch():
    import numpy as np

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    before = list(storage)
    try:
        storage[np.bool_(True)] = 255
    except SystemError as exc:
        assert str(exc) == "error return without exception set"
        assert list(storage) == before
    else:
        raise AssertionError("expected SystemError for numpy.bool_ untyped storage assignment")


def test_untyped_storage_tensor_assignment_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    before = list(storage)
    try:
        storage[torch.tensor(0)] = 255
    except SystemError as exc:
        assert str(exc) == "error return without exception set"
        assert list(storage) == before
    else:
        raise AssertionError("expected SystemError for Tensor untyped storage assignment")


def test_untyped_storage_numpy_int64_setitem_writes_targeted_byte_like_torch():
    import numpy as np

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    storage[np.int64(0)] = 255
    assert list(storage) == [255, 0, 128, 63, 0, 0, 0, 64]


def test_untyped_storage_float_assignment_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    before = list(storage)
    try:
        storage[0.0] = 255
    except SystemError as exc:
        assert str(exc) == "error return without exception set"
        assert list(storage) == before
    else:
        raise AssertionError("expected SystemError for float untyped storage assignment")


def test_untyped_storage_numpy_int64_index_reads_targeted_byte_like_torch():
    import numpy as np

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    assert storage[np.int64(0)] == 0


def test_untyped_storage_numpy_int32_setitem_writes_targeted_byte_like_torch():
    import numpy as np

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    storage[np.int32(0)] = 255
    assert list(storage) == [255, 0, 128, 63, 0, 0, 0, 64]


def test_untyped_storage_bool_assignment_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    before = list(storage)
    try:
        storage[True] = 255
    except SystemError as exc:
        assert str(exc) == "error return without exception set"
        assert list(storage) == before
    else:
        raise AssertionError("expected SystemError for bool untyped storage assignment")


def test_untyped_storage_bytes_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    try:
        _ = storage[b"0"]
    except TypeError as exc:
        assert str(exc) == "can't index a torch.UntypedStorage with bytes"
    else:
        raise AssertionError("expected TypeError for bytes untyped storage index")


def test_untyped_storage_bytes_assignment_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    before = list(storage)
    try:
        storage[b"0"] = 255
    except SystemError as exc:
        assert str(exc) == "error return without exception set"
        assert list(storage) == before
    else:
        raise AssertionError("expected SystemError for bytes untyped storage assignment")


def test_typed_storage_bytes_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[b"0"]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'bytes'>"
    else:
        raise AssertionError("expected RuntimeError for bytes typed storage index")


def test_typed_storage_bytes_setitem_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage[b"0"] = 1.0
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'bytes'>"
    else:
        raise AssertionError("expected RuntimeError for bytes typed storage setitem")


def test_typed_storage_bytearray_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[bytearray(b"0")]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'bytearray'>"
    else:
        raise AssertionError("expected RuntimeError for bytearray typed storage index")


def test_typed_storage_bytearray_setitem_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage[bytearray(b"0")] = 1.0
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'bytearray'>"
    else:
        raise AssertionError("expected RuntimeError for bytearray typed storage setitem")


def test_untyped_storage_bytearray_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    try:
        _ = storage[bytearray(b"0")]
    except TypeError as exc:
        assert str(exc) == "can't index a torch.UntypedStorage with bytearray"
    else:
        raise AssertionError("expected TypeError for bytearray untyped storage index")


def test_untyped_storage_bytearray_assignment_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    before = list(storage)
    try:
        storage[bytearray(b"0")] = 255
    except SystemError as exc:
        assert str(exc) == "error return without exception set"
        assert list(storage) == before
    else:
        raise AssertionError("expected SystemError for bytearray untyped storage assignment")


def test_typed_storage_memoryview_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[memoryview(b"0")]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'memoryview'>"
    else:
        raise AssertionError("expected RuntimeError for memoryview typed storage index")


def test_typed_storage_memoryview_setitem_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage[memoryview(b"0")] = 1.0
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'memoryview'>"
    else:
        raise AssertionError("expected RuntimeError for memoryview typed storage setitem")


def test_typed_storage_range_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[range(1)]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'range'>"
    else:
        raise AssertionError("expected RuntimeError for range typed storage index")


def test_typed_storage_range_setitem_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage[range(1)] = 1.0
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'range'>"
    else:
        raise AssertionError("expected RuntimeError for range typed storage setitem")


def test_typed_storage_foreign_torch_tensor_index_raises_like_torch():
    import torch as pytorch

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        _ = storage[pytorch.tensor(0)]
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'torch.Tensor'>"
    else:
        raise AssertionError("expected RuntimeError for foreign torch Tensor typed storage index")


def test_typed_storage_foreign_torch_tensor_setitem_raises_like_torch():
    import torch as pytorch

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage()
    try:
        storage[pytorch.tensor(0)] = 1.0
    except RuntimeError as exc:
        assert str(exc) == "can't index a <class 'torch.storage.TypedStorage'> with <class 'torch.Tensor'>"
    else:
        raise AssertionError("expected RuntimeError for foreign torch Tensor typed storage setitem")


def test_untyped_storage_memoryview_index_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    try:
        _ = storage[memoryview(b"0")]
    except TypeError as exc:
        assert str(exc) == "can't index a torch.UntypedStorage with memoryview"
    else:
        raise AssertionError("expected TypeError for memoryview untyped storage index")


def test_untyped_storage_memoryview_assignment_raises_like_torch():
    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    before = list(storage)
    try:
        storage[memoryview(b"0")] = 255
    except SystemError as exc:
        assert str(exc) == "error return without exception set"
        assert list(storage) == before
    else:
        raise AssertionError("expected SystemError for memoryview untyped storage assignment")


def test_untyped_storage_numpy_array_index_raises_like_torch():
    import numpy as np

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    try:
        _ = storage[np.array(0)]
    except TypeError as exc:
        assert str(exc) == "can't index a torch.UntypedStorage with numpy.ndarray"
    else:
        raise AssertionError("expected TypeError for numpy.ndarray untyped storage index")


def test_untyped_storage_numpy_array_assignment_raises_like_torch():
    import numpy as np

    storage = torch.tensor([1.0, 2.0], dtype=torch.float32).untyped_storage()
    before = list(storage)
    try:
        storage[np.array(0)] = 255
    except SystemError as exc:
        assert str(exc) == "error return without exception set"
        assert list(storage) == before
    else:
        raise AssertionError("expected SystemError for numpy.ndarray untyped storage assignment")


def test_cpu_from_file(tmp_path):
    path = tmp_path / "storage.bin"
    path.write_bytes(b"\x00" * 16)
    ust = torch.UntypedStorage.from_file(str(path), shared=False)
    assert ust.nbytes() == 16
    assert ust.filename() == str(path)


def test_npu_storage_no_cpu_copy():
    if not torch.npu.is_available():
        return
    t = torch.tensor([1.0, 2.0], device="npu")
    st = t.storage()
    assert st.device.type == "npu"
    assert st.untyped_storage().data_ptr() != 0


def test_meta_storage_no_data_ptr():
    t = torch.tensor([1.0, 2.0], device="meta")
    try:
        _ = t.untyped_storage().data_ptr()
    except RuntimeError:
        pass
    else:
        raise AssertionError("meta storage should not expose data_ptr")


def test_pending_storage_basic():
    from candle._dtype import float32
    from candle._storage import PendingStorage

    storage = PendingStorage((2, 3), float32, "cpu")
    assert storage.nbytes() == 2 * 3 * 4
    try:
        storage.data_ptr()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected data_ptr to raise")
