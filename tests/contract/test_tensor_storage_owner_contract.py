import candle as torch


def test_tensor_impl_declares_runtime_owned_fields():
    x = torch.tensor([1.0, 2.0])

    assert hasattr(x, "_storage")
    assert hasattr(x, "_device_obj")
    assert hasattr(x, "_dtype_obj")
    assert hasattr(x, "_version_value")
    assert hasattr(x, "_base")
    assert hasattr(x, "_view_meta")
    assert hasattr(x, "_dispatch_keys")


def test_tensor_python_shell_still_exposes_public_storage_api():
    x = torch.tensor([1.0, 2.0])

    assert x.storage() is not None
    assert x.untyped_storage() is not None


def test_tensor_data_setter_still_routes_through_runtime_storage_swap():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])

    before = x._version_counter.value
    x.data = y

    assert x.tolist() == [3.0, 4.0]
    assert x._version_counter.value == before + 1


def test_tensor_data_setter_preserves_source_storage_stride_and_offset_truth():
    x = torch.tensor([1.0, 2.0])
    z = torch.tensor([9.0, 3.0, 4.0])
    y = z[1:]

    x.data = y

    assert x.storage().data_ptr() == y.storage().data_ptr()
    assert x.stride == y.stride
    assert x.offset == y.offset
    assert x.tolist() == [3.0, 4.0]


def test_tensor_data_setter_preserves_runtime_device_dtype_caches():
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([3.0, 4.0], dtype=torch.float32)

    x.data = y

    assert x.device == y.device
    assert x.dtype == y.dtype
    assert x.device.type == y.device.type == "cpu"
    assert x.dtype == y.dtype == torch.float32


def test_tensor_shell_device_dtype_helpers_preserve_runtime_cache_values():
    x = torch.tensor([1.0, 2.0])
    before_device = x.device
    before_dtype = x.dtype
    x._set_device_from_storage(before_device)
    x._set_dtype_from_storage(before_dtype)
    assert x.device == before_device
    assert x.dtype == before_dtype


def test_typed_storage_public_api_routes_through_untyped_runtime_owner():
    x = torch.tensor([1.0, 2.0])
    storage = x.storage()
    untyped = storage.untyped_storage()

    assert storage.data_ptr() == untyped.data_ptr()
    assert storage.device == untyped.device
