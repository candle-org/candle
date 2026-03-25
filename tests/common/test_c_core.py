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
        s = StorageImpl.from_device_ptr(ptr, 32, 1, 0, owner=arr)
        assert s.data_ptr() == ptr
        assert s.nbytes() == 32
        assert s.device_type() == 1
        assert s.device_index() == 0

    def test_resizable_flag(self):
        from candle._cython._storage_impl import StorageImpl
        s1 = StorageImpl.alloc_cpu(64)
        assert s1.resizable() == True
        arr = np.zeros(4, dtype=np.float32)
        s2 = StorageImpl.from_numpy(arr)
        assert s2.resizable() == False
