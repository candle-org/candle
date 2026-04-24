import tempfile

import candle as torch
import candle._storage as candle_storage
import torch as pt
from .helpers import assert_torch_error



def test_storage_module_does_not_expose_shared_file_bookkeeping_helpers():
    helper_names = {
        "_register_shared_file",
        "_unregister_shared_file",
        "_cleanup_shared_file",
        "_close_fd_and_cleanup",
        "cleanup_shared_files",
        "shared_files_count",
    }

    exported = {name for name in helper_names if hasattr(candle_storage, name)}

    assert exported == set()



def test_storage_module_still_exposes_public_storage_entry_points():
    public_names = {
        "TypedStorage",
        "UntypedStorage",
        "typed_storage_from_numpy",
        "empty_cpu_typed_storage",
        "meta_typed_storage_from_shape",
    }

    exported = {name for name in public_names if hasattr(candle_storage, name)}

    assert exported == public_names



def test_storage_module_does_not_expose_private_untyped_storage_classes():
    import candle._storage as storage_mod

    assert not hasattr(storage_mod, "_PinnedCPUUntypedStorage")
    assert not hasattr(storage_mod, "_CPUUntypedStorage")



def test_multiprocessing_storage_bookkeeping_surface_still_exists():
    import candle.multiprocessing as mt_mp

    assert hasattr(mt_mp, "cleanup_shared_files")
    assert hasattr(mt_mp, "shared_files_count")



def test_storage_module_exposes_legacy_typed_storage_names():
    for name in {"FloatStorage", "DoubleStorage", "HalfStorage", "LongStorage", "IntStorage", "ByteStorage", "BoolStorage"}:
        assert hasattr(candle_storage, name)


def test_storage_resize_file_backed_error():
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(b"\x00" * 8)
        tmp.flush()
        x = torch.UntypedStorage.from_file(tmp.name, shared=False)
        px = pt.UntypedStorage.from_file(tmp.name, shared=False)

        def mt():
            x.resize_(0)

        def th():
            px.resize_(0)

        assert_torch_error(mt, th)
