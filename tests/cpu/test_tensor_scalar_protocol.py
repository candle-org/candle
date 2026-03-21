import os
import subprocess
import sys
import textwrap

import candle as torch
import pytest


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")


def test_tensor_module_fails_without_compiled_extension():
    env = os.environ.copy()
    python_path = _SRC_DIR
    existing = env.get("PYTHONPATH", "")
    if existing:
        python_path = python_path + os.pathsep + existing
    env["PYTHONPATH"] = python_path

    code = textwrap.dedent("""\
        import importlib
        import sys

        import candle
        import candle._cython as cython_pkg

        sys.modules.pop("candle._tensor", None)
        sys.modules["candle._cython._tensor_impl"] = None
        if hasattr(cython_pkg, "_tensor_impl"):
            delattr(cython_pkg, "_tensor_impl")
        if hasattr(candle, "_tensor"):
            delattr(candle, "_tensor")

        importlib.import_module("candle._tensor")
    """)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    assert result.returncode != 0
    assert "candle._cython._tensor_impl" in result.stderr


def test_tensor_item_int_float_bool_tolist_cpu():
    x = torch.tensor([3.5])
    assert x.item() == 3.5
    assert float(x) == 3.5
    assert int(torch.tensor([3])) == 3
    assert bool(torch.tensor([1])) is True
    assert torch.tensor([[1, 2], [3, 4]]).tolist() == [[1, 2], [3, 4]]


def test_tensor_bool_errors_match_current_behavior():
    with pytest.raises(RuntimeError, match="no values is ambiguous"):
        bool(torch.tensor([]))
    with pytest.raises(RuntimeError, match="more than one value is ambiguous"):
        bool(torch.tensor([1, 2]))


def test_tensor_len_iter_hash_and_properties():
    x = torch.tensor([[1, 2], [3, 4]])
    assert len(x) == 2
    rows = list(iter(x))
    assert len(rows) == 2
    assert rows[0].tolist() == [1, 2]
    assert isinstance(hash(x), int)
    assert x.output_nr == 0
    assert x.is_cpu is True
    assert x.is_cuda is False
    assert x.is_npu is False
    assert x.is_meta is False
    assert x.is_leaf is True
    assert x.is_sparse is False
    assert x.layout == "strided"
    assert x.is_quantized is False
    assert x.storage_offset() == 0
    assert x.get_device() == -1
    assert x.size() == (2, 2)
    assert x.size(0) == 2
    assert x.ndimension() == 2
    assert x.nelement() == 4


def test_tensor_zero_dim_len_and_iter_errors():
    x = torch.tensor(3)
    with pytest.raises(TypeError, match=r"len\(\) of a 0-d tensor"):
        len(x)
    with pytest.raises(TypeError, match="iteration over a 0-d tensor"):
        list(x)


def test_tensor_dtype_shorthand_wrappers():
    x = torch.tensor([1, 2, 3], dtype=torch.int32)
    assert x.float().dtype == torch.float32
    assert x.double().dtype == torch.float64
    assert x.long().dtype == torch.int64
    assert x.byte().dtype == torch.uint8
    assert x.bool().dtype == torch.bool


def test_tensor_comparison_dunders_still_return_tensor():
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([1, 0, 4])
    assert x.eq(y).tolist() == (x == y).tolist()
    assert (x > 1).tolist() == [False, True, True]
