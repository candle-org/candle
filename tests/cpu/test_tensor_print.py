import candle as torch
import torch as ref_torch


def test_tensor_repr_cpu_default_dtype():
    t = torch.ones((2, 2))
    rep = repr(t)
    assert rep.startswith("tensor(")
    assert "dtype=" not in rep
    assert "device=" not in rep
    assert rep == str(t)


def test_tensor_repr_cpu_non_default_dtype():
    t = torch.ones((2, 2), dtype=torch.float16)
    rep = repr(t)
    assert "dtype=torch.float16" in rep


def test_tensor_repr_meta_includes_device():
    t = torch.ones((2, 2), device="meta")
    rep = repr(t)
    assert rep.startswith("tensor(")
    assert "..." in rep
    assert "device='meta'" in rep
    assert "dtype=torch.float32" in rep


def test_tensor_repr_respects_precision():
    prev = torch.get_printoptions()
    try:
        torch.set_printoptions(precision=2)
        t = torch.tensor([1.23456])
        rep = repr(t)
        assert "1.23" in rep
    finally:
        torch.set_printoptions(**prev)


def test_tensor_repr_npu_includes_device():
    if not torch.npu.is_available():
        return
    t = torch.ones((1,))
    t = t.to("npu")
    rep = repr(t)
    assert "device='npu:0'" in rep
    assert "1" in rep


def test_tensor_repr_npu_index():
    if not torch.npu.is_available():
        return
    if torch._C._npu_device_count() < 2:
        return
    t = torch.ones((1,), device="npu:1")
    rep = repr(t)
    assert "device='npu:1'" in rep


def test_printoptions_state_lives_in_tensor_str_module():
    from candle import _tensor_str

    prev = torch.get_printoptions()
    try:
        torch.set_printoptions(precision=2, linewidth=70)
        opts = torch.get_printoptions()
        assert opts["precision"] == 2
        assert opts["linewidth"] == 70
        assert _tensor_str.get_printoptions()["precision"] == 2
        assert _tensor_str.get_printoptions()["linewidth"] == 70
    finally:
        torch.set_printoptions(**prev)


def test_printoptions_context_restores_state():
    prev = torch.get_printoptions()
    with torch.printoptions(precision=2, sci_mode=True):
        inside = torch.get_printoptions()
        assert inside["precision"] == 2
        assert inside["sci_mode"] is True
    assert torch.get_printoptions() == prev


def test_dense_vector_repr_matches_local_torch():
    c = torch.tensor([1.23456, 2.0, -3.5], dtype=torch.float32)
    r = ref_torch.tensor([1.23456, 2.0, -3.5], dtype=ref_torch.float32)
    assert repr(c) == repr(r)


def test_dense_matrix_repr_matches_local_torch():
    c = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    r = ref_torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ref_torch.float32)
    assert repr(c) == repr(r)


def test_complex_repr_matches_local_torch():
    c = torch.tensor([1 + 2j, -3 + 0.5j], dtype=torch.complex64)
    r = ref_torch.tensor([1 + 2j, -3 + 0.5j], dtype=ref_torch.complex64)
    assert repr(c) == repr(r)


def test_threshold_summarization_matches_local_torch():
    from torch._tensor_str import get_printoptions as ref_get, set_printoptions as ref_set

    prev_c = torch.get_printoptions()
    prev_r = ref_get()
    try:
        torch.set_printoptions(threshold=5)
        ref_set(threshold=5)
        c = torch.arange(10)
        r = ref_torch.arange(10)
        assert repr(c) == repr(r)
    finally:
        torch.set_printoptions(**prev_c)
        ref_set(**prev_r)


def test_linewidth_wrapping_matches_local_torch():
    from torch._tensor_str import get_printoptions as ref_get, set_printoptions as ref_set

    prev_c = torch.get_printoptions()
    prev_r = ref_get()
    try:
        torch.set_printoptions(linewidth=30)
        ref_set(linewidth=30)
        c = torch.arange(16).reshape(4, 4)
        r = ref_torch.arange(16).reshape(4, 4)
        assert repr(c) == repr(r)
    finally:
        torch.set_printoptions(**prev_c)
        ref_set(**prev_r)


def test_scientific_mode_matches_local_torch():
    from torch._tensor_str import get_printoptions as ref_get, set_printoptions as ref_set

    prev_c = torch.get_printoptions()
    prev_r = ref_get()
    try:
        torch.set_printoptions(sci_mode=True, precision=2)
        ref_set(sci_mode=True, precision=2)
        c = torch.tensor([1.0e-5, 2.0e6])
        r = ref_torch.tensor([1.0e-5, 2.0e6])
        assert repr(c) == repr(r)
    finally:
        torch.set_printoptions(**prev_c)
        ref_set(**prev_r)
