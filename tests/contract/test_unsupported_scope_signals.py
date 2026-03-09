import pytest

import candle


_MSG = "0.1 NPU-first scope"


def test_compile_raises_for_requested_backend_outside_scope():
    with pytest.raises(NotImplementedError, match=_MSG):
        candle.compile(lambda x: x, backend="inductor")


def test_jit_script_raises_with_scope_signal_when_requested():
    with pytest.raises(NotImplementedError, match=_MSG):
        candle.jit.script(lambda x: x, optimize=True)


def test_onnx_register_custom_op_symbolic_raises_scope_signal():
    with pytest.raises(NotImplementedError, match=_MSG):
        candle.onnx.register_custom_op_symbolic("::op", lambda *a: None, 11)
