from . import symbolic_opset11
from . import symbolic_helper


def register_custom_op_symbolic(*_args, **_kwargs):
    raise NotImplementedError(
        "onnx custom op registration is outside 0.1 NPU-first scope"
    )
