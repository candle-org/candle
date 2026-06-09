from . import symbolic_opset11
from . import symbolic_helper


def register_custom_op_symbolic(*_args, **_kwargs):
    """Register an ONNX symbolic function.

    Candle does not export ONNX graphs yet, but third-party libraries such as
    torchvision call this during import to register optional symbolic handlers.
    Match torch's import-time API shape by accepting the registration without
    affecting eager execution.
    """
    return None
