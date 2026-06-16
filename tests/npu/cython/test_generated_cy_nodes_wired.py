"""Phase 1: prove generated Cython backward nodes are wired into the engine.

Before this phase, _variable_type_cy.pyx imports the PYTHON functions.py as _F,
so the engine only ever sees Python node classes. After the fix, it imports the
compiled _functions_cy module, and a backward node's __module__ is
'candle._generated._functions_cy'.
"""
import numpy as np


def test_generated_backward_node_runs_in_cython_module(npu_device):
    """A simple op (no hand-written NPU fast node) must attach a node whose
    class lives in the compiled _functions_cy module, not the Python functions
    module."""
    import candle as torch

    x = torch.ones(2, 2, device=npu_device, dtype=torch.float32, requires_grad=True)
    # abs has no hand-written NPU fast node, so it uses the generated node.
    y = x.abs()
    assert y.requires_grad
    assert y.grad_fn is not None
    module = type(y.grad_fn).__module__
    assert module == "candle._generated._functions_cy", (
        f"expected generated Cython node, got {module!r} "
        "(engine is still using Python functions.py nodes)"
    )


def test_generated_backward_node_correctness(npu_device):
    """Backward through a generated Cython node must produce correct gradients."""
    import candle as torch

    x = torch.ones(2, 2, device=npu_device, dtype=torch.float32, requires_grad=True)
    y = x.abs()
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.cpu().numpy(), np.ones((2, 2)), rtol=1e-6)


def test_backward_has_backward_method_not_apply(npu_device):
    """The generated node class must resolve `backward` (what the engine calls),
    not `apply`. For nodes whose formula cannot compile in the .pyx, the class
    subclasses the correct Python functions.py node and inherits backward."""
    import candle as torch

    x = torch.ones(2, 2, device=npu_device, dtype=torch.float32, requires_grad=True)
    y = x.abs()
    node_cls = type(y.grad_fn)
    assert hasattr(node_cls, "backward"), (
        f"{node_cls.__name__} must resolve backward() (own or inherited)"
    )
    assert not hasattr(node_cls, "apply") or "apply" not in node_cls.__dict__, (
        f"{node_cls.__name__} should not define apply() in its own __dict__"
    )
