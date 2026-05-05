"""Contract: 5 view ops (flatten, unflatten, squeeze[.dim/.dims], movedim, narrow)
gradient flow goes through engine view-rebase, NOT through hand-added Backward
classes. Regression guard against accidental re-emission by the autograd
generator (tasks #223-#225).
"""
import importlib


REMOVED_BACKWARD_CLASSES = (
    "SqueezeBackward0",
    "SqueezeDimBackward0",
    "SqueezeDimsBackward0",
    "FlattenBackward0",
    "UnflattenBackward0",
    "NarrowBackward0",
    "MovedimBackward0",
)

REMOVED_AUTOGRAD_WRAPPERS = (
    "squeeze_autograd",
    "squeeze_dim_autograd",
    "squeeze_dims_autograd",
    "squeeze__autograd",
    "squeeze__dim_autograd",
    "squeeze__dims_autograd",
    "flatten_autograd",
    "unflatten_autograd",
    "movedim_autograd",
    "narrow_autograd",
)


def test_view_op_backward_classes_are_not_emitted():
    functions = importlib.import_module("candle._generated.functions")
    for name in REMOVED_BACKWARD_CLASSES:
        assert not hasattr(functions, name), (
            f"{name} was deleted in 1B-B because view rebase owns its gradient. "
            f"If the generator re-emitted it, teach the generator to skip "
            f"view-tracked ops."
        )


def test_view_op_autograd_wrappers_are_not_emitted():
    variable_type = importlib.import_module("candle._generated.variable_type")
    for name in REMOVED_AUTOGRAD_WRAPPERS:
        assert not hasattr(variable_type, name), (
            f"{name} was deleted in 1B-B. View rebase owns this op's gradient; "
            f"a wrapper is unnecessary."
        )


def test_view_op_dispatch_registrations_are_absent():
    """The dispatcher registry should have no autograd kernel for these 5 ops.
    Falling through to the default no-autograd kernel is what makes the engine
    rebase block fire on the leaf view.
    """
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey

    for op in ("squeeze", "flatten", "unflatten", "movedim", "narrow"):
        canonical = registry._canonical_name(op)
        entry = registry._ops.get(canonical)
        if entry is None:
            continue  # op not in registry at all is fine
        for key in (DispatchKey.Autograd, DispatchKey.AutogradCPU):
            assert key not in entry.kernels, (
                f"{op} should have NO {key} kernel after 1B-B; view rebase "
                f"owns gradient. Found kernel: {entry.kernels.get(key)}"
            )
