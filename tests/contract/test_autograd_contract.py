import importlib
import sys

import candle as torch
import torch as pt
from .helpers import assert_torch_error


def test_inplace_view_version_error_message():
    def mt():
        x = torch.tensor([1.0], requires_grad=True)
        y = x.view((1,))
        y.add_(1.0)
        y.sum().backward()

    def th():
        x = pt.tensor([1.0], requires_grad=True)
        y = x.view((1,))
        y.add_(1.0)
        y.sum().backward()

    assert_torch_error(mt, th)


def test_autograd_runtime_exports_engine_entrypoints():
    """Contract: candle.autograd.engine must re-export ALL entrypoints that
    candle._cython exports from _autograd_engine as its own public module-level
    names.  The _cython layer provides a richer set of entrypoints
    (current_anomaly_parent, pop_anomaly_config, pop_evaluating_node,
    push_anomaly_config, push_evaluating_node, is_anomaly_enabled,
    is_anomaly_check_nan_enabled) that are NOT currently present in
    engine.__all__ — callers who want these must reach into _cython directly,
    which violates the layering contract.

    This test is expected RED until engine.py re-exports and declares every
    entrypoint that _cython._autograd_engine provides.
    """
    from candle.autograd import engine

    # Full set of names that _cython exports from _autograd_engine
    cython_engine_names = [
        "_GraphTask",
        "_build_dependencies",
        "_run_backward",
        "backward",
        "grad",
        "is_create_graph_enabled",
        # The following are exported by _cython.__init__ but NOT by engine.py
        "current_anomaly_parent",
        "is_anomaly_check_nan_enabled",
        "is_anomaly_enabled",
        "pop_anomaly_config",
        "pop_evaluating_node",
        "push_anomaly_config",
        "push_evaluating_node",
    ]

    assert hasattr(engine, "__all__"), (
        "candle.autograd.engine must define __all__ to make the public "
        "contract explicit and stable."
    )

    missing_from_all = [
        name for name in cython_engine_names if name not in engine.__all__
    ]
    assert not missing_from_all, (
        "candle.autograd.engine.__all__ is missing the following entrypoints "
        "that candle._cython._autograd_engine provides: "
        f"{missing_from_all!r}.  engine.py must re-export these so callers "
        "do not need to reach into candle._cython directly."
    )

    missing_as_attr = [
        name for name in cython_engine_names if not hasattr(engine, name)
    ]
    assert not missing_as_attr, (
        "candle.autograd.engine is missing the following names as actual "
        "module attributes (listing in __all__ is not sufficient): "
        f"{missing_as_attr!r}.  engine.py must import and re-export each name."
    )


def test_autograd_engine_import_failure_is_actionable(monkeypatch):
    """Contract: when candle._cython._autograd_engine is absent the import of
    candle.autograd.engine must raise an ImportError with an actionable message
    that names the missing extension and provides a build hint.  Currently the
    hard import in engine.py would surface a raw low-level ImportError from
    Cython with no user guidance.

    This test is expected RED until engine.py wraps its hard import with an
    actionable error message.
    """
    monkeypatch.setitem(sys.modules, "candle._cython._autograd_engine", None)
    # Also remove the already-loaded engine module so importlib reimports it.
    monkeypatch.delitem(sys.modules, "candle.autograd.engine", raising=False)

    try:
        importlib.import_module("candle.autograd.engine")
    except ImportError as exc:
        msg = str(exc)
        assert "_autograd_engine" in msg, (
            "ImportError must name the missing extension '_autograd_engine', "
            f"got: {msg!r}"
        )
        assert any(
            hint in msg
            for hint in ("build_ext", "pip install", "setup.py", "[cython]")
        ), (
            "ImportError must include a build hint so the user knows how to "
            f"recover, got: {msg!r}"
        )
        return

    raise AssertionError(
        "candle.autograd.engine must raise an actionable ImportError when "
        "_autograd_engine is absent, but no exception was raised."
    )
