try:
    from .._cython._autograd_engine import (  # pylint: disable=import-error,no-name-in-module
        _GraphTask,
        _build_dependencies,
        _run_backward,
        backward,
        current_anomaly_parent,
        grad,
        is_anomaly_check_nan_enabled,
        is_anomaly_enabled,
        is_create_graph_enabled,
        pop_anomaly_config,
        pop_evaluating_node,
        push_anomaly_config,
        push_evaluating_node,
    )
except ImportError as exc:
    raise ImportError(
        "Failed to import candle._cython._autograd_engine. Build the required Cython "
        "runtime core with `python setup.py build_ext --inplace` or install with "
        "a build that includes the compiled extensions."
    ) from exc

__all__ = [
    "_GraphTask",
    "_build_dependencies",
    "_run_backward",
    "backward",
    "current_anomaly_parent",
    "grad",
    "is_anomaly_check_nan_enabled",
    "is_anomaly_enabled",
    "is_create_graph_enabled",
    "pop_anomaly_config",
    "pop_evaluating_node",
    "push_anomaly_config",
    "push_evaluating_node",
]
