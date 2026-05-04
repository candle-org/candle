"""Public shell for autograd anomaly mode.

The runtime owner is ``candle._C._autograd_engine``; this module is only a
thin re-export of the user-facing context managers and helpers so existing
``candle.autograd.anomaly_mode.detect_anomaly`` etc. callers keep working.
"""

from .._C._autograd_engine import (  # pylint: disable=import-error,no-name-in-module
    _AnomalyConfig,
    annotate_node_creation,
    detect_anomaly,
    evaluating_node,
    is_anomaly_check_nan_enabled,
    is_anomaly_enabled,
    pop_anomaly_config,
    pop_evaluating_node,
    push_anomaly_config,
    push_evaluating_node,
    report_anomaly,
    set_detect_anomaly,
)


__all__ = [
    "_AnomalyConfig",
    "annotate_node_creation",
    "detect_anomaly",
    "evaluating_node",
    "is_anomaly_check_nan_enabled",
    "is_anomaly_enabled",
    "pop_anomaly_config",
    "pop_evaluating_node",
    "push_anomaly_config",
    "push_evaluating_node",
    "report_anomaly",
    "set_detect_anomaly",
]
