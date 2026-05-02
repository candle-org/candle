import contextlib
import warnings

from .._C._autograd_engine import (  # pylint: disable=import-error,no-name-in-module
    annotate_node_creation,
    is_anomaly_check_nan_enabled,
    is_anomaly_enabled,
    pop_anomaly_config,
    pop_evaluating_node,
    push_anomaly_config,
    push_evaluating_node,
    report_anomaly,
)


_ENABLE_WARNING = (
    "Anomaly Detection has been enabled. This mode will increase the runtime "
    "and should only be enabled for debugging."
)


class _AnomalyConfig:
    __slots__ = ("enabled", "check_nan")

    def __init__(self, enabled, check_nan):
        self.enabled = bool(enabled)
        self.check_nan = bool(check_nan)


@contextlib.contextmanager
def detect_anomaly(check_nan=True):
    warnings.warn(_ENABLE_WARNING, UserWarning)
    push_anomaly_config(_AnomalyConfig(True, check_nan))
    try:
        yield
    finally:
        pop_anomaly_config()


@contextlib.contextmanager
def set_detect_anomaly(mode, check_nan=True):
    if mode:
        warnings.warn(_ENABLE_WARNING, UserWarning)
    push_anomaly_config(_AnomalyConfig(mode, check_nan))
    try:
        yield
    finally:
        pop_anomaly_config()


@contextlib.contextmanager
def evaluating_node(node):
    push_evaluating_node(node)
    try:
        yield
    finally:
        pop_evaluating_node()
