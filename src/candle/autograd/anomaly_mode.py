import contextlib
import sys
import traceback
import warnings
import weakref

from .._cython._autograd_engine import (
    current_anomaly_parent,
    is_anomaly_check_nan_enabled,
    is_anomaly_enabled,
    pop_anomaly_config,
    pop_evaluating_node,
    push_anomaly_config,
    push_evaluating_node,
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


def annotate_node_creation(node):
    if not is_anomaly_enabled():
        return
    parent = current_anomaly_parent()
    node._anomaly_parent = None if parent is None else weakref.ref(parent)
    node._anomaly_trace = "".join(traceback.format_stack()[:-2])


def _warn_or_stderr(message):
    try:
        warnings.warn(message, UserWarning)
    except Warning as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)


def _node_trace_message(node, *, previous):
    trace = getattr(node, "_anomaly_trace", None)
    node_name = node.name()
    if previous:
        prefix = f"\nPrevious calculation was induced by {node_name}. "
        if trace:
            return prefix + "Traceback of forward call that induced the previous calculation:\n" + trace
        return prefix + (
            "No forward pass information available. Enable detect anomaly during forward pass for more information."
        )
    prefix = f"Error detected in {node_name}. "
    if trace:
        return prefix + "Traceback of forward call that caused the error:\n" + trace
    return prefix + (
        "No forward pass information available. Enable detect anomaly during forward pass for more information."
    )


def report_anomaly(node):
    if not is_anomaly_enabled():
        return
    seen = set()
    current = node
    previous = False
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        _warn_or_stderr(_node_trace_message(current, previous=previous))
        parent_ref = getattr(current, "_anomaly_parent", None)
        current = None if parent_ref is None else parent_ref()
        previous = True
