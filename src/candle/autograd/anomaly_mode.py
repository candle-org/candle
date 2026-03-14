import contextlib
import sys
import threading
import traceback
import warnings
import weakref


_STATE = threading.local()
_ENABLE_WARNING = (
    "Anomaly Detection has been enabled. This mode will increase the runtime "
    "and should only be enabled for debugging."
)


class _AnomalyConfig:
    __slots__ = ("enabled", "check_nan")

    def __init__(self, enabled, check_nan):
        self.enabled = bool(enabled)
        self.check_nan = bool(check_nan)


def _config_stack():
    stack = getattr(_STATE, "config_stack", None)
    if stack is None:
        stack = []
        _STATE.config_stack = stack
    return stack


def _node_stack():
    stack = getattr(_STATE, "node_stack", None)
    if stack is None:
        stack = []
        _STATE.node_stack = stack
    return stack


def is_anomaly_enabled():
    stack = _config_stack()
    return bool(stack and stack[-1].enabled)


def is_anomaly_check_nan_enabled():
    stack = _config_stack()
    return bool(stack and stack[-1].enabled and stack[-1].check_nan)


@contextlib.contextmanager
def detect_anomaly(check_nan=True):
    warnings.warn(_ENABLE_WARNING, UserWarning)
    stack = _config_stack()
    stack.append(_AnomalyConfig(True, check_nan))
    try:
        yield
    finally:
        stack.pop()


@contextlib.contextmanager
def set_detect_anomaly(mode, check_nan=True):
    stack = _config_stack()
    if mode:
        warnings.warn(_ENABLE_WARNING, UserWarning)
    stack.append(_AnomalyConfig(mode, check_nan))
    try:
        yield
    finally:
        stack.pop()


def current_anomaly_parent():
    stack = _node_stack()
    if not stack:
        return None
    return stack[-1]


@contextlib.contextmanager
def evaluating_node(node):
    stack = _node_stack()
    stack.append(node)
    try:
        yield
    finally:
        stack.pop()


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
