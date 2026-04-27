# mypy: allow-untyped-defs
import contextlib
import dataclasses
import math
from typing import Any


@dataclasses.dataclass
class __PrinterOptions:
    precision: int = 4
    threshold: float = 1000
    edgeitems: int = 3
    linewidth: int = 80
    sci_mode: bool | None = None


PRINT_OPTS = __PrinterOptions()


def set_printoptions(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    profile=None,
    sci_mode=None,
):
    if profile is not None:
        if profile == "default":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
        elif profile == "short":
            PRINT_OPTS.precision = 2
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 2
            PRINT_OPTS.linewidth = 80
        elif profile == "full":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = math.inf
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
    if precision is not None:
        PRINT_OPTS.precision = precision
    if threshold is not None:
        PRINT_OPTS.threshold = threshold
    if edgeitems is not None:
        PRINT_OPTS.edgeitems = edgeitems
    if linewidth is not None:
        PRINT_OPTS.linewidth = linewidth
    PRINT_OPTS.sci_mode = sci_mode


def get_printoptions() -> dict[str, Any]:
    return dataclasses.asdict(PRINT_OPTS)


@contextlib.contextmanager
def printoptions(**kwargs):
    old_kwargs = get_printoptions()
    set_printoptions(**kwargs)
    try:
        yield
    finally:
        set_printoptions(**old_kwargs)


def _str(self, *, tensor_contents=None):
    from ._dtype import float32
    from ._device import _default_device

    if tensor_contents is None:
        if self.device.type == "meta":
            data_repr = "..."
        else:
            if self.device.type == "cpu":
                view = self._numpy_view()
            else:
                view = self.to("cpu")._numpy_view()
            data_repr = _format_array(view, PRINT_OPTS)
    else:
        data_repr = tensor_contents

    suffixes = []
    if self.dtype != float32 or self.device.type == "meta":
        suffixes.append(f"dtype={repr(self.dtype)}")
    if self.device.type != _default_device.type:
        device_label = self.device.type
        if self.device.type == "npu":
            device_label = str(self.device)
        suffixes.append(f"device='{device_label}'")
    if self.requires_grad:
        suffixes.append("requires_grad=True")
    if self.grad_fn is not None:
        suffixes.append(f"grad_fn=<{type(self.grad_fn).__name__}>")

    if suffixes:
        return f"tensor({data_repr}, {', '.join(suffixes)})"
    return f"tensor({data_repr})"


def _format_array(arr, options):
    import numpy as np

    formatter = None
    floatmode = None
    if options.sci_mode is True:
        def _format_float(value):
            return np.format_float_scientific(
                value, precision=options.precision, unique=False
            )
        formatter = {"float_kind": _format_float}
    elif options.sci_mode is False:
        floatmode = "fixed"

    kwargs = {
        "precision": options.precision,
        "threshold": options.threshold,
        "edgeitems": options.edgeitems,
        "separator": ", ",
        "prefix": "tensor(",
        "formatter": formatter,
        "floatmode": floatmode,
    }
    try:
        return np.array2string(arr, max_line_width=options.linewidth, **kwargs)
    except TypeError:
        return np.array2string(arr, linewidth=options.linewidth, **kwargs)
