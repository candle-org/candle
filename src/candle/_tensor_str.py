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
    from ._dtype import complex64, float32, int64
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
    if (self.dtype not in (float32, complex64, int64)) or self.device.type == "meta":
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


def _format_real_scalar(value, options):
    import numpy as np

    if options.sci_mode is True:
        return np.format_float_scientific(value, precision=options.precision, unique=False)
    if options.sci_mode is False:
        return f"{value:.{options.precision}f}"
    out = np.format_float_positional(
        value,
        precision=options.precision,
        unique=False,
        fractional=False,
        trim='-'
    )
    if "e" not in out and "." not in out:
        out += "."
    return out


def _format_complex_scalar(value, options, real_width):
    real_str = _format_real_scalar(value.real, options).rjust(real_width)
    imag_str = f"{abs(value.imag):.{options.precision}f}j"
    if value.imag >= 0:
        return f"{real_str}+{imag_str}"
    return f"{real_str}-{imag_str}"


def _format_complex_array(arr, options, real_width):
    import numpy as np

    if arr.ndim == 0:
        return _format_complex_scalar(arr.item(), options, real_width)

    if arr.ndim == 1:
        items = [_format_complex_scalar(v, options, real_width) for v in arr.tolist()]
        return "[" + ", ".join(items) + "]"

    slices = [_format_complex_array(np.asarray(arr[i]), options, real_width) for i in range(arr.shape[0])]
    joiner = ",\n" + " " * 8
    return "[" + joiner.join(slices) + "]"


def _format_array(arr, options):
    import numpy as np

    if np.iscomplexobj(arr):
        flat = np.asarray(arr).reshape(-1)
        real_width = max(len(_format_real_scalar(v.real, options)) for v in flat) if flat.size else 1
        return _format_complex_array(np.asarray(arr), options, real_width)

    formatter = None
    floatmode = "maxprec_equal"
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
        result = np.array2string(arr, max_line_width=options.linewidth, **kwargs)
    except TypeError:
        result = np.array2string(arr, linewidth=options.linewidth, **kwargs)
    # Torch uses double-space before ellipsis; match its summarization separator
    result = result.replace(", ...", ",  ...")
    return result
