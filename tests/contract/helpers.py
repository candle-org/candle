import torch as pt
import numpy as np


def assert_torch_error(fn_mt, fn_torch):
    try:
        fn_torch()
    except Exception as exc_torch:
        torch_exc = exc_torch
    else:
        torch_exc = None

    try:
        fn_mt()
    except Exception as exc_mt:
        mt_exc = exc_mt
    else:
        mt_exc = None

    assert type(mt_exc) is type(torch_exc)
    assert str(mt_exc) == str(torch_exc)


def _dtype_name(dtype):
    return getattr(dtype, "name", str(dtype).split(".")[-1])


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "to") and hasattr(value, "device"):
        dev = getattr(value.device, "type", value.device)
        if dev != "cpu":
            value = value.to("cpu")
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _leaf_grads(values):
    grads = []
    for value in values:
        if not hasattr(value, "requires_grad"):
            continue
        if not getattr(value, "requires_grad", False):
            continue
        grads.append(getattr(value, "grad", None))
    return grads


def _allclose_or_false(a, b, *, rtol, atol):
    try:
        np.testing.assert_allclose(_to_numpy(a), _to_numpy(b), rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False


def run_training_core_parity_case(
    *,
    op_name,
    candle_fn,
    torch_fn,
    candle_inputs,
    torch_inputs,
    expect_error=False,
    check_backward=False,
    check_error_message=False,
    error_message_fragment=None,
    rtol=1e-5,
    atol=1e-8,
):
    del op_name

    mt_args = candle_inputs()
    th_args = torch_inputs()

    if expect_error:
        try:
            torch_fn(*th_args)
        except Exception as th_exc:
            torch_exc = th_exc
        else:
            torch_exc = None

        try:
            candle_fn(*mt_args)
        except Exception as mt_exc:
            candle_exc = mt_exc
        else:
            candle_exc = None

        return {
            "error_type_match": type(candle_exc) is type(torch_exc),
            "error_message_match": (
                True
                if not check_error_message
                else (
                    torch_exc is not None
                    and candle_exc is not None
                    and error_message_fragment is not None
                    and error_message_fragment in str(torch_exc)
                    and error_message_fragment in str(candle_exc)
                )
            ),
            "torch_error": torch_exc,
            "candle_error": candle_exc,
        }

    mt_out = candle_fn(*mt_args)
    th_out = torch_fn(*th_args)

    shape_match = tuple(getattr(mt_out, "shape", ())) == tuple(getattr(th_out, "shape", ()))
    dtype_match = _dtype_name(getattr(mt_out, "dtype", None)) == _dtype_name(getattr(th_out, "dtype", None))

    mt_np = _to_numpy(mt_out)
    th_np = _to_numpy(th_out)
    try:
        np.testing.assert_allclose(mt_np, th_np, rtol=rtol, atol=atol)
        value_match = True
    except AssertionError:
        value_match = False

    grad_count_match = None
    grad_value_match = None
    if check_backward:
        mt_out.backward()
        th_out.backward()
        mt_grads = _leaf_grads(mt_args)
        th_grads = _leaf_grads(th_args)
        grad_count_match = len(mt_grads) == len(th_grads)
        grad_value_match = grad_count_match and all(
            ((mg is None and tg is None) or (mg is not None and tg is not None and _allclose_or_false(mg, tg, rtol=rtol, atol=atol)))
            for mg, tg in zip(mt_grads, th_grads)
        )

    return {
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "value_match": value_match,
        "grad_count_match": grad_count_match,
        "grad_value_match": grad_value_match,
        "candle_output": mt_out,
        "torch_output": th_out,
    }
