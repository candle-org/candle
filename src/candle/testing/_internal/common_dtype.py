"""Dtype enumeration helpers matching torch.testing._internal.common_dtype."""

import candle as torch


def all_types():
    return (torch.float32, torch.float64,
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def all_types_and_complex():
    return all_types() + (torch.complex64, torch.complex128)


def floating_types():
    return (torch.float32, torch.float64)


def floating_types_and_half():
    return (torch.float32, torch.float64, torch.float16)


def floating_and_complex_types():
    return floating_types() + (torch.complex64, torch.complex128)


def integral_types():
    return (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def get_all_int_dtypes():
    return [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


def get_all_fp_dtypes(include_half=True, include_bfloat16=True):
    dtypes = [torch.float32, torch.float64]
    if include_half:
        dtypes.append(torch.float16)
    if include_bfloat16 and hasattr(torch, 'bfloat16'):
        dtypes.append(torch.bfloat16)
    return dtypes


def get_all_complex_dtypes():
    return [torch.complex64, torch.complex128]


def get_all_dtypes(include_half=True, include_bfloat16=True,
                   include_complex=True, include_bool=True):
    dtypes = get_all_int_dtypes() + get_all_fp_dtypes(
        include_half=include_half, include_bfloat16=include_bfloat16
    )
    if include_bool:
        dtypes.append(torch.bool)
    if include_complex:
        dtypes += get_all_complex_dtypes()
    return dtypes


def all_types_and(*extra):
    """all_types() plus any extra dtypes passed as args."""
    return all_types() + tuple(extra)


def all_types_and_complex_and(*extra):
    """all_types_and_complex() plus any extra dtypes passed as args."""
    return all_types_and_complex() + tuple(extra)


def floating_and_complex_types_and(*extra):
    """floating_and_complex_types() plus any extra dtypes passed as args."""
    return floating_and_complex_types() + tuple(extra)


def floating_types_and(*extra):
    """floating_types() plus any extra dtypes passed as args."""
    return floating_types() + tuple(extra)


def integral_types_and(*extra):
    """integral_types() plus any extra dtypes passed as args."""
    return integral_types() + tuple(extra)


def complex_types():
    return (torch.complex64, torch.complex128)


def double_types():
    return (torch.float64, torch.complex128)


# Map from float dtype to its corresponding complex dtype
float_to_corresponding_complex_type_map = {
    torch.float16: torch.complex32 if hasattr(torch, "complex32") else torch.complex64,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def get_all_math_dtypes(device="cpu"):
    """Return all dtypes valid for math operations on the given device."""
    return list(all_types_and_complex()) + [torch.bool, torch.float16]


def get_all_qint_dtypes():
    """Return quantized integer dtypes — empty for candle (no quantization)."""
    return []
