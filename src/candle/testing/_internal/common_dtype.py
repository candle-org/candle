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
