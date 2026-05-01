"""Minimal torch._prims_common stub for candle."""
from typing import Union

DeviceLikeType = Union[str, "candle._device.device", int]


def compute_elementwise_output_logical_to_physical_perm(tensor, ambiguity_check=False):
    del ambiguity_check
    strides = tuple(tensor.stride())
    perm = tuple(sorted(range(len(strides)), key=lambda dim: (-strides[dim], dim)))
    return perm, False
