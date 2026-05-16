"""Minimal torch.utils.dlpack compatibility helpers for candle."""

import enum

from .. import _C


def to_dlpack(tensor):
    return tensor.__dlpack__()


def from_dlpack(dlpack):
    return _C._dlpack_exchange_api.from_dlpack(dlpack)


class DLDeviceType(enum.IntEnum):
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLROCMHost = 11
    kDLExtDev = 12
    kDLOneAPI = 14
