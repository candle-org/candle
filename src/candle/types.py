"""Minimal torch.types compatibility for candle storage."""
from typing import Union, Any as _Any

_bool = Union[bool, "torch.Tensor"]
_int = Union[int, "torch.Tensor"]
Storage = _Any  # forward reference, resolved at runtime
