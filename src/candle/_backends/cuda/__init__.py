from ..common import convert as convert_backend
from ..._dispatch.registry import registry
from . import runtime
from . import storage
from .creation import (
    empty_create,
    empty_like_create,
    full_create,
    full_like_create,
    ones_create,
    ones_like_create,
    tensor_create,
    zeros_create,
    zeros_like_create,
)
from .ops import add


registry.register("add", "cuda", add)
registry.register("to", "cuda", convert_backend.to_device)
registry.register("tensor", "cuda", tensor_create)
registry.register("zeros", "cuda", zeros_create)
registry.register("zeros_like", "cuda", zeros_like_create)
registry.register("ones_like", "cuda", ones_like_create)
registry.register("empty_like", "cuda", empty_like_create)
registry.register("full_like", "cuda", full_like_create)
registry.register("ones", "cuda", ones_create)
registry.register("empty", "cuda", empty_create)
registry.register("full", "cuda", full_create)


__all__ = ["runtime", "storage"]
