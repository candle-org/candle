"""Pure-Python fallback for _allocator.pyx.

Re-exports the existing Python NpuAllocator.
"""

from candle._backends.npu.allocator import NpuAllocator as FastNpuAllocator

__all__ = ["FastNpuAllocator"]
