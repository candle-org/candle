"""Minimal torch.quasirandom stubs for compatibility tests."""

from . import rand
from . import get_default_dtype


class SobolEngine:
    def __init__(self, dimension, scramble=False, seed=None):
        self.dimension = dimension
        self.scramble = scramble
        self.seed = seed

    def draw(self, n=1, dtype=None):
        dt = dtype if dtype is not None else get_default_dtype()
        return rand(n, self.dimension, dtype=dt)
