"""Public Future/collect_all surface backed by the compiled Cython implementation."""

from ._cython._future import Future, collect_all  # pylint: disable=import-error,no-name-in-module
