"""Minimal torch._VF stub for candle compatibility."""


def split(self, split_size, dim=0):
    from ._functional import split as _split
    return _split(self, split_size, dim)


def split_with_sizes(self, split_sizes, dim=0):
    from ._functional import split as _split
    return _split(self, split_sizes, dim)
