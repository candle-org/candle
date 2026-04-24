"""Minimal torch._subclasses.fake_tensor stub."""
from contextlib import contextmanager

class FakeTensor:
    pass
class FakeTensorMode:
    def __enter__(self): return self
    def __exit__(self, *args): pass

@contextmanager
def unset_fake_temporarily():
    yield
