from contextlib import contextmanager


@contextmanager
def no_dispatch():
    yield
