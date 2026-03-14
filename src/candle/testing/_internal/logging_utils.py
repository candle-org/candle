import unittest


class LoggingTestCase(unittest.TestCase):
    pass


def make_logging_test(*args, **kwargs):  # noqa: ARG001
    def decorator(fn):
        return fn

    return decorator
