from .helpers import assert_torch_error


def test_harness_asserts_exact_error():
    def mt():
        raise RuntimeError("X")

    def th():
        raise RuntimeError("Y")

    try:
        assert_torch_error(mt, th)
    except AssertionError:
        assert True
    else:
        assert False
