from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.gen_variable_type import _gen_one_wrapper


def _wrapper_src(func_name: str) -> str:
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    info = next(info for info in infos if info.func_name == func_name)
    return _gen_one_wrapper(info)


def test_gen_variable_type_normalizes_reduction_defaults_to_python_literals():
    src = _wrapper_src('binary_cross_entropy')
    assert 'reduction=Mean' not in src
    assert 'reduction=1' in src
