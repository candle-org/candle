from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.gen_variable_type import _gen_one_wrapper


def _wrapper_src(func_name: str) -> str:
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    info = next(info for info in infos if info.func_name == func_name)
    return _gen_one_wrapper(info)


def test_gen_variable_type_uses_safe_param_names_for_python_keywords():
    src = _wrapper_src('random_.from')
    assert 'def random__from_autograd(' in src
    assert 'from_' in src
    assert 'grad_fn._from = from_' in src
    assert 'def random__from_autograd(self, from,' not in src
