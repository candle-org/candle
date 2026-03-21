from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.gen_functions import _gen_one_node


def _node_src(func_name: str) -> str:
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    info = next(info for info in infos if info.func_name == func_name)
    return _gen_one_node(info)


def test_gen_functions_preserves_ternary_syntax_tokens():
    src = _node_src('acosh')
    assert ' if redispatch("is_complex", keyset, self_) else ' in src
    assert 'if_' not in src
    assert 'else_' not in src
