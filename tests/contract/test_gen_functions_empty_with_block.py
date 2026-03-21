from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.gen_functions import _gen_one_node


def _node_src(func_name: str):
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    return [_gen_one_node(info) for info in infos if info.func_name == func_name]


def test_gen_functions_omits_empty_with_block_for_any_all_like_ops():
    for func_name in ['any', 'all', '_is_all_true', '_is_any_true']:
        for src in _node_src(func_name):
            assert 'with _grad_context(keyset):\n        return (grad,)' not in src
            assert 'return (grad,)' in src
