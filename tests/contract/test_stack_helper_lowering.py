from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.gen_functions import _gen_one_node


def _node_src(func_name: str) -> str:
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    info = next(i for i in infos if i.func_name == func_name)
    return _gen_one_node(info)


def test_stack_backward_uses_local_helper():
    src = _node_src('stack')
    assert 'redispatch("stack_tensors_backward"' not in src
    assert '_stack_backward_helper(' in src
