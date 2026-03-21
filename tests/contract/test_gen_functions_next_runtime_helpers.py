from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.gen_functions import _gen_one_node


def _node_src(func_name: str) -> str:
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    info = next(i for i in infos if i.func_name == func_name)
    return _gen_one_node(info)


def test_matmul_backward_uses_local_helper():
    src = _node_src('matmul')
    assert 'redispatch("matmul_backward"' not in src
    assert '_matmul_backward_helper(' in src


def test_expand_backward_uses_local_helper_not_sum_to_op():
    src = _node_src('expand')
    assert 'redispatch("sum_to"' not in src
    assert '_sum_to_backward_helper(' in src


def test_permute_backward_uses_local_helper():
    src = _node_src('permute')
    assert 'redispatch("permute_backwards"' not in src
    assert '_permute_backward_helper(' in src


def test_select_backward_uses_local_helper():
    src = _node_src('select.int')
    assert 'redispatch("select_backward_symint"' not in src
    assert '_select_backward_symint_helper(' in src


def test_cat_backward_uses_local_helper_without_undefined_tensor_list_name():
    src = _node_src('cat')
    assert 'cat_tensors_backward' not in src
    assert '_cat_backward_helper(' in src
    assert 'to_args_sizes_symint' not in src
    assert 'to_args_scalartypes' not in src
