from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.gen_functions import _gen_one_node


def _node_src(func_name: str) -> str:
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    info = next(info for info in infos if info.func_name == func_name)
    return _gen_one_node(info)


def test_gen_functions_tensor_list_pair_returns_both_tensor_list_grads():
    src = _node_src('_foreach_div.List')
    assert 'grad_self' in src
    assert 'grad_other' in src
    assert 'return (grad_self, grad_other,)' in src


def test_gen_functions_mixed_tensor_and_tensor_lists_return_in_input_order():
    src = _node_src('_lstm_mps')
    assert 'grad_input' in src
    assert 'grad_hx' in src
    assert 'grad_params' in src
    assert 'return (grad_input, grad_hx, grad_params,)' in src


def test_gen_functions_skips_output_derivative_formulas():
    src = _node_src('abs')
    assert 'auto_element_wise' not in src
    assert '\n             = ' not in src
    assert 'grad_self = redispatch("mul", keyset, grad, redispatch("sign", keyset, self_))' in src
