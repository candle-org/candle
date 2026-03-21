from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.formula_transpiler import transpile
from tools.autograd.gen_variable_type import _gen_one_post_wrapper


def _info(func_name: str):
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    return next(i for i in infos if i.func_name == func_name)


def test_transpile_erf_formula_eliminates_M_PI_symbol():
    info = _info('erf')
    formula = info.derivatives[0].formula
    py = transpile(formula)
    assert 'M_PI' not in py


def test_transpile_silu_formula_uses_local_helper():
    info = _info('silu')
    formula = info.derivatives[0].formula
    py = transpile(formula)
    assert 'infinitely_differentiable_silu_backward' not in py
    assert '_silu_grad(' in py or 'silu_backward' in py


def test_transpile_clamp_formula_uses_local_helper():
    info = _info('clamp')
    formula = info.derivatives[0].formula
    py = transpile(formula)
    assert 'redispatch("clamp_backward"' not in py
    assert '_clamp_backward_helper(' in py


def test_tensor_list_post_wrapper_attaches_grad_fn_per_result_item_for_split():
    src = _gen_one_post_wrapper(_info('split.Tensor'))
    assert 'result.grad_fn = grad_fn' not in src
    assert 'result[i].grad_fn = grad_fn' in src or 'for i in range(len(result))' in src


def test_tensor_list_post_wrapper_attaches_grad_fn_per_result_item_for_unbind():
    src = _gen_one_post_wrapper(_info('unbind.int'))
    assert 'result.grad_fn = grad_fn' not in src
    assert 'result[i].grad_fn = grad_fn' in src or 'for i in range(len(result))' in src
