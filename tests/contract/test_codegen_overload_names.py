from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.gen_functions import _gen_one_node
from tools.autograd.gen_variable_type import _gen_one_wrapper
from tools.autograd.gen_registration import gen_registration


def _info(func_name: str):
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    return next(i for i in infos if i.func_name == func_name)


def test_backward_names_are_distinct_across_overloads():
    add_tensor = _info('add.Tensor')
    add_scalar = _info('add.Scalar')
    assert add_tensor.backward_name != add_scalar.backward_name


def test_wrapper_names_are_distinct_across_overloads():
    add_tensor = _info('add.Tensor')
    add_scalar = _info('add.Scalar')
    src_tensor = _gen_one_wrapper(add_tensor)
    src_scalar = _gen_one_wrapper(add_scalar)
    assert 'def add_tensor_autograd(' in src_tensor
    assert 'def add_scalar_autograd(' in src_scalar


def test_registration_uses_canonical_alias_for_overloads():
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    src = gen_registration([i for i in infos if i.func_name in ('add.Tensor', 'add.Scalar')])
    assert '_VT.add_autograd' in src
    assert src.count("register_autograd_kernels('add'") == 1
