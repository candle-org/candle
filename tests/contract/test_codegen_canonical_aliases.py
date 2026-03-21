from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.gen_functions import gen_functions
from tools.autograd.gen_variable_type import gen_variable_type
from tools.autograd.gen_registration import gen_registration


def _subset(*func_names):
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    wanted = set(func_names)
    return [i for i in infos if i.func_name in wanted]


def test_gen_functions_exports_canonical_backward_aliases_for_overloads():
    src = gen_functions(_subset('mul.Tensor', 'mul.Scalar', 'sum', 'sum.dim_IntList'))
    assert 'class MulTensorBackward0' in src
    assert 'class MulScalarBackward0' in src
    assert 'MulBackward0 = MulTensorBackward0' in src
    assert 'class SumBackward0' in src or 'class SumDimIntlistBackward0' in src
    assert 'SumBackward0 = SumBackward0' not in src


def test_gen_variable_type_exports_canonical_wrapper_aliases_for_overloads():
    src = gen_variable_type(_subset('mul.Tensor', 'mul.Scalar', 'sum', 'sum.dim_IntList'))
    assert 'def mul_tensor_autograd(' in src
    assert 'def mul_scalar_autograd(' in src
    assert 'mul_autograd = mul_tensor_autograd' in src
    assert 'def sum_autograd(' in src or 'def sum_dim_intlist_autograd(' in src


def test_gen_registration_registers_each_op_only_once():
    src = gen_registration(_subset('mul.Tensor', 'mul.Scalar', 'sum', 'sum.dim_IntList', 'all', 'all.dim'))
    assert src.count("register_autograd_kernels('mul'") == 1
    assert src.count("register_autograd_kernels('sum'") == 1
    assert src.count("register_autograd_kernels('all'") == 1
    assert "_VT.mul_autograd" in src
    assert "_VT.sum_autograd" in src
    assert "_VT.all_autograd" in src
