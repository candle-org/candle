from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.gen_registration import gen_registration


def test_gen_registration_skips_ops_without_registered_schemas():
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    src = gen_registration(infos)
    # existing Candle schema
    assert "register_autograd_kernels('add'" in src
    # currently missing Candle schema, should not be registered yet
    assert "register_autograd_kernels('addbmm'" not in src
