from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.gen_registration import gen_registration


def test_gen_registration_emits_all_yaml_ops_with_runtime_guard():
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    src = gen_registration(infos)
    # All yaml ops are emitted, with runtime registry.has() guard
    assert "registry.has('add')" in src
    assert "registry.has('addbmm')" in src
    # Legacy delegation is present
    assert "register_legacy_autograd_kernels()" in src


def test_gen_registration_does_not_import_candle_runtime():
    """gen_registration.py must not import candle at module level."""
    import tools.autograd.gen_registration as mod
    src = open(mod.__file__).read()
    # No top-level candle import
    assert "from candle" not in src.split("def gen_registration")[0]
