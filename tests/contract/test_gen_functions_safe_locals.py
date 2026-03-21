from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.gen_functions import _gen_one_node


def _node_src(func_name: str) -> str:
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    info = next(info for info in infos if info.func_name == func_name)
    return _gen_one_node(info)


def test_gen_functions_uses_safe_locals_for_keyword_non_tensor_args():
    src = _node_src('random_.from')
    assert 'from = self._from' not in src
    assert 'from_ = self._from' in src
