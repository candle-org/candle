"""Lock SumBackward0 to expand + contiguous, not the legacy zeros+add path.

The yaml formula for `sum` is `grad.expand_symint(self.sym_sizes())`. Earlier
versions of the preserved generated `functions.py` used a two-step
`add(zeros(self.shape), grad)` workaround which cost ~300us per backward step
on NPU and showed up as one of the largest dispatch overheads in the
linear_bwd_mlp_4096_4096 microbenchmark. Replacing it with `expand` removes
the zeros allocation, and the subsequent `contiguous` ensures downstream
backward consumers receive a materialized (not stride-0) gradient.

This test guards against regressing to the zeros+add pattern.
"""
import inspect

from candle._generated import functions as gen_functions


def test_sum_backward0_uses_expand_not_zeros_add():
    src = inspect.getsource(gen_functions.SumBackward0.backward)
    assert 'redispatch("expand"' in src, (
        f"SumBackward0.backward should use expand:\n{src}"
    )
    assert 'redispatch("zeros"' not in src, (
        f"SumBackward0.backward should not allocate zeros:\n{src}"
    )
    assert 'redispatch("add"' not in src, (
        f"SumBackward0.backward should not add into zeros:\n{src}"
    )
