import candle
import candle.nn.functional as F

from benchmarks.pipeline_npu.cases import CASES


def test_pipeline_case_builders_accept_framework_module():
    for name in ["A1", "A2s", "D2"]:
        case = CASES[name]
        forward = case["builder"](candle, F, "cpu", candle.float32)
        out = forward()
        assert hasattr(out, "shape")


def test_pipeline_cases_do_not_require_global_candle_module_argument():
    case = CASES["A1"]
    forward = case["builder"](candle, F, "cpu", candle.float32)
    out = forward()
    assert tuple(out.shape) == (1, 128, 1024)
