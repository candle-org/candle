import importlib

import candle
import candle.nn.functional as candle_F

from .cases import CASES
from .utils import measure, summarize


def _import_framework(framework):
    if framework == "candle":
        return candle, candle_F, "npu"
    if framework == "torch_npu":
        torch_mod = importlib.import_module("torch")
        importlib.import_module("torch_npu")
        return torch_mod, importlib.import_module("torch.nn.functional"), "npu"
    raise ValueError(f"unknown framework: {framework}")


def _resolve_dtype(torch_mod, dtype_name):
    return getattr(torch_mod, dtype_name)


def _sync_for(torch_mod, device):
    if device == "npu" and hasattr(torch_mod, "npu") and torch_mod.npu.is_available():
        return torch_mod.npu.synchronize
    return None


def run_case(case, *, framework="candle", device="cpu", mode="eager", warmup=5, iters=20):
    torch_mod, F, default_device = _import_framework(framework)
    if device is None:
        device = default_device
    dtype = _resolve_dtype(torch_mod, case["dtype"])
    forward = case["builder"](torch_mod, F, device, dtype)
    sync = _sync_for(torch_mod, device)

    def _run_once():
        if mode == "pipeline":
            if framework != "candle":
                forward()
                return
            with candle.pipeline(max_ops=64):
                forward()
        else:
            forward()

    samples = measure(_run_once, warmup=warmup, iters=iters, sync=sync)
    mean, median, p95 = summarize(samples)

    op_count = 0
    if mode == "pipeline" and framework == "candle":
        with candle.pipeline(max_ops=64) as pipe:
            forward()
            pipe.flush()
            dump = pipe.debug_dump()
            op_count = len(dump.get("entries", []))

    return {
        "framework": framework,
        "case_id": case["case_id"],
        "mode": mode,
        "batch": case["batch"],
        "seq": case["seq"],
        "hidden": case["hidden"],
        "heads": case["heads"],
        "dtype": case["dtype"],
        "mean_ms": float(mean),
        "median_ms": float(median),
        "p95_ms": float(p95),
        "op_count": int(op_count),
        "status": "ok",
    }


def run():
    results = {}
    for name, case in CASES.items():
        results[name] = run_case(case, framework="candle", device="cpu", mode="eager", warmup=1, iters=1)
    return results


__all__ = ["CASES", "run_case", "run"]
