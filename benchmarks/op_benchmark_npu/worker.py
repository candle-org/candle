"""Subprocess worker: import framework, run benchmark cases, print JSON to stdout."""
import argparse
import json
import sys

from .runner import benchmark_op, summarize
from .cases import OP_CASES, SCENARIOS, DTYPES


def _get_framework(framework):
    """Import and return (torch_mod, F, sync_fn, device_name)."""
    if framework == "candle":
        import candle as torch_mod
        import candle.nn.functional as F
        device = "npu"
        if hasattr(torch_mod, "npu") and torch_mod.npu.is_available():
            sync = torch_mod.npu.synchronize
        else:
            print("WARNING: NPU not available for candle", file=sys.stderr)
            sync = None
        return torch_mod, F, sync, device
    elif framework == "torch":
        import torch as torch_mod
        import torch_npu  # noqa: F401  # pylint: disable=import-error
        import torch.nn.functional as F
        device = "npu"
        sync = torch_mod.npu.synchronize
        return torch_mod, F, sync, device
    else:
        raise ValueError(f"Unknown framework: {framework}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", required=True, choices=["candle", "torch"])
    parser.add_argument("--ops", default=None, help="Comma-separated op names")
    parser.add_argument("--scenario", default=None, choices=["infer", "train"])
    parser.add_argument("--dtype", default=None, help="Comma-separated: fp16,bf16,fp32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    torch_mod, F, sync, device = _get_framework(args.framework)

    # Filter ops
    cases = OP_CASES
    if args.ops:
        op_names = set(args.ops.split(","))
        cases = [c for c in cases if c["name"] in op_names]

    # Filter scenarios
    scenarios = SCENARIOS
    if args.scenario:
        scenarios = {args.scenario: SCENARIOS[args.scenario]}

    # Filter dtypes
    dtypes = DTYPES
    if args.dtype:
        dtype_keys = set(args.dtype.split(","))
        dtypes = {k: v for k, v in DTYPES.items() if k in dtype_keys}

    results = []
    for dtype_key, dtype_name in dtypes.items():
        dtype = getattr(torch_mod, dtype_name)
        for scen_key, scen in scenarios.items():
            batch, seq = scen["batch"], scen["seq"]
            for case in cases:
                op_name = case["name"]
                try:
                    fn = case["build"](torch_mod, F, device, dtype, batch, seq)
                    samples = benchmark_op(fn, warmup=args.warmup,
                                           iters=args.iters, sync=sync)
                    mean, median, p95 = summarize(samples)
                    results.append({
                        "op": op_name,
                        "dtype": dtype_key,
                        "scenario": scen_key,
                        "mean_ms": round(mean, 4),
                        "median_ms": round(median, 4),
                        "p95_ms": round(p95, 4),
                        "status": "ok",
                    })
                except Exception as e:
                    results.append({
                        "op": op_name,
                        "dtype": dtype_key,
                        "scenario": scen_key,
                        "mean_ms": 0.0,
                        "median_ms": 0.0,
                        "p95_ms": 0.0,
                        "status": f"error: {e}",
                    })
                    print(f"ERROR [{args.framework}] {op_name} "
                          f"{dtype_key}/{scen_key}: {e}", file=sys.stderr)

    print(json.dumps(results))


if __name__ == "__main__":
    main()
