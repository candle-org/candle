"""Subprocess worker for pipeline NPU benchmark cases."""
import argparse
import json

from .bench import CASES, run_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", required=True, choices=["candle", "torch_npu"])
    parser.add_argument("--cases", default=",".join(CASES.keys()))
    parser.add_argument("--mode", default="eager", choices=["eager", "pipeline"])
    parser.add_argument("--device", default="npu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    selected = [case_id.strip() for case_id in args.cases.split(",") if case_id.strip()]
    results = []
    for case_id in selected:
        case = CASES[case_id]
        try:
            results.append(run_case(
                case,
                framework=args.framework,
                device=args.device,
                mode=args.mode,
                warmup=args.warmup,
                iters=args.iters,
            ))
        except Exception as exc:
            results.append({
                "framework": args.framework,
                "case_id": case_id,
                "mode": args.mode,
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "p95_ms": 0.0,
                "op_count": 0,
                "status": f"error: {exc}",
            })
    print(json.dumps(results))


if __name__ == "__main__":
    main()
