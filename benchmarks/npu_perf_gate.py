"""Run L0/L1/L2 NPU performance gates and write JSON artifacts."""
import argparse
import os
import subprocess
import sys


def build_commands(*, output_dir, warmup, iters, dtype, cases, pipeline_cases):
    return [
        [
            sys.executable,
            "-m", "benchmarks.op_benchmark_npu.run",
            "--mode", "both",
            "--dtype", dtype,
            "--warmup", str(warmup),
            "--iters", str(iters),
            "--max-ratio", "1.0",
            "--json-output", os.path.join(output_dir, "l0_ops.json"),
            "--fail-on-ratio",
        ],
        [
            sys.executable,
            "-m", "benchmarks.pipeline_npu.run",
            "--cases", pipeline_cases,
            "--mode", "eager",
            "--warmup", str(warmup),
            "--iters", str(iters),
            "--max-ratio", "0.99",
            "--json-output", os.path.join(output_dir, "l1_pipeline.json"),
            "--fail-on-ratio",
        ],
        [
            sys.executable,
            "benchmarks/perf_candle_vs_torch_npu.py",
            "--cases", cases,
            "--dtype", "float16" if dtype == "fp16" else dtype,
            "--warmup", str(warmup),
            "--iters", str(iters),
            "--max-fwd-ratio", "0.99",
            "--max-bwd-ratio", "0.99",
            "--max-total-ratio", "0.99",
            "--json-output", os.path.join(output_dir, "l2_models.json"),
            "--fail-on-ratio",
        ],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/npu_perf_gate/latest")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--cases", default="mlp,xfmr,resnet")
    parser.add_argument("--pipeline-cases", default="A1,A2s,A3,B1s,B2,D2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    failures = []
    for command in build_commands(
        output_dir=args.output_dir,
        warmup=args.warmup,
        iters=args.iters,
        dtype=args.dtype,
        cases=args.cases,
        pipeline_cases=args.pipeline_cases,
    ):
        print("$ " + " ".join(command), flush=True)
        proc = subprocess.run(command, check=False)
        if proc.returncode != 0:
            failures.append((command, proc.returncode))

    if failures:
        for command, returncode in failures:
            print(f"gate failed with exit {returncode}: {' '.join(command)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
