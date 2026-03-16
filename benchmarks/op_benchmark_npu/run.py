"""CLI entry point: parse args, spawn candle + torch workers, merge results, report."""
import argparse
import json
import os
import subprocess
import sys

from .cases import OP_CASES, SCENARIOS, DTYPES
from .report import generate_report, print_terminal, write_markdown

# Conda environments for each framework
CONDA_PREFIX = os.environ.get("CONDA_PREFIX_BASE", "/opt/miniconda3")
CONDA_ENVS = {
    "candle": "candle",
    "torch": "mindie",
}


def _run_worker(framework, args):
    """Spawn a subprocess worker in the appropriate conda env."""
    env_name = CONDA_ENVS[framework]
    worker_args = [
        "-m", "benchmarks.op_benchmark_npu.worker",
        "--framework", framework,
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
    ]
    if args.ops:
        worker_args.extend(["--ops", args.ops])
    if args.scenario:
        worker_args.extend(["--scenario", args.scenario])
    if args.dtype:
        worker_args.extend(["--dtype", args.dtype])

    # Source CANN env + conda env in a shell so workers get correct LD_LIBRARY_PATH
    cann_env = os.environ.get(
        "CANN_SET_ENV", "/usr/local/Ascend/cann-8.5.0/set_env.sh"
    )
    conda_sh = os.environ.get(
        "CONDA_SH", "/opt/miniconda3/etc/profile.d/conda.sh"
    )
    worker_args_str = " ".join(worker_args)
    shell_cmd = (
        f"source {cann_env} 2>/dev/null; "
        f"source {conda_sh} && "
        f"conda run -n {env_name} --no-capture-output "
        f"python {worker_args_str}"
    )

    print(f"Running {framework} worker (env={env_name})...", file=sys.stderr)
    proc = subprocess.run(
        ["bash", "-c", shell_cmd],
        capture_output=True, text=True, timeout=1800, check=False,
    )

    if proc.stderr:
        for line in proc.stderr.strip().split("\n"):
            if line:
                print(f"  [{framework}] {line}", file=sys.stderr)

    if proc.returncode != 0:
        print(f"ERROR: {framework} worker exited with code {proc.returncode}",
              file=sys.stderr)
        return []

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        print(f"ERROR: failed to parse {framework} JSON output: {e}",
              file=sys.stderr)
        print(f"  stdout was: {proc.stdout[:500]}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(
        description="NPU Op Benchmark: candle vs torch_npu"
    )
    parser.add_argument("--ops", default=None,
                        help="Comma-separated op names to benchmark")
    parser.add_argument("--scenario", default=None, choices=["infer", "train"],
                        help="Only run one scenario")
    parser.add_argument("--dtype", default=None,
                        help="Comma-separated dtype keys: fp16,bf16,fp32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", default=None,
                        help="Directory to write markdown report")
    args = parser.parse_args()

    # Determine what we're running
    if args.ops:
        op_names = args.ops.split(",")
    else:
        op_names = [c["name"] for c in OP_CASES]

    if args.scenario:
        scen_keys = [args.scenario]
    else:
        scen_keys = list(SCENARIOS.keys())

    if args.dtype:
        dtype_keys = args.dtype.split(",")
    else:
        dtype_keys = list(DTYPES.keys())

    # Run both workers
    candle_results = _run_worker("candle", args)
    torch_results = _run_worker("torch", args)

    if not candle_results and not torch_results:
        print("ERROR: both workers returned no results", file=sys.stderr)
        sys.exit(1)

    # Generate report
    report = generate_report(candle_results, torch_results,
                             op_names, dtype_keys, scen_keys)
    print_terminal(report)

    if args.output:
        path = write_markdown(report, args.output)
        print(f"\nReport saved to: {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
