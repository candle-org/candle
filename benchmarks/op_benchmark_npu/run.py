"""CLI entry point: parse args, spawn candle + torch workers, merge results, report."""
import argparse
import json
import os
import shlex
import subprocess
import sys

from .cases import OP_CASES, SCENARIOS, DTYPES, MODES
from .report import generate_report, print_terminal, ratio_failures, write_markdown

# Conda environments for each framework
CONDA_PREFIX = os.environ.get("CONDA_PREFIX_BASE", "/opt/miniconda3")
CONDA_ENVS = {
    "candle": os.environ.get("CANDLE_CONDA_ENV", "candle"),
    "torch": os.environ.get("TORCH_NPU_CONDA_ENV", "mindie"),
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
    if args.mode:
        worker_args.extend(["--mode", args.mode])

    # Source CANN env + conda env in a shell so workers get correct LD_LIBRARY_PATH
    cann_env = os.environ.get(
        "CANN_SET_ENV", "/usr/local/Ascend/cann-8.5.0/set_env.sh"
    )
    conda_sh = os.environ.get(
        "CONDA_SH", "/opt/miniconda3/etc/profile.d/conda.sh"
    )
    worker_args_str = " ".join(shlex.quote(arg) for arg in worker_args)
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
        return [{
            "framework": "torch_npu" if framework == "torch" else framework,
            "status": f"error: worker exit {proc.returncode}",
            "stderr": proc.stderr,
            "stdout": proc.stdout,
        }]

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        print(f"ERROR: failed to parse {framework} JSON output: {e}",
              file=sys.stderr)
        print(f"  stdout was: {proc.stdout[:500]}", file=sys.stderr)
        return [{
            "framework": "torch_npu" if framework == "torch" else framework,
            "status": f"error: malformed worker JSON: {e}",
            "stderr": proc.stderr,
            "stdout": proc.stdout,
        }]

def _status_failures(candle_results, torch_results):
    """Return infrastructure/case failures from worker result rows."""
    failures = []
    for row in candle_results + torch_results:
        status = row.get("status")
        if status and status != "ok":
            failures.append(
                f"{row.get('op', '-')}/{row.get('mode', '-')}/{row.get('dtype', '-')}/"
                f"{row.get('scenario', '-')}/{row.get('framework', '-')}: {status}"
            )
    return failures

def _output_stream(json_output):
    """Return stream name for human-readable output."""
    return "stderr" if json_output == "-" else "stdout"



def _print_human(text="", stream="stdout"):
    """Print human-readable output to the selected stream."""
    print(text, file=sys.stderr if stream == "stderr" else sys.stdout)


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
    parser.add_argument("--mode", default="fwd", choices=["fwd", "bwd", "both"],
                        help="Run forward cases, backward cases, or both")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", default=None,
                        help="Directory to write markdown report")
    parser.add_argument("--json-output", default=None,
                        help="Path to write JSON results, or '-' for stdout")
    parser.add_argument("--fail-on-ratio", action="store_true",
                        help="Exit nonzero when candle median ratio exceeds --max-ratio")
    parser.add_argument("--max-ratio", type=float, default=1.0,
                        help="Maximum allowed candle/torch_npu median ratio")
    args = parser.parse_args()
    if args.max_ratio <= 0:
        parser.error("--max-ratio must be > 0")

    # Determine what we're running
    if args.mode == "both":
        mode_keys = list(MODES.keys())
    else:
        mode_keys = [args.mode]

    if args.ops:
        op_names = args.ops.split(",")
    else:
        op_names = [c["name"] for c in OP_CASES if c.get("mode", "fwd") in mode_keys]

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

    failures = _status_failures(candle_results, torch_results)
    human_stream = _output_stream(args.json_output)
    if args.fail_on_ratio:
        failures.extend(ratio_failures(
            candle_results,
            torch_results,
            op_names=op_names,
            dtype_keys=dtype_keys,
            scen_keys=scen_keys,
            mode_keys=mode_keys,
            max_ratio=args.max_ratio,
        ))
    if failures:
        _print_human("\n# Gate failures", stream=human_stream)
        for failure in failures:
            _print_human(f"- {failure}", stream=human_stream)

    # Generate report
    report = generate_report(candle_results, torch_results,
                             op_names, dtype_keys, scen_keys, mode_keys)
    if args.json_output != "-":
        print_terminal(report)
    else:
        _print_human(report, stream=human_stream)

    if args.output:
        path = write_markdown(report, args.output)
        print(f"\nReport saved to: {path}", file=sys.stderr)

    if args.json_output:
        payload = {
            "warmup": args.warmup,
            "iters": args.iters,
            "op_names": op_names,
            "dtype_keys": dtype_keys,
            "scenario_keys": scen_keys,
            "mode_keys": mode_keys,
            "max_ratio": args.max_ratio,
            "failures": failures,
            "results": {
                "candle": candle_results,
                "torch_npu": torch_results,
            },
        }
        json_text = json.dumps(payload, indent=2, sort_keys=True)
        if args.json_output == "-":
            print(json_text)
        else:
            with open(args.json_output, "w", encoding="utf-8") as handle:
                handle.write(json_text)
                handle.write("\n")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
