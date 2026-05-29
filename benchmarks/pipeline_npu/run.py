"""Orchestrator for pipeline NPU benchmark comparisons."""
import argparse
import json
import os
import subprocess
import sys

from benchmarks.npu_perf_gates import annotate_ratio_rows, collect_ratio_failures

from .cases import CASES


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _worker_failure_rows(framework, args, status):
    """Return structured rows for every selected pipeline case on worker failure."""
    return [
        {
            "framework": framework,
            "case_id": case_id,
            "mode": args.mode,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "op_count": 0,
            "status": status,
        }
        for case_id in _parse_case_ids(args.cases)
    ]


def _spawn_worker(framework, args):
    repo_root = _repo_root()
    cmd = [
        args.python or sys.executable,
        "-m",
        "benchmarks.pipeline_npu.worker",
        "--framework",
        framework,
        "--cases",
        args.cases,
        "--mode",
        args.mode,
        "--device",
        args.device,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
    ]
    env = os.environ.copy()
    src_path = os.path.join(repo_root, "src")
    existing = env.get("PYTHONPATH")
    paths = [repo_root, src_path]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False, cwd=repo_root)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, file=sys.stderr, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
        return _worker_failure_rows(framework, args, f"error: worker exit {proc.returncode}")
    try:
        rows = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        if proc.stdout:
            print(proc.stdout, file=sys.stderr, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
        return _worker_failure_rows(framework, args, f"error: malformed worker JSON: {exc}")
    if not rows:
        return _worker_failure_rows(framework, args, "error: worker returned no rows")
    return rows


def annotate_pipeline_ratios(rows):
    annotate_ratio_rows(rows, key_fields=("case_id", "mode"), metric="median_ms")


def pipeline_ratio_failures(rows, *, case_ids, mode, max_ratio):
    return collect_ratio_failures(
        rows,
        key_fields=("case_id", "mode"),
        expected_keys=[(case_id, mode) for case_id in case_ids],
        max_ratio=max_ratio,
        ratio_field="median_ratio",
    )


def _print_table(rows, stream=None):
    stream = stream or sys.stdout
    print("case | framework | mode | median ms | ratio | status", file=stream)
    for row in sorted(rows, key=lambda item: (item.get("case_id", ""), item.get("framework", ""), item.get("mode", ""))):
        ratio = row.get("median_ratio")
        ratio_text = "-" if ratio is None else f"{ratio:.2f}x"
        print(
            f"{row.get('case_id', '-')} | "
            f"{row.get('framework', '-')} | "
            f"{row.get('mode', '-')} | "
            f"{row.get('median_ms', 0.0):.4f} | "
            f"{ratio_text} | "
            f"{row.get('status', '-') }",
            file=stream,
        )


def _status_failures(rows):
    failures = []
    for row in rows:
        if row.get("status") != "ok":
            failures.append(
                f"{row.get('case_id', '-')}/{row.get('mode', '-')}/{row.get('framework', '-')}: "
                f"{row.get('status', '-') }"
            )
    return failures


def _write_json_output(path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True)
    if path == "-":
        print(text)
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")


def _parse_case_ids(cases_arg):
    return [case_id.strip() for case_id in cases_arg.split(",") if case_id.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(CASES.keys()))
    parser.add_argument("--mode", default="eager", choices=["eager", "pipeline"])
    parser.add_argument("--device", default="npu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--python")
    parser.add_argument("--json-output")
    parser.add_argument("--fail-on-ratio", action="store_true")
    parser.add_argument("--max-ratio", type=float, default=0.99)
    args = parser.parse_args()

    if args.max_ratio <= 0:
        parser.error("--max-ratio must be > 0")

    case_ids = _parse_case_ids(args.cases)
    unknown = [case_id for case_id in case_ids if case_id not in CASES]
    if unknown:
        parser.error(f"unknown cases: {','.join(unknown)}")

    rows = []
    rows.extend(_spawn_worker("candle", args))
    rows.extend(_spawn_worker("torch_npu", args))
    annotate_pipeline_ratios(rows)

    failures = _status_failures(rows)
    if args.fail_on_ratio:
        failures.extend(pipeline_ratio_failures(rows, case_ids=case_ids, mode=args.mode, max_ratio=args.max_ratio))

    human_stream = sys.stderr if args.json_output == "-" else sys.stdout
    _print_table(rows, stream=human_stream)
    if failures:
        print("failures:", file=human_stream)
        for failure in failures:
            print(f"- {failure}", file=human_stream)

    payload = {
        "warmup": args.warmup,
        "iters": args.iters,
        "cases": case_ids,
        "mode": args.mode,
        "device": args.device,
        "max_ratio": args.max_ratio,
        "frameworks": ["candle", "torch_npu"],
        "rows": rows,
        "failures": failures,
    }
    if args.json_output:
        _write_json_output(args.json_output, payload)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
