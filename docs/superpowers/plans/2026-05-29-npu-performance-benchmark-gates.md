# NPU Performance Benchmark Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 0 benchmark gate infrastructure that measures Candle NPU against torch_npu for L0 single ops, L1 composite blocks, and L2 end-to-end models.

**Architecture:** Keep existing benchmark entry points, add one shared metrics/gating helper, make pipeline cases framework-neutral, and add machine-readable JSON outputs plus ratio gates. This phase does not optimize kernels; it creates the measurement and regression system required before native fast-path and graph work.

**Tech Stack:** Python 3.11, pytest, Candle benchmark scripts, torch_npu subprocess workers, JSON benchmark artifacts, existing conda/CANN environment conventions.

---

## Scope

This plan implements Phase 0 from `docs/superpowers/specs/2026-05-29-npu-performance-core-redesign.md`.

It intentionally does not change NPU kernels, Cython, C++, C, Rust, ACLNN bindings, or autograd execution. Performance results may fail the new gates after this phase; that is expected because the phase creates truthful gates before performance fixes.

## File Structure

Create:

- `benchmarks/npu_perf_gates.py` — shared sample summaries, ratio annotation, and gate failure helpers for benchmark scripts.
- `benchmarks/pipeline_npu/worker.py` — subprocess worker for one framework and selected pipeline cases.
- `benchmarks/pipeline_npu/run.py` — orchestrator that runs Candle and torch_npu pipeline workers, merges ratios, writes JSON, and fails gates.
- `tests/cpu/test_npu_perf_gates.py` — unit tests for shared metrics and ratio helpers.
- `tests/cpu/test_pipeline_npu_framework_neutral.py` — CPU tests that pipeline cases build against a torch-like module argument rather than hard-coded Candle imports.
- `tests/cpu/test_pipeline_npu_report.py` — unit tests for pipeline result ratio annotations and gate failures.
- `tests/cpu/test_perf_candle_vs_torch_npu_ratios.py` — unit tests for L2 ratio annotation and total-ratio gates.

Modify:

- `benchmarks/op_benchmark_npu/runner.py` — use shared summary helper and expose p10/p90/min/max/sample_count.
- `benchmarks/op_benchmark_npu/worker.py` — include extended summary fields in worker JSON rows.
- `benchmarks/op_benchmark_npu/report.py` — include p10/p90, impact, and gate summary from extended rows.
- `benchmarks/op_benchmark_npu/run.py` — add `--json-output`, `--fail-on-ratio`, and `--max-ratio`.
- `benchmarks/pipeline_npu/cases.py` — make builders framework-neutral by passing `torch_mod` and `F` into every case.
- `benchmarks/pipeline_npu/bench.py` — add `framework`, `mode`, and torch_npu support while preserving existing CPU smoke-test behavior.
- `benchmarks/perf_candle_vs_torch_npu.py` — add total latency summaries, p10/p90 fields, total ratio, JSON gate metadata, and total-ratio failure checks.
- `tests/npu/test_pipeline_npu_bench_smoke.py` — assert the new pipeline result fields without requiring NPU hardware.

---

### Task 1: Add shared benchmark metrics and gate helpers

**Files:**
- Create: `benchmarks/npu_perf_gates.py`
- Test: `tests/cpu/test_npu_perf_gates.py`

- [ ] **Step 1: Write failing tests for sample summaries and ratio gates**

Create `tests/cpu/test_npu_perf_gates.py`:

```python
from benchmarks.npu_perf_gates import (
    annotate_ratio_rows,
    collect_ratio_failures,
    summarize_samples,
)


def test_summarize_samples_reports_distribution_fields():
    summary = summarize_samples([4.0, 1.0, 3.0, 2.0, 5.0])

    assert summary == {
        "sample_count": 5,
        "mean_ms": 3.0,
        "median_ms": 3.0,
        "min_ms": 1.0,
        "max_ms": 5.0,
        "p10_ms": 1.0,
        "p90_ms": 5.0,
        "p95_ms": 5.0,
    }


def test_summarize_samples_empty_list_returns_zero_distribution():
    summary = summarize_samples([])

    assert summary == {
        "sample_count": 0,
        "mean_ms": 0.0,
        "median_ms": 0.0,
        "min_ms": 0.0,
        "max_ms": 0.0,
        "p10_ms": 0.0,
        "p90_ms": 0.0,
        "p95_ms": 0.0,
    }


def test_annotate_ratio_rows_adds_candle_ratio_against_torch_npu():
    rows = [
        {"framework": "torch_npu", "case": "add", "median_ms": 2.0, "status": "ok"},
        {"framework": "candle", "case": "add", "median_ms": 3.0, "status": "ok"},
    ]

    annotate_ratio_rows(rows, key_fields=("case",), metric="median_ms")

    candle = next(row for row in rows if row["framework"] == "candle")
    torch_ref = next(row for row in rows if row["framework"] == "torch_npu")
    assert candle["median_ratio"] == 1.5
    assert torch_ref["median_ratio"] == 1.0


def test_collect_ratio_failures_reports_missing_errors_and_slow_rows():
    rows = [
        {"framework": "torch_npu", "case": "add", "median_ms": 2.0, "status": "ok"},
        {"framework": "candle", "case": "add", "median_ms": 3.0, "status": "ok", "median_ratio": 1.5},
        {"framework": "candle", "case": "mul", "median_ms": 0.0, "status": "error: boom"},
    ]

    failures = collect_ratio_failures(
        rows,
        key_fields=("case",),
        expected_keys=[("add",), ("mul",), ("div",)],
        max_ratio=1.0,
        ratio_field="median_ratio",
    )

    assert failures == [
        "add: candle median_ratio 1.50x > 1.00x",
        "mul/candle: error: boom",
        "mul: missing torch_npu result",
        "div: missing candle result",
        "div: missing torch_npu result",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_npu_perf_gates.py -q
```

Expected: FAIL because `benchmarks.npu_perf_gates` does not exist.

- [ ] **Step 3: Implement shared helpers**

Create `benchmarks/npu_perf_gates.py`:

```python
"""Shared metrics and ratio gates for NPU performance benchmarks."""


def _round_ms(value):
    return round(float(value), 4)


def _nearest_percentile(sorted_values, percentile):
    if not sorted_values:
        return 0.0
    index = round((len(sorted_values) - 1) * percentile)
    return sorted_values[index]


def summarize_samples(samples):
    """Return stable millisecond distribution fields for benchmark samples."""
    if not samples:
        return {
            "sample_count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p10_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
        }

    values = sorted(float(sample) for sample in samples)
    return {
        "sample_count": len(values),
        "mean_ms": _round_ms(sum(values) / len(values)),
        "median_ms": _round_ms(_nearest_percentile(values, 0.50)),
        "min_ms": _round_ms(values[0]),
        "max_ms": _round_ms(values[-1]),
        "p10_ms": _round_ms(_nearest_percentile(values, 0.10)),
        "p90_ms": _round_ms(_nearest_percentile(values, 0.90)),
        "p95_ms": _round_ms(_nearest_percentile(values, 0.95)),
    }


def _row_key(row, key_fields):
    return tuple(row[field] for field in key_fields)


def _ratio_field_name(metric):
    if metric.endswith("_ms"):
        return f"{metric[:-3]}_ratio"
    return f"{metric}_ratio"


def annotate_ratio_rows(rows, *, key_fields, metric="median_ms"):
    """Annotate Candle rows with Candle/torch_npu ratio for matching keys."""
    by_key = {}
    for row in rows:
        by_key.setdefault(_row_key(row, key_fields), {})[row.get("framework")] = row

    ratio_field = _ratio_field_name(metric)
    for framework_rows in by_key.values():
        torch_ref = framework_rows.get("torch_npu")
        candle = framework_rows.get("candle")
        if torch_ref is not None and torch_ref.get("status", "ok") == "ok":
            torch_ref[ratio_field] = 1.0
        if candle is None or torch_ref is None:
            continue
        if candle.get("status", "ok") != "ok" or torch_ref.get("status", "ok") != "ok":
            continue
        ref_value = torch_ref.get(metric, 0.0)
        candle_value = candle.get(metric, 0.0)
        if ref_value:
            candle[ratio_field] = round(float(candle_value) / float(ref_value), 4)


def collect_ratio_failures(rows, *, key_fields, expected_keys, max_ratio, ratio_field):
    """Return human-readable gate failures for missing, errored, or slow rows."""
    by_key = {}
    for row in rows:
        by_key.setdefault(_row_key(row, key_fields), {})[row.get("framework")] = row

    failures = []
    for key in expected_keys:
        by_framework = by_key.get(tuple(key), {})
        label = "/".join(str(part) for part in key)
        candle = by_framework.get("candle")
        torch_ref = by_framework.get("torch_npu")

        if candle is None:
            failures.append(f"{label}: missing candle result")
        elif candle.get("status", "ok") != "ok":
            failures.append(f"{label}/candle: {candle['status']}")

        if torch_ref is None:
            failures.append(f"{label}: missing torch_npu result")
        elif torch_ref.get("status", "ok") != "ok":
            failures.append(f"{label}/torch_npu: {torch_ref['status']}")

        if candle is None or torch_ref is None:
            continue
        if candle.get("status", "ok") != "ok" or torch_ref.get("status", "ok") != "ok":
            continue

        ratio = candle.get(ratio_field)
        if ratio is None:
            failures.append(f"{label}: missing candle {ratio_field}")
        elif ratio > max_ratio:
            failures.append(
                f"{label}: candle {ratio_field} {ratio:.2f}x > {max_ratio:.2f}x"
            )

    return failures
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_npu_perf_gates.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1 changes**

```bash
git add benchmarks/npu_perf_gates.py tests/cpu/test_npu_perf_gates.py
git commit -m "bench(npu): add shared performance gate helpers"
```

---

### Task 2: Extend L0 op benchmark JSON and ratio gates

**Files:**
- Modify: `benchmarks/op_benchmark_npu/runner.py:1-40`
- Modify: `benchmarks/op_benchmark_npu/worker.py:6-110`
- Modify: `benchmarks/op_benchmark_npu/report.py:1-135`
- Modify: `benchmarks/op_benchmark_npu/run.py:1-136`
- Test: `tests/cpu/test_npu_op_benchmark_report.py`

- [ ] **Step 1: Write failing tests for op benchmark reports and gates**

Create `tests/cpu/test_npu_op_benchmark_report.py`:

```python
from benchmarks.op_benchmark_npu.report import generate_report, ratio_failures


def test_generate_report_includes_distribution_columns():
    candle = [{
        "framework": "candle",
        "op": "add",
        "mode": "fwd",
        "dtype": "fp16",
        "scenario": "infer",
        "median_ms": 2.0,
        "p10_ms": 1.5,
        "p90_ms": 2.5,
        "status": "ok",
        "median_ratio": 2.0,
    }]
    torch_ref = [{
        "framework": "torch_npu",
        "op": "add",
        "mode": "fwd",
        "dtype": "fp16",
        "scenario": "infer",
        "median_ms": 1.0,
        "p10_ms": 0.8,
        "p90_ms": 1.2,
        "status": "ok",
        "median_ratio": 1.0,
    }]

    report = generate_report(candle, torch_ref, ["add"], ["fp16"], ["infer"], ["fwd"])

    assert "| Op | candle median | torch_npu median | ratio | candle p10/p90 | torch_npu p10/p90 | impact |" in report
    assert "| add | 2.0000 | 1.0000 | 2.00x | 1.5000/2.5000 | 0.8000/1.2000 | 1.0000 |" in report


def test_ratio_failures_reports_slow_op():
    candle = [{
        "framework": "candle",
        "op": "add",
        "mode": "fwd",
        "dtype": "fp16",
        "scenario": "infer",
        "median_ms": 2.0,
        "status": "ok",
        "median_ratio": 2.0,
    }]
    torch_ref = [{
        "framework": "torch_npu",
        "op": "add",
        "mode": "fwd",
        "dtype": "fp16",
        "scenario": "infer",
        "median_ms": 1.0,
        "status": "ok",
        "median_ratio": 1.0,
    }]

    failures = ratio_failures(
        candle,
        torch_ref,
        op_names=["add"],
        dtype_keys=["fp16"],
        scen_keys=["infer"],
        mode_keys=["fwd"],
        max_ratio=1.0,
    )

    assert failures == ["add/fwd/fp16/infer: candle median_ratio 2.00x > 1.00x"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_npu_op_benchmark_report.py -q
```

Expected: FAIL because `ratio_failures` is not exported and the report columns are not extended.

- [ ] **Step 3: Update op benchmark sample summary helper**

Modify `benchmarks/op_benchmark_npu/runner.py` to use the shared summary helper:

```python
import time

from benchmarks.npu_perf_gates import summarize_samples


def benchmark_op(fn, warmup=10, iters=50, sync=None, setup=None, cleanup=None):
    """Run fn with warmup, then time iters iterations. Returns list of ms."""
    for _ in range(warmup):
        if setup:
            setup()
        fn()
        if sync:
            sync()
        if cleanup:
            cleanup()

    times = []
    for _ in range(iters):
        if setup:
            setup()
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        if cleanup:
            cleanup()
    return times


def summarize(samples):
    """Return stable millisecond distribution fields for benchmark samples."""
    return summarize_samples(samples)
```

- [ ] **Step 4: Update op benchmark worker rows**

In `benchmarks/op_benchmark_npu/worker.py`, replace lines 85-95 with:

```python
                    summary = summarize(samples)
                    results.append({
                        "framework": "torch_npu" if args.framework == "torch" else args.framework,
                        "op": op_name,
                        "mode": case.get("mode", "fwd"),
                        "dtype": dtype_key,
                        "scenario": scen_key,
                        **summary,
                        "status": "ok",
                    })
```

Replace lines 97-105 with:

```python
                    results.append({
                        "framework": "torch_npu" if args.framework == "torch" else args.framework,
                        "op": op_name,
                        "mode": case.get("mode", "fwd"),
                        "dtype": dtype_key,
                        "scenario": scen_key,
                        "sample_count": 0,
                        "mean_ms": 0.0,
                        "median_ms": 0.0,
                        "min_ms": 0.0,
                        "max_ms": 0.0,
                        "p10_ms": 0.0,
                        "p90_ms": 0.0,
                        "p95_ms": 0.0,
                        "status": f"error: {e}",
                    })
```

- [ ] **Step 5: Update op benchmark report generation**

At the top of `benchmarks/op_benchmark_npu/report.py`, add the shared helpers import:

```python
from benchmarks.npu_perf_gates import annotate_ratio_rows, collect_ratio_failures
```

Replace `_build_section` with:

```python
def _fmt_ms(value):
    return f"{value:.4f}" if value is not None else "—"


def _build_section(mode_key, dtype_key, scen_key, candle_map, torch_map, op_names):
    dtype_label = DTYPES[dtype_key]
    mode_label = MODES[mode_key]
    scen_label = SCENARIOS[scen_key]["label"]
    header = f"### {mode_label} — {dtype_label} — {scen_label}"

    lines = [header, ""]
    lines.append("| Op | candle median | torch_npu median | ratio | candle p10/p90 | torch_npu p10/p90 | impact |")
    lines.append("|---|---|---|---|---|---|---|")

    ratios = []
    rows = []
    for op in op_names:
        c = candle_map.get((op, mode_key, dtype_key, scen_key))
        t = torch_map.get((op, mode_key, dtype_key, scen_key))

        c_ok = c and c["status"] == "ok"
        t_ok = t and t["status"] == "ok"
        c_med = c["median_ms"] if c_ok else None
        t_med = t["median_ms"] if t_ok else None

        if c_ok and t_ok and t_med > 0:
            ratio = c.get("median_ratio", c_med / t_med)
            impact = c_med - t_med
            ratios.append((op, ratio))
            ratio_str = f"{ratio:.2f}x"
            impact_str = f"{impact:.4f}"
        else:
            ratio = None
            impact = None
            ratio_str = "N/A"
            impact_str = "N/A"

        candle_dist = "ERR" if c and c["status"] != "ok" else f"{_fmt_ms(c.get('p10_ms') if c else None)}/{_fmt_ms(c.get('p90_ms') if c else None)}"
        torch_dist = "ERR" if t and t["status"] != "ok" else f"{_fmt_ms(t.get('p10_ms') if t else None)}/{_fmt_ms(t.get('p90_ms') if t else None)}"
        c_str = c["status"][:30] if c and c["status"] != "ok" else _fmt_ms(c_med)
        t_str = t["status"][:30] if t and t["status"] != "ok" else _fmt_ms(t_med)

        rows.append((
            ratio if ratio is not None else -1.0,
            impact if impact is not None else -1.0,
            f"| {op} | {c_str} | {t_str} | {ratio_str} | {candle_dist} | {torch_dist} | {impact_str} |",
        ))

    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    lines.extend(row for _, _, row in rows)
    lines.append("")
    return lines, ratios
```

Add this function below `generate_report`:

```python
def ratio_failures(candle_results, torch_results, op_names, dtype_keys, scen_keys, mode_keys, max_ratio):
    rows = list(candle_results) + list(torch_results)
    annotate_ratio_rows(rows, key_fields=("op", "mode", "dtype", "scenario"), metric="median_ms")
    expected_keys = [
        (op, mode, dtype, scen)
        for mode in mode_keys
        for dtype in dtype_keys
        for scen in scen_keys
        for op in op_names
    ]
    return collect_ratio_failures(
        rows,
        key_fields=("op", "mode", "dtype", "scenario"),
        expected_keys=expected_keys,
        max_ratio=max_ratio,
        ratio_field="median_ratio",
    )
```

At the start of `generate_report`, after `torch_map` is created, add:

```python
    rows = list(candle_results) + list(torch_results)
    annotate_ratio_rows(rows, key_fields=("op", "mode", "dtype", "scenario"), metric="median_ms")
```

- [ ] **Step 6: Add op benchmark JSON output and ratio gate CLI**

In `benchmarks/op_benchmark_npu/run.py`, update imports:

```python
from .report import generate_report, print_terminal, ratio_failures, write_markdown
```

Add parser arguments after `--output`:

```python
    parser.add_argument("--json-output", default=None,
                        help="Write merged benchmark payload to this JSON path, or '-' for stdout")
    parser.add_argument("--fail-on-ratio", action="store_true",
                        help="Exit nonzero if any Candle op is slower than --max-ratio")
    parser.add_argument("--max-ratio", type=float, default=1.0,
                        help="Maximum allowed Candle/torch_npu median ratio for --fail-on-ratio")
```

After `args = parser.parse_args()`, add:

```python
    if args.max_ratio <= 0:
        parser.error("--max-ratio must be > 0")
```

After `report = generate_report(...)`, add:

```python
    failures = []
    if args.fail_on_ratio:
        failures = ratio_failures(
            candle_results,
            torch_results,
            op_names,
            dtype_keys,
            scen_keys,
            mode_keys,
            args.max_ratio,
        )
```

After `print_terminal(report)`, add:

```python
    if failures:
        print("\n# Ratio gate failures")
        for failure in failures:
            print(f"- {failure}")

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
    if args.json_output:
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.json_output == "-":
            print(text)
        else:
            with open(args.json_output, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.write("\n")
```

Before the end of `main()`, after optional markdown writing, add:

```python
    if failures:
        sys.exit(1)
```

- [ ] **Step 7: Run op benchmark unit tests**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_npu_perf_gates.py tests/cpu/test_npu_op_benchmark_report.py -q
```

Expected: PASS.

- [ ] **Step 8: Run op benchmark import smoke without NPU**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m benchmarks.op_benchmark_npu.run --help
```

Expected: command prints help including `--json-output`, `--fail-on-ratio`, and `--max-ratio`.

- [ ] **Step 9: Commit Task 2 changes**

```bash
git add benchmarks/op_benchmark_npu/runner.py \
  benchmarks/op_benchmark_npu/worker.py \
  benchmarks/op_benchmark_npu/report.py \
  benchmarks/op_benchmark_npu/run.py \
  tests/cpu/test_npu_op_benchmark_report.py
git commit -m "bench(npu): add single-op ratio gates"
```

---

### Task 3: Make pipeline cases framework-neutral

**Files:**
- Modify: `benchmarks/pipeline_npu/cases.py:1-297`
- Test: `tests/cpu/test_pipeline_npu_framework_neutral.py`

- [ ] **Step 1: Write failing tests for framework-neutral builders**

Create `tests/cpu/test_pipeline_npu_framework_neutral.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_pipeline_npu_framework_neutral.py -q
```

Expected: FAIL because builders currently accept `(device, dtype)` and import Candle globally.

- [ ] **Step 3: Update imports and function signatures in pipeline cases**

In `benchmarks/pipeline_npu/cases.py`, delete the first two import lines:

```python
import candle
import candle.nn.functional as F
```

Change every builder signature and call as follows:

```python
def _case_a1(torch_mod, F, device, dtype):
```

```python
def _case_a2(torch_mod, F, device, dtype):
```

```python
def _case_a2s(torch_mod, F, device, dtype):
```

```python
def _case_a3(torch_mod, F, device, dtype):
```

```python
def _block(torch_mod, F, device, dtype, *, b, s, h, heads):
```

```python
def _case_b1(torch_mod, F, device, dtype):
    return _block(torch_mod, F, device, dtype, b=1, s=512, h=2048, heads=16)
```

```python
def _case_b1s(torch_mod, F, device, dtype):
    return _block(torch_mod, F, device, dtype, b=2, s=256, h=512, heads=8)
```

```python
def _case_b2(torch_mod, F, device, dtype):
    return _block(torch_mod, F, device, dtype, b=4, s=128, h=1024, heads=8)
```

```python
def _case_b3(torch_mod, F, device, dtype):
    return _block(torch_mod, F, device, dtype, b=1, s=2048, h=1024, heads=16)
```

```python
def _case_c1(torch_mod, F, device, dtype):
    block = _block(torch_mod, F, device, dtype, b=1, s=512, h=1024, heads=16)
```

```python
def _case_c2(torch_mod, F, device, dtype):
    block = _block(torch_mod, F, device, dtype, b=2, s=256, h=1024, heads=16)
```

```python
def _case_d1(torch_mod, F, device, dtype):
    block = _block(torch_mod, F, device, dtype, b=2, s=256, h=512, heads=8)
```

```python
def _case_d2(torch_mod, F, device, dtype):
```

Inside the file, replace every `candle.randn` with `torch_mod.randn` and every `candle.matmul` with `torch_mod.matmul`.

The `CASE_METADATA` dictionary does not need to change; each `"builder"` value still references the same function names.

- [ ] **Step 4: Run framework-neutral builder tests**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_pipeline_npu_framework_neutral.py -q
```

Expected: PASS.

- [ ] **Step 5: Run existing pipeline smoke tests**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/npu/test_pipeline_npu_bench_smoke.py -q
```

Expected: FAIL until Task 4 updates `bench.py` to call builders with framework arguments.

- [ ] **Step 6: Commit Task 3 changes after Task 4 passes**

Do not commit Task 3 until Task 4 updates the caller and both test files pass together.

---

### Task 4: Add L1 pipeline Candle vs torch_npu runner

**Files:**
- Modify: `benchmarks/pipeline_npu/bench.py:1-61`
- Create: `benchmarks/pipeline_npu/worker.py`
- Create: `benchmarks/pipeline_npu/run.py`
- Modify: `tests/npu/test_pipeline_npu_bench_smoke.py:1-18`
- Test: `tests/cpu/test_pipeline_npu_report.py`

- [ ] **Step 1: Write failing tests for pipeline ratio gates**

Create `tests/cpu/test_pipeline_npu_report.py`:

```python
from benchmarks.pipeline_npu.run import annotate_pipeline_ratios, pipeline_ratio_failures


def test_annotate_pipeline_ratios_adds_candle_ratio():
    rows = [
        {"framework": "torch_npu", "case_id": "A1", "mode": "eager", "median_ms": 10.0, "status": "ok"},
        {"framework": "candle", "case_id": "A1", "mode": "eager", "median_ms": 8.0, "status": "ok"},
    ]

    annotate_pipeline_ratios(rows)

    candle = next(row for row in rows if row["framework"] == "candle")
    assert candle["median_ratio"] == 0.8


def test_pipeline_ratio_failures_require_candle_to_be_faster_than_torch_npu():
    rows = [
        {"framework": "torch_npu", "case_id": "A1", "mode": "eager", "median_ms": 10.0, "status": "ok", "median_ratio": 1.0},
        {"framework": "candle", "case_id": "A1", "mode": "eager", "median_ms": 10.5, "status": "ok", "median_ratio": 1.05},
    ]

    failures = pipeline_ratio_failures(rows, case_ids=["A1"], mode="eager", max_ratio=0.99)

    assert failures == ["A1/eager: candle median_ratio 1.05x > 0.99x"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_pipeline_npu_report.py -q
```

Expected: FAIL because `benchmarks.pipeline_npu.run` does not exist.

- [ ] **Step 3: Update `bench.py` to import frameworks and preserve CPU behavior**

Replace `benchmarks/pipeline_npu/bench.py` with:

```python
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
```

- [ ] **Step 4: Add pipeline worker**

Create `benchmarks/pipeline_npu/worker.py`:

```python
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
```

- [ ] **Step 5: Add pipeline orchestrator**

Create `benchmarks/pipeline_npu/run.py`:

```python
"""Pipeline NPU benchmark orchestrator for Candle vs torch_npu."""
import argparse
import json
import os
import subprocess
import sys

from benchmarks.npu_perf_gates import annotate_ratio_rows, collect_ratio_failures
from .bench import CASES


def _spawn_worker(framework, args):
    cmd = [
        args.python or sys.executable,
        "-m", "benchmarks.pipeline_npu.worker",
        "--framework", framework,
        "--cases", args.cases,
        "--mode", args.mode,
        "--device", args.device,
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
    ]
    env = os.environ.copy()
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
    old_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_dir if not old_path else os.pathsep.join((src_dir, old_path))
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=1800, check=False)
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return []
    return json.loads(proc.stdout)


def annotate_pipeline_ratios(rows):
    annotate_ratio_rows(rows, key_fields=("case_id", "mode"), metric="median_ms")


def pipeline_ratio_failures(rows, *, case_ids, mode, max_ratio):
    expected_keys = [(case_id, mode) for case_id in case_ids]
    return collect_ratio_failures(
        rows,
        key_fields=("case_id", "mode"),
        expected_keys=expected_keys,
        max_ratio=max_ratio,
        ratio_field="median_ratio",
    )


def _print_table(rows):
    print("case | framework | mode | median ms | ratio | status")
    print("---|---|---|---|---|---")
    for row in sorted(rows, key=lambda item: (item["case_id"], item["framework"], item["mode"])):
        ratio = row.get("median_ratio")
        ratio_str = "-" if ratio is None else f"{ratio:.2f}x"
        print(
            f"{row['case_id']} | {row['framework']} | {row['mode']} | "
            f"{row.get('median_ms', 0.0):.4f} | {ratio_str} | {row.get('status', 'ok')}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=",".join(CASES.keys()))
    parser.add_argument("--mode", default="eager", choices=["eager", "pipeline"])
    parser.add_argument("--device", default="npu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--python", default=None, help="Python executable for workers")
    parser.add_argument("--json-output", default=None, help="Write merged payload to path, or '-' for stdout")
    parser.add_argument("--fail-on-ratio", action="store_true")
    parser.add_argument("--max-ratio", type=float, default=0.99)
    args = parser.parse_args()

    if args.max_ratio <= 0:
        parser.error("--max-ratio must be > 0")

    case_ids = [case_id.strip() for case_id in args.cases.split(",") if case_id.strip()]
    unknown = [case_id for case_id in case_ids if case_id not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown}; known: {list(CASES)}")

    rows = _spawn_worker("candle", args) + _spawn_worker("torch_npu", args)
    annotate_pipeline_ratios(rows)
    failures = []
    if args.fail_on_ratio:
        failures = pipeline_ratio_failures(rows, case_ids=case_ids, mode=args.mode, max_ratio=args.max_ratio)

    _print_table(rows)
    if failures:
        print("\n# Ratio gate failures")
        for failure in failures:
            print(f"- {failure}")

    payload = {
        "warmup": args.warmup,
        "iters": args.iters,
        "cases": case_ids,
        "mode": args.mode,
        "device": args.device,
        "max_ratio": args.max_ratio,
        "failures": failures,
        "results": rows,
    }
    if args.json_output:
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.json_output == "-":
            print(text)
        else:
            with open(args.json_output, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.write("\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Update pipeline smoke tests for new result fields**

Modify `tests/npu/test_pipeline_npu_bench_smoke.py`:

```python
from benchmarks.pipeline_npu.bench import CASES, run_case


def test_pipeline_bench_smoke_cpu():
    case = CASES["A1"]
    result = run_case(case, framework="candle", device="cpu", mode="eager", warmup=1, iters=1)
    assert result["framework"] == "candle"
    assert result["case_id"] == "A1"
    assert result["mode"] == "eager"
    assert result["status"] == "ok"
    assert "mean_ms" in result
    assert "median_ms" in result
    assert "p95_ms" in result
    assert isinstance(result["mean_ms"], float)
    assert isinstance(result["median_ms"], float)
    assert isinstance(result["p95_ms"], float)


def test_pipeline_bench_cases_matrix():
    for key in ["A1", "A2", "A2s", "A3", "B1", "B1s", "B2", "B3", "C1", "C2", "D1", "D2"]:
        assert key in CASES
```

- [ ] **Step 7: Run pipeline tests**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_pipeline_npu_framework_neutral.py \
    tests/cpu/test_pipeline_npu_report.py \
    tests/npu/test_pipeline_npu_bench_smoke.py -q
```

Expected: PASS on CPU-only machines because NPU-specific cases are not executed with `device="npu"` in tests.

- [ ] **Step 8: Run pipeline CLI help smoke**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m benchmarks.pipeline_npu.run --help
```

Expected: command prints help including `--json-output`, `--fail-on-ratio`, and `--max-ratio`.

- [ ] **Step 9: Commit Tasks 3 and 4 together**

```bash
git add benchmarks/pipeline_npu/cases.py \
  benchmarks/pipeline_npu/bench.py \
  benchmarks/pipeline_npu/worker.py \
  benchmarks/pipeline_npu/run.py \
  tests/cpu/test_pipeline_npu_framework_neutral.py \
  tests/cpu/test_pipeline_npu_report.py \
  tests/npu/test_pipeline_npu_bench_smoke.py
git commit -m "bench(npu): compare pipeline blocks with torch_npu"
```

---

### Task 5: Extend L2 end-to-end benchmark ratios

**Files:**
- Modify: `benchmarks/perf_candle_vs_torch_npu.py:179-513`
- Test: `tests/cpu/test_perf_candle_vs_torch_npu_ratios.py`

- [ ] **Step 1: Write failing tests for total ratio annotation and failures**

Create `tests/cpu/test_perf_candle_vs_torch_npu_ratios.py`:

```python
from benchmarks.perf_candle_vs_torch_npu import _annotate_ratios, _ratio_failures


def test_annotate_ratios_adds_total_ratio():
    results = [
        {
            "framework": "torch_npu",
            "case": "xfmr",
            "fwd_ms_median": 2.0,
            "bwd_ms_median": 3.0,
            "total_ms_median": 5.0,
        },
        {
            "framework": "candle",
            "case": "xfmr",
            "fwd_ms_median": 1.5,
            "bwd_ms_median": 2.0,
            "total_ms_median": 3.5,
        },
    ]

    _annotate_ratios(results)

    candle = next(row for row in results if row["framework"] == "candle")
    assert candle["fwd_ratio"] == 0.75
    assert candle["bwd_ratio"] == 2.0 / 3.0
    assert candle["total_ratio"] == 0.7


def test_ratio_failures_checks_total_ratio():
    results = [
        {
            "framework": "torch_npu",
            "case": "xfmr",
            "fwd_ms_median": 2.0,
            "bwd_ms_median": 3.0,
            "total_ms_median": 5.0,
        },
        {
            "framework": "candle",
            "case": "xfmr",
            "fwd_ms_median": 2.0,
            "bwd_ms_median": 3.0,
            "total_ms_median": 5.2,
            "fwd_ratio": 1.0,
            "bwd_ratio": 1.0,
            "total_ratio": 1.04,
        },
    ]

    failures = _ratio_failures(
        results,
        cases=["xfmr"],
        max_fwd_ratio=1.0,
        max_bwd_ratio=1.0,
        max_total_ratio=0.99,
    )

    assert failures == ["xfmr: total ratio 1.04x > 0.99x"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_perf_candle_vs_torch_npu_ratios.py -q
```

Expected: FAIL because `_ratio_failures` does not accept `max_total_ratio` and total ratios are not annotated.

- [ ] **Step 3: Add total latency samples and distribution fields**

In `benchmarks/perf_candle_vs_torch_npu.py`, import the shared summary helper near the top:

```python
from benchmarks.npu_perf_gates import summarize_samples
```

In `run_worker`, after `bwd_samples = []`, add:

```python
    total_samples = []
```

After appending `bwd_samples`, add:

```python
        total_samples.append((t2 - t0) * 1000.0)
```

Replace the `result = { ... }` block at lines 221-229 with:

```python
    fwd_summary = summarize_samples(fwd_samples)
    bwd_summary = summarize_samples(bwd_samples)
    total_summary = summarize_samples(total_samples)
    result = {
        "framework": framework,
        "case": case,
        "iters": iters,
        "fwd_ms_median": fwd_summary["median_ms"],
        "bwd_ms_median": bwd_summary["median_ms"],
        "total_ms_median": total_summary["median_ms"],
        "fwd_ms_min": fwd_summary["min_ms"],
        "bwd_ms_min": bwd_summary["min_ms"],
        "total_ms_min": total_summary["min_ms"],
        "fwd_ms_p10": fwd_summary["p10_ms"],
        "bwd_ms_p10": bwd_summary["p10_ms"],
        "total_ms_p10": total_summary["p10_ms"],
        "fwd_ms_p90": fwd_summary["p90_ms"],
        "bwd_ms_p90": bwd_summary["p90_ms"],
        "total_ms_p90": total_summary["p90_ms"],
    }
```

- [ ] **Step 4: Add total ratio annotation**

In `_annotate_ratios`, after `torch_bwd = torch_ref.get("bwd_ms_median")`, add:

```python
        torch_total = torch_ref.get("total_ms_median")
```

After the backward-ratio block, add:

```python
        if torch_total:
            candle["total_ratio"] = candle["total_ms_median"] / torch_total
```

- [ ] **Step 5: Add total ratio failure gate**

Change `_ratio_failures` signature to:

```python
def _ratio_failures(results, cases, max_fwd_ratio, max_bwd_ratio, max_total_ratio):
```

Inside `_ratio_failures`, after `bwd_ratio = candle.get("bwd_ratio")`, add:

```python
        total_ratio = candle.get("total_ratio")
```

Replace the missing-ratio check with:

```python
        if fwd_ratio is None or bwd_ratio is None or total_ratio is None:
            failures.append(f"{case}: unable to compute candle/torch_npu ratio")
            continue
```

After the backward-ratio failure block, add:

```python
        if total_ratio > max_total_ratio:
            failures.append(f"{case}: total ratio {total_ratio:.2f}x > {max_total_ratio:.2f}x")
```

- [ ] **Step 6: Add CLI argument and payload metadata**

In `main()`, after `--max-bwd-ratio`, add:

```python
    parser.add_argument("--max-total-ratio", type=float, default=0.99,
                        help="maximum allowed candle/torch_npu total ratio for --fail-on-ratio")
```

After the `args.max_bwd_ratio` validation, add:

```python
    if args.max_total_ratio <= 0:
        parser.error("--max-total-ratio must be > 0")
```

Change the `_ratio_failures` call to:

```python
        failures = _ratio_failures(
            results,
            cases,
            args.max_fwd_ratio,
            args.max_bwd_ratio,
            args.max_total_ratio,
        )
```

Add `"max_total_ratio": args.max_total_ratio,` to the JSON payload next to `max_fwd_ratio` and `max_bwd_ratio`.

- [ ] **Step 7: Update L2 table output to include total ratio**

In `_print_table`, change:

```python
    header = ["algo", "fwd ms", "bwd ms", "ratio"]
```

to:

```python
    header = ["algo", "fwd ms", "bwd ms", "total ms", "ratio"]
```

Change each successful row from:

```python
                row = [algo, _fmt(fwd), _fmt(bwd), ratio]
```

to:

```python
                total = r.get("total_ms_median")
                row = [algo, _fmt(fwd), _fmt(bwd), _fmt(total), ratio]
```

Change error rows from four fields to five fields:

```python
                row = [algo, "err", "err", "err", "-"]
```

Change the Candle ratio string to:

```python
                    total_part = f" / t {r['total_ratio']:.2f}x" if "total_ratio" in r else ""
                    ratio = f"f {r['fwd_ratio']:.2f}x / b {r['bwd_ratio']:.2f}x{total_part}"
```

- [ ] **Step 8: Run L2 unit tests**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_perf_candle_vs_torch_npu_ratios.py -q
```

Expected: PASS.

- [ ] **Step 9: Run L2 CLI help smoke**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python benchmarks/perf_candle_vs_torch_npu.py --help
```

Expected: command prints help including `--max-total-ratio`.

- [ ] **Step 10: Commit Task 5 changes**

```bash
git add benchmarks/perf_candle_vs_torch_npu.py tests/cpu/test_perf_candle_vs_torch_npu_ratios.py
git commit -m "bench(npu): add end-to-end total ratio gate"
```

---

### Task 6: Add top-level NPU performance gate orchestrator

**Files:**
- Create: `benchmarks/npu_perf_gate.py`
- Test: `tests/cpu/test_npu_perf_gate.py`

- [ ] **Step 1: Write failing tests for orchestrator command construction**

Create `tests/cpu/test_npu_perf_gate.py`:

```python
from benchmarks.npu_perf_gate import build_commands


def test_build_commands_writes_l0_l1_l2_artifacts_under_output_dir():
    commands = build_commands(
        output_dir="results/npu_perf_gate/test",
        warmup=1,
        iters=2,
        dtype="fp16",
        cases="xfmr",
        pipeline_cases="A1,B1s",
    )

    assert commands[0][-5:] == ["--max-ratio", "1.0", "--json-output", "results/npu_perf_gate/test/l0_ops.json", "--fail-on-ratio"]
    assert "benchmarks.op_benchmark_npu.run" in commands[0]
    assert "benchmarks.pipeline_npu.run" in commands[1]
    assert "benchmarks/perf_candle_vs_torch_npu.py" in commands[2]
    assert "results/npu_perf_gate/test/l1_pipeline.json" in commands[1]
    assert "results/npu_perf_gate/test/l2_models.json" in commands[2]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_npu_perf_gate.py -q
```

Expected: FAIL because `benchmarks.npu_perf_gate` does not exist.

- [ ] **Step 3: Implement top-level orchestrator**

Create `benchmarks/npu_perf_gate.py`:

```python
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
```

- [ ] **Step 4: Run orchestrator unit test**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_npu_perf_gate.py -q
```

Expected: PASS.

- [ ] **Step 5: Run orchestrator help smoke**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python benchmarks/npu_perf_gate.py --help
```

Expected: command prints help including `--output-dir`, `--pipeline-cases`, and `--cases`.

- [ ] **Step 6: Commit Task 6 changes**

```bash
git add benchmarks/npu_perf_gate.py tests/cpu/test_npu_perf_gate.py
git commit -m "bench(npu): add performance gate orchestrator"
```

---

### Task 7: Final validation for Phase 0

**Files:**
- No source changes unless validation finds a defect in prior tasks.

- [ ] **Step 1: Run targeted CPU/unit tests**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest \
    tests/cpu/test_npu_perf_gates.py \
    tests/cpu/test_npu_op_benchmark_report.py \
    tests/cpu/test_pipeline_npu_framework_neutral.py \
    tests/cpu/test_pipeline_npu_report.py \
    tests/cpu/test_perf_candle_vs_torch_npu_ratios.py \
    tests/cpu/test_npu_perf_gate.py \
    tests/npu/test_pipeline_npu_bench_smoke.py \
    -q
```

Expected: PASS.

- [ ] **Step 2: Run benchmark CLI help commands**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m benchmarks.op_benchmark_npu.run --help
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m benchmarks.pipeline_npu.run --help
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python benchmarks/perf_candle_vs_torch_npu.py --help
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python benchmarks/npu_perf_gate.py --help
```

Expected: all commands exit 0 and print their CLI help.

- [ ] **Step 3: Run pylint only if implementation touches import style or public modules flagged by local policy**

Run when lint is required for the PR scope:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  pylint benchmarks/ tests/cpu/test_npu_perf_gates.py \
    tests/cpu/test_npu_op_benchmark_report.py \
    tests/cpu/test_pipeline_npu_framework_neutral.py \
    tests/cpu/test_pipeline_npu_report.py \
    tests/cpu/test_perf_candle_vs_torch_npu_ratios.py \
    tests/cpu/test_npu_perf_gate.py \
    --rcfile=.github/pylint.conf
```

Expected: PASS or only pre-existing benchmark lint findings documented before PR creation.

- [ ] **Step 4: On NPU hardware, run a low-iteration non-gating artifact pass**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python benchmarks/npu_perf_gate.py \
    --output-dir results/npu_perf_gate/smoke \
    --warmup 1 \
    --iters 2 \
    --dtype fp16 \
    --cases xfmr \
    --pipeline-cases A1,A2s
```

Expected: The command may exit nonzero because ratio gates reflect current performance gaps. It must write any completed JSON artifacts under `results/npu_perf_gate/smoke/` before the failing gate, and failures must name specific slow or missing rows.

- [ ] **Step 5: Commit final validation fixes if any were needed**

If validation required fixes, commit only the files changed by those fixes:

```bash
git add <fixed-files>
git commit -m "test(npu): stabilize performance gate validation"
```

If no fixes were needed, do not create an empty commit.

---

## Execution Notes

- Use `.worktrees/npu-performance-core-redesign-spec` or a new worktree branched from this plan branch for implementation.
- Do not run destructive git commands.
- Do not push or create a PR unless explicitly requested.
- Current Phase 0 gates are expected to expose failures; do not weaken thresholds to make current performance look better.
- The single-op L0 threshold is `max_ratio=1.0` because single ops must not be slower than torch_npu.
- The L1/L2 model thresholds are `max_ratio=0.99` because composed workloads must be faster than torch_npu.
