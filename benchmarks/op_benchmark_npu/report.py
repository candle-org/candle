"""Report generation: terminal table + markdown file."""
import os
from datetime import datetime

from benchmarks.npu_perf_gates import annotate_ratio_rows, collect_ratio_failures

from .cases import SCENARIOS, DTYPES, MODES


_KEY_FIELDS = ("op", "mode", "dtype", "scenario")


def _row_key(row):
    return tuple(row[field] for field in _KEY_FIELDS)


def _build_maps(candle_results, torch_results):
    return ({_row_key(row): row for row in candle_results},
            {_row_key(row): row for row in torch_results})


def _format_p10_p90(row):
    if row is None or row.get("status") != "ok":
        return "N/A"
    return f"{row.get('p10_ms', 0.0):.4f}/{row.get('p90_ms', 0.0):.4f}"


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
        key = (op, mode_key, dtype_key, scen_key)
        c = candle_map.get(key)
        t = torch_map.get(key)

        c_med = c["median_ms"] if c and c["status"] == "ok" else None
        t_med = t["median_ms"] if t and t["status"] == "ok" else None

        c_str = f"{c_med:.4f}" if c_med is not None else ("ERR" if c else "—")
        t_str = f"{t_med:.4f}" if t_med is not None else ("ERR" if t else "—")

        ratio = c.get("median_ratio") if c else None
        impact = None
        if c_med is not None and t_med is not None:
            impact = c_med - t_med
            if ratio is not None:
                ratios.append((op, ratio))
                r_str = f"{ratio:.2f}x"
            else:
                r_str = "N/A"
            impact_str = f"{impact:.4f}"
        else:
            r_str = "N/A"
            impact_str = "N/A"

        if c and c["status"] != "ok":
            c_str = c["status"][:30]
        if t and t["status"] != "ok":
            t_str = t["status"][:30]

        rows.append((ratio if ratio is not None else -1.0,
                     impact if impact is not None else -1.0,
                     f"| {op} | {c_str} | {t_str} | {r_str} | "
                     f"{_format_p10_p90(c)} | {_format_p10_p90(t)} | "
                     f"{impact_str} |"))

    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    lines.extend(row for _, _, row in rows)
    lines.append("")
    return lines, ratios


def ratio_failures(candle_results, torch_results, op_names, dtype_keys, scen_keys, mode_keys,
                   max_ratio):
    """Return single-op median ratio gate failures for expected benchmark rows."""
    rows = list(candle_results) + list(torch_results)
    annotate_ratio_rows(rows, key_fields=_KEY_FIELDS, metric="median_ms")
    expected_keys = [
        (op, mode_key, dtype_key, scen_key)
        for mode_key in mode_keys
        for dtype_key in dtype_keys
        for scen_key in scen_keys
        for op in op_names
    ]
    return collect_ratio_failures(
        rows,
        key_fields=_KEY_FIELDS,
        expected_keys=expected_keys,
        max_ratio=max_ratio,
        ratio_field="median_ratio",
    )


def generate_report(candle_results, torch_results, op_names, dtype_keys, scen_keys, mode_keys=None):
    """Generate full report. Returns markdown string."""
    if mode_keys is None:
        mode_keys = ["fwd"]
    rows = list(candle_results) + list(torch_results)
    annotate_ratio_rows(rows, key_fields=_KEY_FIELDS, metric="median_ms")
    candle_map, torch_map = _build_maps(candle_results, torch_results)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# NPU Op Benchmark: candle vs torch_npu",
        "",
        f"Date: {now}",
        "",
    ]

    try:
        with open("/usr/local/Ascend/firmware/version.info", "r", encoding="utf-8") as handle:
            lines.append(f"Device info: {handle.read().strip()}")
            lines.append("")
    except (FileNotFoundError, PermissionError):
        pass

    summary_rows = []

    for mode_key in mode_keys:
        for dtype_key in dtype_keys:
            for scen_key in scen_keys:
                mode_ops = [
                    op for op in op_names
                    if ((op, mode_key, dtype_key, scen_key) in candle_map
                        or (op, mode_key, dtype_key, scen_key) in torch_map)
                ]
                if not mode_ops:
                    continue
                section_lines, ratios = _build_section(
                    mode_key, dtype_key, scen_key, candle_map, torch_map, mode_ops
                )
                lines.extend(section_lines)

                if ratios:
                    avg = sum(ratio for _, ratio in ratios) / len(ratios)
                    worst_op, worst_ratio = max(ratios, key=lambda item: item[1])
                    summary_rows.append({
                        "mode": MODES[mode_key],
                        "dtype": DTYPES[dtype_key],
                        "scenario": scen_key,
                        "avg_ratio": avg,
                        "worst_op": worst_op,
                        "worst_ratio": worst_ratio,
                    })

    if summary_rows:
        lines.append("### Summary")
        lines.append("")
        lines.append("| mode | dtype | scenario | avg ratio | worst op (ratio) |")
        lines.append("|---|---|---|---|---|")
        for row in summary_rows:
            lines.append(
                f"| {row['mode']} | {row['dtype']} | {row['scenario']} | "
                f"{row['avg_ratio']:.2f}x | "
                f"{row['worst_op']} ({row['worst_ratio']:.2f}x) |"
            )
        lines.append("")

    return "\n".join(lines)


def print_terminal(report_md):
    """Print report to terminal."""
    print(report_md)


def write_markdown(report_md, output_dir):
    """Write report to markdown file. Returns file path."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"op_benchmark_{ts}.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(report_md)
    return path
