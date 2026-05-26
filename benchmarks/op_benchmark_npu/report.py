"""Report generation: terminal table + markdown file."""
import os
from datetime import datetime

from .cases import SCENARIOS, DTYPES, MODES


def _build_section(mode_key, dtype_key, scen_key, candle_map, torch_map, op_names):
    dtype_label = DTYPES[dtype_key]
    mode_label = MODES[mode_key]
    scen_label = SCENARIOS[scen_key]["label"]
    header = f"### {mode_label} — {dtype_label} — {scen_label}"

    lines = [header, ""]
    lines.append("| Op | candle (ms) | torch_npu (ms) | ratio | impact |")
    lines.append("|---|---|---|---|---|")

    ratios = []
    rows = []
    for op in op_names:
        c = candle_map.get((op, mode_key, dtype_key, scen_key))
        t = torch_map.get((op, mode_key, dtype_key, scen_key))

        c_med = c["median_ms"] if c and c["status"] == "ok" else None
        t_med = t["median_ms"] if t and t["status"] == "ok" else None

        c_str = f"{c_med:.4f}" if c_med is not None else ("ERR" if c else "—")
        t_str = f"{t_med:.4f}" if t_med is not None else ("ERR" if t else "—")

        ratio = None
        impact = None
        if c_med is not None and t_med is not None and t_med > 0:
            ratio = c_med / t_med
            impact = c_med - t_med
            ratios.append((op, ratio))
            r_str = f"{ratio:.2f}x"
            impact_str = f"{impact:.4f}"
        else:
            r_str = "N/A"
            impact_str = "N/A"

        if c and c["status"] != "ok":
            c_str = c["status"][:30]
        if t and t["status"] != "ok":
            t_str = t["status"][:30]

        rows.append((ratio if ratio is not None else -1.0, impact if impact is not None else -1.0,
                     f"| {op} | {c_str} | {t_str} | {r_str} | {impact_str} |"))

    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    lines.extend(row for _, _, row in rows)
    lines.append("")
    return lines, ratios


def generate_report(candle_results, torch_results, op_names, dtype_keys, scen_keys, mode_keys=None):
    """Generate full report. Returns markdown string."""
    if mode_keys is None:
        mode_keys = ["fwd"]
    candle_map = {(r["op"], r.get("mode", "fwd"), r["dtype"], r["scenario"]): r for r in candle_results}
    torch_map = {(r["op"], r.get("mode", "fwd"), r["dtype"], r["scenario"]): r for r in torch_results}

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
