"""Report generation: terminal table + markdown file."""
import os
from datetime import datetime

from .cases import SCENARIOS, DTYPES


def _format_ratio(candle_ms, torch_ms):
    if torch_ms <= 0 or candle_ms <= 0:
        return "N/A"
    ratio = candle_ms / torch_ms
    return f"{ratio:.2f}x"


def _build_section(dtype_key, scen_key, candle_map, torch_map, op_names):
    """Build one table section. Returns (lines, ratios) where ratios is list of floats."""
    dtype_label = DTYPES[dtype_key]
    scen_label = SCENARIOS[scen_key]["label"]
    header = f"### {dtype_label} — {scen_label}"

    lines = [header, ""]
    lines.append("| Op | candle (ms) | torch_npu (ms) | ratio |")
    lines.append("|---|---|---|---|")

    ratios = []
    for op in op_names:
        c = candle_map.get((op, dtype_key, scen_key))
        t = torch_map.get((op, dtype_key, scen_key))

        c_med = c["median_ms"] if c and c["status"] == "ok" else None
        t_med = t["median_ms"] if t and t["status"] == "ok" else None

        c_str = f"{c_med:.4f}" if c_med is not None else ("ERR" if c else "—")
        t_str = f"{t_med:.4f}" if t_med is not None else ("ERR" if t else "—")

        if c_med is not None and t_med is not None and t_med > 0:
            ratio = c_med / t_med
            r_str = f"{ratio:.2f}x"
            ratios.append(ratio)
        else:
            r_str = "N/A"

        # Show error detail if any
        if c and c["status"] != "ok":
            c_str = c["status"][:30]
        if t and t["status"] != "ok":
            t_str = t["status"][:30]

        lines.append(f"| {op} | {c_str} | {t_str} | {r_str} |")

    lines.append("")
    return lines, ratios


def generate_report(candle_results, torch_results, op_names, dtype_keys, scen_keys):
    """Generate full report. Returns markdown string."""
    # Index results by (op, dtype, scenario)
    candle_map = {(r["op"], r["dtype"], r["scenario"]): r for r in candle_results}
    torch_map = {(r["op"], r["dtype"], r["scenario"]): r for r in torch_results}

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# NPU Op Benchmark: candle vs torch_npu",
        "",
        f"Date: {now}",
        "",
    ]

    # Try to get device info
    try:
        with open("/usr/local/Ascend/firmware/version.info", "r") as f:
            lines.append(f"Device info: {f.read().strip()}")
            lines.append("")
    except (FileNotFoundError, PermissionError):
        pass

    summary_rows = []

    for dtype_key in dtype_keys:
        for scen_key in scen_keys:
            section_lines, ratios = _build_section(
                dtype_key, scen_key, candle_map, torch_map, op_names
            )
            lines.extend(section_lines)

            if ratios:
                avg = sum(ratios) / len(ratios)
                worst_idx = max(range(len(ratios)), key=ratios.__getitem__)
                # Find which op corresponds to worst
                valid_ops = []
                for op in op_names:
                    c = candle_map.get((op, dtype_key, scen_key))
                    t = torch_map.get((op, dtype_key, scen_key))
                    if (c and c["status"] == "ok" and t and t["status"] == "ok"
                            and t["median_ms"] > 0):
                        valid_ops.append(op)
                worst_op = valid_ops[worst_idx] if worst_idx < len(valid_ops) else "?"
                worst_ratio = ratios[worst_idx]
                summary_rows.append({
                    "dtype": DTYPES[dtype_key],
                    "scenario": scen_key,
                    "avg_ratio": avg,
                    "worst_op": worst_op,
                    "worst_ratio": worst_ratio,
                })

    # Summary table
    if summary_rows:
        lines.append("### Summary")
        lines.append("")
        lines.append("| dtype | scenario | avg ratio | worst op (ratio) |")
        lines.append("|---|---|---|---|")
        for row in summary_rows:
            lines.append(
                f"| {row['dtype']} | {row['scenario']} | "
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
    with open(path, "w") as f:
        f.write(report_md)
    return path
