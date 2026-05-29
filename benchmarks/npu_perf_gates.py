"""Shared metrics and ratio gates for NPU performance benchmarks."""

import statistics


def _round_ms(value):
    return round(float(value), 4)


def _nearest_percentile(sorted_values, percentile):
    if not sorted_values:
        return 0.0
    index = int((len(sorted_values) - 1) * percentile + 0.5)
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
        "median_ms": _round_ms(statistics.median(values)),
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


def collect_ratio_failures(
    rows, *, key_fields, expected_keys, max_ratio, ratio_field, inclusive=True
):
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
        elif (inclusive and ratio > max_ratio) or (not inclusive and ratio >= max_ratio):
            comparator = ">" if inclusive else ">="
            failures.append(
                f"{label}: candle {ratio_field} {ratio:.2f}x {comparator} {max_ratio:.2f}x"
            )

    return failures
