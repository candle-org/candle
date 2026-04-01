#!/usr/bin/env python3
"""Generate gap summaries and layered 910B diagnostic reports from mapping files."""
import argparse
import fnmatch
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import yaml



def load_mapping_items(map_dir):
    items = []
    for path in sorted(Path(map_dir).glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        for item in payload:
            item["_map_file"] = path.name
            items.append(item)
    return items



def _counter_dict(values):
    return dict(Counter(values))



def write_gap_summary(map_dir, source_status_path, json_path, markdown_path):
    items = load_mapping_items(map_dir)
    source_status = {}
    if source_status_path and Path(source_status_path).exists():
        source_status = json.loads(Path(source_status_path).read_text(encoding="utf-8"))

    payload = {
        "total_items": len(items),
        "by_domain": _counter_dict(item["domain"] for item in items),
        "by_semantic_status": _counter_dict(item["candle"]["current_semantic_status"] for item in items),
        "by_target_impl_tier": _counter_dict(item["candle"]["target_impl_tier"] for item in items),
        "source_status": source_status.get("sources", {}),
        "items": [
            {
                "id": item["id"],
                "domain": item["domain"],
                "priority": item["priority"],
                "semantic_status": item["candle"]["current_semantic_status"],
                "current_impl_tier": item["candle"]["current_impl_tier"],
                "target_impl_tier": item["candle"]["target_impl_tier"],
            }
            for item in items
        ],
    }

    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Reference Gap Summary",
        "",
        f"- Total items: {len(items)}",
        f"- Partial items: {payload['by_semantic_status'].get('partial', 0)}",
        f"- Cython-core targets: {payload['by_target_impl_tier'].get('cython_core', 0)}",
        "",
        "## Items",
        "",
    ]
    for item in items:
        lines.append(
            f"- `{item['id']}` — {item['candle']['current_semantic_status']} -> {item['candle']['target_impl_tier']} ({item['_map_file']})"
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return payload



def _failure_nodeids(junit_path):
    path = Path(junit_path)
    if not path.exists():
        return []
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    nodeids = []
    for testcase in root.iter("testcase"):
        failed = testcase.find("failure") is not None or testcase.find("error") is not None
        if not failed:
            continue
        classname = testcase.attrib.get("classname", "")
        name = testcase.attrib.get("name", "")
        nodeids.append(f"{classname}::{name}" if classname else name)
    return nodeids



def _match_mapping_id(nodeid, items):
    for item in items:
        for pattern in item.get("verification", {}).get("tests", []):
            if fnmatch.fnmatch(nodeid, pattern) or pattern in nodeid:
                return item["id"]
    return None



def write_910b_reports(map_dir, env_json_path, mechanism_junit_path, workload_junit_path, output_dir):
    items = load_mapping_items(map_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_payload = json.loads(Path(env_json_path).read_text(encoding="utf-8"))
    env_payload.setdefault("layer", "infra")
    env_payload["classification"] = "infra_failure" if env_payload.get("npu_smi_rc", 1) != 0 else "infra_ok"

    mechanism_failures = [
        {
            "nodeid": nodeid,
            "mapping_id": _match_mapping_id(nodeid, items),
            "classification": "semantic_mismatch",
        }
        for nodeid in _failure_nodeids(mechanism_junit_path)
    ]
    workload_failures = [
        {
            "nodeid": nodeid,
            "mapping_id": _match_mapping_id(nodeid, items),
            "classification": "workload_regression",
        }
        for nodeid in _failure_nodeids(workload_junit_path)
    ]

    mechanism_payload = {"layer": "mechanism", "failures": mechanism_failures}
    workload_payload = {"layer": "workload", "failures": workload_failures}

    (output_dir / "env-summary.json").write_text(json.dumps(env_payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "mechanism-regressions.json").write_text(json.dumps(mechanism_payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "workload-regressions.json").write_text(json.dumps(workload_payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# 910B Layered Summary",
        "",
        f"- Infra classification: {env_payload['classification']}",
        f"- Mechanism failures: {len(mechanism_failures)}",
        f"- Workload failures: {len(workload_failures)}",
        "",
    ]
    if mechanism_failures:
        lines.append("## Mechanism failures")
        lines.append("")
        for failure in mechanism_failures:
            lines.append(f"- `{failure['nodeid']}` -> `{failure['mapping_id']}`")
        lines.append("")
    if workload_failures:
        lines.append("## Workload failures")
        lines.append("")
        for failure in workload_failures:
            lines.append(f"- `{failure['nodeid']}` -> `{failure['mapping_id']}`")
        lines.append("")
    (output_dir / "latest-summary.md").write_text("\n".join(lines), encoding="utf-8")



def main():
    parser = argparse.ArgumentParser(description="Write parity summary and 910B layered reports")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary")
    summary.add_argument("--map-dir", type=Path, required=True)
    summary.add_argument("--source-status", type=Path, required=True)
    summary.add_argument("--json-out", type=Path, required=True)
    summary.add_argument("--md-out", type=Path, required=True)

    npu = subparsers.add_parser("npu-report")
    npu.add_argument("--map-dir", type=Path, required=True)
    npu.add_argument("--env-json", type=Path, required=True)
    npu.add_argument("--mechanism-junit", type=Path, required=True)
    npu.add_argument("--workload-junit", type=Path, required=True)
    npu.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "summary":
        write_gap_summary(args.map_dir, args.source_status, args.json_out, args.md_out)
    else:
        write_910b_reports(args.map_dir, args.env_json, args.mechanism_junit, args.workload_junit, args.output_dir)



if __name__ == "__main__":
    main()
