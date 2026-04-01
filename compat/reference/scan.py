#!/usr/bin/env python3
"""Collect source-availability status for pinned reference checkouts."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sync import DEFAULT_MANIFEST, PROJECT_ROOT, checkout_path, load_manifest



def _git_output(args, cwd):
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()



def collect_source_status(manifest_path=DEFAULT_MANIFEST, project_root=PROJECT_ROOT):
    manifest = load_manifest(manifest_path)
    payload = {"sources": {}}
    for name, source in manifest["sources"].items():
        checkout = checkout_path(project_root, source["path"])
        if not checkout.exists():
            payload["sources"][name] = {
                "path": str(checkout),
                "present": False,
                "revision": source["revision"],
                "roles": source.get("roles", []),
            }
            continue

        head = _git_output(["rev-parse", "HEAD"], cwd=checkout)
        dirty = bool(_git_output(["status", "--porcelain"], cwd=checkout))
        payload["sources"][name] = {
            "path": str(checkout),
            "present": True,
            "revision": source["revision"],
            "head": head,
            "dirty": dirty,
            "roles": source.get("roles", []),
        }
    return payload



def write_source_status(manifest_path, output_path, project_root=PROJECT_ROOT):
    payload = collect_source_status(manifest_path=manifest_path, project_root=project_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload



def main():
    parser = argparse.ArgumentParser(description="Write source-status JSON for torch / torch_npu references")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "compat" / "reference" / "reports" / "source-status.json")
    args = parser.parse_args()
    payload = write_source_status(args.manifest, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))



if __name__ == "__main__":
    main()
