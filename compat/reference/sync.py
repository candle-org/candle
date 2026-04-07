#!/usr/bin/env python3
"""Manifest-driven sync for pinned torch / torch_npu reference checkouts."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).resolve().with_name("manifest.yaml")


def load_manifest(manifest_path=DEFAULT_MANIFEST):
    with open(manifest_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def checkout_path(project_root, relative_path):
    return Path(project_root) / Path(relative_path)


def _run_git(args, *, cwd=None, capture_output=False):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def _clone_with_mirrors(checkout, revision, mirrors):
    last_error = None
    for mirror in mirrors:
        try:
            _run_git(["clone", "--depth", "1", "--branch", revision, mirror, str(checkout)])
            return mirror
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if checkout.exists():
                shutil.rmtree(checkout)
    raise last_error if last_error is not None else RuntimeError("no mirrors configured")


def ensure_checkout(name, source, *, project_root=PROJECT_ROOT, policy=None, offline=False):
    policy = policy or {}
    checkout = checkout_path(project_root, source["path"])
    revision = source["revision"]
    mirrors = source["mirrors"]

    if checkout.exists():
        if offline:
            return {
                "name": name,
                "status": "reused-offline",
                "path": str(checkout),
                "revision": revision,
            }

        if not policy.get("allow_local_dirty", False):
            status = _run_git(["status", "--porcelain"], cwd=checkout, capture_output=True)
            if status.stdout.strip():
                raise RuntimeError(f"{name} checkout is dirty: {checkout}")

        _run_git(["fetch", "--tags", "--prune"], cwd=checkout)
        _run_git(["checkout", revision], cwd=checkout)
        if policy.get("detached_checkout", True):
            head = _run_git(["rev-parse", "HEAD"], cwd=checkout, capture_output=True).stdout.strip()
            _run_git(["checkout", "--detach", head], cwd=checkout)
        return {
            "name": name,
            "status": "updated",
            "path": str(checkout),
            "revision": revision,
        }

    if offline:
        raise FileNotFoundError(f"offline mode requested but checkout is missing: {checkout}")

    checkout.parent.mkdir(parents=True, exist_ok=True)
    mirror = _clone_with_mirrors(checkout, revision, mirrors)
    if policy.get("detached_checkout", True):
        head = _run_git(["rev-parse", "HEAD"], cwd=checkout, capture_output=True).stdout.strip()
        _run_git(["checkout", "--detach", head], cwd=checkout)
    return {
        "name": name,
        "status": "cloned",
        "path": str(checkout),
        "revision": revision,
        "mirror": mirror,
    }


def ensure_sources(manifest_path=DEFAULT_MANIFEST, *, source_name=None, project_root=PROJECT_ROOT, offline=False):
    manifest = load_manifest(manifest_path)
    policy = manifest.get("policies", {})
    sources = manifest["sources"]
    names = [source_name] if source_name else list(sources.keys())

    results = {}
    for name in names:
        results[name] = ensure_checkout(
            name,
            sources[name],
            project_root=project_root,
            policy=policy,
            offline=offline,
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Sync torch / torch_npu reference checkouts")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source", choices=["pytorch", "torch_npu"], default=None)
    parser.add_argument("--offline", action="store_true", help="Reuse existing checkouts without fetching")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    results = ensure_sources(args.manifest, source_name=args.source, offline=args.offline)
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for name, result in results.items():
            print(f"[reference] {name}: {result['status']} @ {result['path']}")


if __name__ == "__main__":
    main()
