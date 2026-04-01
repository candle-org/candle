import importlib.util
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAN_PATH = PROJECT_ROOT / "compat" / "reference" / "scan.py"
DIFF_PATH = PROJECT_ROOT / "compat" / "reference" / "diff.py"



def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def write_seed_manifest(tmp_path):
    manifest = {
        "sources": {
            "pytorch": {
                "path": "compat/_pytorch",
                "revision": "v2.5.0",
                "mirrors": ["https://github.com/pytorch/pytorch.git"],
                "roles": ["semantic-baseline"],
            },
            "torch_npu": {
                "path": "compat/_torch_npu",
                "revision": "v2.5.1-7.0.0",
                "mirrors": ["https://github.com/Ascend/pytorch.git"],
                "roles": ["implementation-reference"],
            },
        },
        "policies": {
            "detached_checkout": True,
            "allow_local_dirty": False,
            "offline_reuse": True,
            "reports_dir": "compat/reference/reports",
        },
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path



def write_seed_maps(map_dir):
    map_dir.mkdir(parents=True, exist_ok=True)
    (map_dir / "tensor.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "id": "tensor.detach.version_sharing",
                    "domain": "tensor",
                    "priority": "P0",
                    "hot_path": True,
                    "upstream": {
                        "torch": {"semantic_owner": "TensorImpl invariants", "impl_tier": "cpp_core", "refs": ["c10/core/TensorImpl.h"]},
                        "torch_npu": {"semantic_owner": "inherits torch core semantics", "impl_tier": "python_bridge", "refs": []},
                    },
                    "candle": {
                        "modules": ["src/candle/_tensor.py", "src/candle/_cython/_tensor_impl.pyx"],
                        "current_semantic_status": "partial",
                        "current_impl_tier": "python_plus_partial_cython",
                        "target_impl_tier": "cython_core",
                    },
                    "verification": {"tests": ["tests/contract/test_tensor_alias_version_contract.py*"], "reports": []},
                    "gaps": ["detach must share version state with the source tensor"],
                }
            ],
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (map_dir / "op-families.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "id": "op_family.im2col.index_check",
                    "domain": "op_family",
                    "priority": "P0",
                    "hot_path": True,
                    "upstream": {
                        "torch": {"semantic_owner": "aten im2col / unfold", "impl_tier": "cpp_core", "refs": ["aten/src/ATen/native/Im2Col.cpp"]},
                        "torch_npu": {"semantic_owner": "NPU unfold path", "impl_tier": "backend_bridge", "refs": ["torch_npu/csrc"]},
                    },
                    "candle": {
                        "modules": ["src/candle/_backends/npu/ops.py"],
                        "current_semantic_status": "partial",
                        "current_impl_tier": "python_plus_partial_cython",
                        "target_impl_tier": "cython_plus_native_backend",
                    },
                    "verification": {
                        "tests": [
                            "tests/npu/910b/test_910b_watchlist.py::test_910b_im2col",
                            "tests/npu/910b/test_910b_watchlist.py::test_910b_im2col_index_validation_any_regression",
                        ],
                        "reports": [],
                    },
                    "gaps": ["910B index validation must not produce false positives on valid masks"],
                }
            ],
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (map_dir / "backends.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "id": "backends.npu.no_cpu_fallback",
                    "domain": "backends",
                    "priority": "P0",
                    "hot_path": True,
                    "upstream": {
                        "torch": {"semantic_owner": "device placement invariants", "impl_tier": "cpp_core", "refs": ["c10/core/Device.h"]},
                        "torch_npu": {"semantic_owner": "device-only NPU execution", "impl_tier": "backend_bridge", "refs": ["torch_npu/csrc"]},
                    },
                    "candle": {
                        "modules": ["src/candle/_backends/npu/ops.py", "src/candle/_backends/npu/aclnn.py"],
                        "current_semantic_status": "partial",
                        "current_impl_tier": "python_plus_partial_cython",
                        "target_impl_tier": "cython_plus_native_backend",
                    },
                    "verification": {"tests": ["tests/npu/test_no_cpu_fallback_npu.py*"], "reports": []},
                    "gaps": ["all NPU execution paths must remain on-device"],
                }
            ],
            sort_keys=False,
        ),
        encoding="utf-8",
    )



def write_junit(path, cases):
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<testsuite tests="{len(cases)}">',
    ]
    for nodeid, outcome in cases:
        classname, name = nodeid.rsplit("::", 1)
        lines.append(f'  <testcase classname="{classname}" name="{name}">')
        if outcome == "failed":
            lines.append('    <failure message="boom">boom</failure>')
        lines.append("  </testcase>")
    lines.append("</testsuite>")
    path.write_text("\n".join(lines), encoding="utf-8")



def test_collect_source_status_marks_missing_checkout(tmp_path):
    scan = load_module(SCAN_PATH, "compat_reference_scan")
    manifest_path = write_seed_manifest(tmp_path)
    payload = scan.collect_source_status(manifest_path=manifest_path, project_root=tmp_path)
    assert payload["sources"]["pytorch"]["present"] is False
    assert payload["sources"]["torch_npu"]["present"] is False



def test_write_gap_summary_counts_items(tmp_path):
    diff = load_module(DIFF_PATH, "compat_reference_diff")
    map_dir = tmp_path / "map"
    write_seed_maps(map_dir)
    source_status = tmp_path / "source-status.json"
    source_status.write_text(json.dumps({"sources": {"pytorch": {"present": False}}}), encoding="utf-8")
    json_out = tmp_path / "gap-summary.json"
    md_out = tmp_path / "module-diff.md"
    diff.write_gap_summary(map_dir, source_status, json_out, md_out)
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["total_items"] == 3
    assert payload["by_semantic_status"]["partial"] == 3
    assert payload["by_target_impl_tier"]["cython_core"] == 1
    assert "tensor.detach.version_sharing" in md_out.read_text(encoding="utf-8")



def test_write_910b_reports_maps_failures_to_mapping_ids(tmp_path):
    diff = load_module(DIFF_PATH, "compat_reference_diff")
    map_dir = tmp_path / "map"
    write_seed_maps(map_dir)

    env_path = tmp_path / "env-summary.json"
    env_path.write_text(json.dumps({"python": "3.11", "npu_smi_rc": 0}), encoding="utf-8")

    mechanism_junit = tmp_path / "mechanism.xml"
    write_junit(
        mechanism_junit,
        [("tests/npu/910b/test_910b_watchlist.py::test_910b_im2col_index_validation_any_regression", "failed")],
    )

    workload_junit = tmp_path / "workload.xml"
    write_junit(
        workload_junit,
        [("tests/npu/test_npu_golden_training_loop.py::test_golden_training_loop_loss_decreases_and_is_finite", "passed")],
    )

    out_dir = tmp_path / "reports"
    diff.write_910b_reports(map_dir, env_path, mechanism_junit, workload_junit, out_dir)

    mechanism_payload = json.loads((out_dir / "mechanism-regressions.json").read_text(encoding="utf-8"))
    assert mechanism_payload["layer"] == "mechanism"
    assert mechanism_payload["failures"][0]["mapping_id"] == "op_family.im2col.index_check"

    summary = (out_dir / "latest-summary.md").read_text(encoding="utf-8")
    assert "op_family.im2col.index_check" in summary
