from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

ACLNN_PATH = Path(__file__).resolve().parents[1] / "src/candle/_backends/npu/aclnn.py"
HELPER_FUNCTIONS = {
    "_launch_blocking_enabled",
    "_maybe_sync",
    "_get_lib_dirs",
    "_get_lib_names",
    "_require_native_npu_ffi",
    "_infer_ctypes_disabled_op_name",
    "_require_ctypes_npu_path_disabled",
    "_normalize_dtype",
    "_dtype_to_acl",
    "_float32_bits",
    "_float_to_float16_bits",
    "_float_to_bfloat16_bits",
    "_scalar_bytes",
    "_make_int64_array",
    "_make_bool_array",
    "_create_tensor",
    "_create_scalar",
    "_bind_symbol",
    "_optional_symbol",
    "_load_libs",
    "_init_aclnn",
    "_register_cleanup",
    "_cleanup_aclnn",
    "_defer_executor",
    "flush_deferred_executors",
    "get_bindings",
    "symbols_ok",
    "is_available",
    "_contiguous_stride",
    "_unary_call",
    "_create_tensor_list",
    "_create_tensor_list_with_nones",
}
STATUS_ORDER = {"ffi_direct": 0, "legacy_direct": 1, "legacy_via_helper": 2, "other": 3}

def _call_names(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                names.add(f"{func.value.id}.{func.attr}")
    return names

def _classify(func: ast.FunctionDef) -> tuple[str, str]:
    names = _call_names(func)
    if any(name.startswith("_ffi.") for name in names):
        return "ffi_direct", "native cython ffi"
    if "_unary_call" in names:
        return "legacy_via_helper", "shared unary helper -> ctypes descriptors"
    if "_create_tensor" in names or "_create_scalar" in names or "_create_tensor_list" in names or "_create_tensor_list_with_nones" in names:
        return "legacy_direct", "direct legacy ctypes descriptor path"
    return "other", "metadata/capability/helper"

def main() -> None:
    module = ast.parse(ACLNN_PATH.read_text())
    rows = []
    for node in module.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in HELPER_FUNCTIONS or node.name.endswith("_symbols_ok"):
            continue
        status, reason = _classify(node)
        rows.append((node.name, status, reason, node.lineno))

    counts = Counter(status for _, status, _, _ in rows)
    print(f"scanned_file: {ACLNN_PATH}")
    print(f"total_wrappers: {len(rows)}")
    for key in ("ffi_direct", "legacy_direct", "legacy_via_helper", "other"):
        print(f"{key}: {counts.get(key, 0)}")

    print("\npriority_legacy_wrappers:")
    legacy_rows = [row for row in rows if row[1] in {"legacy_direct", "legacy_via_helper"}]
    legacy_rows.sort(key=lambda row: (STATUS_ORDER[row[1]], row[3]))
    for name, status, reason, lineno in legacy_rows[:80]:
        print(f"- {name} [{status}] line={lineno}: {reason}")

if __name__ == "__main__":
    main()
