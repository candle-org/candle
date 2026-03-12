from pathlib import Path


def test_npu_memcpy_calls_use_runtime_helpers():
    repo_root = Path(__file__).resolve().parents[2]
    disallowed = [
        repo_root / "src/candle/_dispatch/functionalize.py",
        repo_root / "src/candle/_storage.py",
        repo_root / "src/candle/_backends/npu/ops/shape.py",
        repo_root / "src/candle/distributed/_process_group.py",
        repo_root / "src/candle/distributed/__init__.py",
    ]
    for path in disallowed:
        text = path.read_text()
        assert "acl.rt.memcpy" not in text, f"direct acl memcpy in {path}"
