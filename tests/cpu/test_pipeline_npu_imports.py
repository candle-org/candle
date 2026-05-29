"""Assert that importing benchmarks.pipeline_npu.bench does not import candle at module load time."""
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_bench_does_not_import_candle_at_module_level():
    code = (
        "import importlib, sys;"
        "sys.path.insert(0, '');"
        "importlib.import_module('benchmarks.pipeline_npu.bench');"
        "assert 'candle' not in sys.modules, 'candle was imported at module level';"
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={
            **__import__("os").environ,
            "PYTHONPATH": f"{_REPO_ROOT}{__import__('os').pathsep}{_REPO_ROOT / 'src'}",
        },
    )
    assert result.returncode == 0, f"subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "OK" in result.stdout
