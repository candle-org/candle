# pylint: disable=missing-function-docstring
"""Contract test for in-place autograd codegen roundtrips.

The checked-in ``src/candle/_generated`` directory currently contains two kinds of
content:

1. pure ``derivatives.yaml`` output; and
2. upstream-maintained Python-side deltas that are still load-bearing today.

Running ``python -m tools.autograd.gen_autograd`` in-place must therefore be a
true roundtrip: it may update generated sections, but it must not silently drop
or rewrite any of the checked-in files.
"""

from __future__ import annotations

import ast
import difflib
import shutil
from pathlib import Path

from tools.autograd.gen_autograd import main as gen_autograd_main


_GENERATED_FILES = (
    "functions.py",
    "variable_type.py",
    "registration.py",
    "_functions_cy.pyx",
    "_variable_type_cy.pyx",
    "view_funcs.py",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _short_diff(expected: str, actual: str) -> str:
    lines = list(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile="expected",
            tofile="actual",
            n=1,
        )
    )
    return "\n".join(lines[:40])


def test_gen_autograd_roundtrips_checked_in_generated_sources(tmp_path):
    root = Path(__file__).resolve().parents[2]
    expected_dir = root / "src" / "candle" / "_generated"
    work_dir = tmp_path / "_generated"
    shutil.copytree(expected_dir, work_dir)

    gen_autograd_main(root / "tools" / "autograd" / "derivatives.yaml", work_dir)

    mismatches = []
    for name in _GENERATED_FILES:
        expected = _read(expected_dir / name)
        actual = _read(work_dir / name)
        if actual == expected:
            continue
        mismatches.append(
            f"{name}: expected {len(expected)} chars, got {len(actual)} chars\n"
            f"{_short_diff(expected, actual)}"
        )

    assert mismatches == [], "\n\n".join(mismatches)


def test_gen_autograd_generates_functions_in_fresh_output_dir(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "generated"
    gen_autograd_main(root / "tools" / "autograd" / "derivatives.yaml", output_dir)

    generated = ast.parse(_read(output_dir / "functions.py"))
    maintained = ast.parse(_read(root / "src" / "candle" / "_generated" / "functions.py"))
    assert any(isinstance(node, ast.ClassDef) and node.name == "MulTensorBackward0" for node in generated.body)

    helpers = (
        "_mul_tensor_backward_helper",
        "_div_tensor_self_backward_helper",
    )
    generated_helpers = {node.name: ast.dump(node) for node in generated.body if isinstance(node, ast.FunctionDef)}
    maintained_helpers = {node.name: ast.dump(node) for node in maintained.body if isinstance(node, ast.FunctionDef)}
    for helper in helpers:
        assert generated_helpers[helper] == maintained_helpers[helper], helper
