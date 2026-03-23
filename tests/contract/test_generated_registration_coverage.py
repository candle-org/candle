# pylint: disable=missing-function-docstring
"""Contract tests that pin the current registration coverage gaps.

These tests document the drift between three files:
  - src/candle/_generated/registration.py   (what is registered at runtime)
  - src/candle/_generated/variable_type.py  (Python defs + canonical aliases)
  - src/candle/_generated/_variable_type_cy.pyx  (Cython defs + canonical aliases)

These tests are source-reading only: they intentionally avoid importing
`candle` so they can be run even in a fresh worktree before compiled `.so`
artifacts exist.
"""

import pathlib
import re

_GEN = pathlib.Path(__file__).parent.parent.parent / "src" / "candle" / "_generated"

LEGACY_MANUAL_WRAPPERS = {
    "sum_to_size_autograd_post",
    "diff_autograd",
    "diff_autograd_post",
}


def _read(name):
    return (_GEN / name).read_text(encoding="utf-8")


def _vt_symbols_from_registration():
    content = _read("registration.py")
    return sorted(set(re.findall(r"_VT(?:_PY)?\.([A-Za-z_]+)", content)))


def _wrapper_symbols_in_file(name):
    content = _read(name)
    defs = set(re.findall(r"^def ([A-Za-z_]+)", content, re.MULTILINE))
    aliases = set(re.findall(r"^([A-Za-z_]+)\s*=\s*[A-Za-z_]+", content, re.MULTILINE))
    return defs | aliases


def test_registration_symbols_exist_in_either_compiled_or_python_surface():
    reg_symbols = _vt_symbols_from_registration()
    vt_symbols = _wrapper_symbols_in_file("variable_type.py")
    cy_symbols = _wrapper_symbols_in_file("_variable_type_cy.pyx")
    missing = [s for s in reg_symbols if s not in vt_symbols and s not in cy_symbols]
    assert missing == [], (
        str(len(missing)) + " symbol(s) referenced in registration.py exist in "
        "neither variable_type.py nor _variable_type_cy.pyx:\n"
        + "\n".join("  " + s for s in missing)
    )


def test_compiled_variable_type_surface_matches_generated_safe_registration_subset():
    cy_symbols = _wrapper_symbols_in_file("_variable_type_cy.pyx")
    vt_symbols = _wrapper_symbols_in_file("variable_type.py")
    reg_symbols = set(_vt_symbols_from_registration())
    assert "sum_to_size_autograd_post" in reg_symbols
    assert "sum_to_size_autograd_post" in vt_symbols
    assert "sum_to_size_autograd_post" not in cy_symbols


def test_registration_does_not_reference_generic_alias_without_backing_wrapper():
    vt_symbols = _wrapper_symbols_in_file("variable_type.py")
    reg_symbols = set(_vt_symbols_from_registration())
    overloaded_ops = ["add", "sub", "mul", "div", "pow"]
    missing = [op + "_autograd" for op in overloaded_ops
               if op + "_autograd" in reg_symbols and op + "_autograd" not in vt_symbols]
    assert missing == []


def test_known_python_only_manual_wrappers_are_tracked_explicitly():
    vt_symbols = _wrapper_symbols_in_file("variable_type.py")
    for name in LEGACY_MANUAL_WRAPPERS:
        assert name in vt_symbols, f"{name} missing from variable_type.py inventory"


def test_legacy_manual_wrapper_inventory_matches_current_cython_gap():
    vt_symbols = _wrapper_symbols_in_file("variable_type.py")
    cy_symbols = _wrapper_symbols_in_file("_variable_type_cy.pyx")
    for name in LEGACY_MANUAL_WRAPPERS:
        assert name in vt_symbols
        assert name not in cy_symbols


def test_overloaded_ops_have_canonical_aliases_in_both_python_and_cython_surfaces():
    vt_symbols = _wrapper_symbols_in_file("variable_type.py")
    cy_symbols = _wrapper_symbols_in_file("_variable_type_cy.pyx")
    required = {"add_autograd", "add_autograd_post", "sub_autograd", "sub_autograd_post",
                "mul_autograd", "mul_autograd_post", "div_autograd", "div_autograd_post",
                "pow_autograd", "pow_autograd_post"}
    assert sorted(required - vt_symbols) == []
    assert sorted(required - cy_symbols) == []


def test_registration_generic_aliases_resolve_against_alias_aware_surface():
    reg_symbols = set(_vt_symbols_from_registration())
    vt_symbols = _wrapper_symbols_in_file("variable_type.py")
    cy_symbols = _wrapper_symbols_in_file("_variable_type_cy.pyx")
    generic = {"add_autograd", "sub_autograd", "mul_autograd", "div_autograd", "pow_autograd"}
    unresolved = sorted(n for n in generic if n in reg_symbols and n not in vt_symbols and n not in cy_symbols)
    assert unresolved == []


# --- Task A4: registration split ---

def test_registration_splits_compiled_safe_and_python_legacy_sections():
    text = _read("registration.py")
    assert "from . import variable_type as _VT_PY" in text
    assert "from . import _variable_type_cy as _VT_CY" in text
    assert "_VT = _VT_CY if _VT_CY is not None else _VT_PY" in text


def test_registration_legacy_section_uses_python_surface():
    text = _read("registration.py")
    marker = "# === UPSTREAM LEGACY REGISTRATIONS ==="
    assert marker in text
    legacy = text.split(marker, 1)[1]
    assert "_VT_PY.sum_to_size_autograd_post" in legacy
    assert "_VT_PY.diff_autograd" in legacy
    assert "_VT_PY.contiguous_autograd" in legacy
    assert "_VT_PY.softmax_autograd" in legacy


def test_registration_generated_safe_section_uses_compiled_candidate():
    text = _read("registration.py")
    head = text.split("# === UPSTREAM LEGACY REGISTRATIONS ===", 1)[0]
    assert "_VT.abs_autograd" in head
    assert "_VT.matmul_autograd" in head
    assert "_VT.relu_autograd" in head
