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

# Explicit inventory of wrappers that are currently hand-maintained on the
# Python side and absent from the Cython generated surface.
LEGACY_MANUAL_WRAPPERS = {
    "sum_to_size_autograd_post",
    "diff_autograd",
    "diff_autograd_post",
}


def _read(name):
    return (_GEN / name).read_text(encoding="utf-8")


def _vt_symbols_from_registration():
    content = _read("registration.py")
    return sorted(set(re.findall(r"_VT\.([A-Za-z_]+)", content)))


def _wrapper_symbols_in_file(name):
    content = _read(name)
    defs = set(re.findall(r"^def ([A-Za-z_]+)", content, re.MULTILINE))
    aliases = set(re.findall(r"^([A-Za-z_]+)\s*=\s*[A-Za-z_]+", content, re.MULTILINE))
    return defs | aliases


# ---------------------------------------------------------------------------
# Task A1
# ---------------------------------------------------------------------------

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
    assert "sum_to_size_autograd_post" not in cy_symbols, (
        "sum_to_size_autograd_post is now present in _variable_type_cy.pyx. "
        "The drift has been resolved; update this sentinel test."
    )


def test_registration_does_not_reference_generic_alias_without_backing_wrapper():
    vt_symbols = _wrapper_symbols_in_file("variable_type.py")
    reg_symbols = set(_vt_symbols_from_registration())

    overloaded_ops = ["add", "sub", "mul", "div", "pow"]
    missing = []
    for op in overloaded_ops:
        name = op + "_autograd"
        if name in reg_symbols and name not in vt_symbols:
            missing.append(name)

    assert missing == [], (
        "registration.py references these generic overload aliases that have "
        "no backing wrapper or canonical alias in variable_type.py:\n"
        + "\n".join("  " + s for s in missing)
    )


# ---------------------------------------------------------------------------
# Task A2 (Option 1): codify current manual Python-only wrappers explicitly
# ---------------------------------------------------------------------------

def test_known_python_only_manual_wrappers_are_tracked_explicitly():
    vt_symbols = _wrapper_symbols_in_file("variable_type.py")
    for name in LEGACY_MANUAL_WRAPPERS:
        assert name in vt_symbols, f"{name} missing from variable_type.py inventory"


def test_legacy_manual_wrapper_inventory_matches_current_cython_gap():
    vt_symbols = _wrapper_symbols_in_file("variable_type.py")
    cy_symbols = _wrapper_symbols_in_file("_variable_type_cy.pyx")
    for name in LEGACY_MANUAL_WRAPPERS:
        assert name in vt_symbols, f"{name} missing from variable_type.py"
        assert name not in cy_symbols, (
            f"{name} is now present in _variable_type_cy.pyx; remove it from "
            "LEGACY_MANUAL_WRAPPERS and update Task A2/A4 expectations"
        )


# ---------------------------------------------------------------------------
# Task A3: alias strategy is currently expressed via canonical aliases
# ---------------------------------------------------------------------------

def test_overloaded_ops_have_canonical_aliases_in_both_python_and_cython_surfaces():
    vt_symbols = _wrapper_symbols_in_file("variable_type.py")
    cy_symbols = _wrapper_symbols_in_file("_variable_type_cy.pyx")

    required_aliases = {
        "add_autograd",
        "add_autograd_post",
        "sub_autograd",
        "sub_autograd_post",
        "mul_autograd",
        "mul_autograd_post",
        "div_autograd",
        "div_autograd_post",
        "pow_autograd",
        "pow_autograd_post",
    }

    missing_py = sorted(required_aliases - vt_symbols)
    missing_cy = sorted(required_aliases - cy_symbols)

    assert missing_py == [], "Missing canonical aliases in variable_type.py: " + ", ".join(missing_py)
    assert missing_cy == [], "Missing canonical aliases in _variable_type_cy.pyx: " + ", ".join(missing_cy)


def test_registration_generic_aliases_resolve_against_alias_aware_surface():
    reg_symbols = set(_vt_symbols_from_registration())
    vt_symbols = _wrapper_symbols_in_file("variable_type.py")
    cy_symbols = _wrapper_symbols_in_file("_variable_type_cy.pyx")

    generic_aliases = {"add_autograd", "sub_autograd", "mul_autograd", "div_autograd", "pow_autograd"}
    unresolved = sorted(name for name in generic_aliases if name in reg_symbols and name not in vt_symbols and name not in cy_symbols)
    assert unresolved == [], "Registration still has unresolved generic aliases: " + ", ".join(unresolved)


# ---------------------------------------------------------------------------
# Task A4 placeholder — registration split not started yet
# ---------------------------------------------------------------------------

def test_registration_split_not_started_yet():
    text = _read("registration.py")
    assert "_VT_CY" not in text and "_VT_PY" not in text
