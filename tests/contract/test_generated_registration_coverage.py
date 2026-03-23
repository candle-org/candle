# pylint: disable=missing-function-docstring
"""Contract tests that pin the current registration coverage gaps.

These tests document the drift between three files:
  - src/candle/_generated/registration.py   (what is registered at runtime)
  - src/candle/_generated/variable_type.py  (Python defs, superset)
  - src/candle/_generated/_variable_type_cy.pyx  (Cython defs, subset)

Failing tests are EXPECTED and record the current state.
Do NOT suppress failures; fix the underlying gaps in follow-up tasks.
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


def _defs_in_file(name):
    content = _read(name)
    return set(re.findall(r"^def ([A-Za-z_]+)", content, re.MULTILINE))


# ---------------------------------------------------------------------------
# Task A1
# ---------------------------------------------------------------------------

def test_registration_symbols_exist_in_either_compiled_or_python_surface():
    reg_symbols = _vt_symbols_from_registration()
    vt_defs = _defs_in_file("variable_type.py")
    cy_defs = _defs_in_file("_variable_type_cy.pyx")

    missing = [s for s in reg_symbols if s not in vt_defs and s not in cy_defs]

    assert missing == [], (
        str(len(missing)) + " symbol(s) referenced in registration.py exist in "
        "neither variable_type.py nor _variable_type_cy.pyx:\n"
        + "\n".join("  " + s for s in missing)
    )


def test_compiled_variable_type_surface_matches_generated_safe_registration_subset():
    cy_defs = _defs_in_file("_variable_type_cy.pyx")
    vt_defs = _defs_in_file("variable_type.py")
    reg_symbols = set(_vt_symbols_from_registration())

    assert "sum_to_size_autograd_post" in reg_symbols
    assert "sum_to_size_autograd_post" in vt_defs
    assert "sum_to_size_autograd_post" not in cy_defs, (
        "sum_to_size_autograd_post is now present in _variable_type_cy.pyx. "
        "The drift has been resolved; update this sentinel test."
    )


def test_registration_does_not_reference_generic_alias_without_backing_wrapper():
    vt_defs = _defs_in_file("variable_type.py")
    reg_symbols = set(_vt_symbols_from_registration())

    overloaded_ops = ["add", "sub", "mul", "div", "pow"]
    missing = []
    for op in overloaded_ops:
        name = op + "_autograd"
        if name in reg_symbols and name not in vt_defs:
            missing.append(name)

    assert missing == [], (
        "registration.py references these generic overload aliases that have "
        "no backing def in variable_type.py:\n"
        + "\n".join("  " + s for s in missing)
    )


# ---------------------------------------------------------------------------
# Task A2 (Option 1): codify current manual Python-only wrappers explicitly
# ---------------------------------------------------------------------------

def test_known_python_only_manual_wrappers_are_tracked_explicitly():
    vt_defs = _defs_in_file("variable_type.py")
    for name in LEGACY_MANUAL_WRAPPERS:
        assert name in vt_defs, f"{name} missing from variable_type.py inventory"


def test_legacy_manual_wrapper_inventory_matches_current_cython_gap():
    vt_defs = _defs_in_file("variable_type.py")
    cy_defs = _defs_in_file("_variable_type_cy.pyx")
    for name in LEGACY_MANUAL_WRAPPERS:
        assert name in vt_defs, f"{name} missing from variable_type.py"
        assert name not in cy_defs, (
            f"{name} is now present in _variable_type_cy.pyx; remove it from "
            "LEGACY_MANUAL_WRAPPERS and update Task A2/A4 expectations"
        )
