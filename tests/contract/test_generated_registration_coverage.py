# pylint: disable=missing-function-docstring
"Contract tests that pin the current registration coverage gaps.

These tests document the drift between three files:
  - src/candle/_generated/registration.py   (what is registered at runtime)
  - src/candle/_generated/variable_type.py  (Python defs, superset)
  - src/candle/_generated/_variable_type_cy.pyx  (Cython defs, subset)

Failing tests are EXPECTED and record the current state.
Do NOT suppress failures; fix the underlying gaps in follow-up tasks.
"
import re
import pathlib

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GEN = pathlib.Path(__file__).parent.parent.parent / "src" / "candle" / "_generated"


def _read(name):
    return (_GEN / name).read_text(encoding="utf-8")


def _vt_symbols_from_registration():
    content = _read("registration.py")
    return sorted(set(re.findall(r"_VT\.([A-Za-z_]+)", content)))


def _defs_in_file(name):
    content = _read(name)
    return set(re.findall(r"^def ([A-Za-z_]+)", content, re.MULTILINE))


# ---------------------------------------------------------------------------
# Test A1-a
# ---------------------------------------------------------------------------

def test_registration_symbols_exist_in_either_compiled_or_python_surface():
    # All symbols referenced by registration.py must be defined in
    # variable_type.py OR _variable_type_cy.pyx.
    #
    # KNOWN FAILURES (55 symbols as of 2026-03-23): overloaded ops such as
    # add_autograd, sub_autograd, mul_autograd, div_autograd, pow_autograd and
    # their _post variants, plus comparison/scatter/norm ops.  These are broken
    # aliases that point to names that do not exist in either surface file.

    reg_symbols = _vt_symbols_from_registration()
    vt_defs = _defs_in_file("variable_type.py")
    cy_defs = _defs_in_file("_variable_type_cy.pyx")

    missing = [s for s in reg_symbols if s not in vt_defs and s not in cy_defs]

    assert missing == [], (
        str(len(missing)) + " symbol(s) referenced in registration.py exist in "
        "neither variable_type.py nor _variable_type_cy.pyx:\n"
        + "\n".join("  " + s for s in missing)
    )


# ---------------------------------------------------------------------------
# Test A1-b
# ---------------------------------------------------------------------------

def test_compiled_variable_type_surface_matches_generated_safe_registration_subset():
    # sum_to_size_autograd_post is referenced by registration.py and defined
    # in variable_type.py, but is ABSENT from _variable_type_cy.pyx.
    #
    # This documents a known drift: the Cython surface is a strict subset and
    # does not yet include sum_to_size_autograd_post.  The assertion that it is
    # MISSING from Cython is intentionally an *inverse* assertion -- it will
    # start failing (and should then be removed) once the Cython file is
    # regenerated to include sum_to_size_autograd_post.

    cy_defs = _defs_in_file("_variable_type_cy.pyx")
    vt_defs = _defs_in_file("variable_type.py")
    reg_symbols = set(_vt_symbols_from_registration())

    # Must be in registration
    assert "sum_to_size_autograd_post" in reg_symbols, (
        "sum_to_size_autograd_post vanished from registration.py -- update this test"
    )
    # Must be in the Python surface
    assert "sum_to_size_autograd_post" in vt_defs, (
        "sum_to_size_autograd_post vanished from variable_type.py -- update this test"
    )
    # Must NOT yet be in the Cython surface (documents the drift)
    assert "sum_to_size_autograd_post" not in cy_defs, (
        "sum_to_size_autograd_post is now present in _variable_type_cy.pyx. "
        "The drift has been resolved -- remove the inverse assertion here and "
        "update Task A4."
    )


# ---------------------------------------------------------------------------
# Test A1-c
# ---------------------------------------------------------------------------

def test_registration_does_not_reference_generic_alias_without_backing_wrapper():
    # For the well-known overloaded ops (add, sub, mul, div, pow), if
    # registration.py references '<op>_autograd', that name must exist as a def
    # in variable_type.py.
    #
    # KNOWN FAILURES (5 ops as of 2026-03-23): add, sub, mul, div, pow.
    # registration.py references e.g. add_autograd but variable_type.py only
    # provides add_tensor_autograd and add_scalar_autograd (overload-specific
    # wrappers).  There is no generic add_autograd alias.

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
