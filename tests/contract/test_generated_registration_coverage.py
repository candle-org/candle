# pylint: disable=missing-function-docstring
"""Contract tests that pin the current registration coverage.

These tests document the structure of the generated autograd system:
  - src/candle/_generated/registration.py      (generated registration)
  - src/candle/_generated/variable_type.py     (generated Python forward wrappers)
  - src/candle/_generated/variable_type_legacy.py (legacy Python forward wrappers)
  - src/candle/_generated/_variable_type_cy.pyx (generated Cython forward wrappers)

These tests are source-reading only: they intentionally avoid importing
`candle` so they can be run even in a fresh worktree before compiled `.so`
artifacts exist.
"""

import pathlib
import re

_GEN = pathlib.Path(__file__).parent.parent.parent / "src" / "candle" / "_generated"


def _read(name):
    return (_GEN / name).read_text(encoding="utf-8")


def _wrapper_symbols_in_file(name):
    content = _read(name)
    defs = set(re.findall(r"^def ([A-Za-z_]+)", content, re.MULTILINE))
    aliases = set(re.findall(r"^([A-Za-z_]+)\s*=\s*[A-Za-z_]+", content, re.MULTILINE))
    return defs | aliases


# --- Registration structure ---

def test_registration_uses_compiled_or_python_generated_surface():
    text = _read("registration.py")
    assert "from . import variable_type as _VT_PY" in text
    assert "from . import _variable_type_cy as _VT_CY" in text
    assert "_VT = _VT_CY if _VT_CY is not None else _VT_PY" in text


def test_registration_delegates_legacy_to_legacy_module():
    text = _read("registration.py")
    assert "register_legacy_autograd_kernels" in text


def test_registration_uses_runtime_registry_has_check():
    text = _read("registration.py")
    assert "from .._dispatch.registry import registry" in text
    assert "registry.has(" in text


def test_registration_generated_section_uses_compiled_candidate():
    text = _read("registration.py")
    assert "_VT.abs_autograd" in text
    assert "_VT.matmul_autograd" in text
    assert "_VT.relu_autograd" in text


# --- Generated vs legacy file separation ---

def test_generated_variable_type_has_no_legacy_wrappers():
    vt_symbols = _wrapper_symbols_in_file("variable_type.py")
    legacy_only = {"contiguous_autograd", "softmax_autograd", "flatten_autograd",
                   "batch_norm_autograd", "layer_norm_autograd"}
    present = legacy_only & vt_symbols
    assert present == set(), (
        f"Legacy wrappers still in generated variable_type.py: {sorted(present)}"
    )


def test_legacy_variable_type_has_expected_wrappers():
    legacy = _wrapper_symbols_in_file("variable_type_legacy.py")
    expected = {"contiguous_autograd", "softmax_autograd", "flatten_autograd",
                "sum_to_size_autograd_post", "diff_autograd"}
    missing = expected - legacy
    assert missing == set(), (
        f"Expected legacy wrappers missing from variable_type_legacy.py: {sorted(missing)}"
    )


def test_legacy_module_has_register_function():
    text = _read("variable_type_legacy.py")
    assert "def register_legacy_autograd_kernels(" in text


# --- Overloaded ops have canonical aliases ---

def test_overloaded_ops_have_canonical_aliases_in_both_surfaces():
    vt_symbols = _wrapper_symbols_in_file("variable_type.py")
    cy_symbols = _wrapper_symbols_in_file("_variable_type_cy.pyx")
    required = {"add_autograd", "add_autograd_post", "sub_autograd", "sub_autograd_post",
                "mul_autograd", "mul_autograd_post", "div_autograd", "div_autograd_post",
                "pow_autograd", "pow_autograd_post"}
    assert sorted(required - vt_symbols) == []
    assert sorted(required - cy_symbols) == []


def test_registration_generic_aliases_resolve_against_alias_aware_surface():
    text = _read("registration.py")
    reg_symbols = set(re.findall(r"_VT\.([A-Za-z_]+)", text))
    vt_symbols = _wrapper_symbols_in_file("variable_type.py")
    cy_symbols = _wrapper_symbols_in_file("_variable_type_cy.pyx")
    generic = {"add_autograd", "sub_autograd", "mul_autograd", "div_autograd", "pow_autograd"}
    unresolved = sorted(n for n in generic if n in reg_symbols and n not in vt_symbols and n not in cy_symbols)
    assert unresolved == []


# --- Functions legacy file ---

def test_functions_legacy_has_expected_backward_nodes():
    text = _read("functions_legacy.py")
    expected_classes = {"Log_softmaxBackward0", "ContiguousBackward0", "FlattenBackward0"}
    for cls in expected_classes:
        assert f"class {cls}" in text, f"Expected {cls} in functions_legacy.py"




def test_generated_functions_has_no_duplicate_backward_class_definitions():
    text = _read("functions.py")
    classes = re.findall(r"^class ([A-Za-z_][A-Za-z0-9_]*)\(Node\):", text, re.MULTILINE)
    duplicates = sorted(name for name in set(classes) if classes.count(name) > 1)
    assert duplicates == [], f"Duplicate backward classes in functions.py: {duplicates}"


def test_generated_functions_cython_has_no_duplicate_backward_class_definitions():
    text = _read("_functions_cy.pyx")
    classes = re.findall(r"^class ([A-Za-z_][A-Za-z0-9_]*)\(_Node\):", text, re.MULTILINE)
    duplicates = sorted(name for name in set(classes) if classes.count(name) > 1)
    assert duplicates == [], f"Duplicate backward classes in _functions_cy.pyx: {duplicates}"
