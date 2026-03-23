# pylint: disable=missing-function-docstring
"""Contract tests that pin the _save() kwarg mismatches between the generated
Cython forward wrappers and the Python backward nodes for the 6 known broken ops.

These tests are source-reading only: they intentionally avoid importing
`candle` so they can be run even in a fresh worktree before compiled `.so`
artifacts exist.

The 6 broken ops (Wave-1 targets in the drift-convergence plan):
  - cumsum  : PYX passes self_=, node expects input_=
  - gather  : PYX passes self_=, node expects input_=
  - prod    : PYX passes self_=, result=; node expects only input_=
  - repeat  : PYX passes self_=, node expects input_=
  - sort    : PYX passes indices=result, node expects result1=
  - topk    : PYX passes indices=result, node expects result1=

Each test encodes the *desired* state (no mismatch) so it FAILS against the
current upstream/main codebase and PASSES once the fix lands.
"""

import pathlib
import re

_GEN = pathlib.Path(__file__).parent.parent.parent / "src" / "candle" / "_generated"


def _read(name: str) -> str:
    return (_GEN / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _pyx_save_kwargs(pyx_text: str, post_fn_name: str) -> set:
    """Return the set of keyword argument *names* that the named `_autograd_post`
    function passes to `grad_fn._save(...)` in the Cython source.

    Searches from the `def <post_fn_name>(...)` line forward until the first
    `grad_fn._save(` call, then parses its argument list.
    """
    pattern = (
        r"def "
        + re.escape(post_fn_name)
        + r"\b[^:]*:[\s\S]*?grad_fn\._save\(([^)]+)\)"
    )
    m = re.search(pattern, pyx_text)
    if m is None:
        raise AssertionError(
            f"Could not find grad_fn._save(...) in {post_fn_name} inside "
            "_variable_type_cy.pyx"
        )
    call_args = m.group(1)
    return set(re.findall(r"(\w+)\s*=", call_args))


def _py_save_kwargs(functions_text: str, class_name: str) -> set:
    """Return the set of keyword argument *names* accepted by `_save(self, *,
    ...)` in the named backward-node class inside functions.py."""
    pattern = (
        r"class "
        + re.escape(class_name)
        + r"\b[\s\S]*?def _save\(self,([^)]+)\)"
    )
    m = re.search(pattern, functions_text)
    if m is None:
        raise AssertionError(
            f"Could not find def _save(...) in class {class_name} inside "
            "functions.py"
        )
    sig = m.group(1)
    return set(re.findall(r"(\w+)\s*=", sig))


# ---------------------------------------------------------------------------
# Fixtures (module-level, loaded once)
# ---------------------------------------------------------------------------

_PYX = _read("_variable_type_cy.pyx")
_FPY = _read("functions.py")


def _assert_save_kwargs_match(post_fn_name: str, class_name: str) -> None:
    """Assert that the kwargs passed by the Cython wrapper exactly match the
    keyword parameters accepted by the Python backward node's _save().
    Raises AssertionError with a clear description of any mismatch."""
    pyx_kwargs = _pyx_save_kwargs(_PYX, post_fn_name)
    py_kwargs = _py_save_kwargs(_FPY, class_name)

    extra_in_pyx = pyx_kwargs - py_kwargs
    extra_in_py = py_kwargs - pyx_kwargs

    msgs = []
    if extra_in_pyx:
        msgs.append(
            f"  CY passes kwargs not accepted by PY node: {sorted(extra_in_pyx)}"
        )
    if extra_in_py:
        msgs.append(
            f"  PY node accepts kwargs never passed by CY: {sorted(extra_in_py)}"
        )

    assert not msgs, (
        f"_save() kwarg mismatch for {post_fn_name} / {class_name}:\n"
        + "\n".join(msgs)
        + f"\n  CY passes : {sorted(pyx_kwargs)}"
        + f"\n  PY accepts: {sorted(py_kwargs)}"
    )


# ---------------------------------------------------------------------------
# Tests for the 6 broken ops
# ---------------------------------------------------------------------------

def test_cumsum_save_kwargs_match():
    """CY passes self_=, but CumsumBackward0._save() expects input_=."""
    _assert_save_kwargs_match("cumsum_autograd_post", "CumsumBackward0")


def test_gather_save_kwargs_match():
    """CY passes self_=, but GatherBackward0._save() expects input_=."""
    _assert_save_kwargs_match("gather_autograd_post", "GatherBackward0")


def test_prod_save_kwargs_match():
    """CY passes self_= and result=, but ProdBackward0._save() expects only input_=."""
    _assert_save_kwargs_match("prod_autograd_post", "ProdBackward0")


def test_repeat_save_kwargs_match():
    """CY passes self_=, but RepeatBackward0._save() expects input_=."""
    _assert_save_kwargs_match("repeat_autograd_post", "RepeatBackward0")


def test_sort_save_kwargs_match():
    """CY passes indices=result, but SortBackward0._save() expects result1=."""
    _assert_save_kwargs_match("sort_autograd_post", "SortBackward0")


def test_topk_save_kwargs_match():
    """CY passes indices=result, but TopkBackward0._save() expects result1=."""
    _assert_save_kwargs_match("topk_autograd_post", "TopkBackward0")
