# pylint: disable=missing-function-docstring
"""Mechanism-level audit contracts (PR 1).

These tests lock down a small set of cross-cutting Storage / Dispatch /
Autograd / NPU invariants that earlier PRs left implicit. They are read-only
audits — none of them mutate runtime behavior. They exist so that drift in
the three mechanisms (storage ownership, dispatch routing priority, NPU
forward/autograd coverage) shows up as an explicit test failure before
further NPU operator intake is attached on top.
"""

import ast
import pathlib
import re

import candle
from candle._dispatch.dispatcher import _kernel_for_entry, _key_order
from candle._dispatch.keys import DispatchKey, DispatchKeySet
from candle._dispatch.registry import registry


_ROOT = pathlib.Path(candle.__file__).resolve().parents[2]
_SRC = _ROOT / "src" / "candle"


def _read(path):
    return (_ROOT / path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Storage: owner identity under resize
# ---------------------------------------------------------------------------


def test_typed_storage_python_wrapper_identity_preserved_across_resize():
    t = candle.tensor([1.0, 2.0, 3.0])
    storage = t.storage()
    before_id = id(storage)
    before_untyped_id = id(storage.untyped_storage())
    storage.resize_(5)
    assert id(t.storage()) == before_id
    assert id(storage.untyped_storage()) == before_untyped_id
    assert storage.size() == 5


def test_untyped_storage_python_wrapper_identity_preserved_across_resize():
    t = candle.tensor([1.0, 2.0])
    untyped = t.storage().untyped_storage()
    before_id = id(untyped)
    old_nbytes = untyped.nbytes()
    untyped.resize_(old_nbytes + 8)
    assert id(t.storage().untyped_storage()) == before_id
    assert untyped.nbytes() == old_nbytes + 8


def test_view_tensor_storage_data_ptr_is_aliased_to_base():
    base = candle.tensor([1.0, 2.0, 3.0, 4.0])
    view = base[1:]
    assert view.storage().data_ptr() == base.storage().data_ptr()
    assert view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()


def test_view_tensor_storage_owner_does_not_match_base_python_wrapper():
    base = candle.tensor([1.0, 2.0, 3.0, 4.0])
    view = base[1:]
    # Each call returns a fresh Python shell, but the underlying StorageImpl
    # is aliased — data_ptr equality is the real ownership invariant.
    assert view.storage().data_ptr() == base.storage().data_ptr()
    assert view.storage().size() == base.storage().size()


# ---------------------------------------------------------------------------
# Dispatch: AutogradNPU routes before NPU when both registered
# ---------------------------------------------------------------------------


def _representative_dual_registered_op():
    for name, entry in registry._ops.items():
        if (
            DispatchKey.NPU in entry.kernels
            and DispatchKey.AutogradNPU in entry.kernels
        ):
            return name, entry
    raise AssertionError("expected at least one op with both NPU and AutogradNPU kernels")


def test_autograd_npu_kernel_routes_before_npu_for_representative_op():
    _name, entry = _representative_dual_registered_op()
    keyset = DispatchKeySet({DispatchKey.AutogradNPU, DispatchKey.NPU})
    kernel, key = _kernel_for_entry(entry, _key_order(keyset))
    assert kernel is entry.kernels[DispatchKey.AutogradNPU]
    assert key is DispatchKey.AutogradNPU


def test_npu_kernel_routes_when_autograd_npu_absent():
    _name, entry = _representative_dual_registered_op()
    keyset = DispatchKeySet({DispatchKey.NPU})
    kernel, key = _kernel_for_entry(entry, _key_order(keyset))
    assert kernel is entry.kernels[DispatchKey.NPU]
    assert key is DispatchKey.NPU


def test_dispatch_key_order_places_all_autograd_keys_strictly_before_backend_keys():
    keyset = DispatchKeySet(set(DispatchKey))
    order = [key.name for key in keyset.iter_keys()]
    autograd_indices = [i for i, n in enumerate(order) if n.startswith("Autograd")]
    backend_indices = [
        i for i, n in enumerate(order) if n in {"CPU", "CUDA", "NPU", "Meta"}
    ]
    assert max(autograd_indices) < min(backend_indices)


# ---------------------------------------------------------------------------
# Autograd: NPU forward op coverage categorization snapshot
# ---------------------------------------------------------------------------


def _schema_short_names():
    tree = ast.parse((_SRC / "_dispatch" / "schemas.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register_schema"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
    return names


def _generated_autograd_registered_short_names():
    text = (_SRC / "_generated" / "registration.py").read_text(encoding="utf-8")
    return set(re.findall(r"register_autograd_kernels\('([^']+)'", text))


def _handwritten_autograd_registered_short_names():
    text = (_SRC / "_backends" / "autograd.py").read_text(encoding="utf-8")
    return (
        set(re.findall(r'\("([A-Za-z0-9_]+)",\s*lambda:', text))
        | set(re.findall(r'\("([A-Za-z0-9_]+)",\s*lambda\s*:', text))
        | set(re.findall(r'\("([A-Za-z0-9_]+)",\s*_[A-Za-z0-9_]+\)', text))
    )


def _npu_forward_short_names():
    names = set()
    for name, entry in registry._ops.items():
        if DispatchKey.NPU in entry.kernels:
            names.add(name.split("::")[-1])
    return names


def test_npu_forward_ops_autograd_coverage_categorization_snapshot():
    """Snapshot the categorization of NPU-forward ops by autograd backing.

    Drift in any single bucket size means a new NPU forward registration
    landed without an explicit autograd story. The numbers themselves are
    not the contract — the contract is that an unexpected delta is loud.
    """
    forward = _npu_forward_short_names()
    generated = _generated_autograd_registered_short_names()
    handwritten = _handwritten_autograd_registered_short_names()

    generated_only = forward & generated - handwritten
    handwritten_only = forward & handwritten - generated
    both = forward & generated & handwritten
    missing = forward - generated - handwritten

    # Every NPU forward op must fall into exactly one of these four buckets.
    assert generated_only | handwritten_only | both | missing == forward
    assert (generated_only & handwritten_only) == set()
    assert (generated_only & both) == set()
    assert (handwritten_only & both) == set()

    assert len(forward) == 432
    assert len(generated_only) == 241
    assert len(handwritten_only) == 33
    assert len(both) == 55
    assert len(missing) == 103


# ---------------------------------------------------------------------------
# Autograd: handwritten cpu_only flag does not silently exclude NPU when
# forward kernel exists. Audit-only — drift surfaces ops that need review.
# ---------------------------------------------------------------------------


def _cpu_only_handwritten_autograd_ops():
    text = (_SRC / "_backends" / "autograd.py").read_text(encoding="utf-8")
    return set(re.findall(r'\("([A-Za-z0-9_]+)".*?cpu_only=True', text))


def test_handwritten_cpu_only_autograd_ops_have_no_npu_forward_kernel():
    """Ops marked `cpu_only=True` in handwritten autograd should not also
    have an NPU forward registration — otherwise calling the NPU forward
    with requires_grad=True would route into a backward that intentionally
    refuses NPU. If this snapshot grows, the new entry needs an explicit
    NPU autograd story (generated derivative or removal of cpu_only).
    """
    cpu_only = _cpu_only_handwritten_autograd_ops()
    npu_forward = _npu_forward_short_names()
    collisions = cpu_only & npu_forward

    # No collisions today. If this snapshot changes, the offending op must
    # be reviewed and either get an NPU-aware autograd or have its NPU
    # forward registration removed.
    assert collisions == set()
