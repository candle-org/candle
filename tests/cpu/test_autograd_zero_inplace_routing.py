"""Boundary tests: in-place autograd routing for zero_ and dead-factory cleanup.

These tests pin three facts after migrating ``zero_`` autograd ownership to
the generator-driven path and removing the unused ``_autograd_view`` helper:

1. ``zero_``'s autograd kernel is registered at every ACTIVE supported autograd
   dispatch key (Autograd / AutogradCPU / AutogradCUDA / AutogradNPU /
   AutogradMeta) and that kernel comes from the generated module
   ``candle._generated.variable_type``.

   Note: ``DispatchKey.PrivateUse3`` (the mps autograd alias) is intentionally
   excluded — it's a global fallthrough placeholder; mps autograd flows through
   the ``default`` (DispatchKey.Autograd) kernel rather than via a per-op mps
   registration. See ``tests/contract/test_codegen_mps_coverage.py``.

2. The dead helper ``_autograd_view`` no longer exists in
   ``candle._backends.autograd`` (it had zero callers after the migration to
   derivatives.yaml).

3. ``zero_`` is no longer registered through the manual ``_autograd_inplace``
   factory — its generator-supplied registration is the single source of truth.
"""
import candle  # noqa: F401  (initializes registry)

from candle._dispatch.keys import DispatchKey
from candle._dispatch.registry import registry


_AUTOGRAD_KEYS_FOR_ZERO_ = (
    DispatchKey.Autograd,
    DispatchKey.AutogradCPU,
    DispatchKey.AutogradCUDA,
    DispatchKey.AutogradNPU,
    DispatchKey.AutogradMeta,
)


def _entry(name):
    return registry._entry(name)  # pylint: disable=protected-access


def test_zero_inplace_autograd_kernel_lives_at_every_active_autograd_key():
    entry = _entry("zero_")
    for key in _AUTOGRAD_KEYS_FOR_ZERO_:
        assert key in entry.kernels, (
            f"zero_ missing autograd kernel for {key.name!r}; "
            f"present: {sorted(k.name for k in entry.kernels)}"
        )


def test_zero_inplace_autograd_kernel_comes_from_generated_module():
    entry = _entry("zero_")
    expected_modules = {
        "candle._generated.variable_type",
        "candle._generated._variable_type_cy",
    }
    for key in _AUTOGRAD_KEYS_FOR_ZERO_:
        fn = entry.kernels.get(key)
        assert fn is not None, f"zero_ has no kernel at {key.name}"
        mod = getattr(fn, "__module__", "")
        assert mod in expected_modules, (
            f"zero_'s autograd kernel at {key.name} is owned by {mod!r}; "
            f"should come from {expected_modules}"
        )


def test_zero_inplace_does_not_register_at_mps_fallthrough_key():
    """PrivateUse3 (mps autograd alias) must remain a fallthrough placeholder.

    Even though ``zero_`` is fully covered by the generator, no kernel must be
    written to PrivateUse3 — that key is reserved as a global fallthrough so
    that mps autograd dispatch falls through to the ``default``
    DispatchKey.Autograd kernel.
    """
    entry = _entry("zero_")
    assert DispatchKey.PrivateUse3 not in entry.kernels, (
        "zero_ has a kernel registered at PrivateUse3 (mps autograd alias); "
        "that key must remain a fallthrough placeholder."
    )


def test_autograd_view_factory_is_removed():
    from candle._backends import autograd as backends_autograd
    assert not hasattr(backends_autograd, "_autograd_view"), (
        "_autograd_view factory has zero callers and should be removed"
    )


def test_autograd_inplace_no_longer_registers_zero_():
    """The Python ``_autograd_inplace('zero_', ...)`` registration is dead
    code (overridden by the generator) and should be removed.
    """
    from pathlib import Path
    src = Path("src/candle/_backends/autograd.py").read_text(encoding="utf-8")
    assert '_autograd_inplace("zero_"' not in src, (
        "src/candle/_backends/autograd.py still registers zero_ via the dead "
        "_autograd_inplace factory; that registration is overridden by the "
        "generator and should be deleted."
    )


def test_inplace_zero_backward_helper_is_removed():
    """The ``_inplace_zero_backward`` helper has no callers after the
    ``_autograd_inplace('zero_', ...)`` line is deleted; it must be removed
    too so dead code does not accumulate.
    """
    from candle._backends import autograd as backends_autograd
    assert not hasattr(backends_autograd, "_inplace_zero_backward"), (
        "_inplace_zero_backward has no callers and must be removed"
    )
