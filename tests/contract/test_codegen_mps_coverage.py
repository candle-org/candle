"""Generator-side contract: PrivateUse3 (mps autograd) is a fallthrough key.

torch's autograd path on MPS does NOT use a per-op AutogradMPS kernel.  Candle
mirrors that design via ``DispatchKey.PrivateUse3``: the dispatch core
registers it as a global fallthrough placeholder (see
``src/candle/_dispatch/registry.py::_register_global_fallthroughs``).  Any
``register_autograd_kernels(name, mps=...)`` call that writes to PrivateUse3
turns the placeholder into an active kernel and breaks fallthrough behaviour
(seen empirically: ``div`` autograd silently returns ``None`` because the
PrivateUse3 entry short-circuits the AutogradCPU path).

This test pins generator output: the autograd codegen must NOT emit ``mps=``
arguments for ``register_autograd_kernels``; instead MPS device autograd is
served by the ``default`` (DispatchKey.Autograd) kernel through fallthrough.
"""
from tools.autograd.load_derivatives import load_derivatives
from tools.autograd.gen_registration import gen_registration


def test_gen_registration_does_not_register_mps_autograd():
    """The generator must NOT write a `mps=` argument for any op.

    Reasoning: ``DispatchKey.PrivateUse3`` (the mps autograd alias) is
    registered as a fallthrough placeholder. Adding a real kernel there breaks
    autograd dispatch on MPS — the mps device's autograd path is served by
    the ``default`` (DispatchKey.Autograd) kernel via fallthrough.
    """
    infos = load_derivatives('tools/autograd/derivatives.yaml')
    src = gen_registration(infos)
    offenders = []
    for line in src.splitlines():
        line = line.strip()
        if not line.startswith("register_autograd_kernels("):
            continue
        if "mps=" in line:
            offenders.append(line)
    assert not offenders, (
        f"Generator emitted {len(offenders)} register_autograd_kernels(...) "
        "calls with `mps=`. PrivateUse3 must remain a fallthrough placeholder. "
        f"First offender: {offenders[0]}"
    )
