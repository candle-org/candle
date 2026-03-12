"""
Bootstrap module for redirecting ``import torch`` to ``candle``.

This file is designed to be installed as a top-level module in site-packages
(via ``py-modules`` in pyproject.toml) and triggered at startup by a ``.pth``
file.  It installs a :class:`importlib.abc.MetaPathFinder` into
``sys.meta_path`` that intercepts ``torch`` and ``torch.*`` imports and
resolves them to ``candle`` / ``candle.*``.

Decision logic (controlled by the ``USE_CANDLE`` environment variable):

* ``USE_CANDLE=1/true/yes`` — always redirect
* ``USE_CANDLE=0/false/no`` — never redirect
* env var not set — redirect only when no real PyTorch is found on sys.path

**Only stdlib imports are used** so that startup cost is negligible when the
hook is inactive.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import threading

# ---------------------------------------------------------------------------
# Alias map: torch submodule name → candle submodule name
# Only entries that differ from a direct 1-to-1 mapping need to be listed.
# ---------------------------------------------------------------------------
_ALIASES = {
    "random": "_random",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_real_torch():
    """Return True if a real PyTorch package exists on sys.path."""
    for entry in sys.path:
        if not entry:
            continue
        candidate = os.path.join(entry, "torch", "__init__.py")
        if os.path.isfile(candidate):
            return True
    return False


def _should_redirect():
    """Decide whether torch imports should be redirected to candle."""
    env = os.environ.get("USE_CANDLE", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    # Auto-detect: redirect only when there is no real PyTorch.
    return not _has_real_torch()


def _resolve_candle_name(fullname):
    """Map a ``torch.*`` fully-qualified name to the corresponding ``candle.*`` name.

    Examples::

        torch            → candle
        torch.nn         → candle.nn
        torch.autograd   → candle.autograd
        torch.random     → candle._random
        torch.nn.functional → candle.nn.functional
        torch.autograd.graph → candle.autograd.graph
    """
    parts = fullname.split(".")
    # parts[0] is "torch"
    parts[0] = "candle"
    # Apply alias to the first submodule component (index 1) if present.
    if len(parts) > 1 and parts[1] in _ALIASES:
        parts[1] = _ALIASES[parts[1]]
    return ".".join(parts)


# ---------------------------------------------------------------------------
# MetaPathFinder / Loader
# ---------------------------------------------------------------------------

class _CandleLoader(importlib.abc.Loader):
    """Loader that imports the candle equivalent and swaps it into sys.modules."""

    def __init__(self, candle_name, torch_name):
        self._candle_name = candle_name
        self._torch_name = torch_name

    def create_module(self, spec):  # noqa: ARG002
        return None  # Use default semantics.

    def exec_module(self, module):  # noqa: ARG002
        candle_mod = importlib.import_module(self._candle_name)
        # Replace the placeholder module that the import machinery created
        # with the real candle module.  PEP 451 guarantees that the import
        # system re-reads sys.modules after exec_module returns.
        sys.modules[self._torch_name] = candle_mod

        # Sync all loaded candle.* submodules → torch.* so that attribute
        # access like ``torch.nn`` on the top-level module works after
        # ``import torch.nn``.
        _sync_submodules()


def _sync_submodules():
    """Mirror loaded ``candle.*`` entries to ``torch.*`` in sys.modules."""
    candle_prefix = "candle."
    # Build reverse alias map: candle internal name → torch public name
    reverse = {v: k for k, v in _ALIASES.items()}

    to_add = {}
    for name, mod in list(sys.modules.items()):
        if not name.startswith(candle_prefix):
            continue
        # candle.nn.functional → torch.nn.functional
        suffix = name[len(candle_prefix):]
        parts = suffix.split(".", 1)
        first = parts[0]
        # Reverse-map aliased submodules.
        if first in reverse:
            parts[0] = reverse[first]
        torch_name = "torch." + ".".join(parts)
        if torch_name not in sys.modules:
            to_add[torch_name] = mod

    sys.modules.update(to_add)


class _CandleTorchFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder that redirects ``torch`` / ``torch.*`` to candle."""

    def __init__(self):
        self._local = threading.local()

    def find_spec(self, fullname, path, target=None):  # noqa: ARG002
        if not (fullname == "torch" or fullname.startswith("torch.")):
            return None

        # Recursion guard: while we are resolving a torch.* import we must
        # not intercept the underlying candle.* import (which might itself
        # trigger torch.* lookups).
        if getattr(self._local, "resolving", False):
            return None
        self._local.resolving = True
        try:
            candle_name = _resolve_candle_name(fullname)
            # Verify that the target candle module actually exists.
            spec = importlib.util.find_spec(candle_name)
            if spec is None:
                return None
            return importlib.machinery.ModuleSpec(
                fullname,
                _CandleLoader(candle_name, fullname),
                origin=spec.origin,
                is_package=(spec.submodule_search_locations is not None),
            )
        finally:
            self._local.resolving = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_finder_instance = None


def _install():
    """Install the import hook (idempotent)."""
    global _finder_instance  # noqa: PLW0603
    if _finder_instance is not None:
        return
    if not _should_redirect():
        return
    _finder_instance = _CandleTorchFinder()
    sys.meta_path.insert(0, _finder_instance)


def _uninstall():
    """Remove the import hook (idempotent)."""
    global _finder_instance  # noqa: PLW0603
    if _finder_instance is None:
        return
    try:
        sys.meta_path.remove(_finder_instance)
    except ValueError:
        pass
    _finder_instance = None


def is_active():
    """Return True if the import hook is currently installed."""
    return _finder_instance is not None and _finder_instance in sys.meta_path


# ---------------------------------------------------------------------------
# Auto-install at module load time (triggered by .pth file).
# ---------------------------------------------------------------------------
_install()
