"""
CLI tool for managing the ``import torch`` → candle compatibility hook.

Usage::

    python -m candle.torch_compat install    # install hook into site-packages
    python -m candle.torch_compat uninstall  # remove hook from site-packages
    python -m candle.torch_compat status     # show current state
"""

import os
import shutil
import site
import sys

_BOOTSTRAP_MODULE = "_candle_torch_compat.py"
_PTH_FILE = "candle-torch-compat.pth"
_PTH_CONTENT = "import _candle_torch_compat\n"


def _find_source_files():
    """Locate the bootstrap module and .pth file shipped with candle."""
    # When installed via pip, both files live in site-packages already.
    # When running from source, they live in src/ (parent of src/candle/).
    candle_pkg = os.path.dirname(os.path.abspath(__file__))  # src/candle/
    src_dir = os.path.dirname(candle_pkg)                    # src/

    bootstrap = os.path.join(src_dir, _BOOTSTRAP_MODULE)
    if os.path.isfile(bootstrap):
        return bootstrap
    return None


def _writable_site_packages():
    """Return the first writable site-packages directory."""
    candidates = []
    # Prefer user site first, then global site-packages.
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        candidates.append(user_site)
    global_sites = site.getsitepackages()
    if isinstance(global_sites, list):
        candidates.extend(global_sites)

    for sp in candidates:
        if os.path.isdir(sp) and os.access(sp, os.W_OK):
            return sp
    return None


def _all_site_packages():
    """Return all site-packages directories."""
    result = []
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        result.append(user_site)
    global_sites = site.getsitepackages()
    if isinstance(global_sites, list):
        result.extend(global_sites)
    return result


def install():
    """Install the bootstrap module and .pth file into site-packages."""
    sp = _writable_site_packages()
    if sp is None:
        print("ERROR: No writable site-packages directory found.", file=sys.stderr)
        print("Try running with --user or as root.", file=sys.stderr)
        return 1

    bootstrap_src = _find_source_files()

    # Install bootstrap module.
    dest_bootstrap = os.path.join(sp, _BOOTSTRAP_MODULE)
    if bootstrap_src:
        shutil.copy2(bootstrap_src, dest_bootstrap)
        print(f"Copied {_BOOTSTRAP_MODULE} → {dest_bootstrap}")
    elif not os.path.isfile(dest_bootstrap):
        print(f"ERROR: Cannot find {_BOOTSTRAP_MODULE} source.", file=sys.stderr)
        return 1
    else:
        print(f"{_BOOTSTRAP_MODULE} already exists at {dest_bootstrap}")

    # Install .pth file.
    dest_pth = os.path.join(sp, _PTH_FILE)
    with open(dest_pth, "w") as f:
        f.write(_PTH_CONTENT)
    print(f"Wrote {_PTH_FILE} → {dest_pth}")

    print("\nHook installed. Restart Python for changes to take effect.")
    return 0


def uninstall():
    """Remove the bootstrap module and .pth file from all site-packages."""
    removed = False
    for sp in _all_site_packages():
        for filename in (_BOOTSTRAP_MODULE, _PTH_FILE):
            path = os.path.join(sp, filename)
            if os.path.isfile(path):
                os.remove(path)
                print(f"Removed {path}")
                removed = True

    if removed:
        print("\nHook uninstalled. Restart Python for changes to take effect.")
    else:
        print("No hook files found in site-packages.")
    return 0


def status():
    """Print current status of the import hook."""
    # Check if hook module is loaded.
    hook_loaded = "_candle_torch_compat" in sys.modules
    print(f"Hook module loaded: {hook_loaded}")

    # Check if the finder is active.
    if hook_loaded:
        import _candle_torch_compat  # noqa: F811
        active = _candle_torch_compat.is_active()
        print(f"Import hook active: {active}")
    else:
        print("Import hook active: N/A (module not loaded)")

    # Environment variable.
    env = os.environ.get("USE_CANDLE", "<not set>")
    print(f"USE_CANDLE env var:  {env}")

    # Real torch detection.
    try:
        # Use a fresh check (can't import _candle_torch_compat if not loaded)
        from _candle_torch_compat import _has_real_torch
        has_torch = _has_real_torch()
    except ImportError:
        # Inline check.
        has_torch = any(
            os.path.isfile(os.path.join(e, "torch", "__init__.py"))
            for e in sys.path if e
        )
    print(f"Real PyTorch found:  {has_torch}")

    # Installed files.
    print("\nInstalled files:")
    for sp in _all_site_packages():
        for filename in (_BOOTSTRAP_MODULE, _PTH_FILE):
            path = os.path.join(sp, filename)
            exists = os.path.isfile(path)
            marker = "  [found]  " if exists else "  [missing]"
            print(f"  {marker} {path}")

    return 0


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 1

    cmd = sys.argv[1].lower()
    if cmd == "install":
        return install()
    if cmd == "uninstall":
        return uninstall()
    if cmd == "status":
        return status()

    print(f"Unknown command: {cmd}", file=sys.stderr)
    print("Usage: python -m candle.torch_compat install|uninstall|status", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
