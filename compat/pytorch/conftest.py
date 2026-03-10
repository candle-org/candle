"""PyTorch test compatibility conftest.

Unlike transformers conftest, this does NOT do import redirection.
Candle's .pth + meta path finder handles torch->candle aliasing.
This conftest only handles:
  - torch.compile no-op
  - xfail injection from xfail.yaml
"""
import sys
from pathlib import Path

import pytest

_COMPAT_DIR = Path(__file__).resolve().parent
_COMPAT_ROOT = _COMPAT_DIR.parent
if str(_COMPAT_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMPAT_ROOT))

from conftest_base import load_xfail_config, match_xfail  # noqa: E402


def pytest_collection_modifyitems(config, items):
    """Mark known failures as xfail from xfail.yaml."""
    xfail_cfg = load_xfail_config(_COMPAT_DIR / "xfail.yaml")
    if not xfail_cfg:
        return
    global_patterns = xfail_cfg.get("_global", [])
    for item in items:
        reason = match_xfail(item.nodeid, global_patterns)
        if reason:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
