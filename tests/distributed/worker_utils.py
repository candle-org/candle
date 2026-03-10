"""Helpers for distributed test worker scripts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_worker_script(script: str, *, name: str | None = None) -> str:
    """Write a temporary worker script and return its path.

    Use a unique path per invocation to avoid collisions across test runs
    or users sharing the same /tmp.
    """

    prefix = "candle_dist_worker_"
    if name:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        prefix = f"{prefix}{safe}_"

    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(script)
    except Exception:
        os.close(fd)
        raise

    return str(Path(path))

