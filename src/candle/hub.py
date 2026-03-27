"""candle.hub — compatibility stub for torch.hub.

Exposes the subset of torch.hub used by third-party libraries such as
torchvision::

    from torch.hub import _get_torch_home
    from torch.hub import load_state_dict_from_url

All heavy operations (network download, checkpoint loading) are delegated to
standard-library / PyTorch-free helpers; this module must not import torch at
runtime.
"""

import errno
import hashlib
import os
import re
import tempfile
import urllib.request
import uuid
import warnings
from typing import Any
from urllib.parse import urlparse

ENV_TORCH_HOME = "TORCH_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
_HASH_REGEX = re.compile(r".*-([0-9a-f]{8,})\..*")
_READ_DATA_CHUNK = 128 * 1024
_hub_dir: str | None = None


def _get_torch_home() -> str:
    """Return the torch home directory."""
    return os.path.expanduser(
        os.getenv(
            ENV_TORCH_HOME,
            os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "torch"),
        )
    )



def get_dir() -> str:
    """Return the hub cache directory."""
    if _hub_dir is not None:
        return _hub_dir
    return os.path.join(_get_torch_home(), "hub")



def set_dir(d: str | os.PathLike[str]) -> None:
    """Set the hub cache directory used by load_state_dict_from_url."""
    global _hub_dir
    _hub_dir = os.path.expanduser(os.fspath(d))



def _download_url_to_file(url: str, dst: str, hash_prefix: str | None = None) -> None:
    """Download *url* to *dst* using a temporary .partial file."""
    dst = os.path.expanduser(dst)
    tmp_dst = None
    handle = None

    for _ in range(tempfile.TMP_MAX):
        candidate = f"{dst}.{uuid.uuid4().hex}.partial"
        try:
            handle = open(candidate, "xb")
            tmp_dst = candidate
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError(errno.EEXIST, "No usable temporary file name found")

    request = urllib.request.Request(url, headers={"User-Agent": "torch.hub"})
    sha256 = hashlib.sha256() if hash_prefix is not None else None

    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310
            while True:
                chunk = response.read(_READ_DATA_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                if sha256 is not None:
                    sha256.update(chunk)

        handle.close()
        handle = None

        if sha256 is not None:
            digest = sha256.hexdigest()
            if not digest.startswith(hash_prefix):
                raise RuntimeError(
                    f'invalid hash value (expected "{hash_prefix}", got "{digest[:len(hash_prefix)]}")'
                )

        os.replace(tmp_dst, dst)
        tmp_dst = None
    finally:
        if handle is not None:
            handle.close()
        if tmp_dst is not None and os.path.exists(tmp_dst):
            os.remove(tmp_dst)



def load_state_dict_from_url(
    url: str,
    model_dir: str | None = None,
    map_location: Any = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: str | None = None,
    weights_only: bool = False,
) -> "dict[str, Any]":
    """Download and load a serialised state-dict from *url*."""
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn(
            "TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead",
            stacklevel=2,
        )

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = file_name or os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)

    hash_prefix = None
    if check_hash:
        match = _HASH_REGEX.search(filename)
        hash_prefix = match.group(1) if match else None

    if not os.path.exists(cached_file):
        if progress:
            warnings.warn(
                "candle.hub: progress display not implemented; downloading silently.",
                stacklevel=2,
            )
        _download_url_to_file(url, cached_file, hash_prefix=hash_prefix)

    if check_hash and hash_prefix is not None:
        sha256 = hashlib.sha256()
        with open(cached_file, "rb") as handle:
            for chunk in iter(lambda: handle.read(_READ_DATA_CHUNK), b""):
                sha256.update(chunk)
        digest = sha256.hexdigest()
        if not digest.startswith(hash_prefix):
            raise RuntimeError(
                f'invalid hash value (expected "{hash_prefix}", got "{digest[:len(hash_prefix)]}")'
            )

    from candle import serialization

    return serialization.load(cached_file, map_location=map_location, weights_only=weights_only)


__all__ = [
    "_get_torch_home",
    "get_dir",
    "set_dir",
    "load_state_dict_from_url",
]
