"""Tests for FSDP shard bookkeeping Cython fastpath.

TDD: these tests are written RED-first. They import from
``candle.distributed._fsdp_fastpath`` which does not yet exist.
"""
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeShape:
    """Minimal shape object."""
    def __init__(self, *dims):
        self._dims = dims
    def __len__(self):
        return len(self._dims)
    def __getitem__(self, i):
        return self._dims[i]


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


class _FakeTensor:
    """Minimal tensor stand-in for shard tests."""
    def __init__(self, *shape, dtype='float32'):
        self.shape = _FakeShape(*shape)
        self.dtype_name = dtype

    def numel(self):
        return _numel(self.shape)


# ---------------------------------------------------------------------------
# Tests for compute_shard_offset
# ---------------------------------------------------------------------------

def test_compute_shard_offset_basic():
    """compute_shard_offset returns cumulative numel of preceding shards."""
    from candle.distributed._fsdp_fastpath import compute_shard_offset

    shapes = [(4, 8), (2, 16), (10,)]
    expected_offsets = [0, 32, 64]
    for i, (shape, expected) in enumerate(zip(shapes, expected_offsets)):
        assert compute_shard_offset(shapes, i) == expected, (
            f"offset[{i}] should be {expected}"
        )


def test_compute_shard_offset_single_param():
    """Single param always has offset 0."""
    from candle.distributed._fsdp_fastpath import compute_shard_offset

    assert compute_shard_offset([(100,)], 0) == 0


def test_compute_shard_offset_total():
    """Offset of one-past-last equals total numel."""
    from candle.distributed._fsdp_fastpath import compute_shard_offset

    shapes = [(4, 8), (2, 16), (10,)]
    total = 32 + 32 + 10
    # simulate 'total' query: pass len(shapes) as idx via helper
    offsets = [compute_shard_offset(shapes, i) for i in range(len(shapes))]
    assert offsets[-1] + _numel(shapes[-1]) == total


# ---------------------------------------------------------------------------
# Tests for build_flat_shard_offsets
# ---------------------------------------------------------------------------

def test_build_flat_shard_offsets_returns_list_of_pairs():
    """build_flat_shard_offsets returns [(start, end)] pairs."""
    from candle.distributed._fsdp_fastpath import build_flat_shard_offsets

    shapes = [(4, 8), (2, 16), (10,)]
    offsets = build_flat_shard_offsets(shapes)
    assert len(offsets) == 3
    assert offsets[0] == (0, 32)
    assert offsets[1] == (32, 64)
    assert offsets[2] == (64, 74)


def test_build_flat_shard_offsets_contiguous():
    """Consecutive offsets must be contiguous (no gaps)."""
    from candle.distributed._fsdp_fastpath import build_flat_shard_offsets

    shapes = [(3, 3), (5,), (2, 2, 2)]
    offsets = build_flat_shard_offsets(shapes)
    for i in range(len(offsets) - 1):
        assert offsets[i][1] == offsets[i + 1][0], (
            f"gap between shard {i} end={offsets[i][1]} and shard {i+1} start={offsets[i+1][0]}"
        )


def test_build_flat_shard_offsets_empty():
    """Empty input returns empty list."""
    from candle.distributed._fsdp_fastpath import build_flat_shard_offsets

    assert build_flat_shard_offsets([]) == []


# ---------------------------------------------------------------------------
# Tests for pack_shards_to_flat / unpack_flat_to_shards
# ---------------------------------------------------------------------------

def test_pack_shards_to_flat_values():
    """pack_shards_to_flat copies shard data into the flat buffer."""
    from candle.distributed._fsdp_fastpath import pack_shards_to_flat

    # Use plain lists as fake tensors — fastpath should call .tolist() / iterate
    shards = [[1.0, 2.0, 3.0], [4.0, 5.0]]
    offsets = [(0, 3), (3, 5)]
    flat = [0.0] * 5
    pack_shards_to_flat(shards, flat, offsets)
    assert flat == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_unpack_flat_to_shards_values():
    """unpack_flat_to_shards copies slices of flat buffer back to shards."""
    from candle.distributed._fsdp_fastpath import unpack_flat_to_shards

    flat = [10.0, 20.0, 30.0, 40.0, 50.0]
    offsets = [(0, 3), (3, 5)]
    shards = [[0.0, 0.0, 0.0], [0.0, 0.0]]
    unpack_flat_to_shards(flat, shards, offsets)
    assert shards[0] == [10.0, 20.0, 30.0]
    assert shards[1] == [40.0, 50.0]


def test_pack_unpack_roundtrip():
    """pack then unpack is identity."""
    from candle.distributed._fsdp_fastpath import (
        pack_shards_to_flat,
        unpack_flat_to_shards,
    )

    shards_orig = [[float(i) for i in range(6)], [float(i * 2) for i in range(4)]]
    offsets = [(0, 6), (6, 10)]
    flat = [0.0] * 10
    pack_shards_to_flat(shards_orig, flat, offsets)

    shards_out = [[0.0] * 6, [0.0] * 4]
    unpack_flat_to_shards(flat, shards_out, offsets)
    assert shards_out == shards_orig


# ---------------------------------------------------------------------------
# Tests for build_param_owner_map
# ---------------------------------------------------------------------------

def test_build_param_owner_map_basic():
    """build_param_owner_map maps each param_name to its group index."""
    from candle.distributed._fsdp_fastpath import build_param_owner_map

    groups = [
        ["fc1.weight", "fc1.bias"],
        ["fc2.weight", "fc2.bias"],
    ]
    mapping = build_param_owner_map(groups)
    assert mapping["fc1.weight"] == 0
    assert mapping["fc1.bias"] == 0
    assert mapping["fc2.weight"] == 1
    assert mapping["fc2.bias"] == 1


def test_build_param_owner_map_empty():
    """Empty groups returns empty dict."""
    from candle.distributed._fsdp_fastpath import build_param_owner_map

    assert build_param_owner_map([]) == {}


def test_build_param_owner_map_single_group():
    """Single group — all params map to 0."""
    from candle.distributed._fsdp_fastpath import build_param_owner_map

    groups = [["a", "b", "c"]]
    mapping = build_param_owner_map(groups)
    assert mapping == {"a": 0, "b": 0, "c": 0}


# ---------------------------------------------------------------------------
# Tests for compute_chunk_size / compute_padded_size
# ---------------------------------------------------------------------------

def test_compute_chunk_size_exact_divisible():
    """When dim_size divisible by world_size, chunk = dim_size // world_size."""
    from candle.distributed._fsdp_fastpath import compute_chunk_size

    assert compute_chunk_size(16, 4) == 4
    assert compute_chunk_size(8, 2) == 4


def test_compute_chunk_size_ceil_division():
    """ceil division: chunk_size = (dim + ws - 1) // ws."""
    from candle.distributed._fsdp_fastpath import compute_chunk_size

    assert compute_chunk_size(10, 3) == 4   # ceil(10/3) = 4
    assert compute_chunk_size(7, 4) == 2    # ceil(7/4) = 2


def test_compute_padded_size_no_padding():
    """Exact divisors produce zero padding."""
    from candle.distributed._fsdp_fastpath import compute_padded_size

    padded, padding = compute_padded_size(16, 4)
    assert padded == 16
    assert padding == 0


def test_compute_padded_size_with_padding():
    """Non-exact dims get padded to next multiple."""
    from candle.distributed._fsdp_fastpath import compute_padded_size

    padded, padding = compute_padded_size(10, 3)
    assert padded == 12   # 4 * 3
    assert padding == 2

    padded, padding = compute_padded_size(7, 4)
    assert padded == 8    # 2 * 4
    assert padding == 1


# ---------------------------------------------------------------------------
# Tests for writeback_shard_grad_flags
# ---------------------------------------------------------------------------

def test_writeback_shard_grad_flags_all_grad():
    """All params with requires_grad=True produce True flags."""
    from candle.distributed._fsdp_fastpath import writeback_shard_grad_flags

    class P:
        def __init__(self, rg):
            self.requires_grad = rg

    params = [P(True), P(True), P(False), P(True)]
    flags = writeback_shard_grad_flags(params)
    assert flags == [True, True, False, True]


def test_writeback_shard_grad_flags_empty():
    from candle.distributed._fsdp_fastpath import writeback_shard_grad_flags

    assert writeback_shard_grad_flags([]) == []
