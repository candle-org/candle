"""Integration tests proving the FSDP Python path exercises Cython fastpath wiring.

These tests verify that:
  - FSDPParamGroup._copy_shards_to_flat calls pack_shards_to_flat (not just
    the Python fallback) when the Cython extension is present.
  - FSDPParamGroup._copy_flat_to_shard_grads calls unpack_flat_to_shards
    when the Cython extension is present.
  - FSDPParamGroup.param_owner_map is built via build_param_owner_map
    and correctly maps every managed parameter name to its group index.

Unlike tests/distributed/test_fsdp_shard_fastpath.py (which tests the Cython
helpers in isolation), these tests exercise the helpers through the real FSDP
Python call stack.
"""
from unittest.mock import patch
import pytest
from candle import nn


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

class MockMeshInfo:
    """Single-rank mesh info sufficient for unit tests."""

    def __init__(self, world_size=1):
        self.shard_mesh_rank = 0
        self.shard_mesh_size = world_size
        self.shard_process_group = None

        ws = world_size

        class _Mesh:
            def size(self, dim=0):
                return ws
            @property
            def ndim(self):
                return 1

        self.mesh = _Mesh()


def _make_fsdp_param_group(mod, world_size=1):
    """Build an FSDPParamGroup directly for unit testing."""
    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam
    from candle.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup

    mesh_info = MockMeshInfo(world_size=world_size)
    params = list(mod.named_parameters(recurse=False))
    fsdp_params = [FSDPParam(p, mod, name, mesh_info) for name, p in params]

    if world_size == 1:
        for fp in fsdp_params:
            fp.unshard = fp._unshard_single_rank

    return FSDPParamGroup(fsdp_params, mod, mesh_info), fsdp_params


# ---------------------------------------------------------------------------
# Guard: skip all tests if Cython extension is absent
# ---------------------------------------------------------------------------

def _fastpath_available():
    try:
        import candle.distributed._fsdp_fastpath  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _fastpath_available(),
    reason="Cython _fsdp_fastpath extension not compiled",
)


# ---------------------------------------------------------------------------
# Test 1: _copy_shards_to_flat calls the Cython pack helper
# ---------------------------------------------------------------------------

def test_copy_shards_to_flat_calls_cython_pack():
    """_copy_shards_to_flat must invoke the Cython pack_shards_to_flat helper.

    We patch the module-level alias _cy_pack_shards that _fsdp_param_group
    imported at load time, then assert it was called.
    """
    import candle.distributed._fsdp_fastpath as _fp_mod
    import candle.distributed._composable.fsdp._fsdp_param_group as _pg_mod

    if not _pg_mod._HAVE_FASTPATH:
        pytest.skip("fastpath not active in _fsdp_param_group")

    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam
    from candle.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup

    # world_size=2, bias=True gives two params -> _use_flat_buffer=True
    mesh_info = MockMeshInfo(world_size=2)
    mod = nn.Linear(8, 4, bias=True)
    params = list(mod.named_parameters(recurse=False))
    fsdp_params = [FSDPParam(p, mod, name, mesh_info) for name, p in params]

    call_count = []
    real_pack = _fp_mod.pack_shards_to_flat

    def spy_pack(shards, flat, offsets):
        call_count.append(1)
        return real_pack(shards, flat, offsets)

    with patch.object(_pg_mod, "_cy_pack_shards", spy_pack):
        group = FSDPParamGroup(fsdp_params, mod, mesh_info)
        # _init_flat_buffer calls _copy_shards_to_flat once during __init__
        assert len(call_count) >= 1, (
            "_cy_pack_shards (pack_shards_to_flat) was not called during "
            "FSDPParamGroup init -- fastpath wiring is broken"
        )
        call_count.clear()

        # Verify explicit call also goes through the Cython path
        group._copy_shards_to_flat()
        assert len(call_count) == 1, (
            "_cy_pack_shards was not called by _copy_shards_to_flat -- "
            "fastpath wiring is broken"
        )


# ---------------------------------------------------------------------------
# Test 2: _copy_flat_to_shard_grads calls the Cython unpack helper
# ---------------------------------------------------------------------------

def test_copy_flat_to_shard_grads_calls_cython_unpack():
    """_copy_flat_to_shard_grads must invoke the Cython unpack_flat_to_shards."""
    import candle.distributed._fsdp_fastpath as _fp_mod
    import candle.distributed._composable.fsdp._fsdp_param_group as _pg_mod

    if not _pg_mod._HAVE_FASTPATH:
        pytest.skip("fastpath not active in _fsdp_param_group")

    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam
    from candle.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
    from candle._creation import zeros

    mesh_info = MockMeshInfo(world_size=2)
    mod = nn.Linear(8, 4, bias=True)
    params = list(mod.named_parameters(recurse=False))
    fsdp_params = [FSDPParam(p, mod, name, mesh_info) for name, p in params]
    group = FSDPParamGroup(fsdp_params, mod, mesh_info)

    flat_src = zeros(group._total_shard_numel, dtype=group._flat_shard.dtype)

    call_count = []
    real_unpack = _fp_mod.unpack_flat_to_shards

    def spy_unpack(flat, shards, offsets):
        call_count.append(1)
        return real_unpack(flat, shards, offsets)

    with patch.object(_pg_mod, "_cy_unpack_shards", spy_unpack):
        group._copy_flat_to_shard_grads(flat_src)
        assert len(call_count) == 1, (
            "_cy_unpack_shards (unpack_flat_to_shards) was not called by "
            "_copy_flat_to_shard_grads -- fastpath wiring is broken"
        )


# ---------------------------------------------------------------------------
# Test 3: param_owner_map built via build_param_owner_map
# ---------------------------------------------------------------------------

def test_param_owner_map_built_via_cython():
    """FSDPParamGroup.param_owner_map must be populated by build_param_owner_map.

    Every param managed by the group must appear in the map at index 0
    (the only group index in this single-group scenario).
    """
    import candle.distributed._fsdp_fastpath as _fp_mod
    import candle.distributed._composable.fsdp._fsdp_param_group as _pg_mod

    if not _pg_mod._HAVE_FASTPATH:
        pytest.skip("fastpath not active in _fsdp_param_group")

    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam
    from candle.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup

    mesh_info = MockMeshInfo(world_size=1)
    mod = nn.Linear(6, 4, bias=True)
    params = list(mod.named_parameters(recurse=False))
    fsdp_params = [FSDPParam(p, mod, name, mesh_info) for name, p in params]

    call_count = []
    real_owner_map = _fp_mod.build_param_owner_map

    def spy_owner_map(groups):
        call_count.append(groups)
        return real_owner_map(groups)

    with patch.object(_pg_mod, "_cy_owner_map", spy_owner_map):
        group = FSDPParamGroup(fsdp_params, mod, mesh_info)

    assert len(call_count) == 1, (
        "build_param_owner_map was not called during FSDPParamGroup init -- "
        "fastpath wiring is broken"
    )

    expected_names = {name for name, _ in params}
    assert expected_names == set(group.param_owner_map.keys()), (
        f"param_owner_map keys {set(group.param_owner_map.keys())} != "
        f"expected {expected_names}"
    )
    for name in expected_names:
        assert group.param_owner_map[name] == 0, (
            f"param_owner_map[{name!r}] should be 0, got {group.param_owner_map[name]}"
        )


# ---------------------------------------------------------------------------
# Test 4: pack/unpack round-trip through real FSDP group methods
# ---------------------------------------------------------------------------

def test_pack_unpack_roundtrip_through_fsdp_group():
    """pack via _copy_shards_to_flat then unpack via _copy_flat_to_shard_grads
    must reproduce the original shard data values (identity round-trip).
    """
    import candle.distributed._composable.fsdp._fsdp_param_group as _pg_mod

    if not _pg_mod._HAVE_FASTPATH:
        pytest.skip("fastpath not active in _fsdp_param_group")

    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam
    from candle.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup

    mesh_info = MockMeshInfo(world_size=2)
    mod = nn.Linear(8, 4, bias=True)  # weight (4x8=32 elems/2=16) + bias (4/2=2)
    params = list(mod.named_parameters(recurse=False))
    fsdp_params = [FSDPParam(p, mod, name, mesh_info) for name, p in params]
    group = FSDPParamGroup(fsdp_params, mod, mesh_info)

    # _init_flat_buffer already packed shards; verify flat_shard is non-trivial
    flat_data = [float(v) for v in group._flat_shard]
    assert any(abs(v) > 0 for v in flat_data), (
        "_flat_shard is all zeros -- pack did not copy shard data"
    )

    # Unpack flat_shard back to shard grads via _copy_flat_to_shard_grads
    group._copy_flat_to_shard_grads(group._flat_shard)

    # Each shard grad must match what was packed
    for fp, (start, end), shape in zip(
        group.fsdp_params, group._shard_offsets, group._shard_shapes
    ):
        grad = fp._sharded_param.to_local().grad
        assert grad is not None, (
            f"grad not written for param {fp._param_name}"
        )
        expected = [float(v) for v in group._flat_shard[start:end]]
        actual = [float(v) for v in grad.reshape(-1)]
        assert expected == pytest.approx(actual, rel=1e-5), (
            f"grad round-trip mismatch for {fp._param_name}: "
            f"expected {expected[:4]}... got {actual[:4]}..."
        )
