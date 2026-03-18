"""Tests for DTensor Checkpointable protocol and DCP planner integration."""
import candle as torch
from candle.distributed.tensor.placement import Shard, Replicate, Partial
from candle.distributed.tensor.dtensor import (
    DTensor, DTensorSpec, TensorMeta, compute_local_shape_and_global_offset,
)
from candle.distributed.checkpoint.metadata import (
    ChunkStorageMetadata, MetadataIndex, TensorStorageMetadata, TensorProperties,
)
from candle.distributed.checkpoint.planner import (
    DefaultSavePlanner, DefaultLoadPlanner, WriteItemType, LoadItemType,
    _chunk_overlap,
)


# ---------------------------------------------------------------------------
# Mock mesh that doesn't require dist.init_process_group
# ---------------------------------------------------------------------------

class MockMesh:
    """Simulates a 1D DeviceMesh with configurable size and local rank."""

    def __init__(self, world_size=2, local_rank=0):
        self._world_size = world_size
        self._local_rank = local_rank

    def size(self, mesh_dim=0):
        return self._world_size

    def get_local_rank(self, mesh_dim=0):
        return self._local_rank

    @property
    def ndim(self):
        return 1


def _make_dtensor(local_data, mesh, placements, global_shape=None):
    """Create a DTensor from raw data + mock mesh."""
    local_tensor = torch.tensor(local_data, dtype=torch.float32)
    if global_shape is None:
        # Infer global shape from placement
        shape = list(local_tensor.shape)
        for mesh_dim, p in enumerate(placements):
            if isinstance(p, Shard):
                shape[p.dim] *= mesh.size(mesh_dim)
        global_shape = tuple(shape)
    meta = TensorMeta(
        shape=global_shape,
        stride=(global_shape[-1], 1) if len(global_shape) == 2 else (1,),
        dtype=torch.float32,
    )
    spec = DTensorSpec(mesh, placements, tensor_meta=meta)
    return DTensor(local_tensor, spec)


# ---------------------------------------------------------------------------
# compute_local_shape_and_global_offset
# ---------------------------------------------------------------------------

class TestComputeLocalShapeAndGlobalOffset:

    def test_shard_dim0_even_split(self):
        mesh = MockMesh(world_size=4, local_rank=0)
        shape, offset = compute_local_shape_and_global_offset(
            (8, 4), mesh, (Shard(0),)
        )
        assert shape == (2, 4)
        assert offset == (0, 0)

    def test_shard_dim0_even_split_rank2(self):
        mesh = MockMesh(world_size=4, local_rank=2)
        shape, offset = compute_local_shape_and_global_offset(
            (8, 4), mesh, (Shard(0),)
        )
        assert shape == (2, 4)
        assert offset == (4, 0)

    def test_shard_dim0_uneven_split(self):
        # 7 rows across 4 ranks: ranks 0-2 get 2, rank 3 gets 1
        mesh = MockMesh(world_size=4, local_rank=0)
        shape, offset = compute_local_shape_and_global_offset(
            (7, 4), mesh, (Shard(0),)
        )
        assert shape == (2, 4)
        assert offset == (0, 0)

    def test_shard_dim0_uneven_split_last_rank(self):
        mesh = MockMesh(world_size=4, local_rank=3)
        shape, offset = compute_local_shape_and_global_offset(
            (7, 4), mesh, (Shard(0),)
        )
        # remainder=3, rank 3 >= remainder -> chunk_size=1
        assert shape == (1, 4)
        assert offset == (6, 0)

    def test_shard_dim1(self):
        mesh = MockMesh(world_size=2, local_rank=1)
        shape, offset = compute_local_shape_and_global_offset(
            (8, 4), mesh, (Shard(1),)
        )
        assert shape == (8, 2)
        assert offset == (0, 2)

    def test_replicate_no_change(self):
        mesh = MockMesh(world_size=4, local_rank=2)
        shape, offset = compute_local_shape_and_global_offset(
            (8, 4), mesh, (Replicate(),)
        )
        assert shape == (8, 4)
        assert offset == (0, 0)

    def test_1d_tensor(self):
        mesh = MockMesh(world_size=2, local_rank=1)
        shape, offset = compute_local_shape_and_global_offset(
            (10,), mesh, (Shard(0),)
        )
        assert shape == (5,)
        assert offset == (5,)


# ---------------------------------------------------------------------------
# DTensor.__create_write_items__
# ---------------------------------------------------------------------------

class TestDTensorCreateWriteItems:

    def test_shard_produces_shard_write_item(self):
        mesh = MockMesh(world_size=2, local_rank=0)
        dt = _make_dtensor([[1, 2, 3, 4]], mesh, (Shard(0),), global_shape=(2, 4))
        items = dt.__create_write_items__("layer.weight", dt)
        assert len(items) == 1
        item = items[0]
        assert item.type == WriteItemType.SHARD
        assert item.index.fqn == "layer.weight"
        assert item.index.offset == (0, 0)
        assert item.tensor_data.chunk.sizes == (1, 4)
        assert item.tensor_data.chunk.offsets == (0, 0)
        assert item.tensor_data.size == (2, 4)

    def test_shard_rank1_offset(self):
        mesh = MockMesh(world_size=2, local_rank=1)
        dt = _make_dtensor([[5, 6, 7, 8]], mesh, (Shard(0),), global_shape=(2, 4))
        items = dt.__create_write_items__("layer.weight", dt)
        item = items[0]
        assert item.index.offset == (1, 0)
        assert item.tensor_data.chunk.offsets == (1, 0)

    def test_replicate_produces_shard_write_item(self):
        mesh = MockMesh(world_size=2, local_rank=0)
        dt = _make_dtensor([[1, 2], [3, 4]], mesh, (Replicate(),), global_shape=(2, 2))
        items = dt.__create_write_items__("layer.bias", dt)
        assert len(items) == 1
        item = items[0]
        assert item.type == WriteItemType.SHARD
        assert item.tensor_data.chunk.sizes == (2, 2)
        assert item.tensor_data.chunk.offsets == (0, 0)

    def test_partial_raises(self):
        mesh = MockMesh(world_size=2, local_rank=0)
        dt = _make_dtensor([[1, 2]], mesh, (Partial(),), global_shape=(1, 2))
        try:
            dt.__create_write_items__("x", dt)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as exc:
            assert "Partial" in str(exc)

    def test_preserves_dtype_and_requires_grad(self):
        mesh = MockMesh(world_size=2, local_rank=0)
        local = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        local.requires_grad = True
        meta = TensorMeta(shape=(2, 2), stride=(2, 1), dtype=torch.float32)
        spec = DTensorSpec(mesh, (Shard(0),), tensor_meta=meta)
        dt = DTensor(local, spec, requires_grad=True)
        items = dt.__create_write_items__("w", dt)
        assert items[0].tensor_data.properties.dtype == torch.float32
        assert items[0].tensor_data.properties.requires_grad is True


# ---------------------------------------------------------------------------
# DTensor.__create_chunk_list__
# ---------------------------------------------------------------------------

class TestDTensorCreateChunkList:

    def test_shard_chunk(self):
        mesh = MockMesh(world_size=4, local_rank=2)
        dt = _make_dtensor([[1, 2, 3, 4], [5, 6, 7, 8]], mesh, (Shard(0),),
                           global_shape=(8, 4))
        chunks = dt.__create_chunk_list__()
        assert len(chunks) == 1
        assert isinstance(chunks[0], ChunkStorageMetadata)
        assert chunks[0].sizes == (2, 4)
        assert chunks[0].offsets == (4, 0)

    def test_replicate_chunk_is_full(self):
        mesh = MockMesh(world_size=2, local_rank=1)
        dt = _make_dtensor([[1, 2], [3, 4]], mesh, (Replicate(),),
                           global_shape=(2, 2))
        chunks = dt.__create_chunk_list__()
        assert chunks[0].sizes == (2, 2)
        assert chunks[0].offsets == (0, 0)


# ---------------------------------------------------------------------------
# DefaultSavePlanner
# ---------------------------------------------------------------------------

class TestDefaultSavePlanner:

    def test_regular_tensor(self):
        planner = DefaultSavePlanner()
        t = torch.randn(3, 4)
        planner.set_up_planner({"weight": t})
        plan = planner.create_local_plan()
        assert len(plan.items) == 1
        item = plan.items[0]
        assert item.type == WriteItemType.TENSOR
        assert item.index.fqn == "weight"
        assert item.tensor_data.size == (3, 4)
        assert item.tensor_data.chunk.offsets == (0, 0)
        assert item.tensor_data.chunk.sizes == (3, 4)

    def test_dtensor_delegates(self):
        mesh = MockMesh(world_size=2, local_rank=0)
        dt = _make_dtensor([[1, 2, 3, 4]], mesh, (Shard(0),), global_shape=(2, 4))
        planner = DefaultSavePlanner()
        planner.set_up_planner({"param": dt})
        plan = planner.create_local_plan()
        assert len(plan.items) == 1
        assert plan.items[0].type == WriteItemType.SHARD

    def test_resolve_data_dtensor_returns_local(self):
        mesh = MockMesh(world_size=2, local_rank=0)
        local = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        dt = _make_dtensor([[1.0, 2.0, 3.0, 4.0]], mesh, (Shard(0),),
                           global_shape=(2, 4))
        planner = DefaultSavePlanner()
        planner.set_up_planner({"p": dt})
        plan = planner.create_local_plan()
        resolved = planner.resolve_data(plan.items[0])
        # Should be the local tensor, not the DTensor
        assert not isinstance(resolved, DTensor)
        assert resolved.shape == (1, 4)

    def test_resolve_data_regular_tensor(self):
        t = torch.randn(3, 4)
        planner = DefaultSavePlanner()
        planner.set_up_planner({"w": t})
        plan = planner.create_local_plan()
        resolved = planner.resolve_data(plan.items[0])
        assert resolved.shape == (3, 4)

    def test_create_global_plan(self):
        mesh0 = MockMesh(world_size=2, local_rank=0)
        mesh1 = MockMesh(world_size=2, local_rank=1)
        dt0 = _make_dtensor([[1, 2, 3, 4]], mesh0, (Shard(0),), global_shape=(2, 4))
        dt1 = _make_dtensor([[5, 6, 7, 8]], mesh1, (Shard(0),), global_shape=(2, 4))

        p0 = DefaultSavePlanner()
        p0.set_up_planner({"w": dt0})
        plan0 = p0.create_local_plan()

        p1 = DefaultSavePlanner()
        p1.set_up_planner({"w": dt1})
        plan1 = p1.create_local_plan()

        # Use p0 to merge
        _, metadata = p0.create_global_plan([plan0, plan1])
        assert "w" in metadata.state_dict_metadata
        tsm = metadata.state_dict_metadata["w"]
        assert isinstance(tsm, TensorStorageMetadata)
        assert tsm.size == (2, 4)
        assert len(tsm.chunks) == 2

    def test_skips_non_tensor(self):
        planner = DefaultSavePlanner()
        planner.set_up_planner({"lr": 0.01, "weight": torch.randn(2, 2)})
        plan = planner.create_local_plan()
        assert len(plan.items) == 1
        assert plan.items[0].index.fqn == "weight"


# ---------------------------------------------------------------------------
# DefaultLoadPlanner
# ---------------------------------------------------------------------------

class TestDefaultLoadPlanner:

    def _make_metadata(self, fqn, size, chunks):
        props = TensorProperties(dtype=torch.float32)
        tsm = TensorStorageMetadata(properties=props, size=size, chunks=chunks)
        return {"state_dict_metadata": {fqn: tsm}}

    def test_regular_tensor_load(self):
        from candle.distributed.checkpoint.metadata import Metadata
        chunk = ChunkStorageMetadata(offsets=(0, 0), sizes=(4, 4))
        meta = Metadata({"w": TensorStorageMetadata(
            properties=TensorProperties(dtype=torch.float32),
            size=(4, 4), chunks=[chunk],
        )})
        t = torch.zeros(4, 4)
        planner = DefaultLoadPlanner()
        planner.set_up_planner({"w": t}, metadata=meta)
        plan = planner.create_local_plan()
        assert len(plan.items) == 1
        item = plan.items[0]
        assert item.type == LoadItemType.TENSOR
        assert item.lengths == (4, 4)

    def test_dtensor_load_same_sharding(self):
        from candle.distributed.checkpoint.metadata import Metadata
        # Saved as 2 shards: [0:2,:] and [2:4,:]
        chunk0 = ChunkStorageMetadata(offsets=(0, 0), sizes=(2, 4))
        chunk1 = ChunkStorageMetadata(offsets=(2, 0), sizes=(2, 4))
        meta = Metadata({"w": TensorStorageMetadata(
            properties=TensorProperties(dtype=torch.float32),
            size=(4, 4), chunks=[chunk0, chunk1],
        )})
        # Loading rank 0: wants [0:2,:]
        mesh = MockMesh(world_size=2, local_rank=0)
        dt = _make_dtensor([[0]*4, [0]*4], mesh, (Shard(0),), global_shape=(4, 4))
        planner = DefaultLoadPlanner()
        planner.set_up_planner({"w": dt}, metadata=meta)
        plan = planner.create_local_plan()
        # Should match exactly one saved chunk
        assert len(plan.items) == 1
        item = plan.items[0]
        assert item.dest_offsets == (0, 0)
        assert item.storage_offsets == (0, 0)
        assert item.lengths == (2, 4)

    def test_dtensor_load_resharding(self):
        """Save with 2 ranks, load with 4 ranks — resharding."""
        from candle.distributed.checkpoint.metadata import Metadata
        # Saved: 2 shards of (4,4) from global (8,4)
        chunk0 = ChunkStorageMetadata(offsets=(0, 0), sizes=(4, 4))
        chunk1 = ChunkStorageMetadata(offsets=(4, 0), sizes=(4, 4))
        meta = Metadata({"w": TensorStorageMetadata(
            properties=TensorProperties(dtype=torch.float32),
            size=(8, 4), chunks=[chunk0, chunk1],
        )})
        # Loading rank 1 of 4: wants rows [2:4,:] — overlaps with saved chunk0
        mesh = MockMesh(world_size=4, local_rank=1)
        dt = _make_dtensor([[0]*4, [0]*4], mesh, (Shard(0),), global_shape=(8, 4))
        planner = DefaultLoadPlanner()
        planner.set_up_planner({"w": dt}, metadata=meta)
        plan = planner.create_local_plan()
        assert len(plan.items) == 1
        item = plan.items[0]
        assert item.lengths == (2, 4)
        # dest_offsets: within local chunk (0,0)
        assert item.dest_offsets == (0, 0)
        # storage_offsets: within saved chunk0, rows 2-3
        assert item.storage_offsets == (2, 0)

    def test_resolve_tensor_dtensor(self):
        mesh = MockMesh(world_size=2, local_rank=0)
        dt = _make_dtensor([[1, 2]], mesh, (Shard(0),), global_shape=(2, 2))
        planner = DefaultLoadPlanner()
        planner.set_up_planner({"w": dt})

        class FakeReadItem:
            def __init__(self):
                self.dest_index = MetadataIndex("w")
        resolved = planner.resolve_tensor(FakeReadItem())
        assert not isinstance(resolved, DTensor)
        assert resolved.shape == (1, 2)

    def test_commit_tensor_dtensor_noop(self):
        mesh = MockMesh(world_size=2, local_rank=0)
        dt = _make_dtensor([[1, 2]], mesh, (Shard(0),), global_shape=(2, 2))
        planner = DefaultLoadPlanner()
        planner.set_up_planner({"w": dt})

        class FakeReadItem:
            def __init__(self):
                self.dest_index = MetadataIndex("w")
        planner.commit_tensor(FakeReadItem(), torch.zeros(1, 2))
        # DTensor should NOT be replaced
        assert isinstance(planner.state_dict["w"], DTensor)

    def test_commit_tensor_regular_replaces(self):
        t = torch.zeros(2, 2)
        planner = DefaultLoadPlanner()
        planner.set_up_planner({"w": t})

        class FakeReadItem:
            def __init__(self):
                self.dest_index = MetadataIndex("w")
        new_t = torch.ones(2, 2)
        planner.commit_tensor(FakeReadItem(), new_t)
        assert planner.state_dict["w"] is new_t

    def test_missing_fqn_skipped(self):
        from candle.distributed.checkpoint.metadata import Metadata
        chunk = ChunkStorageMetadata(offsets=(0,), sizes=(4,))
        meta = Metadata({"w": TensorStorageMetadata(
            properties=TensorProperties(dtype=torch.float32),
            size=(4,), chunks=[chunk],
        )})
        planner = DefaultLoadPlanner()
        planner.set_up_planner({"other": torch.zeros(4)}, metadata=meta)
        plan = planner.create_local_plan()
        assert len(plan.items) == 0


# ---------------------------------------------------------------------------
# _chunk_overlap
# ---------------------------------------------------------------------------

class TestChunkOverlap:

    def test_full_overlap(self):
        a = ChunkStorageMetadata(offsets=(0, 0), sizes=(4, 4))
        b = ChunkStorageMetadata(offsets=(0, 0), sizes=(4, 4))
        result = _chunk_overlap(a, b)
        assert result == ((0, 0), (0, 0), (4, 4))

    def test_no_overlap(self):
        a = ChunkStorageMetadata(offsets=(0, 0), sizes=(2, 4))
        b = ChunkStorageMetadata(offsets=(2, 0), sizes=(2, 4))
        result = _chunk_overlap(a, b)
        assert result is None

    def test_partial_overlap(self):
        a = ChunkStorageMetadata(offsets=(1, 0), sizes=(4, 4))
        b = ChunkStorageMetadata(offsets=(0, 0), sizes=(3, 4))
        result = _chunk_overlap(a, b)
        dest_off, stor_off, lengths = result
        assert lengths == (2, 4)
        assert dest_off == (0, 0)
        assert stor_off == (1, 0)

    def test_1d_overlap(self):
        a = ChunkStorageMetadata(offsets=(5,), sizes=(5,))
        b = ChunkStorageMetadata(offsets=(3,), sizes=(4,))
        result = _chunk_overlap(a, b)
        dest_off, stor_off, lengths = result
        assert lengths == (2,)
        assert dest_off == (0,)
        assert stor_off == (2,)
