"""Tests for minimal pipeline parallel runtime (manual frontend only).

Scope (Task 15 minimal slice):
- PipelineStage
- ScheduleGPipe
- simple microbatch split helper on dim 0
- no tracing frontend
- no 1F1B / interleaving
- no CUDA/NCCL

These tests are CPU/single-process friendly and use stubbed distributed
primitives to validate the scheduling and batch P2P semantics before any
real multi-rank HCCL execution.
"""

import pytest


class _DummyWork:
    def __init__(self):
        self.wait_calls = 0
    def wait(self, timeout=None):
        self.wait_calls += 1
        return True
    def is_completed(self):
        return self.wait_calls > 0


class _Recorder:
    def __init__(self):
        self.ops = []
    def isend(self, tensor, dst=None, group=None, tag=0, group_dst=None):
        self.ops.append(("isend", dst, tensor.shape if hasattr(tensor, "shape") else None))
        return _DummyWork()
    def irecv(self, tensor, src=None, group=None, tag=0, group_src=None):
        self.ops.append(("irecv", src, tensor.shape if hasattr(tensor, "shape") else None))
        return _DummyWork()
    def batch_isend_irecv(self, p2p_ops):
        self.ops.append(("batch", len(p2p_ops)))
        return _DummyWork()


class _IdentityModule:
    def __call__(self, x):
        return x


class _ScaleModule:
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, x):
        return x * self.factor


def _tensor(data):
    import candle
    import numpy as np
    return candle.tensor(np.array(data, dtype="float32"))


def test_pipeline_stage_imports():
    from candle.distributed.pipelining import PipelineStage, ScheduleGPipe
    assert PipelineStage is not None
    assert ScheduleGPipe is not None


def test_tensor_chunk_spec_splits_batch_dim():
    from candle.distributed.pipelining import TensorChunkSpec, split_microbatches

    x = _tensor([[1.0], [2.0], [3.0], [4.0]])
    chunks = split_microbatches((x,), 2, TensorChunkSpec(0))
    assert len(chunks) == 2
    assert chunks[0][0].shape[0] == 2
    assert chunks[1][0].shape[0] == 2


def test_pipeline_stage_first_stage_no_recv_last_stage_no_send():
    from candle.distributed.pipelining import PipelineStage

    stage = PipelineStage(
        submodule=_IdentityModule(),
        stage_index=0,
        num_stages=1,
        input_args=(_tensor([[1.0], [2.0]]),),
        output_args=(_tensor([[1.0], [2.0]]),),
        group=None,
    )
    assert stage.is_first is True
    assert stage.is_last is True


def test_pipeline_stage_middle_flags():
    from candle.distributed.pipelining import PipelineStage

    stage = PipelineStage(
        submodule=_IdentityModule(),
        stage_index=1,
        num_stages=3,
        input_args=(_tensor([[1.0], [2.0]]),),
        output_args=(_tensor([[1.0], [2.0]]),),
        group=None,
    )
    assert stage.is_first is False
    assert stage.is_last is False


def test_pipeline_stage_forward_one_chunk_runs_module():
    from candle.distributed.pipelining import PipelineStage

    stage = PipelineStage(
        submodule=_ScaleModule(2.0),
        stage_index=0,
        num_stages=1,
        input_args=(_tensor([[1.0], [2.0]]),),
        output_args=(_tensor([[2.0], [4.0]]),),
        group=None,
    )
    out = stage.forward_one_chunk(0, (_tensor([[3.0], [5.0]]),), {})
    assert out.shape == (2, 1)


def test_pipeline_stage_builds_p2p_ops_for_non_edge_stage(monkeypatch):
    from candle.distributed.pipelining import PipelineStage
    import candle.distributed as dist

    rec = _Recorder()
    monkeypatch.setattr(dist, "isend", rec.isend, raising=False)
    monkeypatch.setattr(dist, "irecv", rec.irecv, raising=False)

    stage = PipelineStage(
        submodule=_IdentityModule(),
        stage_index=1,
        num_stages=3,
        input_args=(_tensor([[1.0], [2.0]]),),
        output_args=(_tensor([[1.0], [2.0]]),),
        group=None,
    )
    recv_ops = stage.get_fwd_recv_ops(0)
    send_ops = stage.get_fwd_send_ops(0, (_tensor([[1.0], [2.0]]),))
    assert len(recv_ops) == 1
    assert len(send_ops) == 1
    assert recv_ops[0].peer == 0
    assert send_ops[0].peer == 2


def test_schedule_gpipe_step_splits_microbatches_and_runs_stage():
    from candle.distributed.pipelining import PipelineStage, ScheduleGPipe, TensorChunkSpec

    stage = PipelineStage(
        submodule=_ScaleModule(2.0),
        stage_index=0,
        num_stages=1,
        input_args=(_tensor([[1.0], [2.0]]),),
        output_args=(_tensor([[2.0], [4.0]]),),
        group=None,
    )
    sched = ScheduleGPipe(stage, n_microbatches=2, input_chunk_spec=TensorChunkSpec(0))
    x = _tensor([[1.0], [2.0], [3.0], [4.0]])
    out = sched.step(x)
    assert out.shape == (4, 1)


def test_schedule_gpipe_uses_batch_p2p(monkeypatch):
    from candle.distributed.pipelining import PipelineStage, ScheduleGPipe, TensorChunkSpec
    import candle.distributed as dist

    rec = _Recorder()
    monkeypatch.setattr(dist, "batch_isend_irecv", rec.batch_isend_irecv, raising=False)

    stage = PipelineStage(
        submodule=_IdentityModule(),
        stage_index=0,
        num_stages=2,
        input_args=(_tensor([[1.0], [2.0]]),),
        output_args=(_tensor([[1.0], [2.0]]),),
        group=None,
    )
    sched = ScheduleGPipe(stage, n_microbatches=2, input_chunk_spec=TensorChunkSpec(0))
    x = _tensor([[1.0], [2.0], [3.0], [4.0]])
    sched.step(x)
    assert any(op[0] == "batch" for op in rec.ops)


def test_schedule_gpipe_rejects_non_tensor_batch_input():
    from candle.distributed.pipelining import PipelineStage, ScheduleGPipe, TensorChunkSpec

    stage = PipelineStage(
        submodule=_IdentityModule(),
        stage_index=0,
        num_stages=1,
        input_args=(_tensor([[1.0]]),),
        output_args=(_tensor([[1.0]]),),
        group=None,
    )
    sched = ScheduleGPipe(stage, n_microbatches=2, input_chunk_spec=TensorChunkSpec(0))
    with pytest.raises(TypeError):
        sched.step([1.0, 2.0, 3.0, 4.0])
