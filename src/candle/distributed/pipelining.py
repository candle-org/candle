"""Minimal manual-frontend pipeline parallel runtime for candle.

Scope (Task 15):
- manual frontend only
- PipelineStage
- ScheduleGPipe
- TensorChunkSpec + split_microbatches helper

Explicitly deferred:
- tracing frontend / FX / IR
- 1F1B / interleaved schedules
- TP+PP / FSDP+PP composition
- CUDA/NCCL
"""

from dataclasses import dataclass

import candle.distributed as dist


@dataclass(frozen=True)
class TensorChunkSpec:
    split_dim: int = 0


def _normalize_args(args):
    if not isinstance(args, tuple):
        return (args,)
    return args


def split_microbatches(args, n_microbatches, chunk_spec=None):
    """Split tuple of tensor args into microbatches along dim 0.

    Minimal implementation: every arg must be a candle Tensor-like object with
    `.shape` and slicing support; all are chunked evenly along `split_dim`.
    """
    args = _normalize_args(args)
    if chunk_spec is None:
        chunk_spec = TensorChunkSpec(0)
    if not isinstance(chunk_spec, TensorChunkSpec):
        raise TypeError("chunk_spec must be a TensorChunkSpec")

    split_dim = chunk_spec.split_dim
    if not args:
        return []

    first = args[0]
    if not hasattr(first, "shape"):
        raise TypeError("split_microbatches expects tensor inputs")

    total = first.shape[split_dim]
    if total % n_microbatches != 0:
        raise ValueError(
            f"batch dim size {total} must be divisible by n_microbatches={n_microbatches}"
        )
    chunk = total // n_microbatches

    mbs = []
    for mb_idx in range(n_microbatches):
        mb_args = []
        start = mb_idx * chunk
        end = start + chunk
        for arg in args:
            if not hasattr(arg, "shape"):
                raise TypeError("split_microbatches expects tensor inputs")
            slices = [slice(None)] * len(arg.shape)
            slices[split_dim] = slice(start, end)
            mb_args.append(arg[tuple(slices)])
        mbs.append(tuple(mb_args))
    return mbs


class PipelineStage:
    """Minimal manual pipeline stage.

    One stage per rank, 1D pipeline only.
    """

    def __init__(
        self,
        submodule,
        stage_index,
        num_stages,
        input_args,
        output_args=None,
        group=None,
    ):
        self.submodule = submodule
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.group = group
        self.group_rank = dist.get_rank(group) if group is not None else stage_index
        self.group_size = dist.get_world_size(group) if group is not None else num_stages
        self.is_first = stage_index == 0
        self.is_last = stage_index == (num_stages - 1)
        self.inputs = _normalize_args(input_args)
        self.outputs = _normalize_args(output_args) if output_args is not None else None
        self.prev_stage = None if self.is_first else stage_index - 1
        self.next_stage = None if self.is_last else stage_index + 1

    def get_fwd_recv_ops(self, chunk_id):
        if self.is_first:
            return []
        recv_bufs = []
        for inp in self.inputs:
            recv_bufs.append(
                dist.P2POp(dist.irecv, inp, self.prev_stage, self.group)
            )
        return recv_bufs

    def get_fwd_send_ops(self, chunk_id, outputs):
        if self.is_last:
            return []
        outputs = _normalize_args(outputs)
        send_ops = []
        for out in outputs:
            send_ops.append(
                dist.P2POp(dist.isend, out, self.next_stage, self.group)
            )
        return send_ops

    def forward_one_chunk(self, chunk_id, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = _normalize_args(args)
        return self.submodule(*args, **kwargs)


class ScheduleGPipe:
    """Minimal GPipe schedule: fill-drain forward-only manual frontend.

    This first slice focuses on forward microbatch orchestration and P2P send/
    recv behavior. Backward scheduling is intentionally deferred.
    """

    def __init__(self, stage, n_microbatches, input_chunk_spec=None):
        self.stage = stage
        self.n_microbatches = int(n_microbatches)
        self.input_chunk_spec = input_chunk_spec or TensorChunkSpec(0)

    def step(self, *args, **kwargs):
        args = _normalize_args(args)
        if not args or not hasattr(args[0], "shape"):
            raise TypeError("ScheduleGPipe.step expects tensor batch inputs")

        arg_mbs = split_microbatches(args, self.n_microbatches, self.input_chunk_spec)
        out_chunks = []
        send_works = []

        for chunk_id, mb_args in enumerate(arg_mbs):
            recv_ops = self.stage.get_fwd_recv_ops(chunk_id)
            if recv_ops:
                recv_work = dist.batch_isend_irecv(recv_ops)
                recv_work.wait()
            out = self.stage.forward_one_chunk(chunk_id, mb_args, kwargs)
            out_chunks.append(out)
            send_ops = self.stage.get_fwd_send_ops(chunk_id, out)
            if send_ops:
                send_works.append(dist.batch_isend_irecv(send_ops))

        for work in send_works:
            work.wait()

        if len(out_chunks) == 1:
            return out_chunks[0]

        # concatenate outputs along the split dim (dim 0 only in this slice)
        import candle
        return candle.cat(list(out_chunks), dim=self.input_chunk_spec.split_dim)
