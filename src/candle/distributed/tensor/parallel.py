"""Minimal Tensor Parallel entry points for candle.

Scope (Task 13):
  - parallelize_module() for nn.Linear only
  - ColwiseParallel: shard weight along dim 0 (output features)
  - RowwiseParallel: shard weight along dim 1 (input features)

Explicitly deferred:
  - SequenceParallel (class exists but raises NotImplementedError)
  - PrepareModuleInput / PrepareModuleOutput
  - 2D mesh TP
  - non-Linear modules
  - CUDA/NCCL
"""


class _ParallelStyle:
    """Base class for TP placement strategies."""


class ColwiseParallel(_ParallelStyle):
    """Partition Linear weight along dim 0 (output features / columns).

    This matches ``torch.distributed.tensor.parallel.ColwiseParallel``:
    weight shape is (out_features, in_features), sharding along dim 0
    partitions output features across ranks.

    Bias (if present) is sharded along dim 0 as well, matching output
    features on each rank.
    """


class RowwiseParallel(_ParallelStyle):
    """Partition Linear weight along dim 1 (input features / rows).

    This matches ``torch.distributed.tensor.parallel.RowwiseParallel``:
    weight shape is (out_features, in_features), sharding along dim 1
    partitions input features across ranks.

    Bias (if present) is replicated because every rank owns all output
    features and must add the same bias offset.
    """


class SequenceParallel(_ParallelStyle):
    """Sequence-dimension parallel -- deferred, not yet implemented."""


def parallelize_module(module, device_mesh, parallelize_plan):
    """Parallelize *module* according to *parallelize_plan* on *device_mesh*.

    Only ``nn.Linear`` is supported in this Task-13 slice.  Passing any
    other module type raises ``NotImplementedError``.

    ``SequenceParallel`` is also deferred and raises ``NotImplementedError``.

    Args:
        module:           an ``nn.Module`` to parallelize in-place.
        device_mesh:      a ``DeviceMesh`` describing the TP group.
        parallelize_plan: a ``ColwiseParallel`` or ``RowwiseParallel`` instance.

    Returns:
        The same *module* (mutated in-place), for chaining.
    """
    from candle.nn.modules.linear import Linear
    from .dtensor import DTensor
    from .placement import Shard, Replicate

    if isinstance(parallelize_plan, SequenceParallel):
        raise NotImplementedError(
            "SequenceParallel is not yet implemented in candle. "
            "Use ColwiseParallel or RowwiseParallel for nn.Linear."
        )

    if not isinstance(module, Linear):
        raise NotImplementedError(
            f"parallelize_module only supports nn.Linear in this release, "
            f"got {type(module).__name__}. "
            "SequenceParallel, PrepareModuleInput/Output, and non-Linear "
            "modules are deferred."
        )

    if isinstance(parallelize_plan, ColwiseParallel):
        # weight: (out_features, in_features) -> shard along dim 0
        module.weight = DTensor.from_local(
            module.weight, device_mesh, [Shard(0)]
        )
        if module.bias is not None:
            # bias: (out_features,) -> shard along dim 0
            module.bias = DTensor.from_local(
                module.bias, device_mesh, [Shard(0)]
            )
        return module

    if isinstance(parallelize_plan, RowwiseParallel):
        # weight: (out_features, in_features) -> shard along dim 1
        module.weight = DTensor.from_local(
            module.weight, device_mesh, [Shard(1)]
        )
        if module.bias is not None:
            # bias is replicated: every rank holds all out_features
            module.bias = DTensor.from_local(
                module.bias, device_mesh, [Replicate()]
            )
        return module

    raise TypeError(
        f"Unknown parallelize_plan type: {type(parallelize_plan).__name__}. "
        "Expected ColwiseParallel or RowwiseParallel."
    )
