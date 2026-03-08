"""torch.distributed.tensor.parallel stub - not available in candle."""


class SequenceParallel:
    pass


def parallelize_module(*args, **kwargs):
    raise NotImplementedError("parallelize_module is not available in candle.")


def __getattr__(name):
    raise AttributeError(
        f"module 'torch.distributed.tensor.parallel' has no attribute '{name}'. "
        "Tensor parallel is not available in candle."
    )
