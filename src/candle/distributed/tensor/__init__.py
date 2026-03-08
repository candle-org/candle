"""torch.distributed.tensor stub - not available in candle."""


class DTensor:
    pass


class Replicate:
    pass


class Shard:
    pass


class Placement:
    pass


def __getattr__(name):
    raise AttributeError(
        f"module 'torch.distributed.tensor' has no attribute '{name}'. "
        "DTensor is not available in candle."
    )
