"""torch.distributed.fsdp.wrap compatibility module.

Provides the module-level namespace so that ``import candle.distributed.fsdp.wrap``
succeeds.  ``transformer_auto_wrap_policy`` is not yet implemented; all other
attributes raise AttributeError to match PyTorch's behaviour for unknown names.
"""


def transformer_auto_wrap_policy(*args, **kwargs):
    raise NotImplementedError(
        "transformer_auto_wrap_policy is not yet implemented in candle."
    )


def __getattr__(name):
    raise AttributeError(
        f"module 'candle.distributed.fsdp.wrap' has no attribute '{name}'."
    )
