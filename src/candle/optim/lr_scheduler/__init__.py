"""Learning rate schedulers for candle optimizers.

This module provides learning rate scheduling utilities similar to torch.optim.lr_scheduler.
"""

from ._lr_scheduler import (
    LRScheduler,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    LinearLR,
    ConstantLR,
    ReduceLROnPlateau,
    LambdaLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    SequentialLR,
    ChainedScheduler,
    PolynomialLR,
    MultiplicativeLR,
    CyclicLR,
)

__all__ = [
    "LRScheduler",
    "StepLR",
    "MultiStepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "LinearLR",
    "ConstantLR",
    "ReduceLROnPlateau",
    "LambdaLR",
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "SequentialLR",
    "ChainedScheduler",
    "PolynomialLR",
    "MultiplicativeLR",
    "CyclicLR",
]
