import candle as torch


def mask_not_all_zeros(shape, *, dtype=torch.float32):
    return torch.ones(shape, dtype=dtype)
