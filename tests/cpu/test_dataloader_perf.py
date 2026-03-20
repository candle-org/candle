"""Integration tests for DataLoader performance optimizations."""
import numpy as np

import candle as torch
from candle.utils.data import DataLoader, Dataset
from candle.utils.data._utils import default_collate


class TensorRangeDataset(Dataset):
    def __init__(self, n, dim=4):
        self.n = n
        self.dim = dim

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.tensor([float(idx)] * self.dim, dtype=torch.float32)


def test_dataloader_multiprocess_ordered_tensor_batches():
    """Verify multiprocess DataLoader produces correctly ordered tensor batches."""
    ds = TensorRangeDataset(16, dim=2)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
    batches = list(loader)
    assert len(batches) == 4
    for i, batch in enumerate(batches):
        expected = [[float(i * 4 + j)] * 2 for j in range(4)]
        np.testing.assert_array_almost_equal(batch.numpy(), expected)


def test_dataloader_large_batch_shm():
    """Larger batches still work through multiprocessing path."""
    ds = TensorRangeDataset(100, dim=64)
    loader = DataLoader(ds, batch_size=10, shuffle=False, num_workers=4)
    batches = list(loader)
    assert len(batches) == 10
    all_vals = set()
    for batch in batches:
        for row in batch.numpy():
            all_vals.add(row[0])
    assert all_vals == {float(i) for i in range(100)}


def test_default_collate_tensor_fast_path():
    """default_collate uses fast path for all-tensor batches."""
    batch = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    result = default_collate(batch)
    assert result.shape == (2, 2)
    np.testing.assert_array_almost_equal(
        result.numpy(), [[1.0, 2.0], [3.0, 4.0]]
    )
