import candle as torch
from candle.utils.data import Dataset, DataLoader


def test_dataloader_pin_memory_calls_tensor_pin_memory():
    class SampleDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return torch.tensor([idx], device="cpu")

    loader = DataLoader(SampleDataset(), batch_size=2, pin_memory=True, num_workers=0)
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0].is_pinned() is True
