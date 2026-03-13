import math
import random as _random


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class IterableDataset:
    def __iter__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ChainDataset([self, other])


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        if len(tensors) == 0:
            raise ValueError("TensorDataset requires at least one tensor")
        n = tensors[0].shape[0]
        for tensor in tensors[1:]:
            if tensor.shape[0] != n:
                raise ValueError("Size mismatch between tensors")
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = list(datasets)
        if not self.datasets:
            raise ValueError("datasets should not be an empty iterable")
        self.cumulative_sizes = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")
        dataset_idx = 0
        while idx >= self.cumulative_sizes[dataset_idx]:
            dataset_idx += 1
        prev = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        sample_idx = idx - prev
        return self.datasets[dataset_idx][sample_idx]


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class ChainDataset(IterableDataset):
    def __init__(self, datasets):
        self.datasets = list(datasets)
        for ds in self.datasets:
            if not isinstance(ds, IterableDataset):
                raise TypeError(
                    "ChainDataset only supports IterableDataset, "
                    f"but got {type(ds).__name__}"
                )

    def __iter__(self):
        for ds in self.datasets:
            yield from ds

    def __len__(self):
        total = 0
        for ds in self.datasets:
            total += len(ds)
        return total


class StackDataset(Dataset):
    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError(
                "StackDataset accepts either positional or keyword "
                "arguments, not both"
            )
        if not args and not kwargs:
            raise ValueError("StackDataset requires at least one dataset")
        if args:
            self._datasets = list(args)
            self._keys = None
        else:
            self._datasets = list(kwargs.values())
            self._keys = list(kwargs.keys())
        length = len(self._datasets[0])
        for ds in self._datasets[1:]:
            if len(ds) != length:
                raise ValueError("All datasets must have the same length")

    def __getitem__(self, index):
        items = [ds[index] for ds in self._datasets]
        if self._keys is not None:
            return dict(zip(self._keys, items))
        return tuple(items)

    def __getitems__(self, indices):
        samples = []
        for idx in indices:
            samples.append(self[idx])
        return samples

    def __len__(self):
        return len(self._datasets[0])


def random_split(dataset, lengths, generator=None):
    """Randomly split a dataset into non-overlapping new datasets."""
    n = len(dataset)
    if (math.isclose(sum(lengths), 1.0)
            and all(isinstance(x, float) for x in lengths)):
        subset_lengths = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(
                    f"Fraction at index {i} is not between 0 and 1"
                )
            n_items = int(math.floor(n * frac))
            subset_lengths.append(n_items)
        remainder = n - sum(subset_lengths)
        indices_order = sorted(
            range(len(lengths)), key=lambda i: lengths[i], reverse=True
        )
        for i in range(remainder):
            subset_lengths[indices_order[i]] += 1
        lengths = subset_lengths
    if sum(lengths) != n:
        raise ValueError(
            "Sum of input lengths does not equal the length of the "
            f"input dataset (expected {n}, got {sum(lengths)})"
        )
    rng = generator if generator is not None else _random
    indices = list(range(n))
    rng.shuffle(indices)
    subsets = []
    offset = 0
    for length in lengths:
        subsets.append(Subset(dataset, indices[offset:offset + length]))
        offset += length
    return subsets
