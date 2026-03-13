import random


class Sampler:
    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.generator = generator

    def __iter__(self):
        n = len(self.data_source)
        rng = self.generator if self.generator is not None else random
        if self.replacement:
            count = self.num_samples if self.num_samples is not None else n
            return iter([rng.randrange(0, n) for _ in range(count)])
        idx = list(range(n))
        rng.shuffle(idx)
        return iter(idx)

    def __len__(self):
        if self.replacement and self.num_samples is not None:
            return self.num_samples
        return len(self.data_source)


class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        if batch_size <= 0:
            raise ValueError("batch_size should be a positive integer")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = len(self.sampler)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


class SubsetRandomSampler(Sampler):
    def __init__(self, indices, generator=None):
        self.indices = list(indices)
        self.generator = generator

    def __iter__(self):
        rng = self.generator if self.generator is not None else random
        idx = list(self.indices)
        rng.shuffle(idx)
        return iter(idx)

    def __len__(self):
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    def __init__(self, weights, num_samples, replacement=True, generator=None):
        if not isinstance(num_samples, int) or num_samples < 0:
            raise ValueError(
                f"num_samples should be a non-negative integer, got {num_samples}"
            )
        self.weights = list(weights)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self):
        rng = self.generator if self.generator is not None else random
        if self.replacement:
            yield from rng.choices(
                range(len(self.weights)),
                weights=self.weights,
                k=self.num_samples,
            )
        else:
            # Weighted sampling without replacement
            w = list(self.weights)
            indices = list(range(len(w)))
            for _ in range(self.num_samples):
                chosen = rng.choices(
                    indices, weights=[w[i] for i in indices], k=1
                )[0]
                yield chosen
                indices.remove(chosen)

    def __len__(self):
        return self.num_samples
