import time


def benchmark_op(fn, warmup=10, iters=50, sync=None):
    """Run fn with warmup, then time iters iterations. Returns list of ms."""
    for _ in range(warmup):
        fn()
        if sync:
            sync()

    times = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def summarize(samples):
    """Return (mean_ms, median_ms, p95_ms) from a list of ms samples."""
    if not samples:
        return 0.0, 0.0, 0.0
    values = sorted(samples)
    mean = sum(values) / len(values)
    median = values[len(values) // 2]
    p95 = values[int(len(values) * 0.95) - 1] if len(values) >= 2 else values[-1]
    return mean, median, p95
