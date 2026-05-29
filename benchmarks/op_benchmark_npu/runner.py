import time

from benchmarks.npu_perf_gates import summarize_samples


def benchmark_op(fn, warmup=10, iters=50, sync=None, setup=None, cleanup=None):
    """Run fn with warmup, then time iters iterations. Returns list of ms."""
    for _ in range(warmup):
        if setup:
            setup()
        fn()
        if sync:
            sync()
        if cleanup:
            cleanup()

    times = []
    for _ in range(iters):
        if setup:
            setup()
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        if cleanup:
            cleanup()
    return times


def summarize(samples):
    """Return shared distribution fields from a list of ms samples."""
    return summarize_samples(samples)
