"""
Benchmark: per-op latency of binary add on NPU (64x64 float32)
Compares torch_npu (if available) vs candle.

Two modes:
  sync:    synchronize before and after each op (measures single-op wall time including NPU execution)
  no-sync: submit N ops without sync (measures submit/dispatch overhead only)
"""

import time
import statistics

SHAPE = (64, 64)
WARMUP = 100
ITERS = 1000
CHAIN_N = 100  # ops chained without sync for no-sync mode


def _report(label, times_us):
    times_us = sorted(times_us)
    n = len(times_us)
    print(f"  {label:<50} "
          f"median={statistics.median(times_us):7.1f}us  "
          f"p10={times_us[int(n*0.10)]:7.1f}us  "
          f"p90={times_us[int(n*0.90)]:7.1f}us  "
          f"min={times_us[0]:7.1f}us")


def _header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  {'mode':<50} {'median':>9}   {'p10':>9}   {'p90':>9}   {'min':>9}")
    print(f"  {'-'*68}")


# ---------- torch_npu benchmark ----------
def bench_torch_npu():
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError as e:
        print(f"[torch_npu] Skipped: {e}")
        return

    device = torch.device("npu:0")
    a = torch.randn(SHAPE, dtype=torch.float32, device=device)
    b = torch.randn(SHAPE, dtype=torch.float32, device=device)

    for _ in range(WARMUP):
        _ = torch.add(a, b)
    torch.npu.synchronize()

    # --- sync mode: sync before+after each op ---
    times_sync = []
    for _ in range(ITERS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _ = torch.add(a, b)
        torch.npu.synchronize()
        times_sync.append((time.perf_counter() - t0) * 1e6)

    # --- no-sync mode: submit CHAIN_N ops, measure per-op submit cost ---
    # Subtract the sync cost from the total
    times_nosync = []
    for _ in range(ITERS // CHAIN_N):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(CHAIN_N):
            _ = torch.add(a, b)
        t1 = time.perf_counter()
        torch.npu.synchronize()
        times_nosync.append((t1 - t0) * 1e6 / CHAIN_N)

    _header("torch_npu  torch.add((64,64) float32)")
    _report("with sync (add + NPU execute)", times_sync)
    _report(f"no-sync  (submit only, {CHAIN_N} chained)", times_nosync)


# ---------- candle benchmark ----------
def bench_candle():
    try:
        import candle as torch
    except ImportError as e:
        print(f"[candle] Skipped: {e}")
        return

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("[candle] NPU not available, skipping")
        return

    device = torch.device("npu:0")
    a = torch.randn(SHAPE, dtype=torch.float32, device=device)
    b = torch.randn(SHAPE, dtype=torch.float32, device=device)

    for _ in range(WARMUP):
        _ = torch.add(a, b)
    torch.npu.synchronize()

    # --- sync mode ---
    times_sync = []
    for _ in range(ITERS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _ = torch.add(a, b)
        torch.npu.synchronize()
        times_sync.append((time.perf_counter() - t0) * 1e6)

    # --- no-sync mode ---
    times_nosync = []
    for _ in range(ITERS // CHAIN_N):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(CHAIN_N):
            _ = torch.add(a, b)
        t1 = time.perf_counter()
        torch.npu.synchronize()
        times_nosync.append((t1 - t0) * 1e6 / CHAIN_N)

    _header("candle     torch.add((64,64) float32)")
    _report("with sync (add + NPU execute)", times_sync)
    _report(f"no-sync  (submit only, {CHAIN_N} chained)", times_nosync)


if __name__ == "__main__":
    print(f"Shape: {SHAPE}, Warmup: {WARMUP}, Iterations: {ITERS}, Chain: {CHAIN_N}")
    bench_torch_npu()
    bench_candle()
    print("\nDone.")
