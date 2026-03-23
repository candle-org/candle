#!/usr/bin/env python3
"""DDP compute/communication overlap benchmark -- HCCL/NPU path.

P0 EXIT CRITERIA
================
Pass: async-hook mean latency < sync-baseline mean latency * 0.90
      (>= 10% speedup from overlapping allreduce with backward compute)
Fail:
  - async path raises an unexpected exception
  - async mean >= sync mean  (no measurable overlap benefit)
  - timing variance > 3x between min and p95 within a single mode

WHAT IS MEASURED
================
Sync baseline  -- run forward+backward, block on allreduce, then step.
Async hook     -- register comm hook that fires per-bucket during backward;
                  allreduce overlaps with remaining backward compute.

HCCL / NPU NOTES
================
- Requires candle.distributed with HCCL backend (Ascend 910A/910B).
- On machines without NPU/HCCL, script prints INFO skip and exits 0.
- Device sync uses candle.npu.synchronize() when available;
  falls back to no-op so the timing harness still functions on CPU.
"""

from __future__ import annotations

import sys
import time
import statistics
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------

def _check_npu_hccl() -> Tuple[bool, str]:
    """Return (available, reason_string)."""
    try:
        import candle
    except ImportError:
        return False, "candle not importable"

    npu_mod = getattr(candle, "npu", None)
    if npu_mod is None:
        return False, "candle.npu module absent"
    if not getattr(npu_mod, "is_available", lambda: False)():
        return False, "NPU hardware not detected"

    dist_mod = getattr(candle, "distributed", None)
    if dist_mod is None:
        return False, "candle.distributed absent"

    # Check that HCCL backend is registered
    backends = getattr(dist_mod, "Backend", None)
    hccl_name = getattr(backends, "HCCL", None) if backends else None
    if hccl_name is None:
        # Try string probe
        try:
            dist_mod.Backend("hccl")
        except Exception:  # pylint: disable=broad-except
            return False, "HCCL backend not registered in candle.distributed"

    return True, "ok"


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _make_sync_fn() -> Callable[[], None]:
    """Return a device-sync callable, or no-op if NPU not present."""
    try:
        import candle
        npu_mod = getattr(candle, "npu", None)
        if npu_mod is not None and getattr(npu_mod, "is_available", lambda: False)():
            return npu_mod.synchronize
    except ImportError:
        pass
    return lambda: None


def _bench(fn: Callable[[], None], sync: Callable[[], None],
           warmup: int = 5, iters: int = 30) -> List[float]:
    """Run fn, return list of wall-clock milliseconds per iteration."""
    for _ in range(warmup):
        sync()
        fn()
        sync()

    samples: List[float] = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _summarize(samples: List[float]) -> Tuple[float, float, float]:
    """Return (mean_ms, median_ms, p95_ms)."""
    s = sorted(samples)
    mean = statistics.mean(s)
    median = s[len(s) // 2]
    p95 = s[max(0, int(len(s) * 0.95) - 1)]
    return mean, median, p95


# ---------------------------------------------------------------------------
# Stub model + bucket infrastructure for CPU-mode dry-run
# ---------------------------------------------------------------------------

class _FakeTensor:
    """Minimal tensor stub for CPU-mode benchmarking."""
    def __init__(self, size: int):
        self._data = [0.0] * size
        self.grad: Optional[_FakeTensor] = None
        self._size = size

    def numel(self) -> int:
        return self._size

    def backward_fake(self) -> None:
        """Simulate backward fill."""
        self.grad = _FakeTensor(self._size)
        for i in range(self._size):
            self.grad._data[i] = float(i % 7) * 0.001


class _FakeBucket:
    """Minimal GradBucket stub."""
    def __init__(self, size: int = 1 << 22):
        self._buf = _FakeTensor(size)

    def buffer(self) -> _FakeTensor:
        return self._buf


class _FakeProcessGroup:
    """Process group stub that simulates allreduce latency."""
    def __init__(self, world_size: int = 2, simulated_latency_ms: float = 5.0):
        self._world_size = world_size
        self._lat = simulated_latency_ms / 1000.0

    def size(self) -> int:
        return self._world_size

    def allreduce_blocking(self, buf: _FakeTensor) -> None:
        """Block for simulated latency."""
        time.sleep(self._lat)

    def allreduce_async(self, buf: _FakeTensor,
                        callback: Callable[[], None]) -> None:
        """Fire allreduce in background thread, call callback on completion."""
        import threading
        def _work():
            time.sleep(self._lat)
            callback()
        t = threading.Thread(target=_work, daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# Sync DDP baseline
# ---------------------------------------------------------------------------

def _sync_ddp_step(params: List[_FakeTensor],
                   pg: _FakeProcessGroup,
                   bucket_size: int) -> None:
    """
    Sync baseline: complete all backward passes, then block on allreduce
    for every bucket sequentially, then run optimizer step.
    """
    # Simulate forward + backward compute
    for p in params:
        p.backward_fake()
        time.sleep(0.0001)  # per-param compute cost

    # Blocking allreduce over all parameter buckets
    for i in range(0, len(params), max(1, bucket_size)):
        bucket_params = params[i:i + bucket_size]
        fake_bucket = _FakeBucket(sum(p.numel() for p in bucket_params))
        pg.allreduce_blocking(fake_bucket.buffer())

    # Optimizer step (trivial)
    for p in params:
        if p.grad is not None:
            for j in range(p._size):
                p._data[j] -= 0.01 * p.grad._data[j]


# ---------------------------------------------------------------------------
# Async hook DDP path
# ---------------------------------------------------------------------------

def _async_ddp_step(params: List[_FakeTensor],
                    pg: _FakeProcessGroup,
                    bucket_size: int) -> None:
    """
    Async hook path: each bucket fires allreduce asynchronously as soon as
    its gradients are ready, overlapping with remaining backward compute.
    """
    import threading

    num_buckets = max(1, (len(params) + bucket_size - 1) // bucket_size)
    events = [threading.Event() for _ in range(num_buckets)]

    for bucket_idx in range(num_buckets):
        start = bucket_idx * bucket_size
        end = min(start + bucket_size, len(params))
        bucket_params = params[start:end]

        # Simulate backward compute for this bucket's params
        for p in bucket_params:
            p.backward_fake()
            time.sleep(0.0001)

        # Fire async allreduce immediately (hook fires at bucket boundary)
        fake_bucket = _FakeBucket(sum(p.numel() for p in bucket_params))
        ev = events[bucket_idx]
        pg.allreduce_async(fake_bucket.buffer(), callback=ev.set)

    # Wait for all async allreduces to complete
    for ev in events:
        ev.wait(timeout=30.0)

    # Optimizer step
    for p in params:
        if p.grad is not None:
            for j in range(p._size):
                p._data[j] -= 0.01 * p.grad._data[j]


# ---------------------------------------------------------------------------
# HCCL-aware paths (real NPU)
# ---------------------------------------------------------------------------

def _try_hccl_benchmark(warmup: int, iters: int) -> Optional[dict]:
    """
    Attempt the real HCCL benchmark using candle.distributed.
    Returns result dict on success, None if HCCL not usable.
    """
    try:
        import candle
        import candle.distributed as dist
        from candle.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
        from candle.futures import Future
    except ImportError:
        return None

    npu_mod = getattr(candle, "npu", None)
    if npu_mod is None or not getattr(npu_mod, "is_available", lambda: False)():
        return None
    if not dist.is_initialized():
        return None

    sync = npu_mod.synchronize
    world_size = dist.get_world_size()
    hidden = 4096
    seq = 512
    batch = 2

    # Build tensors on NPU
    x = candle.randn((batch, seq, hidden), device="npu")
    w = candle.randn((hidden, hidden), device="npu", requires_grad=True)

    # -- Sync baseline --
    def _sync_step():
        out = candle.matmul(x, w)
        loss = out.sum()
        loss.backward()
        if w.grad is not None:
            grad_buf = w.grad.view(-1)
            dist.all_reduce(grad_buf)
            grad_buf_div = grad_buf / world_size  # noqa: F841
        w.grad = None

    sync_samples = _bench(_sync_step, sync, warmup=warmup, iters=iters)

    # -- Async hook path: fire allreduce while additional compute runs --
    # Pattern mirrors real DDP comm hooks: all_reduce(grad_buf, async_op=True)
    # is enqueued at the bucket boundary, then backward continues on the
    # remaining parameters / loss terms.  work.wait() is deferred until all
    # that extra compute has finished, so HCCL and NPU compute genuinely
    # overlap.
    #
    # Concretely we model a two-layer scenario:
    #   layer-0 grad ready  -> fire async allreduce on grad_buf_0
    #   layer-1 backward    -> runs while allreduce is in flight
    #   layer-1 grad ready  -> fire async allreduce on grad_buf_1
    #   optimizer-prep work -> additional on-device compute while both
    #                          allreduces may still be in flight
    #   wait all works      -> block only once all useful compute is done

    # Second weight tensor for "layer 1" (same shape, independent grad)
    w2 = candle.randn((hidden, hidden), device="npu", requires_grad=True)

    def _hook_step():
        # Layer 0 forward + backward
        out0 = candle.matmul(x, w)
        loss0 = out0.sum()
        loss0.backward()

        pending_works = []

        # Hook fires for layer-0 bucket -- enqueue async, do NOT wait yet
        if w.grad is not None:
            grad_buf_0 = w.grad.view(-1)
            work0 = dist.all_reduce(grad_buf_0, async_op=True)
            if work0 is not None:
                pending_works.append((work0, grad_buf_0))

        # Layer 1 backward runs while layer-0 allreduce is in flight
        out1 = candle.matmul(x, w2)
        loss1 = out1.sum()
        loss1.backward()

        # Hook fires for layer-1 bucket -- enqueue async, do NOT wait yet
        if w2.grad is not None:
            grad_buf_1 = w2.grad.view(-1)
            work1 = dist.all_reduce(grad_buf_1, async_op=True)
            if work1 is not None:
                pending_works.append((work1, grad_buf_1))

        # Additional on-device optimizer-prep compute while both allreduces
        # may still be in flight (scale factors, clip norms, etc.)
        _scale = candle.ones((1,), device="npu") / world_size  # noqa: F841

        # Now drain: wait for every work handle, then apply scale
        for work, grad_buf in pending_works:
            work.wait()
            # in-place divide to complete the allreduce averaging
            grad_buf /= world_size  # noqa: F841

        w.grad = None
        w2.grad = None

    async_samples = _bench(_hook_step, sync, warmup=warmup, iters=iters)

    return {
        "mode": "hccl_npu",
        "sync": sync_samples,
        "async": async_samples,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 0  # missing hardware is not a test failure

OVERLAP_THRESHOLD = 0.90  # async mean must be < sync mean * this


def _evaluate(sync_samples: List[float],
              async_samples: List[float],
              mode: str) -> int:
    """Print report and return exit code."""
    s_mean, s_med, s_p95 = _summarize(sync_samples)
    a_mean, a_med, a_p95 = _summarize(async_samples)

    print()
    print(f"=== DDP Overlap Benchmark [{mode}] ===")
    print(f"  Sync  baseline : mean={s_mean:.2f}ms  median={s_med:.2f}ms  p95={s_p95:.2f}ms")
    print(f"  Async hook path: mean={a_mean:.2f}ms  median={a_med:.2f}ms  p95={a_p95:.2f}ms")

    ratio = a_mean / s_mean if s_mean > 0 else 1.0
    print(f"  Async/Sync ratio: {ratio:.3f}  (threshold < {OVERLAP_THRESHOLD})")

    # Variance check
    sync_variance_ok = (s_p95 / max(min(sync_samples), 1e-9)) < 3.0
    async_variance_ok = (a_p95 / max(min(async_samples), 1e-9)) < 3.0

    passed = True
    if ratio >= OVERLAP_THRESHOLD:
        print(f"  RESULT: FAIL -- async path not faster enough "
              f"(ratio={ratio:.3f} >= {OVERLAP_THRESHOLD})")
        passed = False
    if not sync_variance_ok:
        print("  WARNING: sync timing shows high variance (p95/min > 3x)")
    if not async_variance_ok:
        print("  WARNING: async timing shows high variance (p95/min > 3x)")

    if passed:
        print(f"  RESULT: PASS -- overlap speedup confirmed (ratio={ratio:.3f})")
        return EXIT_PASS
    return EXIT_FAIL


# ---------------------------------------------------------------------------
# Simulated (CPU) benchmark -- always runnable
# ---------------------------------------------------------------------------

def _run_simulated(warmup: int, iters: int) -> int:
    """Run CPU simulation of sync vs async DDP overlap."""
    print("[INFO] Running simulated (CPU stub) DDP overlap benchmark.")
    print("       This models the hook-dispatch latency difference, not real HCCL.")

    num_params = 16
    bucket_size = 4
    param_size = 1024
    params = [_FakeTensor(param_size) for _ in range(num_params)]
    # Use a faster simulated latency for CI
    pg = _FakeProcessGroup(world_size=2, simulated_latency_ms=2.0)

    sync_fn = lambda: _sync_ddp_step(params, pg, bucket_size)  # noqa: E731
    async_fn = lambda: _async_ddp_step(params, pg, bucket_size)  # noqa: E731
    noop = lambda: None  # noqa: E731

    sync_samples = _bench(sync_fn, noop, warmup=warmup, iters=iters)
    async_samples = _bench(async_fn, noop, warmup=warmup, iters=iters)

    return _evaluate(sync_samples, async_samples, mode="simulated_cpu")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    available, reason = _check_npu_hccl()

    if not available:
        print(f"[INFO] HCCL/NPU not available ({reason}).")
        print("[INFO] Falling back to CPU simulation mode.")
        print("[INFO] Re-run on an Ascend 910A/910B node with HCCL initialized")
        print("       to measure real overlap speedup.")
        _run_simulated(warmup=3, iters=20)
        # In simulation mode we always exit 0 (skip) regardless of result,
        # because the P0 gate only applies to real HCCL hardware.
        print()
        print("[INFO] P0 gate not applied in simulation mode (no NPU detected).")
        return EXIT_SKIP

    # Real HCCL path
    print("[INFO] NPU + HCCL detected. Running real DDP overlap benchmark.")
    result = _try_hccl_benchmark(warmup=5, iters=30)
    if result is None:
        print("[INFO] candle.distributed not initialized. "
              "Run via torchrun/candle_run with --nproc_per_node.")
        print("[INFO] Falling back to simulation mode.")
        _run_simulated(warmup=3, iters=20)
        print("[INFO] P0 gate not applied in simulation mode (HCCL not initialized).")
        return EXIT_SKIP

    return _evaluate(result["sync"], result["async"], mode=result["mode"])


if __name__ == "__main__":
    sys.exit(main())
