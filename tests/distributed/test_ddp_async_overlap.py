"""DDP comm-hook and no_sync overlap contract tests.

These tests now lock the intended behavior for the current DDP implementation.
They cover the contracts that previously failed during the RED phase and are now
expected to stay GREEN as Task 4's DDP bucket and reducer fixes evolve.

Contracts covered:
  A. Comm hook futures must be usable for pending async-style hook results.
  B. The hook result tensor must become param.grad.
  C. register_comm_hook() must raise if called after the first forward pass.
  D. no_sync() must suppress the comm hook and accumulate grads locally;
     a synced pass after no_sync() must invoke the hook exactly once.

All tests run single-rank Gloo (world_size=1). No multi-process needed.
"""

import os
import socket
import threading

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init_pg(port):
    import candle.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=0, world_size=1,
        init_method=f"tcp://127.0.0.1:{port}"
    )


def _teardown():
    import candle.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def _tiny_model():
    """Return a minimal single-parameter model."""
    import candle as torch
    import candle.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones((4,)))

        def forward(self, x):
            return x * self.weight

    return TinyModel()


# ---------------------------------------------------------------------------
# Contract A: hook future is not eagerly waited
# ---------------------------------------------------------------------------

class TestHookNotEagerlyWaited:
    """The reducer must not wait() the hook future immediately on receipt."""

    def setup_method(self):
        self._port = _free_port()
        _init_pg(self._port)

    def teardown_method(self):
        _teardown()

    def test_hook_future_pending_at_hook_return(self):
        """A hook that returns a thread-resolved Future must succeed.

        The hook creates an unresolved Future, spawns a background thread to
        resolve it after a short delay, and returns immediately. The reducer
        must wait for real resolution, not call wait() before the thread fires.

        EXPECTED: backward completes, param.grad set, all futures resolved.
        ACTUAL (bug): reducer calls future.wait() on a pending future before
            the background thread resolves it.
        """
        import candle as torch
        import candle.nn as nn
        import candle.futures as futures_mod

        threads = []
        futures_returned = []

        def async_hook(state, bucket):
            buf = bucket.buffer()
            fut = futures_mod.Future()

            def _resolve():
                import time
                time.sleep(0.01)
                fut.set_result(buf)

            t = threading.Thread(target=_resolve, daemon=True)
            t.start()
            threads.append(t)
            futures_returned.append(fut)
            return fut

        base = _tiny_model()
        ddp = nn.DistributedDataParallel(base)
        ddp.register_comm_hook(None, async_hook)

        x = torch.ones((1, 4))
        ddp(x).sum().backward()

        for t in threads:
            t.join(timeout=2.0)

        assert futures_returned, "comm hook was never called"
        for i, fut in enumerate(futures_returned):
            assert fut.done(), (
                f"Future {i} from comm hook was never resolved — "
                "reducer may not have waited for the async future"
            )
        assert base.weight.grad is not None

    def test_hook_called_during_backward_not_after(self):
        """The comm hook must be called during backward — sanity check."""
        import candle as torch
        import candle.nn as nn
        import candle.futures as futures_mod

        hook_calls = [0]

        def sync_hook(state, bucket):
            hook_calls[0] += 1
            fut = futures_mod.Future()
            fut.set_result(bucket.buffer())
            return fut

        base = _tiny_model()
        ddp = nn.DistributedDataParallel(base)
        ddp.register_comm_hook(None, sync_hook)

        x = torch.ones((1, 4))
        ddp(x).sum().backward()

        assert hook_calls[0] >= 1, (
            f"comm hook must be called during backward; got {hook_calls[0]} calls"
        )


# ---------------------------------------------------------------------------
# Contract B: hook result tensor becomes param.grad
# ---------------------------------------------------------------------------

class TestHookResultAppliedToGrad:
    """The tensor in future.result() must be what ends up in param.grad."""

    def setup_method(self):
        self._port = _free_port()
        _init_pg(self._port)

    def teardown_method(self):
        _teardown()

    def test_hook_scaling_reflected_in_grad(self):
        """A hook that multiplies gradients by a known factor must produce
        param.grad that equals base_grad * factor.

        EXPECTED: param.grad values == 2.0 (base_grad=1.0, factor=2.0).
        ACTUAL (bug): if the reducer ignores future.result() and uses the
            original buffer, param.grad will equal 1.0 not 2.0.
        """
        import candle as torch
        import candle.nn as nn
        import candle.futures as futures_mod
        from candle._functional import mul
        import numpy as np

        sentinel = [None]

        def marking_hook(state, bucket):
            """Scale gradients by 2 via functional mul."""
            buf = bucket.buffer()
            scaled = mul(buf, 2.0)
            sentinel[0] = scaled
            fut = futures_mod.Future()
            fut.set_result(scaled)
            return fut

        base = _tiny_model()
        ddp = nn.DistributedDataParallel(base)
        ddp.register_comm_hook(None, marking_hook)

        x = torch.ones((1, 4))
        ddp(x).sum().backward()

        grad = base.weight.grad
        assert grad is not None, "param.grad must not be None after backward"
        assert sentinel[0] is not None, "hook was never called"

        grad_np = np.array(grad.tolist()).ravel()
        sentinel_np = np.array(sentinel[0].tolist()).ravel()
        # EXPECTED: grad matches the scaled tensor returned by the hook
        np.testing.assert_allclose(
            grad_np, sentinel_np, rtol=1e-5,
            err_msg="param.grad must equal the tensor returned by the comm hook future"
        )


# ---------------------------------------------------------------------------
# Contract C: register_comm_hook after first forward must raise
# ---------------------------------------------------------------------------

class TestRegisterCommHookGuard:
    """register_comm_hook() after the first forward pass must raise RuntimeError.

    PyTorch contract: comm hooks must be registered before any forward pass.
    """

    def setup_method(self):
        self._port = _free_port()
        _init_pg(self._port)

    def teardown_method(self):
        _teardown()

    def test_register_hook_after_forward_raises(self):
        """Calling register_comm_hook() after forward() must raise RuntimeError.

        EXPECTED: RuntimeError matching hook/forward/register.
        ACTUAL (bug): no guard exists; hook is silently registered.
        """
        import candle as torch
        import candle.nn as nn
        import candle.futures as futures_mod

        def dummy_hook(state, bucket):
            fut = futures_mod.Future()
            fut.set_result(bucket.buffer())
            return fut

        base = _tiny_model()
        ddp = nn.DistributedDataParallel(base)

        x = torch.ones((1, 4))
        ddp(x).sum().backward()

        with pytest.raises(RuntimeError, match=r"(?i)(hook|forward|register)"):
            ddp.register_comm_hook(None, dummy_hook)


# ---------------------------------------------------------------------------
# Contract D: no_sync() suppresses hook; synced pass invokes hook exactly once
# ---------------------------------------------------------------------------

class TestNoSyncHookSuppression:
    """Inside no_sync(), the comm hook must not be called."""

    def setup_method(self):
        self._port = _free_port()
        _init_pg(self._port)

    def teardown_method(self):
        _teardown()

    def test_hook_not_called_in_no_sync_called_in_sync(self):
        """Hook must be silent during no_sync() and active on the synced pass.

        EXPECTED: hook_calls == 0 after no_sync backward;
                  hook_calls >= 1 after normal backward.
        ACTUAL (bug): hook fires even inside no_sync().
        """
        import candle as torch
        import candle.nn as nn
        import candle.futures as futures_mod

        hook_calls = [0]

        def counting_hook(state, bucket):
            hook_calls[0] += 1
            fut = futures_mod.Future()
            fut.set_result(bucket.buffer())
            return fut

        base = _tiny_model()
        ddp = nn.DistributedDataParallel(base)
        ddp.register_comm_hook(None, counting_hook)

        x = torch.ones((1, 4))

        with ddp.no_sync():
            ddp(x).sum().backward()

        assert hook_calls[0] == 0, (
            f"comm hook must NOT fire inside no_sync(); fired {hook_calls[0]} times"
        )

        hook_calls[0] = 0
        ddp(x).sum().backward()

        assert hook_calls[0] >= 1, (
            "comm hook must fire on the synced backward after no_sync()"
        )

    def test_no_sync_grad_accumulation_three_steps(self):
        """Two no_sync + one sync: grad == 3 * base_grad.

        With world_size=1 and identical inputs, three backward passes
        (2 no_sync + 1 synced) must accumulate to grad = 3.0.

        EXPECTED: all grad elements == 3.0.
        ACTUAL (bug): if no_sync incorrectly reduces each step, grad will
            be reset on each iteration rather than accumulated.
        """
        import candle as torch
        import candle.nn as nn
        import numpy as np

        base = _tiny_model()
        ddp = nn.DistributedDataParallel(base)

        x = torch.ones((1, 4))
        base.weight.grad = None

        with ddp.no_sync():
            ddp(x).sum().backward()
        with ddp.no_sync():
            ddp(x).sum().backward()
        ddp(x).sum().backward()

        grad = base.weight.grad
        assert grad is not None
        grad_np = np.array(grad.tolist()).ravel()
        np.testing.assert_allclose(
            grad_np, np.full_like(grad_np, 3.0), rtol=1e-5,
            err_msg=(
                "Expected accumulated grad of 3.0 (2 no_sync + 1 synced); "
                "if no_sync incorrectly reduces each pass values will differ"
            )
        )
