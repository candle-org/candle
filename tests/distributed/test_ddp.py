"""HCCL 2-card DDP basic smoke test.

Verifies forward, backward, and gradient synchronization across 2 NPU ranks.
"""

import os
import subprocess
import sys
import time

import pytest

from tests.distributed.worker_utils import write_worker_script


SCRIPT = r'''
import os, sys
sys.path.insert(0, os.environ.get("CANDLE_SRC", ""))

import candle as torch
import candle.nn as nn
import candle.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

device = torch.Device(f"npu:{rank}")
dist.init_process_group("hccl", device_id=device)
print(f"[rank {rank}] init OK, world_size={world_size}")


class SimpleModel(nn.Module):
    """Simple model using elementwise ops (avoids matmul)."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((10,)))

    def forward(self, x):
        return x * self.weight


model = SimpleModel().to(device)
ddp_model = nn.DistributedDataParallel(model)

x = torch.ones((4, 10), device=device)
output = ddp_model(x)
loss = output.sum()
loss.backward()

assert model.weight.grad is not None, "weight grad is None"

# Verify grads are synchronized: allreduce(grad) should equal world_size * grad
grad_copy = model.weight.grad.clone()
dist.all_reduce(grad_copy, op=dist.ReduceOp.SUM)
from candle._functional import mul, add, neg
expected = mul(model.weight.grad, float(world_size))
diff = add(grad_copy, neg(expected)).abs().sum().item()
assert diff < 1e-5, f"rank={rank} gradients differ (diff={diff})"

dist.barrier()
dist.destroy_process_group()
print(f"[rank {rank}] HCCL DDP basic {world_size}card PASS")
'''


def _run_once(world_size, master_port):
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(master_port)
    env["WORLD_SIZE"] = str(world_size)
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["CANDLE_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    worker_file = write_worker_script(SCRIPT, name=f"hccl_ddp_{world_size}card")

    procs = []
    for r in range(world_size):
        p = subprocess.Popen(
            [sys.executable, worker_file],
            env={**env, "RANK": str(r)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append(p)

    failed = []
    outputs = []
    timeout = 120
    for r, p in enumerate(procs):
        try:
            out, _ = p.communicate(timeout=timeout)
            txt = out.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
            txt = "TIMEOUT\n" + out.decode("utf-8", errors="replace")
        outputs.append(txt)
        if p.returncode != 0:
            failed.append(r)

    return failed, outputs


def _run_case(world_size, master_port):
    retries = 3
    for attempt in range(1, retries + 1):
        failed, outputs = _run_once(world_size, master_port)
        if not failed:
            return

        joined = "\n".join(outputs)
        transient = "resource unavailable" in joined
        if transient and attempt < retries:
            print(
                f"HCCL transient init failure on {world_size} cards, "
                f"retry {attempt}/{retries}"
            )
            time.sleep(5)
            continue

        for r, txt in enumerate(outputs):
            print(f"=== RANK {r} ===")
            print(txt)
        raise AssertionError(
            f"HCCL DDP basic {world_size}card failed on ranks: {failed}"
        )


def test_ddp():
    import candle as torch
    if torch.npu.device_count() < 2:
        pytest.skip("Need at least 2 NPUs")
    _run_case(2, 29501)
