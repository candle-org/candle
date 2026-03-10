"""Multi-process FSDP2 tests with Gloo backend (world_size=2).

Uses subprocess to run each worker process, avoiding conftest NPU-skip
logic and ensuring clean process isolation.
"""
import os
import sys
import socket
import subprocess
import tempfile
import textwrap

import pytest


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Path to the project root (for PYTHONPATH)
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

_WORKER_SCRIPT = textwrap.dedent(r'''
import os
import sys
import traceback

# Ensure candle is importable
sys.path.insert(0, os.environ["_CANDLE_PROJECT_ROOT"])

import candle as torch
import candle.nn as nn
import candle.distributed as dist
from candle.distributed.device_mesh import DeviceMesh
from candle.distributed._composable.fsdp import fully_shard
from candle.distributed.tensor.dtensor import DTensor

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

try:
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # Create model with same initial weights on all ranks
    torch.manual_seed(42)
    model = nn.Linear(8, 4)

    mesh = DeviceMesh("cpu", (world_size,))
    fully_shard(model, mesh=mesh)

    x = torch.randn(2, 8, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 4), f"rank {rank}: bad shape {out.shape}"

    loss = out.sum()
    loss.backward()

    # Check gradients exist
    local_weight = (
        model.weight.to_local()
        if isinstance(model.weight, DTensor)
        else model.weight
    )
    assert local_weight.grad is not None, f"rank {rank}: no weight gradient"
    print(f"[rank {rank}] FSDP2 forward+backward OK")

    dist.destroy_process_group()
    print(f"[rank {rank}] ALL TESTS PASSED")

except Exception:
    traceback.print_exc()
    # Ensure process group cleanup even on failure
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    sys.exit(1)
''')


def _run_workers(world_size, timeout=120):
    """Spawn *world_size* worker subprocesses and wait for them all to pass."""
    port = _find_free_port()

    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(port)
    env["WORLD_SIZE"] = str(world_size)
    env["_CANDLE_PROJECT_ROOT"] = os.path.join(_PROJECT_ROOT, "src")

    # Write the worker script to a temp file
    fd, worker_file = tempfile.mkstemp(suffix=".py", prefix="_fsdp2_gloo_worker_")
    os.close(fd)
    with open(worker_file, "w") as f:
        f.write(_WORKER_SCRIPT)

    processes = []
    try:
        for rank in range(world_size):
            env_rank = {**env, "RANK": str(rank)}
            p = subprocess.Popen(
                [sys.executable, worker_file],
                env=env_rank,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            processes.append(p)

        outputs = []
        for p in processes:
            out, _ = p.communicate(timeout=timeout)
            outputs.append(out.decode("utf-8", errors="replace"))

        # Print all outputs for debugging
        for rank, out in enumerate(outputs):
            print(f"=== RANK {rank} ===")
            print(out)

        # Assert all exited successfully
        for rank, p in enumerate(processes):
            assert p.returncode == 0, (
                f"Worker rank {rank} exited with code {p.returncode}.\n"
                f"Output:\n{outputs[rank]}"
            )
    finally:
        # Clean up temp file
        try:
            os.unlink(worker_file)
        except OSError:
            pass
        # Kill any leftover processes
        for p in processes:
            if p.poll() is None:
                p.kill()


def test_fsdp2_gloo_forward_backward():
    """FSDP2 forward+backward with 2 processes using Gloo backend."""
    _run_workers(world_size=2)
