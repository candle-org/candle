"""HCCL all_to_all verification on NPU with multiple cards."""

import os
import subprocess
import sys
import time

import pytest

from tests.distributed.worker_utils import write_worker_script


SCRIPT = r'''
import os, sys, time
src_dir = os.environ.get("CANDLE_SRC")
if src_dir:
    sys.path.insert(0, src_dir)

import candle as torch
import candle.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# Stagger init to reduce HCCL burst.
time.sleep(0.05 * rank)

device = torch.Device(f"npu:{rank}")
dist.init_process_group(backend="hccl", device_id=device)
print(f"[rank {rank}] init_process_group OK, backend={dist.get_backend()}, device={device}")

# 1. all_to_all - each rank sends unique data to every other rank
# rank i sends [i*100+j*10, i*100+j*10+1] to rank j
inp = []
for j in range(world_size):
    data = [float(rank * 100 + j * 10), float(rank * 100 + j * 10 + 1)]
    inp.append(torch.tensor(data, device=device))

out = [torch.zeros(2, device=device) for _ in range(world_size)]
dist.all_to_all(out, inp)

# Verify: rank i should receive [j*100+i*10, j*100+i*10+1] from rank j
out_cpu = [o.to("cpu") for o in out]
for j in range(world_size):
    expected = [float(j * 100 + rank * 10), float(j * 100 + rank * 10 + 1)]
    actual = list(out_cpu[j]._numpy_view())
    assert actual == expected, f"rank {rank} from rank {j}: expected {expected}, got {actual}"
print(f"[rank {rank}] all_to_all OK")

# 2. all_to_all_single with equal split
inp_single = torch.tensor([float(rank * world_size + j) for j in range(world_size)], device=device)
out_single = torch.zeros(world_size, device=device)
dist.all_to_all_single(out_single, inp_single)

out_single_cpu = out_single.to("cpu")
expected_single = [float(j * world_size + rank) for j in range(world_size)]
actual_single = list(out_single_cpu._numpy_view())
assert actual_single == expected_single, f"rank {rank} all_to_all_single: expected {expected_single}, got {actual_single}"
print(f"[rank {rank}] all_to_all_single OK")

dist.destroy_process_group()
print(f"[rank {rank}] ALL TESTS PASSED")
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

    worker_file = write_worker_script(SCRIPT, name=f"hccl_all_to_all_{world_size}card")

    failed = []
    outputs = []
    procs = []

    for r in range(world_size):
        p = subprocess.Popen(
            [sys.executable, worker_file],
            env={**env, "RANK": str(r)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append(p)

    timeout = 420 if world_size <= 4 else 900
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
            f"HCCL all_to_all {world_size}card failed on ranks: {failed}"
        )


@pytest.mark.parametrize(
    "world_size,master_port",
    [
        (2, 29750),
        (4, 29760),
        (8, 29770),
    ],
)
def test_hccl_all_to_all_multicard(world_size, master_port):
    import candle as torch
    if torch.npu.device_count() < world_size:
        pytest.skip(f"Need {world_size} NPUs, found {torch.npu.device_count()}")
    _run_case(world_size, master_port)
