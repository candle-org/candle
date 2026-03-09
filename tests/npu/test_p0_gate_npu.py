import os
import subprocess
import sys

import pytest

import candle as torch



def _npu_available() -> bool:
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False



def _npu_count() -> int:
    try:
        return int(torch.npu.device_count())
    except Exception:
        return 0


@pytest.mark.skipif(not _npu_available(), reason="NPU unavailable")
def test_p0_gate_amp_npu_autocast_smoke():
    x = torch.randn((8, 8), device="npu", dtype=torch.float32)
    with torch.amp.autocast("npu", dtype=torch.float16):
        y = torch.matmul(x, x)
        loss = y.sum()

    assert y.device.type == "npu"
    assert loss.device.type == "npu"


@pytest.mark.skipif(not _npu_available(), reason="NPU unavailable")
@pytest.mark.xfail(reason="Known issue: NPU scalar mul path fails with aclnnMul 561000", strict=False)
def test_p0_gate_amp_npu_scaler_smoke_known_issue():
    x = torch.randn((8, 8), device="npu", dtype=torch.float32)
    x.requires_grad = True
    scaler = torch.amp.GradScaler("npu")

    with torch.amp.autocast("npu", dtype=torch.float16):
        y = torch.matmul(x, x)
        loss = y.sum()

    scaled = scaler.scale(loss)
    scaled.backward()

    assert x.grad is not None
    assert x.grad.device.type == "npu"


SCRIPT = r'''
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import candle as torch
import candle.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.Device(f"npu:{rank}")

dist.init_process_group("hccl", device_id=device)

# Repeat to catch transient ordering/race regressions.
for it in range(5):
    inp = []
    for j in range(world_size):
        base = float(1000 * it + rank * 100 + j * 10)
        inp.append(torch.tensor([base, base + 1.0], device=device))
    out = [torch.zeros(2, device=device) for _ in range(world_size)]
    dist.all_to_all(out, inp)

    for j in range(world_size):
        base = float(1000 * it + j * 100 + rank * 10)
        exp = [base, base + 1.0]
        got = list(out[j].to("cpu")._numpy_view())
        assert got == exp, f"iter={it}, rank={rank}, src={j}, got={got}, exp={exp}"

    count = 3
    inp_single = torch.tensor(
        [float(1000 * it + rank * 100 + k) for k in range(world_size * count)],
        device=device,
    )
    out_single = torch.zeros(world_size * count, device=device)
    split_sizes = [count] * world_size
    dist.all_to_all_single(out_single, inp_single, split_sizes, split_sizes)

    got_single = list(out_single.to("cpu")._numpy_view())
    exp_single = []
    for src in range(world_size):
        exp_single.extend([float(1000 * it + src * 100 + rank * count + k) for k in range(count)])
    assert got_single == exp_single, f"iter={it}, rank={rank}, got={got_single}, exp={exp_single}"

dist.destroy_process_group()
'''


@pytest.mark.skipif((not _npu_available()) or _npu_count() < 4, reason="Requires >=4 NPUs")
def test_p0_gate_hccl_all_to_all_4card_stability(tmp_path):
    worker = tmp_path / "_p0_gate_hccl_all_to_all_4card.py"
    worker.write_text(SCRIPT)

    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29731"
    env["WORLD_SIZE"] = "4"
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env["PYTHONPATH"] = root + ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

    procs = []
    for rank in range(4):
        env_rank = {**env, "RANK": str(rank)}
        procs.append(
            subprocess.Popen(
                [sys.executable, str(worker)],
                env=env_rank,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        )

    outputs = []
    failed = False
    for rank, proc in enumerate(procs):
        try:
            out, _ = proc.communicate(timeout=240)
            text = out.decode()
            outputs.append((rank, proc.returncode, text))
            if proc.returncode != 0:
                failed = True
        except subprocess.TimeoutExpired:
            proc.kill()
            outputs.append((rank, -1, "TIMEOUT"))
            failed = True

    if failed:
        details = "\n\n".join(
            f"=== rank {rank} exit={code} ===\n{text}" for rank, code, text in outputs
        )
        pytest.fail(f"4-card HCCL all_to_all gate failed:\n{details}")
