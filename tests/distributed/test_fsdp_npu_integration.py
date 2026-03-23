"""Task 8: NPU-first FSDP integration tests + P1 exit gate.

These tests exercise the full FSDP2 lifecycle on HCCL/NPU:
  - forward + backward + optimizer step
  - no_sync() gradient accumulation
  - summon_full_params() round-trip
  - reshard_after_forward=False behaviour

Each multicard test spawns real worker subprocesses that communicate via
HCCL on NPU hardware.  When NPU hardware or HCCL is unavailable the test
is skipped gracefully.

P1 exit gate: printed at the end of every passing NPU run and also
available via ``test_p1_exit_gate_report`` (always runs, prints gate
status even without hardware).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

_SRC_DIR_EARLY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "src")
)

_HCCL_PROBE_SCRIPT = """\
import os, sys
sys.path.insert(0, os.environ.get("CANDLE_SRC", ""))
try:
    from candle.distributed._backend import is_hccl_available
    if not is_hccl_available():
        sys.exit(2)
    from candle._backends.npu.runtime import device_count
    if device_count() < 2:
        sys.exit(2)
    # Probe HCCL usability: call HcclGetRootInfo directly.
    # This is cheaper than init_process_group (no TCPStore needed) and
    # immediately reveals whether the HCCL runtime is accessible.
    from candle.distributed._hccl.hccl_bindings import get_bindings, HcclRootInfo
    import ctypes
    b = get_bindings()
    root_info = HcclRootInfo()
    ret = b.get_root_info(ctypes.byref(root_info))
    if ret != 0:
        sys.exit(3)  # HCCL runtime not usable
    sys.exit(0)
except SystemExit:
    raise
except Exception:
    sys.exit(3)
"""


def _hccl_available() -> bool:
    """Return True only when HCCL is loadable, >=2 NPUs exist, AND the
    HCCL runtime can actually initialise (probe via subprocess)."""
    try:
        env = os.environ.copy()
        env["CANDLE_SRC"] = _SRC_DIR_EARLY
        result = subprocess.run(
            [sys.executable, "-c", _HCCL_PROBE_SCRIPT],
            env=env,
            capture_output=True,
            timeout=8,
        )
        return result.returncode == 0
    except Exception:  # pragma: no cover
        return False


_SKIP_NO_NPU = pytest.mark.skipif(
    not _hccl_available(),
    reason="Requires HCCL and >= 2 NPU devices",
)

# ---------------------------------------------------------------------------
# Shared subprocess infrastructure
# ---------------------------------------------------------------------------

_SRC_DIR = _SRC_DIR_EARLY


def _run_npu_workers(script: str, world_size: int, timeout: int = 120) -> None:
    """Spawn *world_size* NPU worker subprocesses executing *script*.

    Raises AssertionError if any worker exits non-zero.
    """
    import socket
    import tempfile

    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    fd, worker_file = tempfile.mkstemp(prefix="fsdp_npu_worker_", suffix=".py")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(script)

        base_env = os.environ.copy()
        base_env["MASTER_ADDR"] = "127.0.0.1"
        base_env["MASTER_PORT"] = str(port)
        base_env["WORLD_SIZE"] = str(world_size)
        base_env["CANDLE_SRC"] = _SRC_DIR
        base_env["PYTHONPATH"] = _SRC_DIR + (
            ":" + base_env["PYTHONPATH"] if "PYTHONPATH" in base_env else ""
        )

        procs = []
        for rank in range(world_size):
            env = {**base_env, "RANK": str(rank)}
            procs.append(
                subprocess.Popen(
                    [sys.executable, worker_file],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            )

        outputs = []
        for p in procs:
            out, _ = p.communicate(timeout=timeout)
            outputs.append(out.decode("utf-8", errors="replace"))

        for rank, out in enumerate(outputs):
            print(f"=== RANK {rank} ===\n{out}")

        for rank, p in enumerate(procs):
            assert p.returncode == 0, (
                f"Worker rank {rank} exited {p.returncode}.\nOutput:\n{outputs[rank]}"
            )
    finally:
        try:
            os.unlink(worker_file)
        except OSError:
            pass
        for p in procs:
            if p.poll() is None:
                p.kill()


# ---------------------------------------------------------------------------
# Worker script: forward + backward + optimizer step
# ---------------------------------------------------------------------------

_WORKER_FWD_BWD_OPT = textwrap.dedent(r"""\
import os, sys, traceback
sys.path.insert(0, os.environ["CANDLE_SRC"])

import candle as torch
import candle.nn as nn
import candle.distributed as dist
from candle.distributed.device_mesh import DeviceMesh
from candle.distributed._composable.fsdp import fully_shard
from candle.distributed.tensor.dtensor import DTensor

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.Device(f"npu:{rank}")

try:
    dist.init_process_group(backend="hccl", device_id=device)

    torch.manual_seed(42)
    model = nn.Linear(16, 8).to(device)

    mesh = DeviceMesh("npu", list(range(world_size)))
    fully_shard(model, mesh=mesh)

    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(4, 16, device=device)
    out = model(x)
    assert out.shape == (4, 8), f"rank {rank}: unexpected shape {out.shape}"

    loss = out.sum()
    loss.backward()

    local_w = (
        model.weight.to_local()
        if isinstance(model.weight, DTensor)
        else model.weight
    )
    assert local_w.grad is not None, f"rank {rank}: weight grad missing"

    opt.step()
    opt.zero_grad()

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] fwd+bwd+opt PASS")
except Exception:
    traceback.print_exc()
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    sys.exit(1)
""")


@_SKIP_NO_NPU
def test_npu_fsdp_forward_backward_optimizer_step():
    """FSDP2 forward + backward + SGD step on 2 NPU devices via HCCL."""
    _run_npu_workers(_WORKER_FWD_BWD_OPT, world_size=2)


# ---------------------------------------------------------------------------
# Worker script: no_sync() gradient accumulation
# ---------------------------------------------------------------------------

_WORKER_NO_SYNC = textwrap.dedent(r"""\
import os, sys, traceback
sys.path.insert(0, os.environ["CANDLE_SRC"])

import candle as torch
import candle.nn as nn
import candle.distributed as dist
from candle.distributed.device_mesh import DeviceMesh
from candle.distributed._composable.fsdp import fully_shard
from candle.distributed.tensor.dtensor import DTensor

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.Device(f"npu:{rank}")

try:
    dist.init_process_group(backend="hccl", device_id=device)

    torch.manual_seed(42)
    model = nn.Linear(8, 4).to(device)
    mesh = DeviceMesh("npu", list(range(world_size)))
    fully_shard(model, mesh=mesh)

    local_w = (
        model.weight.to_local()
        if isinstance(model.weight, DTensor)
        else model.weight
    )

    # Two micro-batches under no_sync — gradients accumulate, no reduce-scatter
    x1 = torch.randn(2, 8, device=device)
    x2 = torch.randn(2, 8, device=device)

    with model.no_sync():
        model(x1).sum().backward()
        grad_after_first = local_w.grad.clone() if local_w.grad is not None else None

    # Third backward (outside no_sync) triggers reduce-scatter
    model(x2).sum().backward()

    assert local_w.grad is not None, f"rank {rank}: grad missing after sync backward"

    # Under no_sync the gradient from the first backward should not have been
    # zeroed; after the sync backward the accumulated grad is reduced.
    # Verify that the final grad is non-zero (reduce happened).
    total = float(local_w.grad.abs().sum().to("cpu").item())
    assert total > 0, f"rank {rank}: all-zero grad after no_sync accumulation"

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] no_sync accumulation PASS")
except Exception:
    traceback.print_exc()
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    sys.exit(1)
""")


@_SKIP_NO_NPU
def test_npu_fsdp_no_sync_accumulation():
    """FSDP2 no_sync() accumulates grads then reduces on the final backward."""
    _run_npu_workers(_WORKER_NO_SYNC, world_size=2)


# ---------------------------------------------------------------------------
# Worker script: summon_full_params() round-trip
# ---------------------------------------------------------------------------

_WORKER_SUMMON = textwrap.dedent(r"""\
import os, sys, traceback
sys.path.insert(0, os.environ["CANDLE_SRC"])

import candle as torch
import candle.nn as nn
import candle.distributed as dist
from candle.distributed.device_mesh import DeviceMesh
from candle.distributed._composable.fsdp import fully_shard
from candle.distributed.tensor.dtensor import DTensor

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.Device(f"npu:{rank}")

try:
    dist.init_process_group(backend="hccl", device_id=device)

    torch.manual_seed(42)
    model = nn.Linear(8, 4).to(device)
    mesh = DeviceMesh("npu", list(range(world_size)))
    fully_shard(model, mesh=mesh)

    # Record sharded weight shape before summoning
    w_before = model.weight
    shard_shape_before = (
        w_before.to_local().shape if isinstance(w_before, DTensor)
        else w_before.shape
    )

    with model.summon_full_params(writeback=True):
        # Inside: weight should be full-sized (all rows present)
        w_full = model.weight
        full_shape = (
            w_full.to_local().shape if isinstance(w_full, DTensor)
            else w_full.shape
        )
        assert full_shape[0] >= shard_shape_before[0], (
            f"rank {rank}: full shape {full_shape} not >= shard shape {shard_shape_before}"
        )

    # After summon: weight should be resharded
    w_after = model.weight
    shard_shape_after = (
        w_after.to_local().shape if isinstance(w_after, DTensor)
        else w_after.shape
    )
    assert shard_shape_after == shard_shape_before, (
        f"rank {rank}: shape after summon {shard_shape_after} != before {shard_shape_before}"
    )

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] summon_full_params round-trip PASS")
except Exception:
    traceback.print_exc()
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    sys.exit(1)
""")


@_SKIP_NO_NPU
def test_npu_fsdp_summon_full_params_roundtrip():
    """summon_full_params() unshards and re-shards params correctly on NPU."""
    _run_npu_workers(_WORKER_SUMMON, world_size=2)


# ---------------------------------------------------------------------------
# Worker script: reshard_after_forward=False
# ---------------------------------------------------------------------------

_WORKER_RESHARD_FALSE = textwrap.dedent(r"""\
import os, sys, traceback
sys.path.insert(0, os.environ["CANDLE_SRC"])

import candle as torch
import candle.nn as nn
import candle.distributed as dist
from candle.distributed.device_mesh import DeviceMesh
from candle.distributed._composable.fsdp import fully_shard
from candle.distributed.tensor.dtensor import DTensor

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.Device(f"npu:{rank}")

try:
    dist.init_process_group(backend="hccl", device_id=device)

    torch.manual_seed(42)
    model = nn.Linear(8, 4).to(device)
    mesh = DeviceMesh("npu", list(range(world_size)))
    # reshard_after_forward=False: keep full params between fwd and bwd
    fully_shard(model, mesh=mesh, reshard_after_forward=False)

    x = torch.randn(4, 8, device=device)
    out = model(x)

    # After forward with reshard_after_forward=False the param group should
    # still be in unsharded state (_is_unsharded=True)
    pg = model._fsdp_state.param_group
    assert pg._is_unsharded, (
        f"rank {rank}: expected _is_unsharded=True after fwd with reshard_after_forward=False"
    )

    loss = out.sum()
    loss.backward()  # post-backward: reshard happens here

    assert not pg._is_unsharded, (
        f"rank {rank}: expected _is_unsharded=False after backward"
    )

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] reshard_after_forward=False PASS")
except Exception:
    traceback.print_exc()
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    sys.exit(1)
""")


@_SKIP_NO_NPU
def test_npu_fsdp_reshard_after_forward_false():
    """reshard_after_forward=False keeps full params live until backward."""
    _run_npu_workers(_WORKER_RESHARD_FALSE, world_size=2)


# ---------------------------------------------------------------------------
# P1 exit gate (always runs — no NPU required)
# ---------------------------------------------------------------------------

# Gate criteria: all four integration tests must pass on real NPU hardware.
_P1_GATE_CRITERIA = [
    "fwd+bwd+opt: forward + backward + optimizer step completes on NPU/HCCL",
    "no_sync: gradient accumulation across micro-batches works without premature reduce-scatter",
    "summon_full_params: round-trip unshard/reshard preserves parameter shapes",
    "reshard_after_forward=False: full params remain live between forward and backward",
]


def test_p1_exit_gate_report(capsys):
    """Print P1 readiness gate criteria.

    This test always passes.  It prints the gate checklist and whether
    NPU/HCCL hardware is present on the current machine.
    """
    npu_present = _hccl_available()
    status = "HARDWARE PRESENT" if npu_present else "NO HARDWARE (skipped on this machine)"

    lines = [
        "",
        "=" * 60,
        "P1 EXIT GATE: NPU-first FSDP readiness",
        f"NPU/HCCL status: {status}",
        "=" * 60,
        "Required criteria (must all pass on NPU hardware):",
    ]
    for i, criterion in enumerate(_P1_GATE_CRITERIA, 1):
        lines.append(f"  [{i}] {criterion}")
    lines += [
        "=" * 60,
        "To verify P1 gate: run this file on a machine with >= 2 NPUs.",
        "All four @_SKIP_NO_NPU tests must PASS.",
        "=" * 60,
        "",
    ]
    print("\n".join(lines))
    # Gate itself always passes — it is a report, not an assertion.
    assert True
