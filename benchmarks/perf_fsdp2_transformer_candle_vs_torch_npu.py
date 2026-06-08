#!/usr/bin/env python3
"""Two-card Transformer FSDP2 NPU perf comparison: Candle vs torch_npu."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmarks.npu_perf_gates import summarize_samples


@dataclass(frozen=True)
class FSDP2TransformerCase:
    """Shape/configuration for a Transformer FSDP2 benchmark case."""

    layers: int
    batch_per_rank: int
    seq: int
    hidden: int
    heads: int
    mlp_ratio: int = 4


CASES = {
    "xfmr_fsdp2_small": FSDP2TransformerCase(
        layers=1,
        batch_per_rank=2,
        seq=128,
        hidden=512,
        heads=8,
    ),
    "xfmr_fsdp2_medium": FSDP2TransformerCase(
        layers=4,
        batch_per_rank=1,
        seq=256,
        hidden=1024,
        heads=16,
    ),
}


def _build_transformer_model(torch, cfg):
    """Build the shared Transformer stack used by Candle and torch_npu."""
    nn = torch.nn
    functional = torch.nn.functional

    class TransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(cfg.hidden)
            self.attn = nn.MultiheadAttention(
                cfg.hidden,
                cfg.heads,
                batch_first=True,
                dropout=0.0,
            )
            self.norm2 = nn.LayerNorm(cfg.hidden)
            self.ff1 = nn.Linear(cfg.hidden, cfg.mlp_ratio * cfg.hidden)
            self.ff2 = nn.Linear(cfg.mlp_ratio * cfg.hidden, cfg.hidden)

        def forward(self, x):
            y = self.norm1(x)
            attn_out, _ = self.attn(y, y, y, need_weights=False)
            x = x + attn_out
            z = self.norm2(x)
            z = functional.gelu(self.ff1(z))
            return x + self.ff2(z)

    class TransformerStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([TransformerBlock() for _ in range(cfg.layers)])
            self.final_norm = nn.LayerNorm(cfg.hidden)

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.final_norm(x)

    return TransformerStack()


def _iter_fsdp_transformer_modules(model):
    """Yield modules in bottom-up FSDP wrapping order."""
    for block in model.blocks:
        yield block.norm1
        yield block.attn
        yield block.norm2
        yield block.ff1
        yield block.ff2
        yield block
    yield model.final_norm
    yield model


def _numel(shape):
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def _make_deterministic_tensor(torch, shape, *, start, end, device, dtype,
                               requires_grad=False):
    """Create identical deterministic data in Candle and torch_npu workers."""
    data = torch.linspace(start, end, _numel(shape), device=device, dtype=dtype)
    data = data.reshape(*shape)
    if requires_grad:
        data.requires_grad_(True)
    return data


def _reset_transformer_parameters(torch, model):
    """Initialize Candle and torch_npu models with the same deterministic values."""
    for idx, (name, param) in enumerate(model.named_parameters()):
        if "norm" in name and name.endswith("weight"):
            fill = 1.0
        elif name.endswith("bias"):
            fill = 0.0005 * ((idx % 5) - 2)
        else:
            fill = 0.001 * ((idx % 7) + 1)
        param.data = torch.full_like(param, fill)


def _local_tensor(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _tensor_values(torch, tensor, limit=32):
    tensor = _local_tensor(tensor)
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return []
    flat = flat[:min(limit, flat.numel())].to(torch.float32).to("cpu")
    return [float(value) for value in flat.numpy().reshape(-1).tolist()]


def _tensor_checksum(torch, tensor):
    tensor = _local_tensor(tensor)
    return float(tensor.detach().to(torch.float32).sum().to("cpu").item())


def _named_tensor_checksums(torch, model, *, grads=False):
    names = []
    values = []
    for name, param in model.named_parameters():
        local = _local_tensor(param)
        tensor = None
        if grads:
            tensor = getattr(local, "grad", None)
            if tensor is None and getattr(param, "grad", None) is not None:
                tensor = _local_tensor(param.grad)
        else:
            tensor = local
        names.append(name)
        values.append(0.0 if tensor is None else _tensor_checksum(torch, tensor))
    return names, values


def _run_accuracy_step(torch, dist, model, x, target, optimizer, *, sync_all):
    """Run one deterministic train step and collect cross-framework numerics."""
    sync_all(torch, dist)
    out = model(x)
    diff = out - target
    loss = (diff * diff).mean()
    loss.backward()
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()

    param_names, param_grad_values = _named_tensor_checksums(torch, model, grads=True)
    accuracy = {
        "loss": float(loss.detach().to(torch.float32).to("cpu").item()),
        "output_checksum": _tensor_checksum(torch, out),
        "output_values": _tensor_values(torch, out),
        "input_grad_checksum": _tensor_checksum(torch, x.grad),
        "input_grad_values": _tensor_values(torch, x.grad),
        "param_names": param_names,
        "param_grad_checksum_values": param_grad_values,
    }

    optimizer.step()
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()
    param_names, param_values = _named_tensor_checksums(torch, model, grads=False)
    accuracy["param_names_after_step"] = param_names
    accuracy["param_checksum_after_step_values"] = param_values
    accuracy["param_checksum_after_step"] = sum(param_values)

    optimizer.zero_grad(set_to_none=True)
    x.grad = None
    sync_all(torch, dist)
    return accuracy


def _sync_candle(torch, dist):
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()
    dist.barrier()


def _sync_torch_npu(torch, dist):
    torch.npu.synchronize()
    dist.barrier()


def _run_timed_steps(torch, dist, model, x, optimizer, *, iters, warmup, sync_all,
                     profile_reset=None, profile_summary=None):
    """Run warmup and timed training steps, returning per-rank samples."""
    # Initial runtime/cache warmup outside the requested warmup count.
    out = model(x)
    loss = out.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sync_all(torch, dist)

    for _ in range(warmup):
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    sync_all(torch, dist)
    if profile_reset is not None:
        profile_reset(True)

    fwd_samples = []
    bwd_samples = []
    optim_samples = []
    total_samples = []
    for _ in range(iters):
        sync_all(torch, dist)
        t0 = time.perf_counter()
        out = model(x)
        loss = out.sum()
        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()
        t1 = time.perf_counter()
        loss.backward()
        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()
        t2 = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()
        t3 = time.perf_counter()
        dist.barrier()

        fwd_samples.append((t1 - t0) * 1000.0)
        bwd_samples.append((t2 - t1) * 1000.0)
        optim_samples.append((t3 - t2) * 1000.0)
        total_samples.append((t3 - t0) * 1000.0)

    result = {
        "fwd_ms_samples": fwd_samples,
        "bwd_ms_samples": bwd_samples,
        "optim_ms_samples": optim_samples,
        "total_ms_samples": total_samples,
    }
    if profile_summary is not None:
        result["profile"] = profile_summary()
    if profile_reset is not None:
        profile_reset(False)
    return result


def _write_worker_result(out_dir, rank, payload):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"rank{rank}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_candle_worker(args):
    import candle as torch  # pylint: disable=import-outside-toplevel
    import candle.distributed as dist  # pylint: disable=import-outside-toplevel
    from candle.distributed._composable.fsdp import fully_shard  # pylint: disable=import-outside-toplevel
    from candle.distributed.device_mesh import DeviceMesh  # pylint: disable=import-outside-toplevel

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.Device(f"npu:{rank}")
    cfg = CASES[args.case]

    try:
        dist.init_process_group(backend="hccl", device_id=device)
        torch.manual_seed(args.seed)
        dtype = getattr(torch, args.dtype)
        model = _build_transformer_model(torch, cfg).to(device).to(dtype)
        _reset_transformer_parameters(torch, model)
        mesh = DeviceMesh("npu", (world_size,))
        for module in _iter_fsdp_transformer_modules(model):
            fully_shard(module, mesh=mesh)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        x = _make_deterministic_tensor(
            torch,
            (cfg.batch_per_rank, cfg.seq, cfg.hidden),
            start=-0.75,
            end=0.75,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        target = _make_deterministic_tensor(
            torch,
            (cfg.batch_per_rank, cfg.seq, cfg.hidden),
            start=0.25,
            end=-0.25,
            device=device,
            dtype=dtype,
        )
        accuracy = None
        if args.check_accuracy:
            accuracy = _run_accuracy_step(
                torch, dist, model, x, target, optimizer, sync_all=_sync_candle
            )
        profile_reset = None
        profile_summary = None
        if args.profile_candle:
            from candle.distributed._composable.fsdp import _profile  # pylint: disable=import-outside-toplevel
            profile_reset = _profile.reset
            profile_summary = _profile.summary
        samples = _run_timed_steps(
            torch,
            dist,
            model,
            x,
            optimizer,
            iters=args.iters,
            warmup=args.warmup,
            sync_all=_sync_candle,
            profile_reset=profile_reset,
            profile_summary=profile_summary,
        )
        payload = {
            "rank": rank,
            "framework": "candle",
            "case": args.case,
            "world_size": world_size,
            "global_batch": cfg.batch_per_rank * world_size,
            "batch_per_rank": cfg.batch_per_rank,
            "dtype": args.dtype,
            "config": asdict(cfg),
            **samples,
        }
        if accuracy is not None:
            payload["accuracy"] = accuracy
        _write_worker_result(args.out_dir, rank, payload)
        dist.destroy_process_group()
    except Exception as exc:  # pragma: no cover - diagnostics for benchmark workers
        _write_worker_result(
            args.out_dir,
            rank,
            {"rank": rank, "framework": "candle", "case": args.case, "error": repr(exc)},
        )
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # pylint: disable=broad-except
            pass
        raise


def _run_torch_npu_worker(args):
    import torch  # pylint: disable=import-outside-toplevel
    import torch_npu  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import,import-error
    import torch.distributed as dist  # pylint: disable=import-outside-toplevel
    from torch.distributed._composable.fsdp import fully_shard  # pylint: disable=import-outside-toplevel
    from torch.distributed.device_mesh import init_device_mesh  # pylint: disable=import-outside-toplevel

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    cfg = CASES[args.case]

    try:
        torch.npu.set_device(local_rank)
        dist.init_process_group(backend="hccl")
        torch.manual_seed(args.seed)
        device = torch.device(f"npu:{local_rank}")
        dtype = getattr(torch, args.dtype)
        model = _build_transformer_model(torch, cfg).to(device=device, dtype=dtype)
        _reset_transformer_parameters(torch, model)
        mesh = init_device_mesh("npu", (world_size,))
        for module in _iter_fsdp_transformer_modules(model):
            fully_shard(module, mesh=mesh)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        x = _make_deterministic_tensor(
            torch,
            (cfg.batch_per_rank, cfg.seq, cfg.hidden),
            start=-0.75,
            end=0.75,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        target = _make_deterministic_tensor(
            torch,
            (cfg.batch_per_rank, cfg.seq, cfg.hidden),
            start=0.25,
            end=-0.25,
            device=device,
            dtype=dtype,
        )
        accuracy = None
        if args.check_accuracy:
            accuracy = _run_accuracy_step(
                torch, dist, model, x, target, optimizer, sync_all=_sync_torch_npu
            )
        samples = _run_timed_steps(
            torch,
            dist,
            model,
            x,
            optimizer,
            iters=args.iters,
            warmup=args.warmup,
            sync_all=_sync_torch_npu,
        )
        payload = {
            "rank": rank,
            "framework": "torch_npu",
            "case": args.case,
            "world_size": world_size,
            "global_batch": cfg.batch_per_rank * world_size,
            "batch_per_rank": cfg.batch_per_rank,
            "dtype": args.dtype,
            "config": asdict(cfg),
            **samples,
        }
        if accuracy is not None:
            payload["accuracy"] = accuracy
        _write_worker_result(args.out_dir, rank, payload)
        dist.destroy_process_group()
    except Exception as exc:  # pragma: no cover - diagnostics for benchmark workers
        _write_worker_result(
            args.out_dir,
            rank,
            {"rank": rank, "framework": "torch_npu", "case": args.case, "error": repr(exc)},
        )
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # pylint: disable=broad-except
            pass
        raise


def _max_by_iteration(rank_results, sample_key):
    sample_lists = [row[sample_key] for row in rank_results]
    return [max(values) for values in zip(*sample_lists)]


def _copy_summary_fields(result, prefix, samples):
    summary = summarize_samples(samples)
    for key, value in summary.items():
        if key.endswith("_ms"):
            result[f"{prefix}_ms_{key[:-3]}"] = value
        else:
            result[f"{prefix}_{key}"] = value
    result[f"{prefix}_ms_samples"] = samples


def _aggregate_profile_summaries(rank_results):
    """Aggregate per-rank FSDP profile summaries by phase."""
    phase_names = sorted(
        {
            phase
            for row in rank_results
            for phase in row.get("profile", {})
        }
    )
    if not phase_names:
        return None
    aggregated = {}
    for phase in phase_names:
        counts = []
        totals = []
        avgs = []
        for row in rank_results:
            event = row.get("profile", {}).get(phase, {})
            count = event.get("count", 0)
            total_ms = event.get("total_ms", 0.0)
            avg_ms = event.get("avg_ms", total_ms / count if count else 0.0)
            counts.append(count)
            totals.append(total_ms)
            avgs.append(avg_ms)
        aggregated[phase] = {
            "count_by_rank": counts,
            "count_max": max(counts) if counts else 0,
            "total_ms_by_rank": totals,
            "total_ms_max": max(totals) if totals else 0.0,
            "avg_ms_by_rank": avgs,
            "avg_ms_max": max(avgs) if avgs else 0.0,
        }
    return aggregated


def _aggregate_rank_results(framework, case, rank_results):
    """Aggregate per-rank worker JSON into one framework/case result row."""
    if not rank_results:
        return {"framework": framework, "case": case, "error": "no rank results"}
    errors = [row for row in rank_results if "error" in row]
    if errors:
        return {
            "framework": framework,
            "case": case,
            "error": "; ".join(f"rank {row.get('rank')}: {row['error']}" for row in errors),
            "rank_results": rank_results,
        }

    rank_results = sorted(rank_results, key=lambda row: row["rank"])
    fwd_samples = _max_by_iteration(rank_results, "fwd_ms_samples")
    bwd_samples = _max_by_iteration(rank_results, "bwd_ms_samples")
    optim_samples = _max_by_iteration(rank_results, "optim_ms_samples")
    total_samples = _max_by_iteration(rank_results, "total_ms_samples")

    first = rank_results[0]
    result = {
        "framework": framework,
        "case": case,
        "world_size": first["world_size"],
        "global_batch": first["global_batch"],
        "batch_per_rank": first["batch_per_rank"],
        "dtype": first.get("dtype"),
        "config": first.get("config"),
        "rank_results": rank_results,
    }
    _copy_summary_fields(result, "fwd", fwd_samples)
    _copy_summary_fields(result, "bwd", bwd_samples)
    _copy_summary_fields(result, "optim", optim_samples)
    _copy_summary_fields(result, "total", total_samples)
    profile = _aggregate_profile_summaries(rank_results)
    if profile is not None:
        result["profile"] = profile
    total_ms = result["total_ms_median"]
    result["samples_per_second"] = (
        float(result["global_batch"]) / (float(total_ms) / 1000.0)
        if total_ms else 0.0
    )
    return result


def _base_worker_args(args, framework, case, out_dir):
    worker_args = [
        os.path.abspath(__file__),
        "--worker",
        "--framework", framework,
        "--case", case,
        "--iters", str(args.iters),
        "--warmup", str(args.warmup),
        "--dtype", args.dtype,
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--out-dir", out_dir,
    ]
    if framework == "candle" and getattr(args, "profile_candle", False):
        worker_args.append("--profile-candle")
    if getattr(args, "check_accuracy", False):
        worker_args.append("--check-accuracy")
    return worker_args


def _torch_npu_launch_command(python_exe, script_path, *, world_size, worker_args):
    return [
        python_exe,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        script_path,
        *worker_args,
    ]


def _run_processes(procs, *, timeout):
    outputs = []
    for proc in procs:
        out, _ = proc.communicate(timeout=timeout)
        outputs.append(out.decode("utf-8", errors="replace"))
    return outputs


def _read_rank_results(out_dir, world_size):
    results = []
    for rank in range(world_size):
        path = Path(out_dir) / f"rank{rank}.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            results.append({"rank": rank, "error": f"missing rank output: {path}"})
    return results


def _spawn_candle(case, args, out_dir):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    base_env = os.environ.copy()
    base_env["MASTER_ADDR"] = "127.0.0.1"
    base_env["MASTER_PORT"] = str(port)
    base_env["WORLD_SIZE"] = str(args.world_size)
    src_dir = os.path.join(_REPO_ROOT, "src")
    base_env["CANDLE_SRC"] = src_dir
    old_pythonpath = base_env.get("PYTHONPATH")
    base_env["PYTHONPATH"] = (
        src_dir if not old_pythonpath else os.pathsep.join((src_dir, old_pythonpath))
    )

    procs = []
    for rank in range(args.world_size):
        env = dict(base_env, RANK=str(rank), LOCAL_RANK=str(rank))
        cmd = [args.candle_python or sys.executable, *_base_worker_args(args, "candle", case, out_dir)]
        procs.append(
            subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        )

    outputs = []
    try:
        outputs = _run_processes(procs, timeout=args.timeout)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()

    rank_results = _read_rank_results(out_dir, args.world_size)
    for rank, proc in enumerate(procs):
        if proc.returncode != 0 and "error" not in rank_results[rank]:
            rank_results[rank]["error"] = f"worker exit {proc.returncode}"
            rank_results[rank]["tail"] = outputs[rank].splitlines()[-16:]
    return _aggregate_rank_results("candle", case, rank_results)


def _spawn_torch_npu(case, args, out_dir):
    script_path = os.path.abspath(__file__)
    worker_args = _base_worker_args(args, "torch_npu", case, out_dir)[1:]
    cmd = _torch_npu_launch_command(
        args.torch_npu_python or sys.executable,
        script_path,
        world_size=args.world_size,
        worker_args=worker_args,
    )
    env = os.environ.copy()
    # Keep the benchmark repository importable without forcing Candle's src path
    # into the torch_npu worker environment.
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = _REPO_ROOT if not old_pythonpath else os.pathsep.join((_REPO_ROOT, old_pythonpath))
    try:
        completed = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
        )
        output = completed.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"framework": "torch_npu", "case": case, "error": "timeout"}

    rank_results = _read_rank_results(out_dir, args.world_size)
    if completed.returncode != 0:
        for row in rank_results:
            row.setdefault("error", f"torch.distributed.run exit {completed.returncode}")
            row.setdefault("tail", output.splitlines()[-24:])
    return _aggregate_rank_results("torch_npu", case, rank_results)


def _spawn_framework(framework, case, args):
    with tempfile.TemporaryDirectory(prefix=f"fsdp2_{framework}_{case}_") as tmpdir:
        if framework == "candle":
            return _spawn_candle(case, args, tmpdir)
        if framework == "torch_npu":
            return _spawn_torch_npu(case, args, tmpdir)
    raise ValueError(f"unknown framework: {framework}")


def _index_results(results):
    by_case = {}
    for row in results:
        by_case.setdefault(row["case"], {})[row["framework"]] = row
    return by_case


def _annotate_ratios(results):
    by_case = _index_results(results)
    for by_framework in by_case.values():
        candle = by_framework.get("candle")
        torch_ref = by_framework.get("torch_npu")
        if candle is None or torch_ref is None:
            continue
        if "error" in candle or "error" in torch_ref:
            continue
        metric_pairs = (
            ("fwd_ms_median", "fwd_ratio"),
            ("bwd_ms_median", "bwd_ratio"),
            ("optim_ms_median", "optim_ratio"),
            ("total_ms_median", "total_ratio"),
        )
        for metric, ratio_name in metric_pairs:
            ref_value = torch_ref.get(metric)
            candle_value = candle.get(metric)
            if ref_value and candle_value is not None:
                candle[ratio_name] = candle_value / ref_value
        ref_tput = torch_ref.get("samples_per_second")
        candle_tput = candle.get("samples_per_second")
        if ref_tput and candle_tput is not None:
            candle["throughput_ratio"] = candle_tput / ref_tput


def _max_abs_rel_diff(actual, expected):
    max_abs = 0.0
    max_rel = 0.0
    for actual_value, expected_value in zip(actual, expected):
        abs_diff = abs(float(actual_value) - float(expected_value))
        denom = max(abs(float(expected_value)), 1e-12)
        max_abs = max(max_abs, abs_diff)
        max_rel = max(max_rel, abs_diff / denom)
    return max_abs, max_rel


def _within_tolerance(abs_diff, rel_diff, atol, rtol):
    return abs_diff <= atol or rel_diff <= rtol


def _accuracy_by_rank(row):
    return {
        rank_row.get("rank"): rank_row.get("accuracy")
        for rank_row in row.get("rank_results", [])
        if "accuracy" in rank_row
    }


def _record_accuracy_diff(summary, key, abs_diff, rel_diff):
    abs_key = f"{key}_abs_diff_max"
    rel_key = f"{key}_rel_diff_max"
    max_abs_key = f"{key}_max_abs_diff"
    max_rel_key = f"{key}_max_rel_diff"
    summary[abs_key] = max(summary.get(abs_key, 0.0), abs_diff)
    summary[rel_key] = max(summary.get(rel_key, 0.0), rel_diff)
    summary[max_abs_key] = summary[abs_key]
    summary[max_rel_key] = summary[rel_key]


def _compare_accuracy_vector(summary, failures, case, rank, candle_acc,
                             torch_acc, key, label, atol, rtol):
    candle_values = candle_acc.get(key)
    torch_values = torch_acc.get(key)
    if candle_values is None or torch_values is None:
        failures.append(f"{case}/rank{rank}: missing {label} accuracy values")
        return
    if len(candle_values) != len(torch_values):
        failures.append(
            f"{case}/rank{rank}: {label} length mismatch "
            f"{len(candle_values)} != {len(torch_values)}"
        )
        return
    abs_diff, rel_diff = _max_abs_rel_diff(candle_values, torch_values)
    _record_accuracy_diff(summary, label, abs_diff, rel_diff)
    if not _within_tolerance(abs_diff, rel_diff, atol, rtol):
        failures.append(
            f"{case}/rank{rank}: {label} max diff abs={abs_diff:.6g} "
            f"rel={rel_diff:.6g} exceeds atol={atol:g}, rtol={rtol:g}"
        )


def _compare_accuracy_scalar(summary, failures, case, rank, candle_acc,
                             torch_acc, key, label, atol, rtol):
    if key not in candle_acc or key not in torch_acc:
        failures.append(f"{case}/rank{rank}: missing {label} accuracy value")
        return
    abs_diff, rel_diff = _max_abs_rel_diff([candle_acc[key]], [torch_acc[key]])
    _record_accuracy_diff(summary, label, abs_diff, rel_diff)
    if not _within_tolerance(abs_diff, rel_diff, atol, rtol):
        failures.append(
            f"{case}/rank{rank}: {label} diff abs={abs_diff:.6g} "
            f"rel={rel_diff:.6g} exceeds atol={atol:g}, rtol={rtol:g}"
        )


def _annotate_accuracy_checks(results, cases, *, atol, rtol):
    """Compare Candle and torch_npu same-network numerical results."""
    failures = []
    by_case = _index_results(results)
    scalar_metrics = (
        ("loss", "loss"),
        ("output_checksum", "output_checksum"),
        ("input_grad_checksum", "input_grad_checksum"),
        ("param_checksum_after_step", "param_checksum_after_step"),
    )
    vector_metrics = (
        ("output_values", "output"),
        ("input_grad_values", "input_grad"),
        ("param_grad_checksum_values", "param_grad_checksum"),
        ("param_checksum_after_step_values", "param_checksum_after_step"),
    )
    for case in cases:
        by_framework = by_case.get(case, {})
        candle = by_framework.get("candle")
        torch_ref = by_framework.get("torch_npu")
        if candle is None or torch_ref is None:
            continue
        if "error" in candle or "error" in torch_ref:
            continue
        candle_by_rank = _accuracy_by_rank(candle)
        torch_by_rank = _accuracy_by_rank(torch_ref)
        if not candle_by_rank or not torch_by_rank:
            failures.append(f"{case}: missing Candle or torch_npu accuracy payload")
            continue
        summary = {}
        for rank, torch_acc in sorted(torch_by_rank.items()):
            candle_acc = candle_by_rank.get(rank)
            if candle_acc is None:
                failures.append(f"{case}/rank{rank}: missing Candle accuracy payload")
                continue
            if candle_acc.get("param_names") != torch_acc.get("param_names"):
                failures.append(f"{case}/rank{rank}: parameter name mismatch")
            for key, label in scalar_metrics:
                _compare_accuracy_scalar(
                    summary, failures, case, rank, candle_acc, torch_acc,
                    key, label, atol, rtol,
                )
            for key, label in vector_metrics:
                _compare_accuracy_vector(
                    summary, failures, case, rank, candle_acc, torch_acc,
                    key, label, atol, rtol,
                )
        candle["accuracy"] = summary
    return failures


def _ratio_failures(results, cases, max_total_ratio):
    failures = []
    by_case = _index_results(results)
    for case in cases:
        by_framework = by_case.get(case, {})
        candle = by_framework.get("candle")
        torch_ref = by_framework.get("torch_npu")
        if candle is None or torch_ref is None:
            failures.append(f"{case}: missing candle or torch_npu result")
            continue
        if "error" in candle:
            failures.append(f"{case}/candle: {candle['error']}")
            continue
        if "error" in torch_ref:
            failures.append(f"{case}/torch_npu: {torch_ref['error']}")
            continue
        total_ratio = candle.get("total_ratio")
        if total_ratio is None:
            failures.append(f"{case}: missing candle/torch_npu total ratio")
        elif total_ratio > max_total_ratio:
            failures.append(
                f"{case}: total ratio {total_ratio:.2f}x > {max_total_ratio:.2f}x"
            )
    return failures


def _fmt_ms(value):
    if value is None:
        return "-"
    return f"{value:.2f}"


def _fmt_ratio(row, framework):
    if framework == "torch_npu":
        return "1.00x"
    if "total_ratio" not in row:
        return "-"
    return (
        f"f {row.get('fwd_ratio', 0.0):.2f}x / "
        f"b {row.get('bwd_ratio', 0.0):.2f}x / "
        f"o {row.get('optim_ratio', 0.0):.2f}x / "
        f"t {row['total_ratio']:.2f}x"
    )


def _print_table(results, frameworks, stream=None):
    stream = stream or sys.stdout
    by_case = _index_results(results)
    header = ["algo", "fwd ms", "bwd ms", "optim ms", "total ms", "samples/s", "ratio"]
    print(" | ".join(f"{item:>20}" for item in header), file=stream)
    print("-+-".join("-" * 20 for _ in header), file=stream)
    for case, by_framework in by_case.items():
        for framework in frameworks:
            row = by_framework.get(framework)
            algo = f"{case}/{framework}"
            if row is None or "error" in row:
                values = [algo, "err", "err", "err", "err", "err", "-"]
            else:
                values = [
                    algo,
                    _fmt_ms(row.get("fwd_ms_median")),
                    _fmt_ms(row.get("bwd_ms_median")),
                    _fmt_ms(row.get("optim_ms_median")),
                    _fmt_ms(row.get("total_ms_median")),
                    f"{row.get('samples_per_second', 0.0):.2f}",
                    _fmt_ratio(row, framework),
                ]
            print(" | ".join(f"{item:>20}" for item in values), file=stream)
    for row in results:
        if "error" in row:
            print(f"\n[{row['framework']} / {row['case']}] ERROR: {row['error']}", file=stream)
            for rank_row in row.get("rank_results", []):
                tail = rank_row.get("tail")
                if tail:
                    print(f"  rank {rank_row.get('rank')} tail:", file=stream)
                    for line in tail:
                        print(f"    | {line}", file=stream)


def _write_json_output(path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True)
    if path == "-":
        print(text)
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")


def _parse_cases(value):
    cases = [case.strip() for case in value.split(",") if case.strip()]
    unknown = [case for case in cases if case not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown}; known: {list(CASES)}")
    return cases


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--framework", default="all", choices=("all", "candle", "torch_npu"))
    parser.add_argument("--case", default=None, help="single case for worker mode")
    parser.add_argument("--cases", default=",".join(CASES.keys()))
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--out-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--candle-python", default=None)
    parser.add_argument("--torch-npu-python", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--profile-candle", action="store_true")
    parser.add_argument("--check-accuracy", action="store_true")
    parser.add_argument("--accuracy-atol", type=float, default=1e-2)
    parser.add_argument("--accuracy-rtol", type=float, default=1e-2)
    parser.add_argument("--fail-on-ratio", action="store_true")
    parser.add_argument("--max-total-ratio", type=float, default=1.0)
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.world_size <= 0:
        parser.error("--world-size must be > 0")
    if args.iters <= 0:
        parser.error("--iters must be > 0")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.max_total_ratio <= 0:
        parser.error("--max-total-ratio must be > 0")
    if args.accuracy_atol < 0:
        parser.error("--accuracy-atol must be >= 0")
    if args.accuracy_rtol < 0:
        parser.error("--accuracy-rtol must be >= 0")

    if args.worker:
        if args.case not in CASES:
            parser.error("--case must name a known case in worker mode")
        if not args.out_dir:
            parser.error("--out-dir is required in worker mode")
        if args.framework == "candle":
            _run_candle_worker(args)
        elif args.framework == "torch_npu":
            _run_torch_npu_worker(args)
        else:
            parser.error("worker mode requires --framework candle|torch_npu")
        return

    cases = _parse_cases(args.cases)
    frameworks = ["candle", "torch_npu"] if args.framework == "all" else [args.framework]
    human_stream = sys.stderr if args.json_output == "-" else sys.stdout
    print(
        "# FSDP2 Transformer bench: candle vs torch_npu "
        f"(world_size={args.world_size}, iters={args.iters}, warmup={args.warmup}, "
        f"dtype={args.dtype})",
        file=human_stream,
    )
    print(f"# Cases: {cases}", file=human_stream)

    results = []
    for case in cases:
        for framework in frameworks:
            print(f"  [{framework} / {case}] running...", flush=True, file=human_stream)
            results.append(_spawn_framework(framework, case, args))

    _annotate_ratios(results)
    failures = []
    if args.check_accuracy:
        failures.extend(
            _annotate_accuracy_checks(
                results, cases, atol=args.accuracy_atol, rtol=args.accuracy_rtol
            )
        )
    if args.fail_on_ratio:
        failures.extend(_ratio_failures(results, cases, args.max_total_ratio))

    print(file=human_stream)
    _print_table(results, frameworks, stream=human_stream)
    if failures:
        print("\n# Gate failures", file=human_stream)
        for failure in failures:
            print(f"- {failure}", file=human_stream)

    payload = {
        "cases": cases,
        "frameworks": frameworks,
        "world_size": args.world_size,
        "iters": args.iters,
        "warmup": args.warmup,
        "dtype": args.dtype,
        "lr": args.lr,
        "max_total_ratio": args.max_total_ratio,
        "check_accuracy": args.check_accuracy,
        "accuracy_atol": args.accuracy_atol,
        "accuracy_rtol": args.accuracy_rtol,
        "failures": failures,
        "results": results,
    }
    if args.json_output:
        _write_json_output(args.json_output, payload)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
