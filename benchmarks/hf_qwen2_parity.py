#!/usr/bin/env python3
"""HuggingFace Qwen2 parity runner: Candle vs torch_npu."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")


# ---------------------------------------------------------------------------
# Tiny deterministic model config
# ---------------------------------------------------------------------------


def tiny_qwen2_config_kwargs():
    """Return a small Qwen2 config that still exercises causal GQA attention."""
    return {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "use_cache": True,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


# ---------------------------------------------------------------------------
# Process launch helpers
# ---------------------------------------------------------------------------


def _pythonpath_with(*prefixes, old_pythonpath=None):
    parts = []
    seen = set()
    for path in prefixes:
        if path and path not in seen:
            parts.append(path)
            seen.add(path)
    if old_pythonpath:
        for path in old_pythonpath.split(os.pathsep):
            if path and path not in seen:
                parts.append(path)
                seen.add(path)
    return os.pathsep.join(parts)


def _worker_env(framework):
    """Build an isolated worker environment for *framework*."""
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    if framework == "candle":
        env["USE_CANDLE"] = "1"
        env["CANDLE_SRC"] = _SRC_DIR
        env["PYTHONPATH"] = _pythonpath_with(_SRC_DIR, _REPO_ROOT, old_pythonpath=old_pythonpath)
    elif framework == "torch_npu":
        env.pop("USE_CANDLE", None)
        env.pop("CANDLE_SRC", None)
        # Keep the benchmark package importable without putting Candle's src
        # ahead of the real torch package in the torch_npu worker.
        filtered = None
        if old_pythonpath:
            filtered = os.pathsep.join(
                path for path in old_pythonpath.split(os.pathsep)
                if path and os.path.abspath(path) != os.path.abspath(_SRC_DIR)
            )
        env["PYTHONPATH"] = _pythonpath_with(_REPO_ROOT, old_pythonpath=filtered)
    else:
        raise ValueError(f"unknown framework: {framework}")
    return env


def _base_worker_args(args, framework, out_dir):
    worker_args = [
        os.path.abspath(__file__),
        "--worker",
        "--framework", framework,
        "--mode", args.mode,
        "--cache-mode", args.cache_mode,
        "--device", args.device,
        "--dtype", args.dtype,
        "--seed", str(args.seed),
        "--out-dir", out_dir,
        "--accuracy-atol", str(args.accuracy_atol),
        "--accuracy-rtol", str(args.accuracy_rtol),
    ]
    if getattr(args, "pretrained_path", None):
        worker_args.extend(["--pretrained-path", args.pretrained_path])
    if getattr(args, "local_files_only", False):
        worker_args.append("--local-files-only")
    return worker_args


def _torch_npu_launch_command(python_exe, script_path, *, worker_args):
    return [
        python_exe,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        script_path,
        *worker_args,
    ]


def _write_worker_result(out_dir, payload):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{payload['framework']}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_worker_result(out_dir, framework):
    path = Path(out_dir) / f"{framework}.json"
    if not path.exists():
        return {"framework": framework, "status": "error", "error": f"missing output {path}"}
    return json.loads(path.read_text(encoding="utf-8"))


def _run_process(cmd, env, *, timeout):
    try:
        completed = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        output = completed.stdout.decode("utf-8", errors="replace")
        return completed.returncode, output
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or b"").decode("utf-8", errors="replace")
        return 124, output + "\nTIMEOUT"


def _spawn_framework(framework, args):
    with tempfile.TemporaryDirectory(prefix=f"hf_qwen2_{framework}_") as tmpdir:
        worker_args = _base_worker_args(args, framework, tmpdir)
        python_exe = sys.executable
        if framework == "candle" and args.candle_python:
            python_exe = args.candle_python
        if framework == "torch_npu" and args.torch_npu_python:
            python_exe = args.torch_npu_python

        if framework == "torch_npu":
            cmd = _torch_npu_launch_command(
                python_exe,
                os.path.abspath(__file__),
                worker_args=worker_args[1:],
            )
        else:
            cmd = [python_exe, *worker_args]
        returncode, output = _run_process(
            cmd, _worker_env(framework), timeout=args.timeout
        )
        row = _read_worker_result(tmpdir, framework)
        if returncode != 0 and row.get("status") != "ok":
            row.setdefault("status", "error")
            row.setdefault("error", f"worker exit {returncode}")
            row.setdefault("tail", output.splitlines()[-32:])
        return row


# ---------------------------------------------------------------------------
# Accuracy comparison helpers
# ---------------------------------------------------------------------------


def _max_abs_rel_diff(actual, expected):
    max_abs = 0.0
    max_rel = 0.0
    for actual_value, expected_value in zip(actual, expected):
        actual_value = float(actual_value)
        expected_value = float(expected_value)
        if math.isnan(actual_value) or math.isnan(expected_value):
            abs_diff = 0.0 if math.isnan(actual_value) and math.isnan(expected_value) else math.inf
            rel_diff = abs_diff
        else:
            abs_diff = abs(actual_value - expected_value)
            rel_diff = abs_diff / max(abs(expected_value), 1e-12)
        max_abs = max(max_abs, abs_diff)
        max_rel = max(max_rel, rel_diff)
    return max_abs, max_rel


def _is_numeric(value):
    return isinstance(value, (int, float))


def _is_numeric_vector(value):
    return isinstance(value, list) and all(_is_numeric(item) for item in value)


def _compare_metric(candle_metrics, torch_metrics, key, accuracy, failures, atol, rtol):
    if key not in candle_metrics or key not in torch_metrics:
        return
    candle_value = candle_metrics[key]
    torch_value = torch_metrics[key]
    if isinstance(candle_value, list) or isinstance(torch_value, list):
        if not isinstance(candle_value, list) or not isinstance(torch_value, list):
            failures.append(f"{key}: one side is a vector and the other is not")
            return
        if len(candle_value) != len(torch_value):
            failures.append(f"{key}: length mismatch {len(candle_value)} != {len(torch_value)}")
            return
        if not (_is_numeric_vector(candle_value) and _is_numeric_vector(torch_value)):
            matches = candle_value == torch_value
            accuracy[f"{key}_match"] = matches
            if not matches:
                failures.append(f"{key}: metadata mismatch {candle_value!r} != {torch_value!r}")
            return
        abs_diff, rel_diff = _max_abs_rel_diff(candle_value, torch_value)
        accuracy[f"{key}_max_abs_diff"] = abs_diff
        accuracy[f"{key}_max_rel_diff"] = rel_diff
    else:
        if not (_is_numeric(candle_value) and _is_numeric(torch_value)):
            matches = candle_value == torch_value
            accuracy[f"{key}_match"] = matches
            if not matches:
                failures.append(f"{key}: metadata mismatch {candle_value!r} != {torch_value!r}")
            return
        abs_diff, rel_diff = _max_abs_rel_diff([candle_value], [torch_value])
        accuracy[f"{key}_abs_diff"] = abs_diff
        accuracy[f"{key}_rel_diff"] = rel_diff
    if abs_diff > atol and rel_diff > rtol:
        failures.append(
            f"{key}: abs diff {abs_diff:.6g}, rel diff {rel_diff:.6g} "
            f"exceeds atol={atol:g}, rtol={rtol:g}"
        )


def annotate_accuracy(results, *, atol, rtol):
    """Annotate Candle row with diffs against torch_npu and return failures."""
    by_framework = {row.get("framework"): row for row in results}
    candle = by_framework.get("candle")
    torch_ref = by_framework.get("torch_npu")
    if candle is None or torch_ref is None:
        return []
    if candle.get("status") != "ok" or torch_ref.get("status") != "ok":
        return []

    candle_metrics = candle.get("metrics", {})
    torch_metrics = torch_ref.get("metrics", {})
    metric_keys = sorted(set(candle_metrics) | set(torch_metrics))
    accuracy = {}
    failures = []
    for key in metric_keys:
        _compare_metric(candle_metrics, torch_metrics, key, accuracy, failures, atol, rtol)
    candle["accuracy"] = accuracy
    return failures


# ---------------------------------------------------------------------------
# Worker internals
# ---------------------------------------------------------------------------


def _framework_diagnostics(torch, framework):
    diagnostics = {
        "framework": framework,
        "torch_module": getattr(torch, "__name__", None),
        "torch_file": getattr(torch, "__file__", None),
        "torch_version": getattr(torch, "__version__", None),
    }
    try:
        import transformers  # pylint: disable=import-outside-toplevel
        diagnostics["transformers_version"] = getattr(transformers, "__version__", None)
    except Exception as exc:  # pragma: no cover - diagnostics only
        diagnostics["transformers_error"] = repr(exc)
    return diagnostics


def _prepare_framework(framework, device):
    if framework == "candle":
        # Reuse the generic HuggingFace compatibility patch set so benchmark
        # workers exercise the same torch import stubs as compat/transformers.
        from compat.transformers.conftest import apply_all_patches  # pylint: disable=import-outside-toplevel

        apply_all_patches()
        import candle as torch  # pylint: disable=import-outside-toplevel
        sys.modules["torch"] = torch
        return torch
    if framework == "torch_npu":
        import torch  # pylint: disable=import-outside-toplevel
        import torch_npu  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import,import-error
        if hasattr(torch, "npu"):
            local_device = str(device).split(":")[-1]
            if local_device.isdigit():
                torch.npu.set_device(int(local_device))
        return torch
    raise ValueError(f"unknown framework: {framework}")


def _device_obj(torch, framework, device):
    if framework == "candle":
        return torch.Device(device)
    return torch.device(device)


def _make_inputs(torch, device, *, seq_len=8):
    input_ids = torch.arange(0, seq_len, device=device, dtype=torch.int64).reshape(1, seq_len)
    input_ids = input_ids % tiny_qwen2_config_kwargs()["vocab_size"]
    labels = (input_ids + 1) % tiny_qwen2_config_kwargs()["vocab_size"]
    attention_mask = torch.ones((1, seq_len), device=device, dtype=torch.int64)
    return input_ids, labels, attention_mask


def _tensor_values(torch, tensor, limit=32):
    if tensor is None:
        return []
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return []
    flat = flat[:min(limit, flat.numel())].to(torch.float32).to("cpu")
    return [float(value) for value in flat.numpy().reshape(-1).tolist()]


def _tensor_checksum(torch, tensor):
    if tensor is None:
        return None
    return float(tensor.detach().to(torch.float32).sum().to("cpu").item())


def _grad_values(torch, model, limit=32):
    values = []
    names = []
    for name, param in model.named_parameters():
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        names.append(name)
        values.extend(_tensor_values(torch, grad, limit=max(0, limit - len(values))))
        if len(values) >= limit:
            break
    return names, values


def _shape_numel(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _name_offset(name):
    return (sum(ord(ch) for ch in name) % 997) + 1


def _fill_tiny_qwen2_parameters(torch, model, *, dtype):
    """Fill tiny random-init Qwen2 params identically across frameworks."""
    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            values = torch.ones(shape, device=param.device, dtype=dtype)
        else:
            numel = _shape_numel(shape)
            values = torch.arange(numel, device=param.device, dtype=torch.float32)
            values = (values + _name_offset(name)) * 1e-5
            values = values.reshape(shape)
        param.data = values.to(getattr(param, "dtype", dtype))


def _build_model(torch, args, device):
    from transformers import Qwen2Config, Qwen2ForCausalLM  # pylint: disable=import-outside-toplevel

    dtype = getattr(torch, args.dtype)
    if args.pretrained_path:
        return Qwen2ForCausalLM.from_pretrained(
            args.pretrained_path,
            local_files_only=args.local_files_only,
            torch_dtype=dtype,
        ).to(device)
    config = Qwen2Config(**tiny_qwen2_config_kwargs())
    torch.manual_seed(args.seed)
    model = Qwen2ForCausalLM(config).to(device).to(dtype)
    _fill_tiny_qwen2_parameters(torch, model, dtype=dtype)
    model.eval() if args.mode == "forward" else model.train()
    return model


def _run_model(torch, model, args, device):
    input_ids, labels, attention_mask = _make_inputs(torch, device)
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": args.cache_mode != "none",
    }
    if args.mode in ("loss", "train-step"):
        kwargs["labels"] = labels

    if args.cache_mode == "dynamic":
        first_kwargs = dict(kwargs)
        if "labels" in first_kwargs:
            first_kwargs.pop("labels")
        current_tokens = 2 if args.mode in ("loss", "train-step") else 1
        first_kwargs["input_ids"] = input_ids[:, :-current_tokens]
        first_kwargs["attention_mask"] = attention_mask[:, :-current_tokens]
        first = model(**first_kwargs)
        kwargs["input_ids"] = input_ids[:, -current_tokens:]
        kwargs["attention_mask"] = attention_mask
        if "labels" in kwargs:
            kwargs["labels"] = labels[:, -current_tokens:]
        kwargs["past_key_values"] = first.past_key_values
    elif args.cache_mode == "static":
        raise NotImplementedError("static cache parity mode is not implemented yet")

    out = model(**kwargs)
    loss = getattr(out, "loss", None)
    if args.mode == "train-step":
        if loss is None:
            loss = out.logits.to(torch.float32).sum()
        loss.backward()
        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()
    metrics = {
        "logits_shape": list(out.logits.shape),
        "logits_values": _tensor_values(torch, out.logits),
        "logits_checksum": _tensor_checksum(torch, out.logits),
    }
    if loss is not None:
        metrics["loss"] = float(loss.detach().to(torch.float32).to("cpu").item())
    if args.mode == "train-step":
        grad_names, grad_values = _grad_values(torch, model)
        metrics["grad_names"] = grad_names
        metrics["grad_values"] = grad_values
    if args.cache_mode != "none" and getattr(out, "past_key_values", None) is not None:
        metrics["cache_type"] = type(out.past_key_values).__name__
    return metrics


def _run_worker(args):
    framework = args.framework
    try:
        torch = _prepare_framework(framework, args.device)
        device = _device_obj(torch, framework, args.device)
        model = _build_model(torch, args, device)
        metrics = _run_model(torch, model, args, device)
        payload = {
            "framework": framework,
            "status": "ok",
            "mode": args.mode,
            "cache_mode": args.cache_mode,
            "device": args.device,
            "dtype": args.dtype,
            "diagnostics": _framework_diagnostics(torch, framework),
            "metrics": metrics,
        }
    except Exception as exc:  # pragma: no cover - diagnostics for subprocess workers
        payload = {
            "framework": framework,
            "status": "error",
            "mode": args.mode,
            "cache_mode": args.cache_mode,
            "device": args.device,
            "dtype": args.dtype,
            "error": repr(exc),
        }
    _write_worker_result(args.out_dir, payload)
    if payload["status"] != "ok":
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_json_output(path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True)
    if path == "-":
        print(text)
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--framework", default="all", choices=("all", "candle", "torch_npu"))
    parser.add_argument("--mode", default="forward", choices=("forward", "loss", "train-step"))
    parser.add_argument("--cache-mode", default="none", choices=("none", "dynamic", "static"))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--candle-python", default=None)
    parser.add_argument("--torch-npu-python", default=None)
    parser.add_argument("--pretrained-path", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--accuracy-atol", type=float, default=5e-2)
    parser.add_argument("--accuracy-rtol", type=float, default=5e-2)
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.accuracy_atol < 0:
        parser.error("--accuracy-atol must be >= 0")
    if args.accuracy_rtol < 0:
        parser.error("--accuracy-rtol must be >= 0")
    if args.iters <= 0:
        parser.error("--iters must be > 0")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.worker:
        if not args.out_dir:
            parser.error("--out-dir is required in worker mode")
        if args.framework == "all":
            parser.error("worker mode requires --framework candle|torch_npu")
        _run_worker(args)
        return

    frameworks = ["candle", "torch_npu"] if args.framework == "all" else [args.framework]
    results = [_spawn_framework(framework, args) for framework in frameworks]
    failures = annotate_accuracy(results, atol=args.accuracy_atol, rtol=args.accuracy_rtol)
    payload = {
        "frameworks": frameworks,
        "mode": args.mode,
        "cache_mode": args.cache_mode,
        "device": args.device,
        "dtype": args.dtype,
        "accuracy_atol": args.accuracy_atol,
        "accuracy_rtol": args.accuracy_rtol,
        "failures": failures,
        "results": results,
    }
    if args.json_output:
        _write_json_output(args.json_output, payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)
    for row in results:
        if row.get("status") != "ok":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
