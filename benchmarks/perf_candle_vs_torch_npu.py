"""End-to-end NPU perf comparison: candle vs torch_npu.

Requires an Ascend NPU runtime and runs each framework in a separate subprocess.
The same model code is executed with `import candle as torch` or real torch plus
torch_npu; weights and inputs are initialized independently, so the table is for
latency comparison rather than numerical equivalence.
"""

import argparse
import json
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmarks.npu_perf_gates import summarize_samples


def build_mlp(torch, device, dtype):
    nn = torch.nn
    f = torch.nn.functional

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(1024, 4096)
            self.l2 = nn.Linear(4096, 4096)
            self.l3 = nn.Linear(4096, 1024)

        def forward(self, x):
            x = f.gelu(self.l1(x))
            x = f.gelu(self.l2(x))
            return self.l3(x)

    model = MLP().to(device).to(dtype)
    x = torch.randn(32, 1024, device=device, dtype=dtype, requires_grad=True)
    return model, (x,)


def build_xfmr(torch, device, dtype):
    nn = torch.nn
    f = torch.nn.functional

    b, s, h, heads = 2, 128, 512, 8
    head_dim = h // heads

    class XfmrBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(h)
            self.norm2 = nn.LayerNorm(h)
            self.wq = nn.Linear(h, h)
            self.wk = nn.Linear(h, h)
            self.wv = nn.Linear(h, h)
            self.wo = nn.Linear(h, h)
            self.ff1 = nn.Linear(h, 4 * h)
            self.ff2 = nn.Linear(4 * h, h)

        def forward(self, x):
            y = self.norm1(x)
            q = self.wq(y).reshape(b, s, heads, head_dim).transpose(1, 2).contiguous()
            k = self.wk(y).reshape(b, s, heads, head_dim).transpose(1, 2).contiguous()
            v = self.wv(y).reshape(b, s, heads, head_dim).transpose(1, 2).contiguous()
            attn = torch.matmul(q, k.transpose(-2, -1).contiguous())
            attn = f.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().reshape(b, s, h)
            x = x + self.wo(out)
            z = self.norm2(x)
            z = f.gelu(self.ff1(z))
            return x + self.ff2(z)

    model = XfmrBlock().to(device).to(dtype)
    x = torch.randn(b, s, h, device=device, dtype=dtype, requires_grad=True)
    return model, (x,)


def build_sdpa(torch, device, dtype):
    nn = torch.nn
    f = torch.nn.functional

    b, heads, s, head_dim = 2, 8, 128, 64

    class SDPABlock(nn.Module):
        def forward(self, q, k, v):
            return f.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    model = SDPABlock().to(device).to(dtype)
    q = torch.randn(b, heads, s, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(b, heads, s, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(b, heads, s, head_dim, device=device, dtype=dtype, requires_grad=True)
    return model, (q, k, v)


def build_mha_sdpa(torch, device, dtype):
    nn = torch.nn

    b, s, h, heads = 2, 128, 512, 8

    class MhaSdpBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.mha = nn.MultiheadAttention(h, heads, batch_first=True, dropout=0.0)

        def forward(self, x):
            out, _ = self.mha(x, x, x, need_weights=False)
            return out

    model = MhaSdpBlock().to(device).to(dtype)
    x = torch.randn(b, s, h, device=device, dtype=dtype, requires_grad=True)
    return model, (x,)


def _build_xfmr_sdpa(torch, device, dtype, b, s, h, heads):
    nn = torch.nn
    f = torch.nn.functional

    class XfmrSdpaBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(h)
            self.attn = nn.MultiheadAttention(h, heads, batch_first=True, dropout=0.0)
            self.norm2 = nn.LayerNorm(h)
            self.ff1 = nn.Linear(h, 4 * h)
            self.ff2 = nn.Linear(4 * h, h)

        def forward(self, x):
            y = self.norm1(x)
            attn_out, _ = self.attn(y, y, y, need_weights=False)
            x = x + attn_out
            z = self.norm2(x)
            z = f.gelu(self.ff1(z))
            return x + self.ff2(z)

    model = XfmrSdpaBlock().to(device).to(dtype)
    x = torch.randn(b, s, h, device=device, dtype=dtype, requires_grad=True)
    return model, (x,)


def build_xfmr_sdpa(torch, device, dtype):
    return _build_xfmr_sdpa(torch, device, dtype, b=2, s=128, h=512, heads=8)


def build_xfmr_sdpa_large(torch, device, dtype):
    return _build_xfmr_sdpa(torch, device, dtype, b=2, s=256, h=1024, heads=16)


def build_resnet(torch, device, dtype):
    nn = torch.nn
    f = torch.nn.functional

    class ResBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(64)

        def forward(self, x):
            h = f.relu(self.bn1(self.conv1(x)))
            h = self.bn2(self.conv2(h))
            return f.relu(h + x)

    model = ResBlock().to(device).to(dtype)
    x = torch.randn(8, 64, 32, 32, device=device, dtype=dtype, requires_grad=True)
    return model, (x,)


CASES = {
    "mlp": build_mlp,
    "xfmr": build_xfmr,
    "sdpa": build_sdpa,
    "mha_sdpa": build_mha_sdpa,
    "xfmr_sdpa": build_xfmr_sdpa,
    "xfmr_sdpa_large": build_xfmr_sdpa_large,
    "resnet": build_resnet,
}


def _import_framework(framework):
    if framework == "candle":
        import candle as torch  # pylint: disable=import-outside-toplevel
        return torch
    if framework == "torch_npu":
        import torch  # pylint: disable=import-outside-toplevel
        import torch_npu  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import,import-error
        return torch
    raise ValueError(f"unknown framework: {framework}")


def _sync(torch):
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()


def _resolve_dtype(torch, name):
    return getattr(torch, name)


def _clear_grads(model, inputs):
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    for x in inputs:
        if getattr(x, "grad", None) is not None:
            x.grad = None


def _profile_top_rows(prof, row_limit):
    rows = list(prof.key_averages())
    rows.sort(key=lambda row: (row.cpu_time_total, row.self_cpu_time_total, row.key), reverse=True)
    out = []
    for row in rows[: max(0, int(row_limit))]:
        out.append(
            {
                "name": row.key,
                "device": row.device_type,
                "count": row.count,
                "self_us": row.self_cpu_time_total,
                "total_us": row.cpu_time_total,
                "avg_us": row.cpu_time,
            }
        )
    return out


def _profile_candle_step(torch, model, inputs, case, profile_dir, profile_topk):
    from candle import profiler  # pylint: disable=import-outside-toplevel

    with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]) as prof:
        with profiler.record_function(f"{case}.forward"):
            out = model(*inputs)
            loss = out.sum()
            _sync(torch)
        with profiler.record_function(f"{case}.backward"):
            loss.backward()
            _sync(torch)

    trace_path = None
    if profile_dir:
        os.makedirs(profile_dir, exist_ok=True)
        trace_path = os.path.join(profile_dir, f"{case}-candle-trace.json")
        prof.export_chrome_trace(trace_path)

    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=max(0, int(profile_topk)))
    return {
        "trace_path": trace_path,
        "table": table,
        "top": _profile_top_rows(prof, profile_topk),
    }


def _profile_candle_graph_step(torch, graph, case, profile_dir, profile_topk):
    from candle import profiler  # pylint: disable=import-outside-toplevel

    with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]) as prof:
        with profiler.record_function(f"{case}.graph_step"):
            graph.replay()
            _sync(torch)

    trace_path = None
    if profile_dir:
        os.makedirs(profile_dir, exist_ok=True)
        trace_path = os.path.join(profile_dir, f"{case}-candle-graph-trace.json")
        prof.export_chrome_trace(trace_path)

    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=max(0, int(profile_topk)))
    return {
        "trace_path": trace_path,
        "table": table,
        "top": _profile_top_rows(prof, profile_topk),
    }


def _capture_graph_step(torch, model, inputs):
    if not hasattr(torch.npu, "NPUGraph") or not hasattr(torch.npu, "graph"):
        raise RuntimeError("torch.npu.NPUGraph/graph is not available")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = model(*inputs)
        loss = out.sum()
        loss.backward()
    _sync(torch)
    return graph


def _summarize_graph_step_result(framework, case, iters, total_samples, profile):
    total_summary = summarize_samples(total_samples)
    result = {
        "framework": framework,
        "case": case,
        "iters": iters,
        "graph_step": True,
        "fwd_ms_median": None,
        "bwd_ms_median": None,
        "total_ms_median": total_summary["median_ms"],
        "fwd_ms_min": None,
        "bwd_ms_min": None,
        "total_ms_min": total_summary["min_ms"],
        "fwd_ms_p10": None,
        "bwd_ms_p10": None,
        "total_ms_p10": total_summary["p10_ms"],
        "fwd_ms_p90": None,
        "bwd_ms_p90": None,
        "total_ms_p90": total_summary["p90_ms"],
    }
    if profile is not None:
        result["profile"] = profile
    return result


def run_worker(framework, case, iters, warmup, dtype_name, profile_candle=False,
               profile_dir=None, profile_topk=20, graph_step=False):
    torch = _import_framework(framework)
    dtype = _resolve_dtype(torch, dtype_name)

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return {"error": f"NPU not available under {framework}"}

    device = "npu:0"
    builder = CASES[case]
    model, inputs = builder(torch, device, dtype)

    for _ in range(warmup):
        out = model(*inputs)
        loss = out.sum()
        loss.backward()
        _clear_grads(model, inputs)
    _sync(torch)

    profile = None
    if graph_step:
        _clear_grads(model, inputs)
        graph = _capture_graph_step(torch, model, inputs)
        total_samples = []
        if framework == "candle" and profile_candle:
            profile = _profile_candle_graph_step(torch, graph, case, profile_dir, profile_topk)
        for _ in range(iters):
            _sync(torch)
            t0 = time.perf_counter()
            graph.replay()
            _sync(torch)
            t1 = time.perf_counter()
            total_samples.append((t1 - t0) * 1000.0)
        return _summarize_graph_step_result(framework, case, iters, total_samples, profile)

    if framework == "candle" and profile_candle:
        profile = _profile_candle_step(torch, model, inputs, case, profile_dir, profile_topk)
        _clear_grads(model, inputs)
        _sync(torch)

    fwd_samples = []
    bwd_samples = []
    total_samples = []
    for _ in range(iters):
        _sync(torch)
        t0 = time.perf_counter()
        out = model(*inputs)
        loss = out.sum()
        _sync(torch)
        t1 = time.perf_counter()
        loss.backward()
        _sync(torch)
        t2 = time.perf_counter()

        fwd_samples.append((t1 - t0) * 1000.0)
        bwd_samples.append((t2 - t1) * 1000.0)
        total_samples.append((t2 - t0) * 1000.0)
        _clear_grads(model, inputs)

    fwd_summary = summarize_samples(fwd_samples)
    bwd_summary = summarize_samples(bwd_samples)
    total_summary = summarize_samples(total_samples)
    result = {
        "framework": framework,
        "case": case,
        "iters": iters,
        "fwd_ms_median": fwd_summary["median_ms"],
        "bwd_ms_median": bwd_summary["median_ms"],
        "total_ms_median": total_summary["median_ms"],
        "fwd_ms_min": fwd_summary["min_ms"],
        "bwd_ms_min": bwd_summary["min_ms"],
        "total_ms_min": total_summary["min_ms"],
        "fwd_ms_p10": fwd_summary["p10_ms"],
        "bwd_ms_p10": bwd_summary["p10_ms"],
        "total_ms_p10": total_summary["p10_ms"],
        "fwd_ms_p90": fwd_summary["p90_ms"],
        "bwd_ms_p90": bwd_summary["p90_ms"],
        "total_ms_p90": total_summary["p90_ms"],
    }
    if profile is not None:
        result["profile"] = profile
    return result


def _spawn_worker(framework, case, iters, warmup, dtype_name, python_exe=None,
                  profile_candle=False, profile_dir=None, profile_topk=20,
                  graph_step=False):
    cmd = [
        python_exe or sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--framework", framework,
        "--case", case,
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--dtype", dtype_name,
        "--profile-topk", str(profile_topk),
    ]
    if framework == "candle" and profile_candle:
        cmd.append("--profile-candle")
    if graph_step:
        cmd.append("--graph-step")
    if profile_dir is not None:
        cmd.extend(("--profile-dir", profile_dir))

    env = os.environ.copy()
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    old_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_dir if not old_path else os.pathsep.join((src_dir, old_path))
    try:
        out = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT, timeout=600)
    except subprocess.CalledProcessError as exc:
        return {
            "framework": framework,
            "case": case,
            "error": f"worker exit {exc.returncode}",
            "tail": exc.output.decode(errors="replace").splitlines()[-12:],
        }
    except subprocess.TimeoutExpired:
        return {"framework": framework, "case": case, "error": "timeout"}

    last_json = None
    for line in out.decode(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                last_json = json.loads(line)
            except json.JSONDecodeError:
                continue
    if last_json is None:
        return {"framework": framework, "case": case, "error": "no json from worker"}
    if isinstance(last_json, dict):
        last_json.setdefault("framework", framework)
        last_json.setdefault("case", case)
    return last_json


def _fmt(x):
    if x is None:
        return "    -"
    return f"{x:7.2f}"


def _index_results(results):
    by_case = {}
    for r in results:
        by_case.setdefault(r["case"], {})[r["framework"]] = r
    return by_case


def _annotate_ratios(results):
    by_case = _index_results(results)
    for by_fw in by_case.values():
        candle = by_fw.get("candle")
        torch_ref = by_fw.get("torch_npu")
        if not candle or not torch_ref:
            continue
        if "error" in candle or "error" in torch_ref:
            continue
        torch_fwd = torch_ref.get("fwd_ms_median")
        torch_bwd = torch_ref.get("bwd_ms_median")
        torch_total = torch_ref.get("total_ms_median")
        candle_fwd = candle.get("fwd_ms_median")
        candle_bwd = candle.get("bwd_ms_median")
        candle_total = candle.get("total_ms_median")
        if torch_fwd and candle_fwd is not None:
            candle["fwd_ratio"] = candle_fwd / torch_fwd
        if torch_bwd and candle_bwd is not None:
            candle["bwd_ratio"] = candle_bwd / torch_bwd
        if torch_total and candle_total is not None:
            candle["total_ratio"] = candle_total / torch_total


def _ratio_failures(results, cases, max_fwd_ratio, max_bwd_ratio, max_total_ratio):
    failures = []
    by_case = _index_results(results)
    for case in cases:
        by_fw = by_case.get(case, {})
        candle = by_fw.get("candle")
        torch_ref = by_fw.get("torch_npu")
        if candle is None or torch_ref is None:
            failures.append(f"{case}: missing candle or torch_npu result for ratio gate")
            continue
        if "error" in candle:
            failures.append(f"{case}/candle: {candle['error']}")
            continue
        if "error" in torch_ref:
            failures.append(f"{case}/torch_npu: {torch_ref['error']}")
            continue
        fwd_ratio = candle.get("fwd_ratio")
        bwd_ratio = candle.get("bwd_ratio")
        total_ratio = candle.get("total_ratio")
        if total_ratio is None:
            failures.append(f"{case}: unable to compute candle/torch_npu total ratio")
            continue
        if not (candle.get("graph_step") or torch_ref.get("graph_step")):
            if fwd_ratio is None or bwd_ratio is None:
                failures.append(f"{case}: unable to compute candle/torch_npu ratio")
                continue
            if fwd_ratio > max_fwd_ratio:
                failures.append(f"{case}: fwd ratio {fwd_ratio:.2f}x > {max_fwd_ratio:.2f}x")
            if bwd_ratio > max_bwd_ratio:
                failures.append(f"{case}: bwd ratio {bwd_ratio:.2f}x > {max_bwd_ratio:.2f}x")
        if total_ratio > max_total_ratio:
            failures.append(f"{case}: total ratio {total_ratio:.2f}x > {max_total_ratio:.2f}x")
    return failures


def _print_table(results, frameworks, stream=None):
    stream = stream or sys.stdout
    by_case = _index_results(results)

    header = ["algo", "fwd ms", "bwd ms", "total ms", "ratio"]
    print(" | ".join(f"{h:>18}" for h in header), file=stream)
    print("-+-".join("-" * 18 for _ in header), file=stream)

    for case, by_fw in by_case.items():
        for fw in frameworks:
            r = by_fw.get(fw)
            algo = f"{case}/{fw}"
            if r is None or "error" in r:
                row = [algo, "err", "err", "err", "-"]
            else:
                fwd = r["fwd_ms_median"]
                bwd = r["bwd_ms_median"]
                total = r.get("total_ms_median")
                if fw == "torch_npu":
                    ratio = "1.00x"
                elif "fwd_ratio" in r and "bwd_ratio" in r:
                    total_part = f" / t {r['total_ratio']:.2f}x" if "total_ratio" in r else ""
                    ratio = f"f {r['fwd_ratio']:.2f}x / b {r['bwd_ratio']:.2f}x{total_part}"
                elif "total_ratio" in r:
                    ratio = f"t {r['total_ratio']:.2f}x"
                else:
                    ratio = "-"
                row = [algo, _fmt(fwd), _fmt(bwd), _fmt(total), ratio]
            print(" | ".join(f"{c:>18}" for c in row), file=stream)

    for r in results:
        if "error" in r:
            print(f"\n[{r['framework']} / {r['case']}] ERROR: {r['error']}", file=stream)
            tail = r.get("tail")
            if tail:
                for line in tail:
                    print(f"  | {line}", file=stream)


def _print_profile_tables(results, stream=None):
    stream = stream or sys.stdout
    for result in results:
        profile = result.get("profile")
        if not profile:
            continue
        print(f"\n# Candle profiler: {result['case']}", file=stream)
        trace_path = profile.get("trace_path")
        if trace_path:
            print(f"# Trace: {trace_path}", file=stream)
        print(profile["table"], file=stream)


def _write_json_output(path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True)
    if path == "-":
        print(text)
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")


def _worker_failures(results):
    failures = []
    for result in results:
        if "error" in result:
            failures.append(f"{result.get('case', '-')}/{result.get('framework', '-')}: {result['error']}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--framework", default="all", help="candle|torch_npu|all")
    parser.add_argument("--case", default=None, help="single case for worker mode")
    parser.add_argument("--cases", default=",".join(CASES.keys()),
                        help="comma-separated case list for orchestrator")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--candle-python", default=None,
                        help="Python executable for candle workers; defaults to current Python")
    parser.add_argument("--torch-npu-python", default=None,
                        help="Python executable for torch_npu workers; defaults to current Python")
    parser.add_argument("--fail-on-ratio", action="store_true",
                        help="exit nonzero when candle is slower than the configured torch_npu ratio")
    parser.add_argument("--max-fwd-ratio", type=float, default=1.0,
                        help="maximum allowed candle/torch_npu forward ratio for --fail-on-ratio")
    parser.add_argument("--max-bwd-ratio", type=float, default=1.0,
                        help="maximum allowed candle/torch_npu backward ratio for --fail-on-ratio")
    parser.add_argument("--max-total-ratio", type=float, default=0.99,
                        help="maximum allowed candle/torch_npu total ratio for --fail-on-ratio")
    parser.add_argument("--json-output", default=None,
                        help="write final results payload to this path, or '-' for stdout")
    parser.add_argument("--profile-candle", action="store_true",
                        help="run one extra profiled Candle iteration after warmup")
    parser.add_argument("--profile-dir", default=None,
                        help="directory for Candle profiler Chrome trace JSON files")
    parser.add_argument("--profile-topk", type=int, default=20,
                        help="number of Candle profiler rows to keep and print")
    parser.add_argument("--print-candle-profile-table", action="store_true",
                        help="print Candle profiler key_averages tables for profiled workers")
    parser.add_argument("--graph-step", action="store_true",
                        help="capture one forward+backward step with torch.npu.NPUGraph and time replay")
    parser.add_argument("--candle-graph-step", action="store_true",
                        help="capture only Candle workers with torch.npu.NPUGraph; compare against torch_npu eager")
    args = parser.parse_args()

    if args.max_fwd_ratio <= 0:
        parser.error("--max-fwd-ratio must be > 0")
    if args.max_bwd_ratio <= 0:
        parser.error("--max-bwd-ratio must be > 0")
    if args.max_total_ratio <= 0:
        parser.error("--max-total-ratio must be > 0")

    if args.worker:
        result = run_worker(
            args.framework,
            args.case,
            args.iters,
            args.warmup,
            args.dtype,
            profile_candle=args.profile_candle,
            profile_dir=args.profile_dir,
            profile_topk=args.profile_topk,
            graph_step=args.graph_step or (args.candle_graph_step and args.framework == "candle"),
        )
        print(json.dumps(result))
        return

    if args.framework == "all":
        frameworks = ["candle", "torch_npu"]
    else:
        frameworks = [args.framework]
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in cases if c not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown}; known: {list(CASES)}")

    human_stream = sys.stderr if args.json_output == "-" else sys.stdout
    print(f"# Perf bench: candle vs torch_npu (iters={args.iters}, warmup={args.warmup}, "
          f"dtype={args.dtype})", file=human_stream)
    print(f"# Cases: {cases}", file=human_stream)

    python_for = {
        "candle": args.candle_python,
        "torch_npu": args.torch_npu_python,
    }
    results = []
    for case in cases:
        for fw in frameworks:
            print(f"  [{fw} / {case}] running...", flush=True, file=human_stream)
            results.append(_spawn_worker(
                fw,
                case,
                args.iters,
                args.warmup,
                args.dtype,
                python_for.get(fw),
                profile_candle=args.profile_candle,
                profile_dir=args.profile_dir,
                profile_topk=args.profile_topk,
                graph_step=args.graph_step or (args.candle_graph_step and fw == "candle"),
            ))

    _annotate_ratios(results)
    failures = _worker_failures(results)
    if args.fail_on_ratio:
        failures.extend(_ratio_failures(
            results,
            cases,
            args.max_fwd_ratio,
            args.max_bwd_ratio,
            args.max_total_ratio,
        ))

    print(file=human_stream)
    _print_table(results, frameworks, stream=human_stream)
    if args.print_candle_profile_table:
        _print_profile_tables(results, stream=human_stream)

    if failures:
        print("\n# Gate failures", file=human_stream)
        for failure in failures:
            print(f"- {failure}", file=human_stream)

    payload = {
        "iters": args.iters,
        "warmup": args.warmup,
        "dtype": args.dtype,
        "cases": cases,
        "frameworks": frameworks,
        "max_fwd_ratio": args.max_fwd_ratio,
        "max_bwd_ratio": args.max_bwd_ratio,
        "max_total_ratio": args.max_total_ratio,
        "graph_step": args.graph_step,
        "candle_graph_step": args.candle_graph_step,
        "failures": failures,
        "results": results,
    }
    if args.json_output:
        _write_json_output(args.json_output, payload)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
