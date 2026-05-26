"""End-to-end NPU perf comparison: candle vs torch_npu.

Requires an Ascend NPU runtime and runs each framework in a separate subprocess.
The same model code is executed with `import candle as torch` or real torch plus
torch_npu; weights and inputs are initialized independently, so the table is for
latency comparison rather than numerical equivalence.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

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
    "resnet": build_resnet,
}


def _import_framework(framework):
    if framework == "candle":
        import candle as torch  # pylint: disable=import-outside-toplevel
        return torch
    if framework == "torch_npu":
        import torch  # pylint: disable=import-outside-toplevel
        import torch_npu  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
        return torch
    raise ValueError(f"unknown framework: {framework}")


def _sync(torch):
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()


def _resolve_dtype(torch, name):
    return getattr(torch, name)


def run_worker(framework, case, iters, warmup, dtype_name):
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
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        for x in inputs:
            if getattr(x, "grad", None) is not None:
                x.grad = None
    _sync(torch)

    fwd_samples = []
    bwd_samples = []
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

        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        for x in inputs:
            if getattr(x, "grad", None) is not None:
                x.grad = None

    return {
        "framework": framework,
        "case": case,
        "iters": iters,
        "fwd_ms_median": statistics.median(fwd_samples),
        "bwd_ms_median": statistics.median(bwd_samples),
        "fwd_ms_min": min(fwd_samples),
        "bwd_ms_min": min(bwd_samples),
    }


def _spawn_worker(framework, case, iters, warmup, dtype_name, python_exe=None):
    cmd = [
        python_exe or sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--framework", framework,
        "--case", case,
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--dtype", dtype_name,
    ]
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
    return last_json


def _fmt(x):
    if x is None:
        return "    -"
    return f"{x:7.2f}"


def _print_table(results, frameworks):
    by_case = {}
    for r in results:
        by_case.setdefault(r["case"], {})[r["framework"]] = r

    header = ["algo", "fwd ms", "bwd ms", "ratio"]
    print(" | ".join(f"{h:>18}" for h in header))
    print("-+-".join("-" * 18 for _ in header))

    for case, by_fw in by_case.items():
        torch_ref = by_fw.get("torch_npu")
        torch_fwd = torch_ref.get("fwd_ms_median") if torch_ref and "error" not in torch_ref else None
        torch_bwd = torch_ref.get("bwd_ms_median") if torch_ref and "error" not in torch_ref else None
        for fw in frameworks:
            r = by_fw.get(fw)
            algo = f"{case}/{fw}"
            if r is None or "error" in r:
                row = [algo, "err", "err", "-"]
            else:
                fwd = r["fwd_ms_median"]
                bwd = r["bwd_ms_median"]
                if fw == "torch_npu":
                    ratio = "1.00x"
                elif torch_fwd and torch_bwd:
                    ratio = f"f {fwd / torch_fwd:.2f}x / b {bwd / torch_bwd:.2f}x"
                else:
                    ratio = "-"
                row = [algo, _fmt(fwd), _fmt(bwd), ratio]
            print(" | ".join(f"{c:>18}" for c in row))

    for r in results:
        if "error" in r:
            print(f"\n[{r['framework']} / {r['case']}] ERROR: {r['error']}")
            tail = r.get("tail")
            if tail:
                for line in tail:
                    print(f"  | {line}")


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
    args = parser.parse_args()

    if args.worker:
        result = run_worker(args.framework, args.case, args.iters, args.warmup, args.dtype)
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

    print(f"# Perf bench: candle vs torch_npu (iters={args.iters}, warmup={args.warmup}, "
          f"dtype={args.dtype})")
    print(f"# Cases: {cases}")

    python_for = {
        "candle": args.candle_python,
        "torch_npu": args.torch_npu_python,
    }
    results = []
    for case in cases:
        for fw in frameworks:
            print(f"  [{fw} / {case}] running...", flush=True)
            results.append(_spawn_worker(
                fw, case, args.iters, args.warmup, args.dtype, python_for.get(fw)
            ))

    print()
    _print_table(results, frameworks)


if __name__ == "__main__":
    main()
