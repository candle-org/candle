<div align="center">

<img src="assets/logo.png" alt="Candle" width="200">

# Candle

**C**ANN c**an** han**dle** — Not as bright as a torch, but light enough to carry anywhere.

A pure-Python deep learning framework that runs your PyTorch code — no rewrite needed.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![CI](https://github.com/candle-org/candle/actions/workflows/ci.yaml/badge.svg)](https://github.com/candle-org/candle/actions)
[![GitHub stars](https://img.shields.io/github/stars/candle-org/candle?style=social)](https://github.com/candle-org/candle)

[Getting Started](#getting-started) | [Why Candle](#why-candle) | [Backends](#backends) | [Roadmap](#roadmap) | [Contributing](#contributing)

**[English](README.md)** | [中文](README_zh.md)

</div>

---

## Why Candle

PyTorch is powerful — but it's also **2GB+ of C++ binaries**, hard to install on edge devices, and locked to CUDA. Candle takes a different approach:

| | PyTorch | Candle |
|---|---|---|
| Install size | ~2 GB | ~10 MB |
| Build from source | C++ toolchain required | `pip install candle-python` |
| Ascend NPU | Community fork | First-class ACLNN kernels |
| Apple MPS | Partial | Native Metal shaders |
| Run existing `import torch` code | — | Zero-change drop-in |

## Getting Started

### Install

```bash
pip install candle-python
```

Or install from source:

```bash
git clone https://github.com/candle-org/candle.git
cd candle
pip install -e ".[test]"
```

Ascend NPU users must install CANN first and follow [docs/install-npu.md](docs/install-npu.md) before running device code.

### Write code the way you already know

```python
import candle as torch
import candle.nn as nn

# Tensors, autograd, nn — all the APIs you're used to
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

x = torch.randn(2, 784, requires_grad=True)
out = model(x)
out.sum().backward()
print(x.grad.shape)  # (2, 784)
```

### Ascend NPU quick start (0.1 GA path)

See [docs/install-npu.md](docs/install-npu.md) for the full environment prerequisites. On a supported Ascend 910B host:

```bash
if [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:${PYTHONPATH:-}
python - <<'PY'
import candle as torch

assert torch.npu.is_available(verbose=True)
x = torch.randn((4, 8), device='npu')
w = torch.randn((8, 2), device='npu', requires_grad=True)
y = torch.matmul(x, w)
y.sum().backward()
print(torch.npu.get_device_name())
print(y.device)
PY
```

### Or just `import torch` — seriously

Candle ships an import hook. Existing PyTorch code runs **without changing a single line**:

```python
import torch                    # resolved to candle
import torch.nn.functional as F # resolved to candle.nn.functional

x = torch.randn(3, 4)
y = F.relu(x)
```

**How it works:**

| `USE_CANDLE` env var | PyTorch installed? | `import torch` gives you |
|---|---|---|
| `1` / `true` / `yes` | doesn't matter | Candle |
| `0` / `false` / `no` | doesn't matter | PyTorch (or ImportError) |
| *not set* | No | Candle |
| *not set* | Yes | PyTorch |

If PyTorch isn't installed, Candle picks up automatically. If both are installed, set `USE_CANDLE=1`:

```bash
USE_CANDLE=1 python train.py
```

## Backends

Candle runs on multiple hardware backends with a single API:

```
candle.device("cpu")    # NumPy — works everywhere
candle.device("cuda")   # NVIDIA GPU
candle.device("mps")    # Apple Silicon GPU (Metal)
candle.device("npu")    # Huawei Ascend (ACLNN)
```

### Ascend NPU — First-Class Support

Candle's current `0.1.x` GA target is a single-card Ascend 910B training path. Install and runtime prerequisites are documented in [docs/install-npu.md](docs/install-npu.md), and the validated release surface is summarized in [docs/support-matrix.md](docs/support-matrix.md).

Unlike wrapper libraries, Candle calls **ACLNN large kernels directly** via ctypes — no framework overhead, no Python-to-C++ bridge:

```python
import candle as torch

x = torch.randn(1024, 1024, device="npu")
y = torch.matmul(x, x.T)  # runs native ACLNN kernel
```

## Features

- **Pure Python** — No C++ extensions. Install in seconds, debug in Python, deploy anywhere.
- **PyTorch-Compatible API** — `Tensor`, `nn.Module`, `autograd`, `optim` — the full stack.
- **`import torch` Drop-in** — Built-in import hook. Zero code changes for existing projects.
- **Multi-Backend** — CPU, CUDA, Apple MPS, Ascend NPU from one codebase.
- **Ascend NPU Native** — Direct ACLNN kernel integration, not a bolted-on afterthought.
- **Agentic AI Ready** — Lightweight enough to embed in AI agent runtimes.

## Roadmap

Most frameworks stop at tensors. Candle doesn't — the end goal is a **self-hosting agentic kernel**: a system that deploys local models on Ascend, then uses those same models to debug, optimize, and evolve itself.

### Phase 1 — Foundation (current)

> A PyTorch-compatible pure-Python framework with multi-backend support.

- [x] Core tensor ops, autograd, `nn.Module`, optimizers
- [x] CPU backend (NumPy)
- [x] Ascend NPU backend (ACLNN native kernels)
- [x] Apple MPS backend (Metal shaders)
- [x] `import torch` zero-change drop-in hook
- [ ] CUDA backend
- [ ] `torch.compile` graph-mode acceleration
- [ ] Distributed training (`DistributedDataParallel`)
- [ ] Full TorchVision / TorchAudio model compatibility

### Phase 2 — Cognitive Runtime

> Local model deployment becomes a first-class primitive, not an afterthought.

- [ ] Local model loading, quantization & serving on Ascend
- [ ] Role-based model router (debug model, generation model, judge model)
- [ ] Multi-model inference policy (local-first, cloud fallback)
- [ ] Self-hosted reasoning & tool-use runtime

### Phase 3 — Agentic Kernel

> The framework gains the ability to observe, diagnose, and act on itself.

- [ ] **Dev Layer** — bug detection, repro construction, fix suggestions (powered by local debug model)
- [ ] **Bootstrap Layer** — candidate generation, distillation, config search (powered by local generation model)
- [ ] **ModelOps Layer** — evaluation, promotion, rollback, lineage tracking (powered by local judge model)

### Phase 4 — Self-Hosting

> Local models are no longer just managed artifacts — they become the execution engine of the kernel itself.

- [ ] Local-first agentic execution: Dev / Bootstrap / ModelOps agents default to self-deployed models
- [ ] Continuous self-improvement loop: trace → evaluate → generate fix → test → promote
- [ ] Model-as-kernel: the system uses its own models to improve its own models

```
┌─────────────────────────────────────────┐
│            Applications                 │
│   train · debug · infer · deploy        │
├─────────────────────────────────────────┤
│          Agentic Kernel                 │
│   Dev Layer · Bootstrap · ModelOps      │
├─────────────────────────────────────────┤
│        Cognitive Runtime                │
│   Local Model RT · Router · Policy      │
├─────────────────────────────────────────┤
│      Intelligence Substrate             │
│   TraceStore · Evaluator · Registry     │
├─────────────────────────────────────────┤
│           Foundation                    │
│   tensor · autograd · nn · compiler     │
│   CPU · CUDA · MPS · Ascend NPU        │
└─────────────────────────────────────────┘
```

See [docs/support-matrix.md](docs/support-matrix.md) for the full `0.1.x` op support matrix.

## Used By

> Building something with Candle? [Open an issue](https://github.com/candle-org/candle/issues/new?labels=ecosystem&title=Add+my+project+to+Used+By) and we'll add you here!

<!--
<div align="center">
<a href="https://example.com"><img src="https://img.shields.io/badge/YourProject-blue" height="30"></a>
</div>
-->

## Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=candle-org/candle&type=Date)](https://star-history.com/#candle-org/candle&Date)

</div>

## Contributing

```bash
# Clone and install in dev mode
git clone https://github.com/candle-org/candle.git
cd candle
pip install -e ".[test]"

# Run tests
pytest tests/cpu/ tests/contract/ -v --tb=short

# Lint
pip install -e ".[lint]"
pylint src/candle --rcfile=.github/pylint.conf
```

We welcome contributions! Whether it's new ops, backend support, bug fixes, or docs — open an issue or submit a PR.

## License

[MIT](LICENSE)
