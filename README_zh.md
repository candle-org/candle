<div align="center">

<img src="assets/logo.png" alt="Candle" width="200">

# Candle

**C**ANN c**an** han**dle** — 没有火炬那么亮，但足够轻，走到哪带到哪。

纯 Python 深度学习框架，直接运行你的 PyTorch 代码 — 无需任何改动。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![CI](https://github.com/candle-org/candle/actions/workflows/ci.yaml/badge.svg)](https://github.com/candle-org/candle/actions)
[![GitHub stars](https://img.shields.io/github/stars/candle-org/candle?style=social)](https://github.com/candle-org/candle)

[快速开始](#快速开始) | [为什么选 Candle](#为什么选-candle) | [硬件后端](#硬件后端) | [路线图](#路线图) | [参与贡献](#参与贡献)

[English](README.md) | **[中文](README_zh.md)**

</div>

---

## 为什么选 Candle

PyTorch 很强大 — 但它也意味着 **2GB+ 的 C++ 二进制文件**、边缘设备上装不动、被 CUDA 绑死。Candle 走了一条不同的路：

| | PyTorch | Candle |
|---|---|---|
| 安装体积 | ~2 GB | ~10 MB |
| 源码编译 | 需要 C++ 工具链 | `pip install candle-python` |
| 昇腾 NPU | 社区分支 | 原生 ACLNN 大算子 |
| Apple MPS | 部分支持 | 原生 Metal 着色器 |
| 运行现有 `import torch` 代码 | — | 零改动直接跑 |

## 快速开始

### 安装

```bash
pip install candle-python
```

就这一行。不需要 CUDA 工具包，不需要编译器，不用等 10 分钟。

### 用你熟悉的方式写代码

```python
import candle as torch
import candle.nn as nn

# Tensor、autograd、nn —— 你习惯的 API 全都有
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

### 或者直接 `import torch` — 真的可以

Candle 内置了 import hook，现有 PyTorch 代码**一行都不用改**就能跑：

```python
import torch                    # 自动解析为 candle
import torch.nn.functional as F # 自动解析为 candle.nn.functional

x = torch.randn(3, 4)
y = F.relu(x)
```

**工作原理：**

| `USE_CANDLE` 环境变量 | 是否安装了 PyTorch | `import torch` 得到的是 |
|---|---|---|
| `1` / `true` / `yes` | 无所谓 | Candle |
| `0` / `false` / `no` | 无所谓 | PyTorch（或 ImportError） |
| *未设置* | 未安装 | Candle |
| *未设置* | 已安装 | PyTorch |

如果没装 PyTorch，Candle 自动接管。如果两个都装了，设置 `USE_CANDLE=1` 即可：

```bash
USE_CANDLE=1 python train.py
```

## 硬件后端

一套 API，多种硬件后端：

```
candle.device("cpu")    # NumPy —— 随处可用
candle.device("cuda")   # NVIDIA GPU
candle.device("mps")    # Apple Silicon GPU (Metal)
candle.device("npu")    # 华为昇腾 (ACLNN)
```

### 昇腾 NPU — 一等公民

Candle 不是套壳，而是通过 ctypes **直接调用 ACLNN 大算子** — 没有框架开销，没有 Python-to-C++ 桥接层：

```python
import candle as torch

x = torch.randn(1024, 1024, device="npu")
y = torch.matmul(x, x.T)  # 运行原生 ACLNN 算子
```

## 特性

- **纯 Python** — 没有 C++ 扩展。秒级安装，Python 里调试，部署到任何地方。
- **PyTorch 兼容 API** — `Tensor`、`nn.Module`、`autograd`、`optim` — 全家桶。
- **`import torch` 直接替换** — 内置 import hook，现有项目零改动。
- **多后端** — CPU、CUDA、Apple MPS、昇腾 NPU，一份代码跑所有硬件。
- **昇腾 NPU 原生** — ACLNN 大算子直调，不是事后拼凑。
- **Agentic AI 就绪** — 足够轻量，可以嵌入 AI Agent 运行时。

## 路线图

多数框架止步于张量计算。Candle 不会 — 最终目标是**自托管智能内核 (Self-Hosting Agentic Kernel)**：在昇腾上部署本地模型，再用这些模型反过来驱动框架自身的调试、优化和演进。

### 第一阶段 — 基础框架（当前）

> PyTorch 兼容的纯 Python 框架，多硬件后端支持。

- [x] 核心张量算子、autograd、`nn.Module`、优化器
- [x] CPU 后端 (NumPy)
- [x] 昇腾 NPU 后端（ACLNN 原生大算子）
- [x] Apple MPS 后端（Metal 着色器）
- [x] `import torch` 零改动替换 hook
- [ ] CUDA 后端
- [ ] `torch.compile` 图模式加速
- [ ] 分布式训练 (`DistributedDataParallel`)
- [ ] TorchVision / TorchAudio 模型完整兼容

### 第二阶段 — 认知运行时 (Cognitive Runtime)

> 本地模型部署成为框架的一等原语，而不是外挂。

- [ ] 本地模型加载、量化与昇腾上部署 serving
- [ ] 基于角色的模型路由（调试模型、生成模型、评判模型）
- [ ] 多模型推理策略（本地优先，云端兜底）
- [ ] 自托管推理与工具调用运行时

### 第三阶段 — 智能内核 (Agentic Kernel)

> 框架获得观测、诊断和自我修复的能力。

- [ ] **Dev Layer** — bug 检测、复现构造、修复建议（由本地调试模型驱动）
- [ ] **Bootstrap Layer** — 候选生成、蒸馏、配置搜索（由本地生成模型驱动）
- [ ] **ModelOps Layer** — 评测、晋升、回滚、血统追踪（由本地评判模型驱动）

### 第四阶段 — 自托管自举 (Self-Hosting)

> 本地模型不再只是被管理的产物 — 它们成为内核自身的执行引擎。

- [ ] 本地优先的智能体执行：Dev / Bootstrap / ModelOps 默认调用自部署的模型
- [ ] 持续自我改进闭环：trace → 评测 → 生成修复 → 测试 → 晋升
- [ ] 模型即内核：系统用自己的模型来改进自己的模型

```
┌─────────────────────────────────────────┐
│              应用层                      │
│    训练 · 调试 · 推理 · 部署             │
├─────────────────────────────────────────┤
│            智能内核                      │
│    Dev Layer · Bootstrap · ModelOps     │
├─────────────────────────────────────────┤
│           认知运行时                     │
│    本地模型 RT · 路由 · 推理策略          │
├─────────────────────────────────────────┤
│           智能基座                       │
│    TraceStore · 评测器 · 制品仓库        │
├─────────────────────────────────────────┤
│             基础层                       │
│    tensor · autograd · nn · compiler    │
│    CPU · CUDA · MPS · 昇腾 NPU          │
└─────────────────────────────────────────┘
```

完整的 `0.1.x` 算子支持矩阵见 [docs/support-matrix.md](docs/support-matrix.md)。

## 谁在使用

> 用 Candle 构建了什么？[提个 issue](https://github.com/candle-org/candle/issues/new?labels=ecosystem&title=Add+my+project+to+Used+By) 告诉我们，我们会把你加到这里！

<!--
<div align="center">
<a href="https://example.com"><img src="https://img.shields.io/badge/YourProject-blue" height="30"></a>
</div>
-->

## Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=candle-org/candle&type=Date)](https://star-history.com/#candle-org/candle&Date)

</div>

## 参与贡献

```bash
# 克隆并以开发模式安装
git clone https://github.com/candle-org/candle.git
cd candle
pip install -e ".[test]"

# 运行测试
pytest tests/cpu/ tests/contract/ -v --tb=short

# 代码检查
pip install -e ".[lint]"
pylint src/candle --rcfile=.github/pylint.conf
```

欢迎任何形式的贡献！无论是新算子、后端支持、bug 修复还是文档改进 — 提 issue 或者发 PR 都行。

## 许可证

[MIT](LICENSE)
