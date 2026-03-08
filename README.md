# Candle

> **C**ANN c**AN** han**DLE** — A lightweight, pure-Python deep learning framework with Ascend NPU optimization.

Candle is a lightweight deep learning framework written entirely in Python. While not as bright as a torch, a candle is light enough to carry anywhere — and it gets the job done.

## Features

- **Pure Python** — No C++ extensions, no build complexity. Just Python.
- **Ascend NPU First** — Native ACLNN kernel integration for Huawei Ascend devices.
- **PyTorch-Compatible API** — Familiar `torch`-like interface for tensors, autograd, nn modules, and optimizers.
- **Multi-Backend** — CPU (NumPy), CUDA, Ascend NPU, Apple MPS.
- **Agentic AI Ready** — Built as a native foundation for agentic AI workflows.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import candle

# Create tensors
x = candle.randn(3, 4)
y = candle.randn(4, 5)

# Matrix operations
z = candle.matmul(x, y)
print(z.shape)  # (3, 5)

# Neural networks
import candle.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# Autograd
x = candle.randn(2, 784, requires_grad=True)
out = model(x)
out.sum().backward()
print(x.grad.shape)  # (2, 784)
```

## Development

### Install dependencies

```bash
pip install -r requirements/requirements-test.txt
```

### Run tests

```bash
pytest tests/ -v --tb=short
```

### Lint

```bash
pip install -r requirements/requirements-lint.txt
pylint src/candle --rcfile=.github/pylint.conf
```

## License

MIT
