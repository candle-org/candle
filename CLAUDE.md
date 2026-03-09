# Candle Project - Claude Code Configuration

## Project Overview

Candle is a PyTorch-compatible ML framework (`import candle as torch`) with a custom dispatch system, autograd engine, and multi-backend support (CPU, MPS, CUDA, NPU).

## Directory Structure

```
candle/
├── src/candle/                # Source code
│   ├── _backends/             # Backend implementations
│   │   ├── cpu/               #   CPU ops (numpy-based)
│   │   ├── mps/               #   MPS ops (Metal GPU + numpy fallback)
│   │   ├── npu/               #   NPU ops (ACLNN ctypes bindings)
│   │   ├── cuda/              #   CUDA ops
│   │   ├── common/            #   Shared view/convert ops
│   │   └── autograd.py        #   Backward implementations for all ops
│   ├── _dispatch/             # Dispatch system & schema validation
│   │   ├── dispatcher.py      #   Core dispatcher
│   │   ├── schema.py          #   Schema validation
│   │   ├── schemas.py         #   Op schema definitions
│   │   └── registry.py        #   Op registry
│   ├── _autograd/             # Autograd engine
│   ├── nn/                    # Neural network modules
│   ├── _tensor.py             # Tensor class
│   ├── _functional.py         # Functional API (dispatch wrappers)
│   └── _creation.py           # Tensor creation functions
├── tests/
│   ├── conftest.py            # Auto-skip MPS/NPU tests when hardware unavailable
│   ├── cpu/                   # CPU tests
│   ├── mps/                   # MPS tests
│   ├── contract/              # API contract tests
│   ├── npu/                   # NPU tests
│   ├── cuda/                  # CUDA tests
│   └── distributed/           # Distributed tests
├── requirements/
│   ├── requirements.txt       # Base dependencies
│   ├── requirements-test.txt  # CPU test deps (CPU-only PyTorch)
│   └── requirements-test-mps.txt  # MPS test deps (standard PyTorch + pyobjc)
├── examples/
│   └── ascendc/               # AscendC custom operator examples
├── .github/workflows/ci.yaml  # CI: pylint → test-cpu + test-mps
└── CLAUDE.md                  # This file
```

## Environment

- **Conda**: `source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp ...`
- **Platform**: macOS Apple Silicon (Darwin), local MPS hardware available
- **Python**: 3.11

## CI Pipeline

CI runs on every PR and push to `main`:

1. **pylint-check** (ubuntu-latest) — lint with pylint
2. **test-cpu** (ubuntu-latest) — `pytest tests/cpu/ tests/contract/ -v --tb=short`
3. **test-mps** (macos-14, M1) — `pytest tests/mps/ -v --tb=short`

Jobs 2 and 3 run in parallel after pylint passes.

---

## Important Constraints

### Core Design Principle: General-Purpose PyTorch Compatibility

Candle must remain a **general-purpose PyTorch compatibility layer**.

- **NEVER** add application-specific hacks or special cases to candle code
- All fixes must be generic PyTorch API implementations
- If a test fails due to application-specific behavior, document it rather than adding special cases

### Core Design Principle: Candle is Independent of PyTorch

Candle does NOT depend on PyTorch at runtime. PyTorch is only used in tests for result validation.

- **NEVER** import `torch` in candle source code (`src/candle/`)
- All computation must be implemented via the internal dispatch mechanism and backend-specific kernels
- Allowed dependencies: `numpy`, `scipy`, `ctypes`, and the Python standard library

### Core Design Principle: No Fallback to CPU on GPU/NPU

For MPS/CUDA/NPU devices, **NEVER** fall back to CPU (numpy) to work around kernel bugs or missing functionality.

- MPS ops must stay on the Metal GPU path
- NPU ops must use ACLNN kernels via ctypes
- CUDA ops must use CUDA kernels
- NumPy is ONLY acceptable for the CPU backend

**When a native kernel has a bug or limitation:**

1. **Composite workaround allowed**: You MAY reimplement the op as a composition of smaller on-device ops that already work correctly. All computation must remain on the same device.
2. **Preserve the native kernel entry point**: Do NOT delete the broken native kernel call. Keep it in the code behind a clear guard (e.g., a flag or commented-out block) so it can be re-enabled and tested when the underlying platform (CANN SDK / CUDA toolkit / macOS) is updated.
3. **Document the issue**: Record every known kernel issue in `docs/known-kernel-issues.md` with: op name, backend, error description, workaround used, and the platform version that exhibits the bug.
4. **Never silently degrade**: Moving computation to CPU is never an acceptable workaround — it hides the real problem and breaks device-placement guarantees.

### Core Design Principle: Schema Validation is Intentional

Schema validation errors are design guardrails, not bugs to suppress.

- **NEVER** bypass or disable schema validation to make tests pass
- If an op needs to handle a case the schema rejects (e.g., `squeeze(dim=None)`), fix at the functional layer before dispatch, not by weakening the schema

### Kernel Implementation Priority

For each backend, follow this priority order:

1. **Native device kernels** (Metal shaders for MPS, ACLNN for NPU, CUDA kernels) — always preferred
2. **Accelerate BLAS** (for MPS matmul) or equivalent hardware-accelerated libraries
3. **Composite of existing dispatched ops** — build complex ops from smaller on-device ops that already work. This is the **only acceptable workaround** when a native kernel has a bug. All ops in the composite must run on the same device.
4. **NumPy fallback** — ONLY for CPU backend, NEVER for MPS/CUDA/NPU

When using option 3 as a workaround for a broken native kernel:
- Keep the native kernel code in place (guarded, not deleted)
- Add an entry to `docs/known-kernel-issues.md`
- Add a `# TODO: re-enable native kernel when <platform> fixes <issue>` comment

### For Bug Fixes

- **Fix source bugs** over working around them in tests
- When a test reveals a bug, fix the source code in `src/candle/`, don't modify the test

---

## Git Configuration

### Remotes

- **origin**: `lvyufeng/candle` (fork, push target)
- **upstream**: `candle-org/candle` (upstream, PR target)

### PR Workflow

```bash
# 1. Create feature branch from main
git checkout -b feat/<name>

# 2. Push to origin
git push -u origin feat/<name>

# 3. Create PR to upstream
gh pr create --repo candle-org/candle --head lvyufeng:feat/<name> --base main
```

### Merge Convention

- Squash merge PRs into main
- After merge: sync local main, delete feature branch

```bash
git checkout main && git pull upstream main && git push origin main
git branch -d feat/<name> && git push origin --delete feat/<name>
```

---

## Test Execution

### Run Tests Locally

```bash
# CPU tests
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/cpu/ tests/contract/ -v --tb=short

# MPS tests (macOS Apple Silicon only)
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python -m pytest tests/mps/ -v --tb=short
```

### Test Organization

- Tests in `tests/<device>/` are auto-skipped when the device is unavailable (via `conftest.py`)
- CPU tests: always run
- MPS tests: skip if no Apple GPU
- NPU tests: skip if no Ascend hardware

---

## Backend Development Guide

### Adding a New Backend

1. **Create `src/candle/_backends/<device>/ops.py`**: Implement op kernels
2. **Register ops** in the dispatch registry
3. **Add device detection** in `src/candle/_backends/<device>/runtime.py`
4. **Add tests** in `tests/<device>/`
5. **Update CI** in `.github/workflows/ci.yaml`

### MPS Backend Pattern (reference)

- GPU path: Metal compute shaders via `_can_use_gpu()` check
- `_can_use_gpu(t)` requires: float32/16, contiguous, numel > 0, has metal_buffer
- Binary ops use `dispatch_binary` (same shape) or `dispatch_binary_scalar`
- For commutative ops (add, mul): swap operands when `a` is smaller for correct broadcast shape
- Metal runtime: pyobjc preferred, ctypes fallback for systems without pyobjc

### NPU Backend Pattern (reference)

- Use ACLNN large kernels via ctypes bindings to `libopapi.so`
- **Always prefer a single ACLNN kernel** over compositing multiple small ops
- Compositing small ops incurs kernel launch overhead and prevents hardware-level fusion
- Check `_backends/npu/aclnn.py` before implementing any NPU op as a composite

---

## Troubleshooting

- **Tests not running**: Ensure conda env is activated (`conda run -n mindnlp`)
- **MPS tests skipped locally**: Verify `pyobjc-framework-Metal` is installed
- **MPS tests skipped on CI**: Check that `requirements-test-mps.txt` includes `pyobjc-framework-Metal`
- **Git push fails**: Check push access, uncommitted changes, branch existence on remote
- **Pylint fails on CI**: macOS-only imports need `# pylint: disable=import-error`
