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

- **Conda**: `source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle ...`
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

### Worktree & Branch Rules (Mandatory)

These rules apply to Claude Code, Codex, and all coding agents. No exceptions.

#### 1. Always use a worktree

Before making ANY code change, create an isolated worktree and sync upstream:

```bash
git fetch upstream main
git worktree add .worktrees/<branch-name> -b <branch-name> upstream/main
cd .worktrees/<branch-name>
```

Never edit files on `main` directly. If you find yourself on `main` with uncommitted changes, stop immediately, create a worktree, and move the changes there.

#### 2. Never modify main

The `main` branch is read-only during development. All commits go on feature branches inside `.worktrees/`. If a command would alter `main`, abort it.

#### 3. Rebase upstream before PR

Before pushing and opening a PR, rebase onto the latest upstream main:

```bash
git fetch upstream main
git rebase upstream/main
```

Resolve any conflicts before proceeding.

#### 4. Pre-PR validation gate

**Determine validation scope based on what changed.** Do NOT blindly run pylint and the full test suite on every PR — only run what the change actually requires.

| Change scope | Pylint | CPU + contract tests | Local backend tests (MPS/NPU) |
|---|---|---|---|
| Only docs, markdown, scripts, CI yaml | Skip | Skip | Skip |
| Only `tests/` (no `src/candle/` changes) | Skip | Run affected test files | Run if touching backend tests |
| `src/candle/` code changes | **Required** | **Required** | **Required** if local hardware available |

**When pylint is required:**

```bash
pylint src/candle/ --rcfile=.github/pylint.conf
```

Do NOT open a PR if pylint fails. Fix all issues first.

**When tests are required:**

```bash
# CPU + contract tests (always required for src/candle/ changes)
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/ tests/contract/ -v --tb=short

# MPS tests (required on macOS Apple Silicon when src/candle/ changes)
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/mps/ -v --tb=short
```

Do NOT open a PR if tests fail. Do NOT rely on CI as your test runner — CI is a safety net, not a substitute for local validation.

#### 5. Clean up after merge

After a PR is merged (manually or via command), delete ONLY the worktree and branch YOU created:

```bash
# From the repo root (not inside the worktree)
git worktree remove .worktrees/<your-branch-name>
git branch -d <your-branch-name>
git push origin --delete <your-branch-name>
```

NEVER touch worktrees or branches created by other agents or contributors.

### PR Workflow

```bash
# Push to origin
git push -u origin <branch-name>

# Create PR to upstream
gh pr create --repo candle-org/candle --head lvyufeng:<branch-name> --base main
```

### Merge Convention

- Squash merge PRs into main
- After merge: clean up worktree and branch (Rule 5 above), then sync main:

```bash
git checkout main && git pull upstream main && git push origin main
```

---

## Test Execution

### When to run tests

- **`src/candle/` changes**: Always run CPU + contract tests, plus local backend tests (MPS or NPU) if hardware is available.
- **`tests/` only changes**: Run the affected test files.
- **Docs, scripts, CI yaml only**: No tests needed.

CI is a safety net, NOT a substitute for local testing. Do NOT push untested code and rely on CI to catch failures.

### Run Tests Locally

```bash
# CPU tests
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/ tests/contract/ -v --tb=short

# MPS tests (macOS Apple Silicon only)
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
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

### NPU Transformer / SDPA Performance Playbook

Use this checklist for Transformer and attention performance work. It captures the lessons from the NPU MLP optimization batches and is mandatory for SDPA-related development.

#### Goal and acceptance bar

- Target API: PyTorch-compatible `torch.nn.functional.scaled_dot_product_attention` (SDPA), plus any generic functional/module paths that route through it.
- Functional requirement: support autograd for train-mode Transformer workloads; inference-only speedups are insufficient.
- Performance requirement: end-to-end Transformer forward, backward, and total step time on NPU must be **at parity with or faster than `torch_npu`** for the benchmarked shapes and dtype.
- Implementation must remain generic Candle behavior: no Transformer-shape hacks, no benchmark-only branches, no model-specific fusion.

#### Required development order

1. **Benchmark first**
   - Add or reuse a Transformer/SDPA benchmark that reports forward, backward, total, and profiler top rows for both Candle and `torch_npu`.
   - Use identical shapes, dtype, dropout mode, mask/causal settings, and synchronization boundaries for both frameworks.
   - Always set `PYTHONPATH=<your-worktree>/src` in `candle311`; otherwise benchmarks may import the wrong editable install.
2. **Profile before optimizing**
   - Identify whether the gap is in SDPA kernels, matmul/bmm, softmax/dropout/mask ops, autograd graph overhead, tensor views/transposes, allocator/runtime/deferred cleanup, or CPU-visible dispatch overhead.
   - Do not guess. Record the hot rows before choosing an implementation path.
3. **Write RED tests before production code**
   - Correctness tests must compare against PyTorch in tests only.
   - Routing tests should prove hot paths avoid Python wrappers/redispatch where that is the intended performance property.
   - Autograd tests must verify query/key/value gradients stay on NPU and match tolerances.
   - Include mask/causal/dropout/scale combinations supported by the implementation; unsupported cases must fall back through normal generic dispatch, not silently return wrong results.
4. **Prefer native fused kernels**
   - For NPU SDPA, first check ACLNN/Ascend availability for fused attention forward/backward kernels.
   - If native fused forward/backward is usable, bind it in `_C`/Cython and keep Python-side API matching PyTorch.
   - If native fused kernels are absent or incorrect, use an on-device composite of generic ops only as a temporary generic fallback. Never move NPU tensors to CPU.
5. **Keep autograd fused/generic**
   - SDPA forward fast paths must attach a tested autograd node or route through generated autograd without losing graph semantics.
   - Backward should prefer native fused SDPA backward; otherwise implement a generic on-device backward composite.
   - Preserve `create_graph`, saved tensor hooks, public `next_functions`, retain-grad, and leaf `.grad` semantics. Internal fast caches may defer public metadata creation only if public introspection still materializes PyTorch-compatible state.
6. **Use proven NPU hot-path techniques**
   - Use exact base `TensorImpl` guards before bypassing dispatch.
   - Read cached shape/stride/device/dtype fields directly in Cython.
   - Use cached stream/runtime/allocator helpers and fast large-pool tensor wrappers for fresh outputs.
   - Add PTA executor-cache keys for stable native kernels. Keys must include all semantic attributes: op name, tensor descriptors, dtype(s), scale, dropout probability, causal flag, mask type/shape/stride, output descriptor, alias bits where relevant, and stream.
   - Keep descriptor/workspace/executor lifetimes valid until deferred execution cleanup.
7. **Validate broadly before PR**
   - For any `src/candle/` change: rebuild Cython, run affected NPU tests, full NPU suite, CPU + contract tests, pylint, and repeated Transformer benchmarks.
   - Do not claim parity from one noisy run. Use enough repeats to distinguish improvement from `torch_npu` run-to-run variance.
   - If eager op-level optimization cannot reach parity, move to a generic graph/capture/fused-backend plan rather than adding application-specific fusion.

#### Useful validation commands

```bash
# Rebuild Cython extensions
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311 && \
PYTHONPATH=<your-worktree>/src python setup.py build_ext --inplace

# NPU focused/full tests
source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311 && \
PYTHONPATH=<your-worktree>/src python -m pytest tests/npu/ -v --tb=short

# CPU + contract tests
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311 && \
PYTHONPATH=<your-worktree>/src python -m pytest tests/cpu/ tests/contract/ -v --tb=short

# Pylint gate
source /home/jenkins/anaconda3/etc/profile.d/conda.sh && conda activate candle311 && \
PYTHONPATH=<your-worktree>/src python -m pylint src/candle/ --rcfile=.github/pylint.conf
```

---

## Troubleshooting

- **Tests not running**: Ensure conda env is activated (`conda run -n candle`)
- **MPS tests skipped locally**: Verify `pyobjc-framework-Metal` is installed
- **MPS tests skipped on CI**: Check that `requirements-test-mps.txt` includes `pyobjc-framework-Metal`
- **Git push fails**: Check push access, uncommitted changes, branch existence on remote
- **Pylint fails on CI**: macOS-only imports need `# pylint: disable=import-error`
