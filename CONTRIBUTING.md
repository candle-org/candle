# Contributing to Candle

Thank you for your interest in Candle! We welcome contributions of all kinds — bug fixes, new operators, backend improvements, tests, documentation, and tooling.

This guide covers everything you need to go from "I want to help" to "I opened a clean PR." For deeper architecture notes and backend internals, see the [project Wiki](https://github.com/candle-org/candle/wiki).

---

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Development Environment](#development-environment)
- [Repository Workflow](#repository-workflow)
- [Project Rules](#project-rules)
- [Validation Before Opening a PR](#validation-before-opening-a-pr)
- [Pull Request Guidelines](#pull-request-guidelines)
- [For AI Contributors](#for-ai-contributors)

---

## Ways to Contribute

| Type | Examples |
|------|----------|
| **Bug fix** | Fix an op that returns wrong results, fix a dispatch error |
| **Operator parity** | Implement a missing PyTorch op on CPU/MPS/NPU |
| **Backend support** | Add or improve Metal shaders, ACLNN kernels, CUDA kernels |
| **Tests** | Add contract tests, improve coverage, add edge-case tests |
| **Documentation** | Fix typos, improve docstrings, write guides |
| **Tooling** | CI improvements, developer scripts, linting rules |

Not sure where to start? Look for issues labeled [`good first issue`](https://github.com/candle-org/candle/labels/good%20first%20issue).

---

## Development Environment

### Prerequisites

- Python 3.11+
- Git
- A C compiler (for Cython extensions)
- [Conda](https://docs.conda.io/) (recommended) or a virtual environment

### Setup

```bash
# 1. Fork candle-org/candle on GitHub, then clone your fork
git clone https://github.com/<your-username>/candle.git
cd candle
git remote add upstream https://github.com/candle-org/candle.git

# 2. Create a conda environment (recommended)
conda create -n candle python=3.11
conda activate candle

# 3. Install build dependencies and the package in editable mode
pip install 'cython>=3.0' 'setuptools>=68.0' wheel
pip install -r requirements/requirements-test.txt   # includes numpy, scipy, pytest, torch (CPU)
pip install -e . --no-build-isolation
```

Candle includes Cython extensions that are compiled during install. The `--no-build-isolation` flag lets the build reuse your environment's Cython and setuptools.

#### MPS development (macOS Apple Silicon)

```bash
# Use the MPS requirements instead (includes pyobjc-framework-Metal)
pip install -r requirements/requirements-test-mps.txt
pip install -e . --no-build-isolation
```

### Verify your setup

```bash
# Run CPU + contract tests
python -m pytest tests/cpu/ tests/contract/ -v --tb=short

# Run lint
pylint src/candle/ --rcfile=.github/pylint.conf

# Run MPS tests (macOS Apple Silicon only)
python -m pytest tests/mps/ -v --tb=short
```

If all tests pass and pylint reports zero errors, you're ready to contribute.

---

## Repository Workflow

Candle uses a **worktree-based branch workflow**. This keeps `main` clean and makes it easy to work on multiple changes in parallel.

### Step-by-step

```bash
# 1. Sync with upstream
git fetch upstream main

# 2. Create a worktree with a feature branch based on upstream/main
git worktree add .worktrees/<branch-name> -b <branch-name> upstream/main
cd .worktrees/<branch-name>

# 3. Make your changes, commit as you go

# 4. Before pushing, rebase onto latest upstream
git fetch upstream main
git rebase upstream/main

# 5. Push to your fork
git push -u origin <branch-name>

# 6. Open a PR against upstream
gh pr create --repo candle-org/candle --head <your-username>:<branch-name> --base main
```

> **Tip:** If you prefer a simpler workflow, you can also use a regular feature branch (`git checkout -b <branch-name> upstream/main`). Worktrees are recommended but not strictly required for human contributors.

### Key rules

- **Never develop on `main`.** The `main` branch is read-only during development. All work goes on feature branches.
- **One branch per change.** Don't mix unrelated features in one branch.
- **Clean up after merge.** Once your PR is merged:

```bash
# From the repo root (not inside the worktree)
git worktree remove .worktrees/<branch-name>
git branch -d <branch-name>
git push origin --delete <branch-name>

# Sync main
git checkout main && git pull upstream main && git push origin main
```

---

## Project Rules

These are non-negotiable design invariants. PRs that violate them will not be merged.

### 1. Candle is independent of PyTorch at runtime

PyTorch is only used in tests for result validation. **Never** import `torch` in source code under `src/candle/`.

Allowed runtime dependencies: `numpy`, `scipy`, `ctypes`, and the Python standard library.

### 2. No CPU fallback for GPU/NPU backends

MPS, CUDA, and NPU ops must stay on their native device. **Never** fall back to NumPy to work around a kernel bug.

When a native kernel has a bug:

1. **Composite workaround** — Reimplement the op using smaller on-device ops that already work. All computation must remain on the same device.
2. **Preserve the native kernel** — Keep the broken kernel code guarded (not deleted) so it can be re-enabled later. Mark with `# TODO: re-enable native kernel when <platform> fixes <issue>`.
3. **Document it** — Add an entry to [`docs/known-kernel-issues.md`](docs/known-kernel-issues.md) with: op name, backend, error description, workaround used, and platform version.

### 3. Schema validation is a guardrail, not an obstacle

**Never** bypass or disable schema validation to make tests pass. If an op needs to handle a case the schema rejects, fix it at the functional layer before dispatch — not by weakening the schema.

### 4. Fix source bugs, not tests

When a test reveals a bug, fix the source code in `src/candle/`. Don't modify the test to hide the problem.

### 5. General-purpose only

Candle is a general-purpose PyTorch compatibility layer. **Never** add application-specific hacks or special cases. All fixes must be generic PyTorch API implementations.

### 6. Schema-first development

For any new operator, follow this order:

1. Register the schema in `src/candle/_dispatch/schemas.py`
2. Add or update contract tests in `tests/contract/`
3. Register backend kernels (CPU / MPS / NPU / Autograd / Functionalize)
4. Add or update functional / tensor API exports

Do not register a kernel before its schema exists.

### 7. Kernel implementation priority

For each backend, prefer implementations in this order:

1. **Native device kernels** — Metal shaders for MPS, ACLNN for NPU, CUDA kernels
2. **Hardware-accelerated libraries** — Accelerate BLAS for MPS matmul, etc.
3. **Composite of existing dispatched ops** — the only acceptable workaround when a native kernel has a bug; all ops must run on the same device
4. **NumPy** — only for the CPU backend, never for MPS/CUDA/NPU

---

## Validation Before Opening a PR

Every PR touching `src/candle/` must pass these checks locally:

```bash
# Required: lint must be clean
pylint src/candle/ --rcfile=.github/pylint.conf

# Required: contract tests
python -m pytest tests/contract/ -v --tb=short

# Recommended: full CPU test suite
python -m pytest tests/cpu/ tests/contract/ -v --tb=short

# If your change touches MPS code (macOS Apple Silicon):
python -m pytest tests/mps/ -v --tb=short
```

**Do not open a PR if pylint fails.** Fix all issues first.

### CI pipeline

CI runs automatically on every PR that touches `src/candle/`, `tests/`, or `.github/workflows/`:

| Job | Runner | Scope |
|-----|--------|-------|
| **pylint-check** | Ubuntu | Lint gate — must pass before tests run |
| **test-cpu** | Ubuntu | `tests/cpu/` + `tests/contract/` |
| **test-mps** | macOS 14 (M1) | `tests/mps/` |
| **test-npu-*** | Self-hosted (Ascend 910A/910B/310B) | `tests/npu/` + `tests/distributed/` |

All test jobs run in parallel after pylint passes. If any job fails, the entire workflow is cancelled.

---

## Pull Request Guidelines

### Keep PRs small and focused

- One logical change per PR.
- Don't mix unrelated fixes, refactors, or features.
- Prefer multiple small PRs over one large one.

### Write a clear description

Use the [PR template](.github/pull_request_template.md). At minimum, include:

- **Summary** — What does this PR do and why? One or two sentences.
- **Linked issue** — Reference the issue with `Closes #...` or `Related to #...`.
- **Test plan** — Which tests did you run? Paste the commands.

### Match PyTorch semantics

- Match PyTorch dispatch semantics first (schema binding, error class, dispatch path), then optimize.
- Error message wording can differ slightly unless a contract test requires exact match.

### Code style

- Follow existing patterns in the codebase.
- Pylint with `.github/pylint.conf` is the style authority.
- Don't add unnecessary comments, docstrings, or type annotations to code you didn't change.

---

## For AI Contributors

This section applies to contributions generated by AI coding tools (Claude Code, Codex, Copilot, Cursor, etc.). Human contributors can skip this section.

### Disclosure

- **Always** disclose the AI tool used in your PR description.
- Use the [AI PR template](.github/PULL_REQUEST_TEMPLATE/ai.md) instead of the default template.

### Evidence requirements

AI-generated PRs are held to a higher evidence bar:

- **Paste exact commands** you ran — do not abbreviate or paraphrase.
- **Paste actual terminal output** — do not summarize or edit.
- **List all files changed** and the areas they affect.
- **Include reviewer notes** about remaining risks or edge cases.

### Mandatory checklist

Before opening a PR, confirm:

- [ ] I disclosed the AI tool used
- [ ] I linked the relevant issue(s)
- [ ] I included a concise change summary
- [ ] I pasted actual validation output (not paraphrased)
- [ ] I did **not** bypass schema validation to make tests pass
- [ ] I did **not** introduce CPU fallback for GPU/NPU behavior
- [ ] I did **not** add unrelated changes

### Worktree workflow is mandatory for AI agents

AI agents **must** use the worktree workflow described in [Repository Workflow](#repository-workflow). Never commit directly to `main`. Never touch worktrees or branches created by other agents or contributors.

For detailed agent-specific rules (development order, verification gates, kernel fallback policy), see [`AGENTS.md`](AGENTS.md).

---

## Getting Help

- **Issues**: [github.com/candle-org/candle/issues](https://github.com/candle-org/candle/issues)
- **Wiki**: [github.com/candle-org/candle/wiki](https://github.com/candle-org/candle/wiki) — architecture docs, backend guides, debugging tips

Thank you for contributing to Candle!
