# MindTorch v2 → Candle Migration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the mindtorch_v2 codebase from the mindnlp repository to the standalone candle repository, renaming all references from `mindtorch_v2` to `candle`, and setting up CI/CD, tests, and packaging.

**Architecture:** The candle repository will be a standalone Python package with `src/candle/` layout. All internal references to `mindtorch_v2` will be replaced with `candle`. GitHub Actions will provide pylint checks, unit testing, and wheel releases. The torch proxy system will be preserved for transparent PyTorch compatibility.

**Tech Stack:** Python 3.9+, pytest, pylint, GitHub Actions, setuptools/pyproject.toml

---

## Source Analysis

- **Source code:** `/Users/lvyufeng/Projects/mindnlp/src/mindtorch_v2/` (384 files)
- **Tests:** `/Users/lvyufeng/Projects/mindnlp/tests/mindtorch_v2/` (306 files)
- **Target:** `/Users/lvyufeng/Projects/candle/` (empty repo, MIT license)
- **References to rename:** 192 files, ~584 occurrences of `mindtorch_v2`
- **Also rename:** `MINDTORCH_TEST_FORCE_CPU_ONLY` → `CANDLE_TEST_FORCE_CPU_ONLY`

---

### Task 1: Copy Source Code to Candle Repository

**Files:**
- Copy: `mindnlp/src/mindtorch_v2/**` → `candle/src/candle/**`

**Step 1: Copy the source tree**

```bash
cp -r /Users/lvyufeng/Projects/mindnlp/src/mindtorch_v2/ /Users/lvyufeng/Projects/candle/src/candle/
```

**Step 2: Remove `__pycache__` directories**

```bash
find /Users/lvyufeng/Projects/candle/src/candle/ -type d -name __pycache__ -exec rm -rf {} +
```

**Step 3: Verify file count**

```bash
find /Users/lvyufeng/Projects/candle/src/candle/ -name "*.py" | wc -l
```

Expected: ~384 .py files

**Step 4: Commit**

```bash
cd /Users/lvyufeng/Projects/candle
git add src/candle/
git commit -m "feat: copy mindtorch_v2 source code as candle base"
```

---

### Task 2: Copy Tests to Candle Repository

**Files:**
- Copy: `mindnlp/tests/mindtorch_v2/**` → `candle/tests/`

**Step 1: Copy the test tree**

```bash
cp -r /Users/lvyufeng/Projects/mindnlp/tests/mindtorch_v2/ /Users/lvyufeng/Projects/candle/tests/
```

**Step 2: Remove `__pycache__` directories**

```bash
find /Users/lvyufeng/Projects/candle/tests/ -type d -name __pycache__ -exec rm -rf {} +
```

**Step 3: Verify file count**

```bash
find /Users/lvyufeng/Projects/candle/tests/ -name "*.py" | wc -l
```

Expected: ~108 test files

**Step 4: Commit**

```bash
cd /Users/lvyufeng/Projects/candle
git add tests/
git commit -m "feat: copy mindtorch_v2 tests as candle test base"
```

---

### Task 3: Rename All `mindtorch_v2` References to `candle` in Source Code

**Files:**
- Modify: All 44 files in `src/candle/` that reference `mindtorch_v2`

**Step 1: Batch rename all occurrences in source files**

```bash
cd /Users/lvyufeng/Projects/candle
find src/candle/ -name "*.py" -exec sed -i '' 's/mindtorch_v2/candle/g' {} +
```

**Step 2: Verify no remaining references**

```bash
grep -r "mindtorch_v2" src/candle/ | wc -l
```

Expected: 0

**Step 3: Spot check key files**

- `src/candle/__init__.py` - should have no `mindtorch_v2` references
- `src/candle/distributed/__init__.py` - docstrings should say `candle`
- `src/candle/nn/parallel/distributed.py` - should say `candle`

**Step 4: Commit**

```bash
cd /Users/lvyufeng/Projects/candle
git add src/candle/
git commit -m "refactor: rename all mindtorch_v2 references to candle in source"
```

---

### Task 4: Rename All `mindtorch_v2` References to `candle` in Tests

**Files:**
- Modify: All 148 files in `tests/` that reference `mindtorch_v2`

**Step 1: Batch rename all occurrences in test files**

```bash
cd /Users/lvyufeng/Projects/candle
find tests/ -name "*.py" -exec sed -i '' 's/mindtorch_v2/candle/g' {} +
```

**Step 2: Rename `MINDTORCH_TEST_FORCE_CPU_ONLY` to `CANDLE_TEST_FORCE_CPU_ONLY`**

```bash
find tests/ -name "*.py" -exec sed -i '' 's/MINDTORCH_TEST_FORCE_CPU_ONLY/CANDLE_TEST_FORCE_CPU_ONLY/g' {} +
```

**Step 3: Verify no remaining references**

```bash
grep -r "mindtorch_v2" tests/ | wc -l
grep -r "MINDTORCH" tests/ | wc -l
```

Expected: 0 for both

**Step 4: Commit**

```bash
cd /Users/lvyufeng/Projects/candle
git add tests/
git commit -m "refactor: rename all mindtorch_v2 references to candle in tests"
```

---

### Task 5: Create `pyproject.toml` for Candle Package

**Files:**
- Create: `candle/pyproject.toml`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "candle-python"
version = "0.1.0"
description = "A PyTorch-compatible deep learning framework"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
authors = [
    {name = "candle-org"},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "numpy",
    "ml_dtypes",
    "safetensors",
    "tqdm",
    "requests",
    "pillow>=10.0.0",
]

[project.optional-dependencies]
test = [
    "pytest>=7.2.0,<9.0.0",
    "torch",
]
lint = [
    "pylint>=2.17",
]
all = [
    "candle-python[test,lint]",
]

[project.urls]
Homepage = "https://github.com/lvyufeng/candle"
Repository = "https://github.com/lvyufeng/candle"
Issues = "https://github.com/lvyufeng/candle/issues"

[tool.setuptools.packages.find]
where = ["src"]
include = ["candle*"]

[tool.setuptools.package-data]
candle = ["*.py", "*/*.py", "*/*/*.py", "*/*/*/*.py"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

**Step 2: Commit**

```bash
cd /Users/lvyufeng/Projects/candle
git add pyproject.toml
git commit -m "feat: add pyproject.toml for candle package"
```

---

### Task 6: Create Requirements Files

**Files:**
- Create: `candle/requirements/requirements.txt`
- Create: `candle/requirements/requirements-test.txt`
- Create: `candle/requirements/requirements-lint.txt`

**Step 1: Create requirements directory and files**

`requirements/requirements.txt`:
```
numpy
ml_dtypes
safetensors
tqdm
requests
pillow>=10.0.0
```

`requirements/requirements-test.txt`:
```
-r requirements.txt
pytest>=7.2.0,<9.0.0
--extra-index-url https://download.pytorch.org/whl/cpu
torch
```

`requirements/requirements-lint.txt`:
```
pylint>=2.17
--extra-index-url https://download.pytorch.org/whl/cpu
torch
```

**Step 2: Commit**

```bash
cd /Users/lvyufeng/Projects/candle
git add requirements/
git commit -m "feat: add requirements files"
```

---

### Task 7: Create pytest Configuration

**Files:**
- Create: `candle/pytest.ini`

**Step 1: Create pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

**Step 2: Commit**

```bash
cd /Users/lvyufeng/Projects/candle
git add pytest.ini
git commit -m "feat: add pytest configuration"
```

---

### Task 8: Create Pylint Configuration

**Files:**
- Create: `candle/.github/pylint.conf`

**Step 1: Copy pylint.conf from mindnlp**

```bash
mkdir -p /Users/lvyufeng/Projects/candle/.github
cp /Users/lvyufeng/Projects/mindnlp/.github/pylint.conf /Users/lvyufeng/Projects/candle/.github/pylint.conf
```

**Step 2: Commit**

```bash
cd /Users/lvyufeng/Projects/candle
git add .github/pylint.conf
git commit -m "feat: add pylint configuration"
```

---

### Task 9: Create GitHub Actions CI Workflow

**Files:**
- Create: `candle/.github/workflows/ci.yaml`

**Step 1: Create CI workflow**

```yaml
name: CI

on:
  pull_request:
    branches: ["main"]
    paths:
      - 'src/candle/**'
      - 'tests/**'
      - '.github/workflows/**'
  push:
    branches: ["main"]
    paths:
      - 'src/candle/**'
      - 'tests/**'
      - '.github/workflows/**'

permissions:
  contents: read

jobs:
  pylint-check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements/requirements-lint.txt
    - name: Lint with pylint
      run: |
        pylint --jobs=1 src/candle --rcfile=.github/pylint.conf

  test:
    needs: pylint-check
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11"]
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements/requirements-test.txt
    - name: Install candle
      run: |
        pip install -e .
    - name: Run tests
      run: |
        cd tests
        pytest . -v --tb=short \
          --ignore=test_ops_npu.py \
          --ignore=test_ddp.py \
          --ignore=test_ddp_cpu.py \
          --ignore=test_ddp_builtin_hooks.py \
          --ignore=test_ddp_comm_hook.py \
          --ignore=test_ddp_unused_params.py \
          --ignore=test_ddp_static_graph.py \
          --ignore=test_ddp_bucket_view.py \
          --ignore=test_gloo_ddp.py \
          --ignore=test_hccl_all_to_all_multicard.py
```

**Step 2: Commit**

```bash
cd /Users/lvyufeng/Projects/candle
git add .github/workflows/ci.yaml
git commit -m "feat: add GitHub Actions CI workflow"
```

---

### Task 10: Create GitHub Actions Release Workflow

**Files:**
- Create: `candle/.github/workflows/release.yaml`

**Step 1: Create release workflow**

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    name: Build and Release

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.x"

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build

    - name: Build package
      run: python -m build --wheel

    - name: Upload artifact
      uses: actions/upload-artifact@v4
      with:
        name: candle-whl
        path: dist/*

    - name: Release and Upload Assets
      uses: ncipollo/release-action@v1
      with:
        bodyFile: ""
        allowUpdates: true
        removeArtifacts: true
        artifacts: |
          dist/*.whl
```

**Step 2: Commit**

```bash
cd /Users/lvyufeng/Projects/candle
git add .github/workflows/release.yaml
git commit -m "feat: add GitHub Actions release workflow"
```

---

### Task 11: Create Test Runner for Transformers Compatibility

**Files:**
- Create: `candle/tests/run_test.py`

**Step 1: Create run_test.py**

Adapt from `mindnlp/tests/run_test_v2.py`, replacing all `mindtorch_v2` references with `candle`:

- Line 34: `from candle._torch_proxy import install` (if torch_proxy exists)
- Line 39: docstrings referencing `candle` instead of `mindtorch_v2`
- Line 72: error message: `"not available in candle"`
- Line 288: print: `"[candle] Default device:"`
- Line 289: print: `"[candle] NPU available:"`

**Step 2: Commit**

```bash
cd /Users/lvyufeng/Projects/candle
git add tests/run_test.py
git commit -m "feat: add test runner for transformers compatibility"
```

---

### Task 12: Update README.md

**Files:**
- Modify: `candle/README.md`

**Step 1: Write README with project description and quickstart**

Include:
- Project name and description
- Installation instructions (`pip install -e .`)
- Basic usage example
- Development setup (requirements, running tests, linting)
- License info

**Step 2: Commit**

```bash
cd /Users/lvyufeng/Projects/candle
git add README.md
git commit -m "docs: add project README"
```

---

### Task 13: Final Verification

**Step 1: Verify no `mindtorch_v2` references remain**

```bash
cd /Users/lvyufeng/Projects/candle
grep -r "mindtorch_v2" src/ tests/ | grep -v __pycache__ | wc -l
grep -r "mindtorch" src/ tests/ | grep -v __pycache__ | wc -l
```

Expected: 0 for both

**Step 2: Verify directory structure**

```bash
ls -la /Users/lvyufeng/Projects/candle/
ls -la /Users/lvyufeng/Projects/candle/src/candle/
ls -la /Users/lvyufeng/Projects/candle/tests/
ls -la /Users/lvyufeng/Projects/candle/.github/workflows/
```

**Step 3: Verify package can be found by setuptools**

```bash
cd /Users/lvyufeng/Projects/candle
python -c "from setuptools import find_packages; print(find_packages(where='src'))"
```

Expected: `['candle', 'candle.nn', 'candle.optim', ...]`

**Step 4: Run a quick import test**

```bash
cd /Users/lvyufeng/Projects/candle
PYTHONPATH=src python -c "import candle; print(candle.__version__)"
```

Expected: `0.1.0`

**Step 5: Commit any final fixes**

```bash
cd /Users/lvyufeng/Projects/candle
git add -A
git commit -m "chore: final cleanup and verification"
```

---

## Summary of Changes

| Component | Source (mindnlp) | Target (candle) |
|-----------|-----------------|-----------------|
| Source code | `src/mindtorch_v2/` | `src/candle/` |
| Tests | `tests/mindtorch_v2/` | `tests/` |
| Package name | `mindtorch_v2` | `candle` |
| CI workflow | `.github/workflows/ci_pipeline.yaml` | `.github/workflows/ci.yaml` |
| Release workflow | `.github/workflows/make_wheel_releases.yml` | `.github/workflows/release.yaml` |
| Setup | `setup.py` (shared with mindnlp) | `pyproject.toml` (standalone) |
| Pylint config | `.github/pylint.conf` | `.github/pylint.conf` |
| Test runner | `tests/run_test_v2.py` | `tests/run_test.py` |
| Env var | `MINDTORCH_TEST_FORCE_CPU_ONLY` | `CANDLE_TEST_FORCE_CPU_ONLY` |
