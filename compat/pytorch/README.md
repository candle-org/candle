# PyTorch Official Test Compatibility

Run PyTorch's own test suite against candle to measure API compatibility.

## How it works

1. `run.py` shallow-clones PyTorch at a pinned tag (`v2.5.0`) into `_pytorch/`
2. A bridge `conftest.py` is generated inside `_pytorch/test/` that loads xfail rules
3. Candle's `.pth` + meta path finder handles `import torch` -> `import candle` aliasing
4. pytest runs the selected test files with `--confcutdir` and `--continue-on-collection-errors`
5. Results are collected as JSON and summarized as a human-readable table

## Prerequisites

```bash
pip install -e .
pip install -r compat/pytorch/requirements.txt
```

## Usage

### Default: tier 1 mechanism tests

```bash
python compat/pytorch/run.py
```

### Specific tier

```bash
python compat/pytorch/run.py --tier mechanism:1   # test_tensor, test_torch, test_autograd
python compat/pytorch/run.py --tier mechanism:2   # adds test_nn, test_ops, test_modules, test_linalg
python compat/pytorch/run.py --tier gpu:1         # test_cuda (CUDA->NPU mapping)
python compat/pytorch/run.py --tier gpu:2         # adds test_ops (GPU variant)
python compat/pytorch/run.py --tier mps           # test_mps
python compat/pytorch/run.py --tier distributed   # distributed tests
```

### Single file

```bash
python compat/pytorch/run.py --file test_tensor.py
python compat/pytorch/run.py --file test_tensor.py -k "test_add"
```

### Setup only (clone without running tests)

```bash
python compat/pytorch/run.py --setup-only
```

### Gate only (run pass_gate tests from xfail.yaml)

```bash
python compat/pytorch/run.py --gate-only
```

### View a previous report

```bash
python compat/pytorch/run.py --summarize compat/pytorch/_reports/latest.json
```

### Shell script (for CI / agents)

```bash
./compat/pytorch/test-and-report.sh                     # default tier
./compat/pytorch/test-and-report.sh test_tensor.py      # single file
./compat/pytorch/test-and-report.sh --tier mechanism:2   # specific tier
```

## Tier descriptions

| Tier | Category | Files | Description |
|------|----------|-------|-------------|
| `mechanism:1` | Core | test_tensor, test_torch, test_autograd | Fundamental tensor ops, torch API, autograd |
| `mechanism:2` | Extended | test_nn, test_ops, test_modules, test_linalg | Neural network, ops, modules, linear algebra |
| `gpu:1` | GPU core | test_cuda | CUDA tests (mapped to available GPU backend) |
| `gpu:2` | GPU extended | test_ops | GPU-specific op tests |
| `mps` | Apple GPU | test_mps | Metal Performance Shaders tests |
| `distributed` | Multi-process | test_c10d_gloo, test_c10d_nccl | Distributed communication |

## xfail workflow

1. Run tests -- many will fail initially
2. Add failing patterns to `xfail.yaml` with reasons
3. Fix candle bugs -- failures become XPASS (unexpected pass)
4. Remove fixed entries from `xfail.yaml`
5. Over time `xfail.yaml` shrinks as compatibility improves

### xfail.yaml format

```yaml
_global:
  - pattern: "test_.*compile.*"
    reason: "torch.compile not implemented"
```

## Adding new tiers

1. Add a new key to `tests.yaml` (e.g., `tier3_mechanism`)
2. List the test file names under it
3. Update `run.py:get_test_files_for_tier()` if the category is new
4. Update this README

## Directory structure

```
compat/pytorch/
  tests.yaml          # test file tiers and deselect/skip config
  xfail.yaml          # known failures
  conftest.py         # xfail injection hook
  run.py              # main entry script
  requirements.txt    # pip dependencies
  test-and-report.sh  # shell wrapper for CI/agents
  README.md           # this file
  _pytorch/           # (gitignored) cloned PyTorch repo
  _reports/           # (gitignored) JSON test reports
```
