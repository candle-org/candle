# PyTorch Test Migration Design

**Date**: 2026-03-10
**Status**: Approved

## Goal

Run PyTorch's official Python test suite against candle with zero modifications
to the test files, to verify full `import candle as torch` drop-in compatibility.

## Two Focus Areas

1. **Hardware-independent mechanism tests** — test_tensor.py, test_torch.py,
   test_autograd.py, etc. These validate core semantics and let agents develop
   alignment fixes.
2. **NPU tests** — PyTorch's CUDA tests with `cuda` → `npu` device mapping.
   These let agents develop NPU backend alignment.

## Architecture: Two Layers

### Layer 1 — `src/candle/testing/_internal/` (shipped with candle)

A from-scratch implementation of `torch.testing._internal` APIs. Candle's
existing `.pth` + meta path finder ensures `from torch.testing._internal
import TestCase` resolves to `candle.testing._internal.TestCase`. No import
redirection hacks in conftest.

**Implementation priority:**

| Priority | Module | Core APIs |
|---|---|---|
| P0 | `common_utils.py` | `TestCase`, `run_tests()`, `IS_WINDOWS`, `TEST_CUDA`, `skipIfNoXXX`, `slowTest` |
| P0 | `common_device_type.py` | `instantiate_device_type_tests()`, `@dtypes`, `@onlyCPU`, `@onlyCUDA`, `deviceCountAtLeast` |
| P0 | `common_dtype.py` | `all_types()`, `floating_types()`, `integral_types()`, `get_all_dtypes()` |
| P1 | `common_utils.py` ext | `make_tensor()`, `parametrize()`, `subtest()`, `freeze_rng_state()` |
| P2 | `common_cuda.py` | `TEST_CUDA`, `TEST_MULTIGPU`, `_get_torch_cuda_version()` |
| P3 | `opinfo/core.py` | `OpInfo`, `SampleInput`, `DecorateInfo` |
| P3 | `common_nn.py` | `NNTestCase`, NN test helpers |
| P3 | `common_distributed.py` | distributed test helpers |

**`instantiate_device_type_tests` design:**

- Reads `CANDLE_TEST_DEVICES` env var or auto-detects available devices
- For each device, generates a subclass (e.g. `TestFooCPU`, `TestFooNPU`)
  with `device` parameter injected
- `@dtypes` expands to parametrize over dtype combinations
- `@onlyCUDA` matches both `cuda` and `npu` (configurable)

**NPU device mapping:**

```python
DEVICE_MAP = {"cuda": "npu", "cuda:0": "npu:0", "cuda:1": "npu:1"}
```

Applied inside `instantiate_device_type_tests` — when generating device
variants, `cuda` is transparently replaced with `npu`.

### Layer 2 — `compat/pytorch/` (test runner)

Runtime-clones PyTorch at a pinned tag, runs its test files via pytest.

**Directory structure:**

```
compat/
├── conftest_base.py             # shared patches (extracted from existing conftest.py)
│
├── transformers/                # existing — moved from compat/ root
│   ├── run.py, conftest.py, models.yaml, xfail.yaml, ...
│   ├── _transformers/           # gitignored clone
│   └── _reports/
│
├── pytorch/                     # NEW
│   ├── run.py                   # clone pytorch, select tests, run pytest
│   ├── conftest.py              # device mapping, compile no-op, xfail injection
│   ├── tests.yaml               # test file tiers + skip rules
│   ├── xfail.yaml               # known failures
│   ├── requirements.txt
│   ├── README.md
│   ├── test-and-report.sh       # shell entry for agents
│   ├── _pytorch/                # gitignored clone
│   └── _reports/
```

**`tests.yaml` tiers:**

```yaml
pytorch_ref: "v2.5.0"

tier1_mechanism:       # hardware-independent, CPU
  - test_tensor.py
  - test_torch.py
  - test_autograd.py

tier2_mechanism:
  - test_nn.py
  - test_ops.py
  - test_modules.py
  - test_linalg.py

tier1_gpu:             # CUDA→NPU mapped
  - test_cuda.py

tier2_gpu:
  - test_ops.py        # re-run with device=npu

mps:
  - test_mps.py

distributed:
  - distributed/test_c10d_gloo.py
  - distributed/test_c10d_nccl.py

deselect_patterns:
  - "*dynamo*"
  - "*inductor*"
  - "*compile*"
  - "*export*"
  - "*fx*"
  - "*quantization*"
  - "*onnx*"
```

**`conftest.py` responsibilities (minimal — no import redirection):**

| Patch | Purpose |
|---|---|
| device mapping | `cuda` → `npu` in `instantiate_device_type_tests` |
| `torch.compile` no-op | compile decorators become identity |
| xfail injection | mark known failures from `xfail.yaml` |
| skip markers | skip dynamo/inductor/quantization tests |

**`run.py` CLI:**

```
python compat/pytorch/run.py                              # tier1_mechanism
python compat/pytorch/run.py --tier mechanism:2           # tier1+2 mechanism
python compat/pytorch/run.py --tier gpu:1                 # CUDA→NPU
python compat/pytorch/run.py --file test_tensor.py        # single file
python compat/pytorch/run.py --file test_tensor.py -k add # single test
python compat/pytorch/run.py --summarize report.json      # view report
```

## Refactoring Existing `compat/`

Current `compat/` root files move to `compat/transformers/`:

| Before | After |
|---|---|
| `compat/run.py` | `compat/transformers/run.py` |
| `compat/conftest.py` | `compat/transformers/conftest.py` |
| `compat/models.yaml` | `compat/transformers/models.yaml` |
| `compat/xfail.yaml` | `compat/transformers/xfail.yaml` |
| `compat/requirements.txt` | `compat/transformers/requirements.txt` |
| `compat/README.md` | `compat/transformers/README.md` |
| `compat/test-and-report.sh` | `compat/transformers/test-and-report.sh` |

Shared logic (version spoof, meta path finder, etc.) extracted to
`compat/conftest_base.py`.

## CI Strategy

### PR Gate (every PR)

Runs `pass_gate` tests from `xfail.yaml` — tests that are known to pass
and must not regress:

```yaml
# xfail.yaml
pass_gate:
  test_tensor.py:
    - "test_fill_*"
    - "test_zeros_*"
```

Fast (< 5 min), blocks PR on regression.

### Nightly

Full run of all tiers. Agent diffs against previous results, files issues
for new failures.

```yaml
# .github/workflows/pytorch-tests.yaml
on:
  pull_request:        # PR gate only
  schedule:
    - cron: "0 3 * * *"
  workflow_dispatch:
```

## Agent & Skill

**`/pytorch-test` slash command** — runs tests, summarizes results.

**`pytorch-tester` agent** — same workflow as `compat-tester`:

```
run tests → parse report → deduplicate by root cause
→ check existing issues → file new issues → update xfail.yaml
```

**Issue labels:**

| Label | Meaning |
|---|---|
| `pytorch-compat/import-error` | missing torch.* module/attribute |
| `pytorch-compat/missing-op` | operator not implemented |
| `pytorch-compat/wrong-result` | computation produces wrong result |
| `pytorch-compat/testing-infra` | candle.testing._internal missing API |
| `pytorch-compat/device-mapping` | NPU device mapping issue |

## End-to-End Workflow

```
Trigger (developer, CI, agent)
    │
    ▼
/pytorch-test or nightly CI
    │
    ▼
run.py: clone pytorch v2.5.0 → pytest → JSON report
    │
    ▼
pytorch-tester agent analyzes report
    │
    ├─ Collection error → issue (import-error or testing-infra)
    ├─ Missing op       → issue (missing-op)
    ├─ Wrong result     → issue (wrong-result)
    └─ Device issue     → issue (device-mapping)
    │
    ▼
Update xfail.yaml (reference issue #)
    │
    ▼
Other agents pick up issues → fix candle source → xfail→XPASS → remove entry
```

## Implementation Order

1. Refactor `compat/` — move transformers to subdirectory, extract shared base
2. Implement `candle.testing._internal` P0 (TestCase, instantiate_device_type_tests, dtypes)
3. Create `compat/pytorch/` scaffolding (run.py, conftest.py, tests.yaml)
4. Smoke test: run `test_tensor.py` end-to-end
5. Implement P1 (make_tensor, parametrize) based on smoke test gaps
6. Add CI workflow, slash command, agent
7. First full nightly run → populate xfail.yaml
8. Implement P2 (common_cuda, NPU mapping) → run GPU tests
9. Implement P3 (OpInfo, NN) as needed
