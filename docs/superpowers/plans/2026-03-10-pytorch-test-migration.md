# PyTorch Test Migration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run PyTorch's official Python test suite against candle with zero modifications to test files.

**Architecture:** Two layers — (1) `src/candle/testing/_internal/` implements PyTorch's test infrastructure APIs as a shipped part of candle, (2) `compat/pytorch/` clones PyTorch at runtime and runs its tests via pytest. The existing `compat/` root files move into `compat/transformers/` to create a unified multi-library compat directory.

**Tech Stack:** Python 3.10+, pytest, PyYAML, numpy, candle

---

## File Structure

### New files to create

```
src/candle/testing/_internal/
├── __init__.py                    # Re-exports from submodules
├── common_utils.py                # TestCase, run_tests, env flags, skip decorators
├── common_device_type.py          # instantiate_device_type_tests, @dtypes, @onlyCPU...
└── common_dtype.py                # all_types, floating_types, integral_types...

compat/
├── conftest_base.py               # Shared patches extracted from existing conftest.py

compat/transformers/               # Existing files moved here
├── run.py
├── conftest.py                    # Updated to import from conftest_base
├── models.yaml
├── xfail.yaml
├── requirements.txt
├── README.md
└── test-and-report.sh

compat/pytorch/                    # New
├── run.py                         # Clone pytorch, select tests, run pytest
├── conftest.py                    # Device mapping, compile no-op, xfail injection
├── tests.yaml                     # Test file tiers + skip rules
├── xfail.yaml                     # Known failures
├── requirements.txt
├── README.md
└── test-and-report.sh

.github/workflows/pytorch-tests.yaml
.claude/commands/pytorch-test.md
.claude/agents/pytorch-tester.md
```

### Existing files to modify

```
src/candle/testing/__init__.py     # Add _internal re-export
compat/conftest.py                 # DELETE (moved to compat/transformers/)
compat/run.py                      # DELETE (moved to compat/transformers/)
compat/models.yaml                 # DELETE (moved)
compat/xfail.yaml                  # DELETE (moved)
compat/requirements.txt            # DELETE (moved)
compat/README.md                   # DELETE (moved), replaced with index README
compat/test-and-report.sh          # DELETE (moved)
.gitignore                         # Update patterns
.claude/commands/compat.md         # Update paths
```

---

## Chunk 1: Refactor `compat/` + `compat/pytorch/` scaffolding

### Task 1: Move existing compat files to `compat/transformers/`

**Files:**
- Move: `compat/*.py`, `compat/*.yaml`, `compat/*.txt`, `compat/*.md`, `compat/*.sh` → `compat/transformers/`
- Create: `compat/README.md` (index)
- Modify: `.gitignore`

- [ ] **Step 1: Create `compat/transformers/` and move files**

```bash
mkdir -p compat/transformers
git mv compat/conftest.py compat/transformers/
git mv compat/run.py compat/transformers/
git mv compat/models.yaml compat/transformers/
git mv compat/xfail.yaml compat/transformers/
git mv compat/requirements.txt compat/transformers/
git mv compat/README.md compat/transformers/
git mv compat/test-and-report.sh compat/transformers/
```

- [ ] **Step 2: Fix internal path references in moved files**

In `compat/transformers/conftest.py`, update `_COMPAT_DIR` to point to
`compat/transformers/` instead of `compat/`:

```python
_COMPAT_DIR = Path(__file__).resolve().parent          # now compat/transformers/
_PROJECT_ROOT = _COMPAT_DIR.parent.parent              # was .parent
_SRC_DIR = _PROJECT_ROOT / "src"
```

In `compat/transformers/run.py`, update all path constants:

```python
COMPAT_DIR = Path(__file__).resolve().parent            # compat/transformers/
PROJECT_ROOT = COMPAT_DIR.parent.parent                 # was .parent
SRC_DIR = PROJECT_ROOT / "src"
TRANSFORMERS_DIR = COMPAT_DIR / "_transformers"
REPORTS_DIR = COMPAT_DIR / "_reports"
```

In `compat/transformers/test-and-report.sh`, update the cd path:

```bash
cd "$(dirname "$0")/../.."    # was "../.."
```

And the run.py path:

```bash
python compat/transformers/run.py ...  # was compat/run.py
```

- [ ] **Step 3: Update `.gitignore`**

Replace the existing compat entries with:

```
# Compat test artifacts
compat/*/_transformers/
compat/*/_pytorch/
compat/*/_reports/
```

- [ ] **Step 4: Update `.claude/commands/compat.md`**

Update the command to use `compat/transformers/run.py`:

```bash
python compat/transformers/run.py $PARSED_ARGS ...
```

- [ ] **Step 5: Write `compat/README.md` index**

```markdown
# Compatibility Tests

Run third-party library test suites against candle to verify drop-in compatibility.

| Directory | Library | Status |
|---|---|---|
| `transformers/` | HuggingFace transformers | Active |
| `pytorch/` | PyTorch official tests | In progress |

See each subdirectory's README for usage.
```

- [ ] **Step 6: Verify transformers compat still works**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  env USE_CANDLE=1 python compat/transformers/run.py --model bert \
  --json-report /tmp/tf-verify.json --tb=line
python compat/transformers/run.py --summarize /tmp/tf-verify.json
```

Expected: Same output as before (1 collection error for bert).

- [ ] **Step 7: Commit**

```bash
git add -A compat/ .gitignore .claude/commands/compat.md
git commit -m "refactor: move compat/ files to compat/transformers/ for multi-library support"
```

### Task 2: Extract `compat/conftest_base.py`

**Files:**
- Create: `compat/conftest_base.py`
- Modify: `compat/transformers/conftest.py`

- [ ] **Step 1: Create `compat/conftest_base.py`**

Extract shared logic from `compat/transformers/conftest.py`:
- `_apply_version_spoof()` (version spoofing)
- `_CandleTorchFinder` class (meta path finder) — **Note: this is the compat-layer finder, not candle's built-in one. Kept for transformers which needs extra stubs.**
- `_install_torch_finder()`
- `_bypass_transformers_dep_check()` — rename to `_bypass_dep_check(lib_name)`
- `_apply_safetensors_patch()`
- `_apply_torch_npu_shim()`
- xfail loading/matching helpers (`_load_xfail_config`, `_match_xfail`, `pytest_collection_modifyitems`)

The base file exports these as importable functions. Each `conftest.py` calls
the ones it needs.

- [ ] **Step 2: Update `compat/transformers/conftest.py`**

Replace inline implementations with imports from `conftest_base`:

```python
import sys
from pathlib import Path

_COMPAT_DIR = Path(__file__).resolve().parent
_COMPAT_ROOT = _COMPAT_DIR.parent
sys.path.insert(0, str(_COMPAT_ROOT))

from conftest_base import (
    apply_version_spoof,
    install_torch_finder,
    apply_module_stubs_for_transformers,
    apply_safetensors_patch,
    apply_torch_npu_shim,
    bypass_dep_check,
    load_xfail_config,
    match_xfail,
)
```

Keep transformers-specific stubs (module stubs for `torch.backends.cuda`,
`torch.hub`, `torch.library`, etc.) in `compat/transformers/conftest.py`.

- [ ] **Step 3: Verify transformers compat still works**

Same verification command as Task 1 Step 6.

- [ ] **Step 4: Commit**

```bash
git add compat/conftest_base.py compat/transformers/conftest.py
git commit -m "refactor: extract shared compat patches to conftest_base.py"
```

### Task 3: Create `compat/pytorch/` scaffolding

**Files:**
- Create: `compat/pytorch/run.py`
- Create: `compat/pytorch/conftest.py`
- Create: `compat/pytorch/tests.yaml`
- Create: `compat/pytorch/xfail.yaml`
- Create: `compat/pytorch/requirements.txt`
- Create: `compat/pytorch/README.md`
- Create: `compat/pytorch/test-and-report.sh`

- [ ] **Step 1: Create `compat/pytorch/tests.yaml`**

```yaml
pytorch_ref: "v2.5.0"

tier1_mechanism:
  - test_tensor.py
  - test_torch.py
  - test_autograd.py

tier2_mechanism:
  - test_nn.py
  - test_ops.py
  - test_modules.py
  - test_linalg.py

tier1_gpu:
  - test_cuda.py

tier2_gpu:
  - test_ops.py

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
  - "*functorch*"

skip_markers:
  - slow
  - skipIfTorchDynamo
```

- [ ] **Step 2: Create `compat/pytorch/xfail.yaml`**

```yaml
# Known failures — populated after first run
# Format: test node ID pattern + reason

_global:
  - pattern: "test_.*compile.*"
    reason: "torch.compile not implemented"
  - pattern: "test_.*dynamo.*"
    reason: "torch._dynamo not implemented"
  - pattern: "test_.*export.*"
    reason: "torch.export not implemented"
```

- [ ] **Step 3: Create `compat/pytorch/run.py`**

Same structure as `compat/transformers/run.py` but adapted for PyTorch:
- Clones `pytorch/pytorch` at `v2.5.0` to `compat/pytorch/_pytorch/`
- Maps test file names to `_pytorch/test/<file>`
- Generates bridge conftest in `_pytorch/test/`
- Supports `--tier mechanism:1`, `--tier gpu:1`, `--file test_tensor.py`
- Supports `--summarize`
- Uses `--continue-on-collection-errors`

Key differences from transformers run.py:
- Clone URL: `https://github.com/pytorch/pytorch.git`
- Test dir: `_pytorch/test/` (not `_pytorch/tests/models/`)
- Tier structure: `mechanism:N`, `gpu:N`, `mps`, `distributed`
- No `--model` flag; uses `--file` and `--tier`

- [ ] **Step 4: Create `compat/pytorch/conftest.py`**

Minimal — no import redirection (candle's `.pth` handles that):

```python
"""PyTorch test compatibility conftest.

Unlike transformers conftest, this does NOT do import redirection.
Candle's .pth + meta path finder handles torch→candle aliasing.
This conftest only handles:
  - Device mapping (cuda → npu)
  - torch.compile no-op
  - xfail injection from xfail.yaml
  - Skip markers from tests.yaml
"""
import fnmatch
from pathlib import Path
import yaml

_COMPAT_DIR = Path(__file__).resolve().parent

def _load_xfail_config():
    xfail_path = _COMPAT_DIR / "xfail.yaml"
    if not xfail_path.exists():
        return {}
    with open(xfail_path) as f:
        return yaml.safe_load(f) or {}

def _match_xfail(nodeid, entries):
    for entry in entries:
        if isinstance(entry, str):
            if fnmatch.fnmatch(nodeid, entry):
                return "known failure"
        elif isinstance(entry, dict):
            pattern = entry.get("pattern", "")
            reason = entry.get("reason", "known failure")
            if fnmatch.fnmatch(nodeid, f"*{pattern}*"):
                return reason
    return None

def pytest_collection_modifyitems(config, items):
    import pytest
    xfail_cfg = _load_xfail_config()
    if not xfail_cfg:
        return
    global_patterns = xfail_cfg.get("_global", [])
    for item in items:
        reason = _match_xfail(item.nodeid, global_patterns)
        if reason:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
```

- [ ] **Step 5: Create `compat/pytorch/requirements.txt`**

```
pyyaml>=6.0
pytest>=7.2.0,<9.0.0
pytest-json-report>=1.5
```

- [ ] **Step 6: Create `compat/pytorch/README.md`**

Document usage, tiers, xfail workflow (similar to transformers README).

- [ ] **Step 7: Create `compat/pytorch/test-and-report.sh`**

Shell script wrapper similar to `compat/transformers/test-and-report.sh`.

- [ ] **Step 8: Commit**

```bash
git add compat/pytorch/
git commit -m "feat: add compat/pytorch/ scaffolding for PyTorch test runner"
```

---

## Chunk 2: `candle.testing._internal` P0 implementation

### Task 4: Create `common_dtype.py`

**Files:**
- Create: `src/candle/testing/_internal/__init__.py`
- Create: `src/candle/testing/_internal/common_dtype.py`
- Modify: `src/candle/testing/__init__.py`

This is the simplest module — pure data, no dependencies on TestCase.

- [ ] **Step 1: Create `_internal/__init__.py`**

```python
"""candle.testing._internal — PyTorch-compatible test infrastructure."""
```

- [ ] **Step 2: Write `common_dtype.py`**

Implement dtype enumeration functions that match PyTorch's API:

```python
import candle as torch

def all_types():
    return (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float16, torch.float32, torch.float64)

def all_types_and_complex():
    return all_types() + (torch.complex64, torch.complex128)

def floating_types():
    return (torch.float32, torch.float64)

def floating_types_and_half():
    return (torch.float16, torch.float32, torch.float64)

def floating_and_complex_types():
    return floating_types() + (torch.complex64, torch.complex128)

def integral_types():
    return (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

def get_all_dtypes(include_half=True, include_bfloat16=True,
                   include_complex=True, include_bool=True):
    dtypes = list(all_types())
    if include_bool:
        dtypes.append(torch.bool)
    if include_half:
        dtypes.append(torch.float16)
    if include_bfloat16 and hasattr(torch, 'bfloat16'):
        dtypes.append(torch.bfloat16)
    if include_complex:
        dtypes.extend([torch.complex64, torch.complex128])
    return tuple(dtypes)
```

- [ ] **Step 3: Update `_internal/__init__.py`**

```python
from .common_dtype import (
    all_types, all_types_and_complex, floating_types,
    floating_types_and_half, floating_and_complex_types,
    integral_types, get_all_dtypes,
)
```

- [ ] **Step 4: Update `src/candle/testing/__init__.py`**

Add `_internal` subpackage:

```python
from . import _internal
```

- [ ] **Step 5: Verify import works**

```bash
USE_CANDLE=1 python -c "from torch.testing._internal.common_dtype import all_types; print(all_types())"
```

Expected: tuple of candle dtype objects.

- [ ] **Step 6: Commit**

```bash
git add src/candle/testing/
git commit -m "feat: add candle.testing._internal.common_dtype"
```

### Task 5: Create `common_utils.py` — TestCase + environment flags

**Files:**
- Create: `src/candle/testing/_internal/common_utils.py`
- Modify: `src/candle/testing/_internal/__init__.py`

This is the largest P0 module. `TestCase` is the base class for all PyTorch tests.

- [ ] **Step 1: Write `common_utils.py`**

Core contents:

```python
"""Common test utilities — TestCase, run_tests, environment detection."""
import os
import sys
import unittest
import functools
import contextlib
import tempfile

import candle as torch
import numpy as np

# ---------------------------------------------------------------------------
# Environment flags (match PyTorch's names)
# ---------------------------------------------------------------------------
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform == "linux"

TEST_CUDA = torch.cuda.is_available() if hasattr(torch, "cuda") else False
TEST_MPS = torch.mps.is_available() if hasattr(torch, "mps") else False
TEST_NPU = torch.npu.is_available() if hasattr(torch, "npu") else False

# Map CUDA tests to NPU when appropriate
TEST_CUDA = TEST_CUDA or TEST_NPU

TEST_MULTIGPU = False
if TEST_CUDA and hasattr(torch.cuda, "device_count"):
    TEST_MULTIGPU = torch.cuda.device_count() > 1

# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------
class TestCase(unittest.TestCase):
    """PyTorch-compatible test case base class."""
    precision = 1e-5
    rel_tol = 0

    def setUp(self):
        super().setUp()

    def assertTensorsEqual(self, a, b, prec=None):
        if prec is None:
            prec = self.precision
        np.testing.assert_allclose(
            a.detach().cpu().numpy(),
            b.detach().cpu().numpy(),
            atol=prec, rtol=self.rel_tol,
        )

    def assertEqual(self, x, y, msg=None, *, atol=None, rtol=None):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            _atol = atol if atol is not None else self.precision
            _rtol = rtol if rtol is not None else self.rel_tol
            torch.testing.assert_close(x, y, atol=_atol, rtol=_rtol, msg=msg)
        else:
            super().assertEqual(x, y, msg=msg)

# ---------------------------------------------------------------------------
# Skip decorators
# ---------------------------------------------------------------------------
def skipIfNoCuda(fn):
    return unittest.skipIf(not TEST_CUDA, "No CUDA/NPU device")(fn)

def skipIfNoMPS(fn):
    return unittest.skipIf(not TEST_MPS, "No MPS device")(fn)

def slowTest(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if os.environ.get("PYTORCH_TEST_SKIP_SLOW", "0") == "1":
            raise unittest.SkipTest("slow test")
        return fn(*args, **kwargs)
    return wrapper

# ---------------------------------------------------------------------------
# run_tests
# ---------------------------------------------------------------------------
def run_tests():
    unittest.main()

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def make_tensor(*shape, dtype, device="cpu", low=None, high=None,
                requires_grad=False, **kwargs):
    return torch.testing.make_tensor(
        *shape, dtype=dtype, device=device, low=low, high=high,
        requires_grad=requires_grad, **kwargs
    )

@contextlib.contextmanager
def freeze_rng_state():
    # Save and restore RNG state
    rng_state = torch.random.get_rng_state() if hasattr(torch.random, "get_rng_state") else None
    try:
        yield
    finally:
        if rng_state is not None:
            torch.random.set_rng_state(rng_state)

def parametrize(arg_name, arg_values):
    """Simple parametrize decorator compatible with unittest.TestCase."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self):
            for val in arg_values:
                with self.subTest(**{arg_name: val}):
                    fn(self, **{arg_name: val})
        return wrapper
    return decorator

def subtest(arg_values):
    """Decorator form of subTest for parametrize-like usage."""
    return parametrize("x", arg_values)

# Re-export commonly used names
TEST_WITH_ROCM = False
TEST_WITH_ASAN = False
TEST_WITH_TSAN = False
```

- [ ] **Step 2: Update `_internal/__init__.py`**

Add imports from `common_utils`:

```python
from .common_utils import (
    TestCase, run_tests, make_tensor,
    IS_WINDOWS, IS_MACOS, IS_LINUX,
    TEST_CUDA, TEST_MPS, TEST_NPU, TEST_MULTIGPU,
    skipIfNoCuda, skipIfNoMPS, slowTest,
    freeze_rng_state, parametrize, subtest,
    TEST_WITH_ROCM, TEST_WITH_ASAN, TEST_WITH_TSAN,
)
```

- [ ] **Step 3: Verify import works**

```bash
USE_CANDLE=1 python -c "
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_CUDA
print('TestCase:', TestCase)
print('TEST_CUDA:', TEST_CUDA)
"
```

- [ ] **Step 4: Commit**

```bash
git add src/candle/testing/_internal/
git commit -m "feat: add candle.testing._internal.common_utils (TestCase, run_tests, env flags)"
```

### Task 6: Create `common_device_type.py` — device parameterization

**Files:**
- Create: `src/candle/testing/_internal/common_device_type.py`
- Modify: `src/candle/testing/_internal/__init__.py`

This is the most critical module — `instantiate_device_type_tests` generates
device-specific test classes from generic test methods.

- [ ] **Step 1: Write `common_device_type.py`**

Core contents:

```python
"""Device-type test infrastructure — instantiate_device_type_tests, @dtypes, etc."""
import os
import unittest
import functools
from typing import List

import candle as torch

from .common_utils import TEST_CUDA, TEST_MPS, TEST_NPU
from .common_dtype import all_types, floating_types

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
_DEVICE_MAP = {
    "cuda": "npu" if TEST_NPU and not TEST_CUDA else "cuda",
}

def _get_available_devices():
    devices = ["cpu"]
    if TEST_CUDA:
        devices.append("cuda")
    elif TEST_NPU:
        devices.append("npu")
    if TEST_MPS:
        devices.append("mps")
    env_devices = os.environ.get("CANDLE_TEST_DEVICES", "")
    if env_devices:
        devices = [d.strip() for d in env_devices.split(",")]
    return devices

# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
def dtypes(*dtype_args):
    """Parametrize a test over multiple dtypes."""
    def decorator(fn):
        fn._dtypes = dtype_args
        return fn
    return decorator

def onlyCPU(fn):
    fn._only_device = "cpu"
    return fn

def onlyCUDA(fn):
    fn._only_device = "cuda"
    return fn

def onlyNativeDeviceTypes(fn):
    fn._only_native = True
    return fn

def deviceCountAtLeast(count):
    def decorator(fn):
        fn._min_device_count = count
        return fn
    return decorator

def skipCPUIf(condition, reason=""):
    def decorator(fn):
        fn._skip_cpu_if = (condition, reason)
        return fn
    return decorator

def skipCUDAIf(condition, reason=""):
    def decorator(fn):
        fn._skip_cuda_if = (condition, reason)
        return fn
    return decorator

# ---------------------------------------------------------------------------
# instantiate_device_type_tests
# ---------------------------------------------------------------------------
def instantiate_device_type_tests(test_class, scope, except_for=None, only_for=None):
    """Generate device-specific test classes from a generic test class.

    For a class TestFoo with test_bar(self, device, dtype), generates:
      - TestFooCPU.test_bar_cpu_float32(self)
      - TestFooCPU.test_bar_cpu_float64(self)
      - TestFooNPU.test_bar_npu_float32(self) (if NPU available)
      - etc.
    """
    devices = _get_available_devices()
    if only_for:
        devices = [d for d in devices if d in only_for]
    if except_for:
        devices = [d for d in devices if d not in except_for]

    for device in devices:
        device_suffix = device.upper()
        class_name = f"{test_class.__name__}{device_suffix}"

        # Create a new class inheriting from the test class
        device_class = type(class_name, (test_class,), {"device_type": device})

        # For each test method, generate device+dtype variants
        for attr_name in list(dir(test_class)):
            if not attr_name.startswith("test_"):
                continue
            fn = getattr(test_class, attr_name)
            if not callable(fn):
                continue

            # Check device-only filters
            only_device = getattr(fn, "_only_device", None)
            if only_device and only_device != device and not (
                only_device == "cuda" and device == "npu"
            ):
                continue

            # Check skip conditions
            skip_cpu = getattr(fn, "_skip_cpu_if", None)
            if skip_cpu and device == "cpu" and skip_cpu[0]:
                continue
            skip_cuda = getattr(fn, "_skip_cuda_if", None)
            if skip_cuda and device in ("cuda", "npu") and skip_cuda[0]:
                continue

            # Get dtype list
            dtype_list = getattr(fn, "_dtypes", None)

            if dtype_list:
                # Generate one test per dtype
                for dt in dtype_list:
                    dt_name = str(dt).split(".")[-1]
                    test_name = f"{attr_name}_{device}_{dt_name}"

                    def make_test(f, d, dtype):
                        @functools.wraps(f)
                        def test_fn(self):
                            return f(self, device=d, dtype=dtype)
                        return test_fn

                    setattr(device_class, test_name, make_test(fn, device, dt))
                # Remove original
                if hasattr(device_class, attr_name):
                    delattr(device_class, attr_name)
            else:
                # Single test with device arg
                test_name = f"{attr_name}_{device}"

                def make_test_no_dtype(f, d):
                    @functools.wraps(f)
                    def test_fn(self):
                        import inspect
                        sig = inspect.signature(f)
                        if "dtype" in sig.parameters:
                            return f(self, device=d, dtype=torch.float32)
                        return f(self, device=d)
                    return test_fn

                setattr(device_class, test_name, make_test_no_dtype(fn, device))
                if hasattr(device_class, attr_name):
                    delattr(device_class, attr_name)

        # Register in caller's scope
        scope[class_name] = device_class

    # Remove original class from scope
    if test_class.__name__ in scope:
        del scope[test_class.__name__]
```

- [ ] **Step 2: Update `_internal/__init__.py`**

Add imports from `common_device_type`.

- [ ] **Step 3: Verify import and basic functionality**

```bash
USE_CANDLE=1 python -c "
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, dtypes, onlyCPU
)
from torch.testing._internal.common_utils import TestCase

class TestFoo(TestCase):
    def test_bar(self, device):
        print(f'  test_bar on {device}')

scope = {}
instantiate_device_type_tests(TestFoo, scope)
print('Generated classes:', list(scope.keys()))
"
```

Expected: `Generated classes: ['TestFooCPU']` (CPU only on this machine).

- [ ] **Step 4: Commit**

```bash
git add src/candle/testing/_internal/
git commit -m "feat: add candle.testing._internal.common_device_type (instantiate_device_type_tests)"
```

---

## Chunk 3: Smoke test + CI + slash command + agent

### Task 7: End-to-end smoke test with `test_tensor.py`

**Files:**
- Modify: `compat/pytorch/run.py` (if needed based on smoke test)
- Modify: `compat/pytorch/xfail.yaml` (populate from results)

- [ ] **Step 1: Clone pytorch and setup**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  pip install -r compat/pytorch/requirements.txt
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  python compat/pytorch/run.py --setup-only
```

- [ ] **Step 2: Run test_tensor.py smoke test**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp \
  env USE_CANDLE=1 python compat/pytorch/run.py \
  --file test_tensor.py \
  --json-report /tmp/pytorch-smoke.json \
  --tb=line
```

- [ ] **Step 3: Analyze results and populate xfail.yaml**

```bash
python compat/pytorch/run.py --summarize /tmp/pytorch-smoke.json
```

Based on results, add collection errors and systematic failures to `xfail.yaml`.

- [ ] **Step 4: Commit**

```bash
git add compat/pytorch/xfail.yaml
git commit -m "feat: populate pytorch xfail.yaml from initial test_tensor.py run"
```

### Task 8: CI workflow + slash command + agent

**Files:**
- Create: `.github/workflows/pytorch-tests.yaml`
- Create: `.claude/commands/pytorch-test.md`
- Create: `.claude/agents/pytorch-tester.md`

- [ ] **Step 1: Create `.github/workflows/pytorch-tests.yaml`**

```yaml
name: PyTorch Compat

on:
  schedule:
    - cron: "0 3 * * *"
  workflow_dispatch:
    inputs:
      tier: { required: false, default: "mechanism:1" }
      file: { required: false, default: "" }
  pull_request:
    paths:
      - "src/candle/**"
      - "compat/pytorch/**"

jobs:
  pr-gate:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e . && pip install -r compat/pytorch/requirements.txt
      - run: USE_CANDLE=1 python compat/pytorch/run.py --tier mechanism:1 --gate-only --tb=short

  nightly:
    if: github.event_name != 'pull_request'
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e . && pip install -r compat/pytorch/requirements.txt
      - run: |
          TIER="${{ github.event.inputs.tier || 'mechanism:1' }}"
          FILE="${{ github.event.inputs.file || '' }}"
          ARGS="--tier $TIER"
          [ -n "$FILE" ] && ARGS="--file $FILE"
          USE_CANDLE=1 python compat/pytorch/run.py $ARGS \
            --json-report compat/pytorch/_reports/report.json -v --tb=short
      - uses: actions/upload-artifact@v4
        if: always()
        with: { name: pytorch-compat-report, path: "compat/pytorch/_reports/" }
      - if: always()
        run: python compat/pytorch/run.py --summarize compat/pytorch/_reports/report.json >> "$GITHUB_STEP_SUMMARY"
```

- [ ] **Step 2: Create `.claude/commands/pytorch-test.md`**

```markdown
Run PyTorch official tests against candle and report results.

## Usage
/pytorch-test                     # run tier1 mechanism tests
/pytorch-test test_tensor.py      # run single file
/pytorch-test --tier gpu:1        # CUDA→NPU tests

## Instructions
1. Parse $ARGUMENTS (empty → --tier mechanism:1, file name → --file <name>)
2. Run: USE_CANDLE=1 python compat/pytorch/run.py $ARGS --json-report /tmp/pt-report.json -v --tb=short
3. Summarize: python compat/pytorch/run.py --summarize /tmp/pt-report.json
4. For each failure: report root cause, suggest which candle file needs fixing
5. Suggest xfail.yaml entries and issues to file

Labels: pytorch-compat/import-error, pytorch-compat/missing-op,
        pytorch-compat/wrong-result, pytorch-compat/testing-infra
```

- [ ] **Step 3: Create `.claude/agents/pytorch-tester.md`**

Agent definition following same pattern as `compat-tester.md`:
- Run tests → parse report → deduplicate by root cause
- Check existing issues → file new issues → update xfail.yaml
- Issue labels: `pytorch-compat/*`
- DO NOT modify candle source — only file issues

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/pytorch-tests.yaml .claude/commands/pytorch-test.md .claude/agents/pytorch-tester.md
git commit -m "feat: add CI workflow, slash command, and agent for PyTorch compat tests"
```

---

## Implementation Order Summary

| Task | Description | Depends on |
|---|---|---|
| 1 | Move compat/ to compat/transformers/ | — |
| 2 | Extract conftest_base.py | 1 |
| 3 | Create compat/pytorch/ scaffolding | 1 |
| 4 | Implement common_dtype.py | — |
| 5 | Implement common_utils.py (TestCase) | 4 |
| 6 | Implement common_device_type.py | 5 |
| 7 | Smoke test test_tensor.py | 3, 6 |
| 8 | CI + slash command + agent | 7 |
