# Cython Runtime Core Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Candle's eager dispatch and autograd runtime hot paths Cython-first, replacing the torch C++-style runtime-critical layers with a canonical Cython Runtime Core while preserving PyTorch-compatible behavior.

**Architecture:** Keep public API, schema definitions, registry mutation, and control-plane policy in Python, but make tensor runtime state, keyset construction, eager dispatch execution, autograd node storage, and generated autograd post-processing use a canonical Cython implementation. Migrate in phases: first freeze the runtime boundary and unify dispatch semantics, then standardize forward autograd on single-pass `autograd_post`, then move node/runtime hot paths deeper into Cython, and only after that optimize backward scheduler internals.

**Tech Stack:** Cython 3, setuptools `Extension` modules, generated autograd code from `tools/autograd/*.py`, pytest contract/cpu tests, Python benchmark helpers under `benchmarks/`

---

### Task 1: Add failing runtime-core import contract tests

**Files:**
- Modify: `tests/contract/test_dispatch_contract.py`
- Modify: `tests/contract/test_autograd_contract.py`
- Test: `tests/contract/test_dispatch_contract.py`
- Test: `tests/contract/test_autograd_contract.py`

**Step 1: Write the failing tests**

Append import/fallback contract tests that make the runtime-core policy explicit:

```python
import importlib
import sys

import pytest


def test_dispatcher_module_exports_cython_dispatch_entrypoints():
    from candle._dispatch import dispatcher

    assert hasattr(dispatcher, "dispatch")
    assert hasattr(dispatcher, "dispatch_with_keyset")
    assert callable(dispatcher.dispatch)
    assert callable(dispatcher.dispatch_with_keyset)


def test_autograd_runtime_exports_engine_entrypoints():
    from candle.autograd import engine

    assert hasattr(engine, "backward")
    assert hasattr(engine, "grad")
    assert hasattr(engine, "_run_backward")


def test_runtime_core_import_failure_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "candle._cython._dispatcher_core", None)
    monkeypatch.setitem(sys.modules, "candle._cython._autograd_engine", None)

    with pytest.raises((ImportError, ModuleNotFoundError)) as exc:
        importlib.reload(importlib.import_module("candle._dispatch.dispatcher"))

    assert "candle._cython" in str(exc.value) or "runtime core" in str(exc.value)
```

Add a matching autograd-side test that verifies the failure mode is intentional and descriptive rather than a raw unrelated import crash.

**Step 2: Run tests to verify they fail**

Run:
```bash
python -m pytest tests/contract/test_dispatch_contract.py::test_runtime_core_import_failure_is_actionable tests/contract/test_autograd_contract.py::test_autograd_runtime_exports_engine_entrypoints -v --tb=short
```

Expected: FAIL because the current runtime-core import policy is implicit and the failure mode is not fully codified.

**Step 3: Commit**

```bash
git add tests/contract/test_dispatch_contract.py tests/contract/test_autograd_contract.py
git commit -m "test: codify runtime core import contracts"
```

---

### Task 2: Add dispatch-key constant parity tests

**Files:**
- Create: `tests/contract/test_dispatch_key_constant_parity.py`
- Test: `tests/contract/test_dispatch_key_constant_parity.py`

**Step 1: Write the failing test**

Create `tests/contract/test_dispatch_key_constant_parity.py`:

```python
from candle._dispatch.keys import DispatchKey


def test_dispatch_key_numeric_values_match_runtime_core_expectations():
    expected = {
        "CPU": 1 << 15,
        "NPU": 1 << 13,
        "CUDA": 1 << 14,
        "Meta": 1 << 12,
        "AutogradCPU": 1 << 6,
        "AutogradNPU": 1 << 7,
        "AutogradCUDA": 1 << 8,
        "AutogradMeta": 1 << 10,
        "ADInplaceOrView": 1 << 4,
        "Autograd": 1 << 11,
        "Functionalize": 1 << 3,
        "Autocast": 1 << 19,
        "Pipeline": 1 << 1,
        "Python": 1 << 2,
        "PrivateUse2": 1 << 21,
        "PrivateUse3": 1 << 22,
    }
    actual = {name: int(getattr(DispatchKey, name)) for name in expected}
    assert actual == expected
```

Then extend the file with a second test that verifies `DispatchKeySet.from_tensors(...)` produces masks matching the same numeric values for representative CPU/MPS/meta/autograd combinations.

**Step 2: Run test to verify it passes or expose drift**

Run:
```bash
python -m pytest tests/contract/test_dispatch_key_constant_parity.py -v --tb=short
```

Expected: either PASS immediately or expose drift. If it passes immediately, keep it as a regression guard before touching runtime-core constants.

**Step 3: Commit**

```bash
git add tests/contract/test_dispatch_key_constant_parity.py
git commit -m "test: guard dispatch key constant parity"
```

---

### Task 3: Add failing Python-dispatch vs Cython-dispatch equivalence tests

**Files:**
- Create: `tests/cpu/test_dispatch_runtime_core_equivalence.py`
- Test: `tests/cpu/test_dispatch_runtime_core_equivalence.py`

**Step 1: Write the failing tests**

Create `tests/cpu/test_dispatch_runtime_core_equivalence.py` with a helper that invokes both paths:

```python
import candle as torch
from candle._dispatch.dispatcher import _py_dispatch_with_keyset, dispatch, current_dispatch_keyset
from candle._dispatch.keys import DispatchKeySet


def _fresh_keyset(*tensors):
    return DispatchKeySet.from_tensors(tensors, grad_enabled=False, pipeline_enabled=False, functionalize_enabled=False, device=None, autocast_enabled=False)


def test_cython_and_python_dispatch_match_plain_binary_result():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    keyset = _fresh_keyset(a, b)

    py = _py_dispatch_with_keyset("add", keyset, None, a, b)
    cy = dispatch("add", None, a, b)

    torch.testing.assert_close(py, cy)


def test_cython_and_python_dispatch_match_inplace_version_bump():
    a1 = torch.tensor([1.0])
    a2 = torch.tensor([1.0])
    inc = torch.tensor([2.0])

    keyset = _fresh_keyset(a1, inc)
    _py_dispatch_with_keyset("add_", keyset, None, a1, inc)
    dispatch("add_", None, a2, inc)

    assert a1._version_counter.value == a2._version_counter.value
```

Add one more case for a `requires_grad=True` op and assert that `type(result.grad_fn).__name__`, `result.requires_grad`, and `len(result.grad_fn.next_functions)` match.

**Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/cpu/test_dispatch_runtime_core_equivalence.py -v --tb=short
```

Expected: FAIL on at least one case because the two dispatch paths are currently maintained separately.

**Step 3: Commit**

```bash
git add tests/cpu/test_dispatch_runtime_core_equivalence.py
git commit -m "test: compare python and cython dispatch semantics"
```

---

### Task 4: Reduce `dispatcher.py` to a policy veneer around canonical Cython dispatch

**Files:**
- Modify: `src/candle/_dispatch/dispatcher.py`
- Test: `tests/cpu/test_dispatch_runtime_core_equivalence.py`
- Test: `tests/contract/test_dispatch_key_order.py`
- Test: `tests/cpu/test_torch_dispatch.py`

**Step 1: Make the canonical ownership explicit**

At the top of `src/candle/_dispatch/dispatcher.py`, add a short module docstring note that this file is the policy/fallback veneer and that `_cython._dispatcher_core` is the canonical eager runtime when available.

**Step 2: Keep one Python fallback, but stop duplicating hot-path policy**

Preserve `_py_dispatch_with_keyset` as a reference/fallback implementation, but restructure the public functions so that:

- `dispatch(...)` selects the Cython runtime-core path by default.
- `dispatch_with_keyset(...)` selects the Cython runtime-core path by default.
- the Python implementation remains available only as a fallback and explicit test reference.

Concretely, keep the existing Python implementation body but rename/publicly retain:

```python
_py_dispatch = dispatch
_py_dispatch_with_keyset = dispatch_with_keyset
```

Then assign the public names from `_cython._dispatcher_core` only once, near the bottom, and make the fallback explicit with comments.

**Step 3: Remove duplicated helper ownership where possible**

Do **not** delete helpers used by the fallback path, but stop treating them as the primary implementation. Keep these helpers as policy/control-plane helpers only:
- `_dispatch_torch_dispatch`
- `_infer_dispatch_device`
- `_wrap_dispatch_error`
- `_check_inplace_targets`
- `_pending_from_meta`

**Step 4: Run tests**

Run:
```bash
python -m pytest tests/cpu/test_dispatch_runtime_core_equivalence.py tests/contract/test_dispatch_key_order.py tests/cpu/test_torch_dispatch.py -v --tb=short
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/candle/_dispatch/dispatcher.py tests/cpu/test_dispatch_runtime_core_equivalence.py
git commit -m "refactor: make cython dispatcher core canonical"
```

---

### Task 5: Centralize runtime-core dispatch-key constants and assert parity in Cython-facing code

**Files:**
- Modify: `src/candle/_dispatch/keys.py`
- Modify: `src/candle/_cython/_dispatch.pyx`
- Modify: `src/candle/_cython/_dispatcher_core.pyx`
- Modify: `src/candle/_cython/_tensor_impl.pyx`
- Test: `tests/contract/test_dispatch_key_constant_parity.py`

**Step 1: Define a single Python-side constant mapping**

In `src/candle/_dispatch/keys.py`, add a constant dict near `DispatchKey`:

```python
DISPATCH_KEY_BITS = {
    "Pipeline": int(DispatchKey.Pipeline),
    "Python": int(DispatchKey.Python),
    "Functionalize": int(DispatchKey.Functionalize),
    "ADInplaceOrView": int(DispatchKey.ADInplaceOrView),
    "AutogradOther": int(DispatchKey.AutogradOther),
    "AutogradCPU": int(DispatchKey.AutogradCPU),
    "AutogradNPU": int(DispatchKey.AutogradNPU),
    "AutogradCUDA": int(DispatchKey.AutogradCUDA),
    "AutogradXPU": int(DispatchKey.AutogradXPU),
    "AutogradMeta": int(DispatchKey.AutogradMeta),
    "Autograd": int(DispatchKey.Autograd),
    "Meta": int(DispatchKey.Meta),
    "NPU": int(DispatchKey.NPU),
    "CUDA": int(DispatchKey.CUDA),
    "CPU": int(DispatchKey.CPU),
    "Autocast": int(DispatchKey.Autocast),
    "PrivateUse2": int(DispatchKey.PrivateUse2),
    "PrivateUse3": int(DispatchKey.PrivateUse3),
}
```

**Step 2: Add a Cython validation helper instead of trusting duplicated literals silently**

In `_dispatch.pyx`, `_dispatcher_core.pyx`, and `_tensor_impl.pyx`, keep the `DEF` literals for compile-time speed, but add small debug-only assertions during first import that compare the literals against `DISPATCH_KEY_BITS`.

The helper should raise a targeted `RuntimeError` if any bit drifts.

**Step 3: Run tests**

Run:
```bash
python -m pytest tests/contract/test_dispatch_key_constant_parity.py tests/contract/test_dispatch_key_order.py -v --tb=short
```

Expected: PASS.

**Step 4: Commit**

```bash
git add src/candle/_dispatch/keys.py src/candle/_cython/_dispatch.pyx src/candle/_cython/_dispatcher_core.pyx src/candle/_cython/_tensor_impl.pyx tests/contract/test_dispatch_key_constant_parity.py
git commit -m "refactor: guard runtime core dispatch key parity"
```

---

### Task 6: Add failing single-pass `autograd_post` equivalence tests

**Files:**
- Create: `tests/cpu/test_autograd_post_equivalence.py`
- Test: `tests/cpu/test_autograd_post_equivalence.py`

**Step 1: Write the failing tests**

Create `tests/cpu/test_autograd_post_equivalence.py`:

```python
import candle as torch
from candle._generated import variable_type as _VT


def test_autograd_post_matches_wrapper_for_unary_op():
    x1 = torch.tensor([1.0, 2.0], requires_grad=True)
    x2 = torch.tensor([1.0, 2.0], requires_grad=True)

    wrapped = _VT.exp_autograd(x1)
    raw = x2.exp()
    posted = _VT.exp_autograd_post(raw, x2, raw_keyset=None, active_keyset=None)

    assert type(wrapped.grad_fn).__name__ == type(posted.grad_fn).__name__
    assert wrapped.requires_grad is posted.requires_grad is True
    assert len(wrapped.grad_fn.next_functions) == len(posted.grad_fn.next_functions)


def test_autograd_post_matches_wrapper_for_binary_gradients():
    a1 = torch.tensor([1.0, 2.0], requires_grad=True)
    b1 = torch.tensor([3.0, 4.0], requires_grad=True)
    a2 = torch.tensor([1.0, 2.0], requires_grad=True)
    b2 = torch.tensor([3.0, 4.0], requires_grad=True)

    out1 = _VT.add_autograd(a1, b1)
    out1.sum().backward()

    raw = a2 + b2
    out2 = _VT.add_autograd_post(raw, a2, b2, raw_keyset=None, active_keyset=None)
    out2.sum().backward()

    torch.testing.assert_close(a1.grad, a2.grad)
    torch.testing.assert_close(b1.grad, b2.grad)
```

**Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/cpu/test_autograd_post_equivalence.py -v --tb=short
```

Expected: FAIL because direct calls with `raw_keyset` / `active_keyset` placeholders are not yet normalized and/or the two paths are not fully equivalent.

**Step 3: Commit**

```bash
git add tests/cpu/test_autograd_post_equivalence.py
git commit -m "test: add autograd post equivalence coverage"
```

---

### Task 7: Make `autograd_post` the canonical generated forward-attach path for simple ops

**Files:**
- Modify: `src/candle/_cython/_dispatcher_core.pyx`
- Modify: `src/candle/_dispatch/registration.py`
- Modify: `src/candle/_generated/registration.py`
- Test: `tests/cpu/test_autograd_post_equivalence.py`
- Test: `tests/cpu/test_dispatch_autograd_wrappers.py`
- Test: `tests/cpu/test_autograd.py`

**Step 1: Keep the current single-pass preference, but codify it**

In `_dispatcher_core.pyx`, leave the existing fast path:

```python
if has_autograd and autograd_post_fn is not None:
    ...
    result = autograd_post_fn(result, *args, raw_keyset=raw_keyset, active_keyset=active_keyset, **kwargs)
```

but add comments and an invariant check so this path is explicitly the preferred path for generated ops.

**Step 2: Add a marker helper in registration**

In `_dispatch/registration.py`, extend `register_autograd_post_kernels` to set a small boolean marker, for example:

```python
entry.has_single_pass_autograd = True
```

Then in `_dispatcher_core.pyx`, use that marker only for diagnostics/comments, not to change behavior.

**Step 3: Keep legacy autograd kernel registration as fallback**

Do **not** remove `register_autograd_kernels(...)` yet. The goal of this task is to make the primary path explicit, not to delete fallbacks prematurely.

**Step 4: Run tests**

Run:
```bash
python -m pytest tests/cpu/test_autograd_post_equivalence.py tests/cpu/test_dispatch_autograd_wrappers.py tests/cpu/test_autograd.py -v --tb=short
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/candle/_cython/_dispatcher_core.pyx src/candle/_dispatch/registration.py src/candle/_generated/registration.py tests/cpu/test_autograd_post_equivalence.py
git commit -m "refactor: codify single-pass autograd post path"
```

---

### Task 8: Add failing FastNode saved-state hot-path tests

**Files:**
- Create: `tests/contract/test_autograd_node_runtime_core.py`
- Test: `tests/contract/test_autograd_node_runtime_core.py`

**Step 1: Write the failing tests**

Create `tests/contract/test_autograd_node_runtime_core.py`:

```python
import candle as torch
from candle.autograd.node import Node


def test_node_saved_tensors_round_trip():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    node = Node(lambda grad: (grad,), (x,), name="DummyBackward0")
    node.save_for_backward(x)
    saved, = node.saved_tensors()
    assert saved is x


def test_node_release_saved_tensors_blocks_materialization():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    node = Node(lambda grad: (grad,), (x,), name="DummyBackward0")
    node.save_for_backward(x)
    node.release_saved_tensors()

    try:
        node.saved_tensors()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "Trying to backward through the graph a second time" in str(exc)
```

Add a third test that stores `node._saved_fields["self"]` and checks `_saved_self` resolution works through the Cython-backed `Node.__getattr__` path.

**Step 2: Run test to verify it fails or exposes gaps**

Run:
```bash
python -m pytest tests/contract/test_autograd_node_runtime_core.py -v --tb=short
```

Expected: PASS or expose gaps. Keep the file either way as a hot-path regression suite.

**Step 3: Commit**

```bash
git add tests/contract/test_autograd_node_runtime_core.py
git commit -m "test: guard cython node saved-state semantics"
```

---

### Task 9: Move more Node saved-state behavior into Cython-backed helpers

**Files:**
- Modify: `src/candle/_cython/_autograd_node.pyx`
- Modify: `src/candle/autograd/node.py`
- Test: `tests/contract/test_autograd_node_runtime_core.py`
- Test: `tests/contract/test_autograd_graph_node.py`
- Test: `tests/cpu/test_autograd_inplace.py`

**Step 1: Keep Python `autograd/node.py` as a veneer**

Do not reintroduce a Python implementation. Keep the public file as a re-export veneer over `_cython._autograd_node`.

**Step 2: Strengthen the Cython implementation**

In `_autograd_node.pyx`, make sure the Cython-owned `Node` remains the source of truth for:
- `save_for_backward`
- `saved_tensors`
- `release_saved_tensors`
- `_saved_*` and `_raw_saved_*` access
- `next_functions`

If any of those behaviors are still partially duplicated in Python elsewhere, remove the duplication and rely on the Cython type.

**Step 3: Run tests**

Run:
```bash
python -m pytest tests/contract/test_autograd_node_runtime_core.py tests/contract/test_autograd_graph_node.py tests/cpu/test_autograd_inplace.py -v --tb=short
```

Expected: PASS.

**Step 4: Commit**

```bash
git add src/candle/_cython/_autograd_node.pyx src/candle/autograd/node.py tests/contract/test_autograd_node_runtime_core.py
git commit -m "refactor: deepen cython node runtime ownership"
```

---

### Task 10: Add failing backward-engine runtime-core equivalence tests

**Files:**
- Create: `tests/contract/test_autograd_engine_runtime_core.py`
- Test: `tests/contract/test_autograd_engine_runtime_core.py`

**Step 1: Write the failing tests**

Create `tests/contract/test_autograd_engine_runtime_core.py`:

```python
import candle as torch
from candle.autograd import engine



def test_run_backward_accumulates_leaf_grad_once_per_path():
    a = torch.ones((2, 2), requires_grad=True)
    b = a * a
    c = b + b
    engine._run_backward((c.sum(),), (torch.tensor(1.0),), retain_graph=False, create_graph=False, accumulate_grad=True)
    assert a.grad is not None



def test_run_backward_respects_input_filtering():
    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([2.0], requires_grad=True)
    out = (a * a).sum()
    grad_a, grad_b = engine.grad((out,), (a, b), allow_unused=True)
    assert grad_a is not None
    assert grad_b is None
```

Add a third test that exercises reentrant backward through a hook, matching the contract already covered in `test_autograd_engine_topo.py`.

**Step 2: Run test to verify baseline**

Run:
```bash
python -m pytest tests/contract/test_autograd_engine_runtime_core.py tests/contract/test_autograd_engine_topo.py -v --tb=short
```

Expected: PASS or expose scheduler gaps. Keep the file as the direct runtime-core guard suite.

**Step 3: Commit**

```bash
git add tests/contract/test_autograd_engine_runtime_core.py
git commit -m "test: add runtime core backward engine coverage"
```

---

### Task 11: Cythonize backward scheduler internals without changing public semantics

**Files:**
- Modify: `src/candle/_cython/_autograd_engine.pyx`
- Modify: `src/candle/autograd/engine.py`
- Test: `tests/contract/test_autograd_engine_runtime_core.py`
- Test: `tests/contract/test_autograd_engine_topo.py`
- Test: `tests/contract/test_autograd_create_graph.py`
- Test: `tests/contract/test_autograd_retain_graph.py`

**Step 1: Keep `autograd/engine.py` as a veneer**

Leave `src/candle/autograd/engine.py` as the thin re-export layer it already is.

**Step 2: Improve Cython-owned scheduling internals incrementally**

In `_autograd_engine.pyx`, focus only on hot scheduler structures first:
- `_build_dependencies`
- `_GraphTask.received`
- `_GraphTask.node_grads`
- `_GraphTask.ready`
- the main `run()` loop bookkeeping

Do **not** redesign user-visible anomaly, hook, or error semantics in this task.

**Step 3: Run tests**

Run:
```bash
python -m pytest tests/contract/test_autograd_engine_runtime_core.py tests/contract/test_autograd_engine_topo.py tests/contract/test_autograd_create_graph.py tests/contract/test_autograd_retain_graph.py -v --tb=short
```

Expected: PASS.

**Step 4: Commit**

```bash
git add src/candle/_cython/_autograd_engine.pyx src/candle/autograd/engine.py tests/contract/test_autograd_engine_runtime_core.py
git commit -m "perf: move backward scheduler hot paths into cython"
```

---

### Task 12: Add eager runtime-core microbenchmarks

**Files:**
- Create: `benchmarks/runtime_core/__init__.py`
- Create: `benchmarks/runtime_core/runner.py`
- Create: `benchmarks/runtime_core/dispatch_cases.py`
- Create: `benchmarks/runtime_core/autograd_cases.py`
- Create: `benchmarks/runtime_core/report.py`
- Create: `tests/cpu/test_runtime_core_bench_smoke.py`
- Test: `tests/cpu/test_runtime_core_bench_smoke.py`

**Step 1: Write the smoke test first**

Create `tests/cpu/test_runtime_core_bench_smoke.py`:

```python
from benchmarks.runtime_core.runner import benchmark_op



def test_runtime_core_benchmark_smoke():
    samples = benchmark_op(lambda: 1 + 1, warmup=1, iters=3)
    assert len(samples) == 3
    assert all(sample >= 0 for sample in samples)
```

**Step 2: Implement the minimal benchmark runner**

In `benchmarks/runtime_core/runner.py`, implement a tiny helper modeled after `benchmarks/op_benchmark_npu/runner.py`:

```python
import time


def benchmark_op(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1_000_000.0)
    return samples
```

Add minimal dispatch and autograd benchmark case modules for:
- tiny `add`
- tiny unary chain
- tiny `requires_grad=True` binary chain
- tiny forward+backward shared-subgraph case

**Step 3: Run smoke test**

Run:
```bash
python -m pytest tests/cpu/test_runtime_core_bench_smoke.py -v --tb=short
```

Expected: PASS.

**Step 4: Commit**

```bash
git add benchmarks/runtime_core tests/cpu/test_runtime_core_bench_smoke.py
git commit -m "bench: add eager runtime core microbenchmarks"
```

---

### Task 13: Add generated-autograd metadata scaffolding for future Cython consumption

**Files:**
- Modify: `tools/autograd/gen_variable_type.py`
- Modify: `tools/autograd/gen_registration.py`
- Modify: `src/candle/_generated/variable_type.py`
- Modify: `src/candle/_generated/registration.py`
- Test: `tests/cpu/test_declarative_autograd.py`

**Step 1: Add a small generated metadata structure**

Update `gen_variable_type.py` so each generated op emits a metadata constant alongside the wrapper/post-wrapper, for example:

```python
ADD_AUTOGRAD_META = {
    "op": "add",
    "backward_cls": "AddBackward0",
    "has_post": True,
    "differentiable_inputs": ("self", "other"),
    "saved_inputs": (),
    "saved_outputs": (),
    "non_tensor_args": (),
    "multi_output": False,
}
```

This is not yet consumed by Cython runtime, but it prepares the generator for later descriptor-driven node creation.

**Step 2: Regenerate the files**

Run the project’s generation command or script for autograd codegen so `src/candle/_generated/variable_type.py` and `src/candle/_generated/registration.py` are updated.

**Step 3: Run tests**

Run:
```bash
python -m pytest tests/cpu/test_declarative_autograd.py tests/cpu/test_autograd_post_equivalence.py -v --tb=short
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tools/autograd/gen_variable_type.py tools/autograd/gen_registration.py src/candle/_generated/variable_type.py src/candle/_generated/registration.py
git commit -m "refactor: emit autograd post metadata for runtime core"
```

---

### Task 14: Verify the whole runtime-core migration slice

**Files:**
- Test: `tests/contract/test_dispatch_contract.py`
- Test: `tests/contract/test_autograd_contract.py`
- Test: `tests/contract/test_dispatch_key_order.py`
- Test: `tests/contract/test_dispatch_key_constant_parity.py`
- Test: `tests/cpu/test_dispatch_runtime_core_equivalence.py`
- Test: `tests/cpu/test_torch_dispatch.py`
- Test: `tests/cpu/test_autograd_post_equivalence.py`
- Test: `tests/contract/test_autograd_engine_runtime_core.py`
- Test: `tests/contract/test_autograd_engine_topo.py`
- Test: `tests/cpu/test_autograd_inplace.py`
- Test: `tests/cpu/test_runtime_core_bench_smoke.py`

**Step 1: Run the focused runtime-core test suite**

Run:
```bash
python -m pytest \
  tests/contract/test_dispatch_contract.py \
  tests/contract/test_autograd_contract.py \
  tests/contract/test_dispatch_key_order.py \
  tests/contract/test_dispatch_key_constant_parity.py \
  tests/cpu/test_dispatch_runtime_core_equivalence.py \
  tests/cpu/test_torch_dispatch.py \
  tests/cpu/test_autograd_post_equivalence.py \
  tests/contract/test_autograd_engine_runtime_core.py \
  tests/contract/test_autograd_engine_topo.py \
  tests/cpu/test_autograd_inplace.py \
  tests/cpu/test_runtime_core_bench_smoke.py \
  -v --tb=short
```

Expected: PASS.

**Step 2: Run pylint gate**

Run:
```bash
pylint src/candle/ --rcfile=.github/pylint.conf
```

Expected: PASS with zero errors.

**Step 3: Record benchmark numbers**

Run the new runtime-core microbenchmarks and save a brief markdown summary under `results/` or another existing non-doc artifact location already used in the repo.

**Step 4: Commit final verification changes**

```bash
git add tests/contract tests/cpu benchmarks/runtime_core src/candle tools/autograd src/candle/_generated
git commit -m "feat(cython): establish canonical runtime core for dispatch and autograd"
```

---

Plan complete and saved to `docs/plans/2026-03-21-cython-runtime-core-alignment.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
