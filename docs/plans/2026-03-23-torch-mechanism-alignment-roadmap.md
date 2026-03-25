# Torch Mechanism Alignment Roadmap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the highest-impact mechanism gaps between candle and torch by first fixing correctness-critical alias/version/autograd invariants, then filling runtime stream gaps and finally hardening dispatcher/forward-AD semantics, while also aligning implementation strategy with torch’s performance profile by preferring Cython implementations for hot-path mechanisms that torch implements in C++.

**Architecture:** Treat this as a staged parity program, not a broad refactor. Preserve the current Storage/TensorImpl + dispatcher + Cython autograd architecture, but tighten the invariants that connect them: aliasing, view tracking, version bumping, output-slot identity, custom Function mutation semantics, and stream-aware runtime behavior. For any mechanism that sits on a hot path in torch’s C++ runtime (TensorImpl/autograd/dispatcher/runtime kernels/stream state), the default target in candle should be Cython rather than pure Python; Python shims remain acceptable only for orchestration, tests, and non-hot control paths.

**Tech Stack:** Python, Cython, candle dispatch registry/schema, Cython autograd runtime, backend runtimes (CPU/CUDA/NPU/MPS), pytest

---

## Priority Order

- **P0:** Correctness-critical semantic gaps that can silently produce wrong alias/version/autograd behavior.
- **P1:** Runtime/stream semantics needed for torch-like execution ordering and memory lifetime, especially CUDA.
- **P2:** Dispatcher and forward-AD completeness/hardening work that improves coverage and long-tail parity.
- **Cross-cutting performance rule:** When a task touches a hot-path mechanism that torch implements in C++, candle should land the steady-state implementation in Cython (`src/candle/_cython/`) or in backend kernels invoked from Cython-friendly hot paths. Avoid ending in Python-only implementations for hot-path semantics unless the task is explicitly scoped as a temporary correctness-first checkpoint with a follow-up Cython task in the same plan.

---

### Task 1: P0 — Lock in contract tests for alias/version/autograd invariants

**Files:**
- Modify: `tests/cpu/test_autograd_function.py`
- Modify: `tests/contract/test_inplace_view_rules.py`
- Create: `tests/contract/test_tensor_alias_version_contract.py`
- Create: `tests/contract/test_autograd_output_slots_contract.py`
- Reference: `src/candle/_tensor.py`
- Reference: `src/candle/_backends/common/view.py`
- Reference: `src/candle/_cython/_autograd_function.pyx`
- Reference: `src/candle/_cython/_autograd_node.pyx`
- Reference: `src/candle/_cython/_autograd_engine.pyx`

**Step 1: Write failing alias/version contract tests**

Create `tests/contract/test_tensor_alias_version_contract.py` with focused parity tests:

```python
import pytest
import candle as torch


def test_detach_shares_version_counter_with_source():
    base = torch.arange(4.0, requires_grad=True)
    detached = base.detach()
    before = detached._version_counter.value
    base._version_counter.bump()
    assert detached._version_counter.value == before + 1


def test_unary_inplace_preserves_view_aliasing():
    x = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
    v = x.view((4,))
    x.abs_()
    assert v.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_setitem_bumps_version_counter():
    x = torch.tensor([1.0, 2.0, 3.0])
    before = x._version_counter.value
    x[0] = torch.tensor(9.0)
    assert x._version_counter.value == before + 1
```

**Step 2: Write failing custom Function / output-slot tests**

Create `tests/contract/test_autograd_output_slots_contract.py`:

```python
import pytest
import candle as torch
from candle.autograd import Function
from candle.autograd.engine import backward


class _MarkDirtyIdentity(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.mark_dirty(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output,)


class _Duplicate(Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone(), x.clone()

    @staticmethod
    def backward(ctx, ga, gb):
        return (ga + gb,)


def test_mark_dirty_bumps_saved_tensor_version():
    x = torch.tensor([1.0], requires_grad=True)

    class _SaveThenRead(Function):
        @staticmethod
        def forward(ctx, t):
            ctx.save_for_backward(t)
            return t.clone()

        @staticmethod
        def backward(ctx, grad_output):
            ctx.saved_tensors
            return (grad_output,)

    y = _SaveThenRead.apply(x)
    _MarkDirtyIdentity.apply(x)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        backward(y.sum())


def test_duplicate_outputs_route_gradients_to_same_input():
    x = torch.tensor([3.0], requires_grad=True)
    a, b = _Duplicate.apply(x)
    backward((a + b).sum())
    assert x.grad.item() == 2.0
```

**Step 3: Run targeted tests to verify failure**

Run:
- `PYTHONPATH=src pytest tests/contract/test_tensor_alias_version_contract.py -v --tb=short`
- `PYTHONPATH=src pytest tests/contract/test_autograd_output_slots_contract.py -v --tb=short`
- `PYTHONPATH=src pytest tests/cpu/test_autograd_function.py -k "version_check or DuplicateOutput" -v --tb=short`

Expected: FAIL on current detach/version, unary inplace aliasing, mark_dirty handling, or output-slot semantics.

**Step 4: Commit test-only red state if desired**

```bash
git add tests/contract/test_tensor_alias_version_contract.py tests/contract/test_autograd_output_slots_contract.py tests/cpu/test_autograd_function.py tests/contract/test_inplace_view_rules.py
git commit -m "test(contract): encode tensor alias and autograd invariants"
```

---

### Task 2: P0 — Fix detach/version sharing semantics

**Files:**
- Modify: `src/candle/_tensor.py`
- Modify: `src/candle/_cython/_tensor_impl.pyx`
- Test: `tests/contract/test_tensor_alias_version_contract.py`

**Step 1: Confirm the exact current detach behavior**

The current implementation snapshots `_version_value` instead of sharing a version source:
- `src/candle/_tensor.py:478-484`
- `src/candle/_cython/_tensor_impl.pyx:228-247`

**Step 2: Implement minimal version-sharing fix**

Choose one mechanism and keep it consistent everywhere:

Option A (recommended): make detach point at a shared root version source by introducing an explicit `_version_owner`/shared-proxy concept in `TensorImpl`.

Option B: extend the existing `_base`/proxy delegation idea so detached tensors can share version state without being treated as views.

Constraints:
- Do **not** mark detached tensors as views.
- Do **not** change storage aliasing.
- Keep `requires_grad=False`, `grad_fn=None` semantics.
- **Cython-first:** land the steady-state version-sharing machinery in `src/candle/_cython/_tensor_impl.pyx`; keep `src/candle/_tensor.py` as a thin adapter only.

**Step 3: Re-run targeted tests**

Run:
- `PYTHONPATH=src pytest tests/contract/test_tensor_alias_version_contract.py::test_detach_shares_version_counter_with_source -v --tb=short`
- `PYTHONPATH=src pytest tests/cpu/test_autograd_function.py::test_version_check -v --tb=short`

Expected: PASS.

**Step 4: Commit**

```bash
git add src/candle/_tensor.py src/candle/_cython/_tensor_impl.pyx tests/contract/test_tensor_alias_version_contract.py tests/cpu/test_autograd_function.py
git commit -m "fix(autograd): share version state across detach aliases"
```

---

### Task 3: P0 — Replace storage-swap unary inplace ops with true storage-preserving mutation

**Files:**
- Modify: `src/candle/_tensor.py`
- Modify: `src/candle/_dispatch/schemas.py`
- Modify: `src/candle/_backends/cpu/ops.py`
- Modify: `src/candle/_backends/cpu/__init__.py`
- Modify: `src/candle/_backends/meta/ops.py`
- Modify: `src/candle/_backends/meta/__init__.py`
- Modify: `src/candle/_cython/_cpu_kernels.pyx`
- Modify: `src/candle/_cython/_tensor_api.pyx` (if hot Tensor forwarding needs updating)
- Test: `tests/contract/test_tensor_alias_version_contract.py`
- Test: `tests/contract/test_inplace_view_rules.py`

**Step 1: Inventory current fake-inplace ops**

The following pattern currently swaps storage:
- `src/candle/_tensor.py:694-836`
- `src/candle/_tensor.py:1683-1713`

At minimum cover:
- `abs_`, `neg_`, `exp_`, `log_`, `log2_`, `log10_`, `sqrt_`
- `sin_`, `cos_`, `tan_`, `tanh_`, `sigmoid_`
- `floor_`, `ceil_`, `round_`, `trunc_`, `pow_`, `reciprocal_`
- `bitwise_and_`, `bitwise_or_`, `bitwise_xor_`

**Step 2: Write/extend failing tests per family**

Add targeted tests asserting both:
- returned object is `self`
- existing views/aliases see updated values after inplace op

Example:

```python
def test_abs_inplace_updates_existing_view_storage():
    x = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
    v = x.view((4,))
    out = x.abs_()
    assert out is x
    assert v.tolist() == [1.0, 2.0, 3.0, 4.0]
```

**Step 3: Implement true inplace kernels**

Preferred approach:
- Add Cython CPU kernels in `src/candle/_cython/_cpu_kernels.pyx` that support in-place unary mutation on the same underlying buffer.
- Make `src/candle/_backends/cpu/ops.py` call those kernels for steady-state CPU execution.
- Register real mutating kernels (`abs_`, `neg_`, etc.) in backend registries.
- Make tensor methods dispatch those kernels instead of computing out-of-place and reassigning `_storage`.
- Preserve dispatcher-driven version bumping through mutating schema entries.

Constraints:
- Do **not** silently leave fallback storage-swap logic in place for covered ops.
- Keep meta kernels returning correct shape/dtype.
- **Cython-first:** hot-path mutation must not end in Python elementwise loops or Python-only orchestration when a Cython kernel path is feasible.
- Scope CPU first if needed, but leave clear backend TODOs for CUDA/MPS/NPU if not implemented in same change.

**Step 4: Re-run targeted tests**

Run:
- `PYTHONPATH=src pytest tests/contract/test_tensor_alias_version_contract.py -k "unary_inplace or setitem" -v --tb=short`
- `PYTHONPATH=src pytest tests/contract/test_inplace_view_rules.py -v --tb=short`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/candle/_tensor.py src/candle/_dispatch/schemas.py src/candle/_backends/cpu/ops.py src/candle/_backends/cpu/__init__.py src/candle/_backends/meta/ops.py src/candle/_backends/meta/__init__.py src/candle/_cython/_cpu_kernels.pyx src/candle/_cython/_tensor_api.pyx tests/contract/test_tensor_alias_version_contract.py tests/contract/test_inplace_view_rules.py
git commit -m "fix(inplace): preserve storage aliasing for unary mutations"
```

---

### Task 4: P0 — Make `__setitem__` version semantics explicit and test-backed

**Files:**
- Modify: `src/candle/_tensor.py`
- Modify: `src/candle/_dispatch/schemas.py` (only if schema gap found)
- Modify: `src/candle/_cython/_tensor_api.pyx` (only if Cython method wrappers need parity)
- Modify: `tests/contract/test_tensor_alias_version_contract.py`
- Reference: `src/candle/_dispatch/dispatcher.py:100-121,399-443`

**Step 1: Decide the intended invariant**

Invariant: every data mutation must bump version exactly once, regardless of whether bump happens in the tensor method or dispatcher.

**Step 2: Implement the smallest robust fix**

Recommended approach:
- Leave the single source of truth in dispatcher/schema if `setitem` is already modeled as mutating.
- Add a targeted regression test to ensure version changes exactly once.
- If `setitem` is not schema-backed as mutating on every path, register/fix schema rather than adding a second bump path in `Tensor.__setitem__`.
- If a Cython fast-path bypasses the Python method in hot execution, keep its semantics aligned too.

Constraint:
- **Do not** move hot mutation semantics into a Python-only workaround if the active fast path runs through Cython.

**Step 3: Re-run targeted tests**

Run:
- `PYTHONPATH=src pytest tests/contract/test_tensor_alias_version_contract.py::test_setitem_bumps_version_counter -v --tb=short`

Expected: PASS.

**Step 4: Commit**

```bash
git add src/candle/_tensor.py src/candle/_dispatch/schemas.py src/candle/_cython/_tensor_api.pyx tests/contract/test_tensor_alias_version_contract.py
git commit -m "fix(dispatch): guarantee version bump for setitem mutations"
```

---

### Task 5: P0 — Implement `mark_dirty()` semantics for custom Function

**Files:**
- Modify: `src/candle/_cython/_autograd_function.pyx`
- Modify: `src/candle/_cython/_autograd_node.pyx`
- Modify: `src/candle/_tensor.py`
- Test: `tests/contract/test_autograd_output_slots_contract.py`
- Test: `tests/cpu/test_autograd_function.py`

**Step 1: Write/extend failing tests**

Augment custom Function tests to assert:
- `ctx.mark_dirty(x)` bumps version on the dirtied tensor
- saved-tensor version mismatch is detected after dirty mutation
- view/custom-function safety rules still raise when expected

**Step 2: Implement minimal correct semantics**

In `_function_apply`:
- consume `ctx._dirty`
- for each dirtied tensor, bump version exactly once
- ensure the returned mutated tensor keeps the correct autograd linkage for backward

Constraints:
- Keep `mark_non_differentiable` behavior unchanged.
- Do not overreach into unrelated Function API cleanup.
- Prefer correctness over trying to emulate every obscure PyTorch edge case in the first patch.
- **Cython-first:** the steady-state dirty/version/autograd-edge behavior must live in the Cython Function runtime, not in a Python shim around `Function.apply`.

**Step 3: Re-run targeted tests**

Run:
- `PYTHONPATH=src pytest tests/contract/test_autograd_output_slots_contract.py -k "dirty" -v --tb=short`
- `PYTHONPATH=src pytest tests/cpu/test_autograd_function.py -k "version_check" -v --tb=short`

Expected: PASS.

**Step 4: Commit**

```bash
git add src/candle/_cython/_autograd_function.pyx src/candle/_cython/_autograd_node.pyx src/candle/_tensor.py tests/contract/test_autograd_output_slots_contract.py tests/cpu/test_autograd_function.py
git commit -m "fix(autograd): honor mark_dirty version semantics"
```

---

### Task 6: P0 — Introduce real output-slot identity (`output_nr`) for autograd edges

**Files:**
- Modify: `src/candle/_cython/_tensor_impl.pyx`
- Modify: `src/candle/_cython/_autograd_node.pyx`
- Modify: `src/candle/_cython/_autograd_function.pyx`
- Modify: `src/candle/_cython/_autograd_engine.pyx`
- Modify: `src/candle/autograd/graph.py`
- Test: `tests/contract/test_autograd_output_slots_contract.py`
- Test: `tests/cpu/test_autograd_function.py`
- Test: `tests/contract/test_autograd_graph_node.py`

**Step 1: Write failing multi-output slot tests**

Add a contract test that verifies different outputs of the same op/function carry distinct output slot ids and that the engine respects them.

Example direction:

```python
def test_duplicate_outputs_have_distinct_output_slots():
    x = torch.tensor([1.0], requires_grad=True)
    a, b = _Duplicate.apply(x)
    assert a.output_nr != b.output_nr
```

Also add a backward-routing test where only one output receives grad and confirm the correct slot is used.

**Step 2: Implement output-slot plumbing**

Required changes:
- Add stored `output_nr` metadata to tensors produced by multi-output nodes.
- In `_function_apply`, create one logical node with slot-aware outputs **or** keep current structure but still assign and consume real `output_nr` values consistently.
- In `next_functions`, stop hardcoding `(fn, 0)` everywhere.
- In engine traversal, consume `_output_nr` meaningfully instead of ignoring it.

Constraints:
- **Cython-first:** store and route output-slot identity in the Cython tensor/node/engine runtime; do not implement this only in Python graph wrappers.

Recommendation: if the smallest correct patch is still substantial, do it in two commits:
1. store/propagate output slot ids
2. consume them in engine routing

**Step 3: Re-run targeted tests**

Run:
- `PYTHONPATH=src pytest tests/contract/test_autograd_output_slots_contract.py -v --tb=short`
- `PYTHONPATH=src pytest tests/cpu/test_autograd_function.py -k "DuplicateOutput" -v --tb=short`
- `PYTHONPATH=src pytest tests/contract/test_autograd_graph_node.py -v --tb=short`

Expected: PASS.

**Step 4: Commit**

```bash
git add src/candle/_cython/_tensor_impl.pyx src/candle/_cython/_autograd_node.pyx src/candle/_cython/_autograd_function.pyx src/candle/_cython/_autograd_engine.pyx src/candle/autograd/graph.py tests/contract/test_autograd_output_slots_contract.py tests/cpu/test_autograd_function.py tests/contract/test_autograd_graph_node.py
git commit -m "fix(autograd): track and honor output slots for multi-output nodes"
```

---

### Task 7: P1 — Add CUDA current/default stream semantics and context management

**Files:**
- Modify: `src/candle/cuda.py`
- Create: `src/candle/_backends/cuda/state.py`
- Modify: `src/candle/_backends/cuda/runtime.py`
- Modify: `src/candle/_backends/cuda/ops.py`
- Test: `tests/cpu/test_cuda_stream_state.py` (new, mocked runtime)
- Reference: `src/candle/npu.py`
- Reference: `src/candle/_backends/npu/state.py`
- Reference: `tests/npu/test_npu_streams.py`

**Step 1: Write failing CUDA stream state tests**

Mirror the existing NPU test style with a fake runtime:

```python
def test_cuda_default_stream_is_thread_local_current_default(monkeypatch):
    s0 = torch.cuda.default_stream()
    assert torch.cuda.current_stream().stream == s0.stream


def test_cuda_stream_context_switches_current_stream(monkeypatch):
    s = torch.cuda.Stream()
    cur = torch.cuda.current_stream()
    with torch.cuda.stream(s):
        assert torch.cuda.current_stream() is s
    assert torch.cuda.current_stream() is cur
```

Add at least:
- current stream thread-local isolation
- default stream per-device behavior
- context manager restore semantics

**Step 2: Implement CUDA TLS stream state**

Create `src/candle/_backends/cuda/state.py` analogous to NPU:
- thread-local `current_device`
- per-device `current_streams`
- per-device `default_streams`
- `device_guard`
- `current_stream` / `default_stream` / `set_current_stream`

Expose from `src/candle/cuda.py`:
- `default_stream(device=None)`
- `current_stream(device=None)`
- `set_stream(stream)`
- `class stream` context manager

**Step 3: Route CUDA ops through current stream**

Audit `_backends/cuda/ops.py` and ensure launch helpers use the current stream handle instead of implicit/default runtime behavior.

Constraints:
- **Cython-first where hot:** if stream lookup/launch-path overhead becomes hot, move the finalized fast path into Cython/runtime bindings rather than leaving repeated Python dispatch on the steady-state path.

**Step 4: Re-run targeted tests**

Run:
- `PYTHONPATH=src pytest tests/cpu/test_cuda_stream_state.py -v --tb=short`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/candle/cuda.py src/candle/_backends/cuda/state.py src/candle/_backends/cuda/runtime.py src/candle/_backends/cuda/ops.py tests/cpu/test_cuda_stream_state.py
git commit -m "feat(cuda): add current stream state and context management"
```

---

### Task 8: P1 — Make CUDA memory lifetime stream-aware (`record_stream` / deferred free)

**Files:**
- Modify: `src/candle/_tensor.py`
- Modify: `src/candle/_backends/cuda/storage.py`
- Create or modify: `src/candle/_backends/cuda/allocator.py`
- Modify: `src/candle/_backends/cuda/runtime.py`
- Test: `tests/cpu/test_cuda_allocator_streams.py` (new, mocked runtime)
- Reference: `src/candle/_backends/npu/allocator.py`
- Reference: `src/candle/_tensor.py:418-424`

**Step 1: Write failing allocator-lifetime tests**

Use a fake allocator/runtime to assert:
- `tensor.record_stream(cuda_stream)` records stream usage
- frees are deferred until an event/query indicates completion
- `record_stream` is no longer NPU-only

**Step 2: Implement minimal CUDA allocator stream tracking**

Do not attempt a full caching allocator rewrite in the first patch. Minimal acceptable scope:
- per-allocation stream-use metadata
- `record_stream` support on CUDA tensors
- event-backed deferred free for reused/deleted blocks

Constraints:
- **Cython-first where hot:** if allocator bookkeeping or event polling sits on a hot path, land the steady-state bookkeeping in Cython/runtime-facing code rather than a Python-only allocator loop.

**Step 3: Re-run targeted tests**

Run:
- `PYTHONPATH=src pytest tests/cpu/test_cuda_allocator_streams.py -v --tb=short`

Expected: PASS.

**Step 4: Commit**

```bash
git add src/candle/_tensor.py src/candle/_backends/cuda/storage.py src/candle/_backends/cuda/allocator.py src/candle/_backends/cuda/runtime.py tests/cpu/test_cuda_allocator_streams.py
git commit -m "feat(cuda): make tensor lifetime stream-aware"
```

---

### Task 9: P1 — Define realistic MPS execution semantics and add minimal stream/event roadmap hooks

**Files:**
- Modify: `src/candle/mps.py`
- Modify: `src/candle/_backends/mps/runtime.py`
- Modify: `src/candle/_backends/mps/metal_compute.py`
- Create: `tests/cpu/test_mps_runtime_contract.py` (mock-only)

**Step 1: Write contract tests for current intended behavior**

Pick one of two scopes and encode it explicitly:

**Recommended narrow scope:**
- keep MPS synchronous for now
- add tests documenting that `synchronize()` is explicit and kernels may still commit eagerly
- add TODO contract placeholders for future `Stream/Event`

Do **not** promise Torch-level async streams unless you intend to build them immediately.

**Step 2: Minimal implementation/doc fix**

If retaining sync behavior for now:
- make the API/docs explicit
- add internal runtime hooks so a future `Stream/Event` abstraction can be layered without rewriting call sites again

If building the async path now, do it as a separate design doc before code.

Constraint:
- If and when MPS gains a hot-path stream/event abstraction, target Cython/runtime bindings for the steady-state path rather than Python-only coordination.

**Step 3: Re-run tests**

Run:
- `PYTHONPATH=src pytest tests/cpu/test_mps_runtime_contract.py -v --tb=short`

Expected: PASS.

**Step 4: Commit**

```bash
git add src/candle/mps.py src/candle/_backends/mps/runtime.py src/candle/_backends/mps/metal_compute.py tests/cpu/test_mps_runtime_contract.py
git commit -m "refactor(mps): clarify runtime synchronization contract"
```

---

### Task 10: P2 — Harden dispatcher alias/composite semantics

**Files:**
- Modify: `src/candle/_dispatch/dispatcher.py`
- Modify: `src/candle/_dispatch/schema.py`
- Modify: `src/candle/_dispatch/schemas.py`
- Modify: `src/candle/_dispatch/registration.py`
- Test: `tests/contract/test_schema_view_alias.py`
- Test: `tests/contract/test_dispatch_contract.py`
- Test: `tests/contract/test_dispatch_keyset_construction.py`

**Step 1: Write failing contract tests for alias metadata usage**

Add tests ensuring:
- return alias metadata is preserved/observable where relevant
- functionalization/writeback respects aliasing for mutating ops
- keyset construction does not rely on PrivateUse placeholders for long-term backend identity in tests that inspect semantics

**Step 2: Tighten alias handling without broad refactor**

Goals:
- use schema alias info in more than just version bumping where needed
- make mutating op legality less dependent on scattered ad-hoc checks
- keep existing dispatcher structure intact

Non-goals for this task:
- full Torch boxed/unboxed parity
- total key-system rewrite

Constraints:
- **Cython-first where hot:** if dispatcher fast-path logic is touched repeatedly on hot execution paths, prefer landing the steady-state path in `src/candle/_cython/_dispatcher_core.pyx` or adjacent Cython runtime code, with Python remaining orchestration/configuration only.

**Step 3: Re-run targeted tests**

Run:
- `PYTHONPATH=src pytest tests/contract/test_schema_view_alias.py tests/contract/test_dispatch_contract.py tests/contract/test_dispatch_keyset_construction.py -v --tb=short`

Expected: PASS.

**Step 4: Commit**

```bash
git add src/candle/_dispatch/dispatcher.py src/candle/_dispatch/schema.py src/candle/_dispatch/schemas.py src/candle/_dispatch/registration.py tests/contract/test_schema_view_alias.py tests/contract/test_dispatch_contract.py tests/contract/test_dispatch_keyset_construction.py
git commit -m "refactor(dispatch): tighten alias and mutation semantics"
```

---

### Task 11: P2 — Expand forward-mode AD from proof-of-concept to useful parity subset

**Files:**
- Modify: `src/candle/autograd/forward_ad.py`
- Modify: `src/candle/_dispatch/dispatcher.py`
- Create: `tests/cpu/test_forward_ad_contract.py`
- Reference: `docs/plans/2026-03-12-forward-ad-cpu-implementation.md`

**Step 1: Write failing tests for missing high-frequency JVPs**

Cover at least:
- unary pointwise (`neg`, `exp`, `sin`, `cos`, `tanh`)
- reductions (`mean` if intended, not just `sum`)
- simple view-preserving ops (`reshape`, `view`) if they should propagate tangents

**Step 2: Implement the smallest useful JVP expansion**

- Add rule coverage for high-frequency training math first.
- Preserve the current dispatcher integration model.
- Raise explicit `RuntimeError` for uncovered ops rather than silent wrong tangents.

Constraint:
- For hot-path JVP propagation and forward-AD metadata access, prefer extending the Cython dispatcher/autograd runtime once semantics are stable, rather than leaving steady-state propagation entirely in Python.

**Step 3: Re-run targeted tests**

Run:
- `PYTHONPATH=src pytest tests/cpu/test_forward_ad_contract.py -v --tb=short`

Expected: PASS.

**Step 4: Commit**

```bash
git add src/candle/autograd/forward_ad.py src/candle/_dispatch/dispatcher.py tests/cpu/test_forward_ad_contract.py
git commit -m "feat(forward-ad): expand JVP coverage for core ops"
```

---

### Task 12: Full verification gate after each priority band

**Files:**
- No code changes required unless failures are found.

**Step 1: After P0, run the focused correctness suite**

Run:
- `PYTHONPATH=src pytest tests/contract/test_tensor_alias_version_contract.py tests/contract/test_autograd_output_slots_contract.py tests/contract/test_inplace_view_rules.py tests/cpu/test_autograd_function.py -v --tb=short`

**Step 2: After P1, run runtime-focused suites**

Run:
- `PYTHONPATH=src pytest tests/npu/test_npu_streams.py tests/cpu/test_cuda_stream_state.py tests/cpu/test_cuda_allocator_streams.py tests/cpu/test_mps_runtime_contract.py -v --tb=short`

**Step 3: After P2, run broader contract + CPU suites**

Run:
- `PYTHONPATH=src pytest tests/contract/ tests/cpu/ -v --tb=short`

**Step 4: Pylint gate**

Run:
- `pylint src/candle/ --rcfile=.github/pylint.conf`

Expected: zero errors.

**Step 5: Commit only after green verification**

Use narrow commits per task; do **not** batch unrelated mechanism fixes into one commit.

---

## Execution Notes

- **Recommended execution order:** Task 1 → 2 → 3 → 4 → 5 → 6 → 12(P0 gate) → 7 → 8 → 9 → 12(P1 gate) → 10 → 11 → 12(P2/final gate)
- **Why this order:** P0 fixes close correctness holes first; P1 makes runtime semantics less surprising; P2 expands completeness only after invariants are trustworthy.
- **Keep scope tight:** do not simultaneously redesign the entire dispatcher, allocator, and autograd engine.
- **Prefer red/green/refactor:** each task should start with one or two failing tests, then minimal code, then immediate re-run.
- **Frequent commits:** one mechanism fix per commit keeps regressions bisectable.

---

Plan complete and saved to `docs/plans/2026-03-23-torch-mechanism-alignment-roadmap.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
