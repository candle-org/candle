# View-Tracking 1B-A: Conditional-View Forward for `contiguous`/`flatten`/`unflatten`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `contiguous` / `flatten` / `unflatten` return a real view (or the input itself for `contiguous`) when the input is already contiguous, instead of always copying — matching PyTorch's conditional-view semantics. No autograd or codegen changes.

**Architecture:** A small forward-path fix in three backends. CPU and NPU `contiguous` add a `if a.is_contiguous(): return a` short-circuit. CPU `flatten`/`unflatten` and MPS `flatten`/`unflatten` GPU paths route through `view_backend._make_view` for contiguous inputs, falling back to existing copy paths otherwise. NPU `flatten_op`/`unflatten_op` already delegate to `view_backend.reshape` and need no change. After this plan, the engine rebase block landed in PR #372 still has no live consumers — that's 1B-B's job.

**Tech Stack:** Python (numpy for CPU), Metal/Cython for MPS, ctypes for NPU. View infrastructure: `src/candle/_backends/common/view.py:_make_view` and `_get_base` and `_contiguous_stride`.

---

## Spec reference

Spec: `docs/superpowers/specs/2026-05-04-view-tracking-1b-six-ops-design.md`. Read it before starting any task.

## Pre-task setup

You are working inside the existing worktree at `/home/jenkins/lvyufeng/candle/.worktrees/view-tracking-1b` on branch `view-tracking-1b`. The worktree was set up by an earlier session and is already rebased onto `upstream/main`. Do NOT create a new worktree.

Verify before starting:

```bash
cd /home/jenkins/lvyufeng/candle/.worktrees/view-tracking-1b
git rev-parse --abbrev-ref HEAD   # should print: view-tracking-1b
git status --short                # should print nothing (clean tree)
```

## File map

| File | Responsibility | Action |
|---|---|---|
| `tests/cpu/test_view_fastpath.py` | Pin conditional-view forward semantics for the 3 ops | Create |
| `src/candle/_backends/cpu/ops.py` | CPU forward kernels for `contiguous`, `flatten`, `unflatten` | Modify lines 1351-1364 (`contiguous`), 3058-3074 (`flatten`, `unflatten`) |
| `src/candle/_backends/mps/ops/shape.py` | MPS GPU+CPU paths for `flatten`, `unflatten` | Modify lines 1099-1135 |
| `src/candle/_backends/npu/ops/shape.py` | NPU `contiguous` adds fast-path | Modify lines 1428-1444 |

Out of scope: any change to `_generated/`, any `view_func`/`rev_view_func` wiring, any CUDA backend file (no `_backends/cuda/ops/shape.py` for these ops).

## Conventions used by this codebase

- Tests live in `tests/cpu/` and import `candle as torch` (not real PyTorch).
- Storage-share assertion idiom is `a.storage() is b.storage()` — NOT `a.storage().data_ptr() == b.storage().data_ptr()`.
- `_make_view(base, shape, stride, offset, op_name, source=a)` — `source=a` lets it inherit `creation_kind` from the input view metadata.
- `_get_base(a)` returns `a._base if a._base is not None else a` — always use it before passing to `_make_view`, so view-of-view chains collapse to the storage root.
- All Python view ops in `_backends/common/view.py` already use `_get_base` + `_make_view`; the new code in this plan follows that pattern verbatim.
- Pylint must pass on `src/candle/`. The conda env is `candle`. The pyobjc imports for MPS need `# pylint: disable=import-error`.

---

## Task 1: Red tests for the conditional-view fast-path

**Files:**
- Create: `tests/cpu/test_view_fastpath.py`

This task is TDD-style: write failing tests first to pin the desired behavior. The tests will fail today because CPU `contiguous`/`flatten`/`unflatten` always copy.

- [ ] **Step 1: Write the failing test file**

Create `tests/cpu/test_view_fastpath.py` with this exact content:

```python
"""Pin conditional-view forward semantics for contiguous/flatten/unflatten.

After Sub-batch 1B-A:
- contiguous(a) returns `a` itself when a is already contiguous in default
  format. Non-contiguous input still copies.
- flatten(a, ...) and unflatten(a, ...) return a real view (storage shared,
  _base set, _view_meta["op"] populated) when a is contiguous. Non-contiguous
  input still copies.

Tests stay on CPU — MPS and NPU regression coverage rides on the cpu/contract
gate plus the existing per-backend test suites.
"""
import numpy as np
import candle as torch


# ---------------------------------------------------------------------------
# contiguous
# ---------------------------------------------------------------------------

def test_contiguous_returns_self_when_already_contiguous():
    t = torch.randn(3, 4)
    assert t.is_contiguous()
    u = t.contiguous()
    assert u is t


def test_contiguous_copies_when_input_is_non_contiguous():
    t = torch.randn(4, 4).t()
    assert not t.is_contiguous()
    u = t.contiguous()
    assert u is not t
    assert u.storage() is not t.storage()
    assert u.is_contiguous()


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------

def test_flatten_shares_storage_when_input_is_contiguous():
    t = torch.randn(3, 4)
    u = t.flatten()
    assert u.storage() is t.storage()
    assert u._base is not None
    assert u._view_meta is not None
    assert u._view_meta["op"] == "flatten"
    assert u._view_func is None
    assert u._rev_view_func is None


def test_flatten_view_mutation_propagates_to_input():
    t = torch.randn(3, 4)
    u = t.flatten()
    u[0] = 99.0
    np.testing.assert_array_equal(t.flatten().numpy()[0], 99.0)


def test_flatten_copies_when_input_is_non_contiguous():
    t = torch.randn(4, 4).t()
    assert not t.is_contiguous()
    u = t.flatten()
    assert u.storage() is not t.storage()


# ---------------------------------------------------------------------------
# unflatten
# ---------------------------------------------------------------------------

def test_unflatten_shares_storage_when_input_is_contiguous():
    t = torch.randn(2, 6)
    u = t.unflatten(1, (2, 3))
    assert u.storage() is t.storage()
    assert u._base is not None
    assert u._view_meta is not None
    assert u._view_meta["op"] == "unflatten"
    assert u._view_func is None
    assert u._rev_view_func is None


def test_unflatten_view_mutation_propagates_to_input():
    t = torch.randn(2, 6)
    u = t.unflatten(1, (2, 3))
    u[0, 0, 0] = 99.0
    np.testing.assert_array_equal(t.numpy()[0, 0], 99.0)


def test_unflatten_copies_when_input_is_non_contiguous():
    t = torch.randn(4, 4).t()
    assert not t.is_contiguous()
    u = t.unflatten(0, (2, 2))
    assert u.storage() is not t.storage()


# ---------------------------------------------------------------------------
# Backward through the fast-path is unchanged: hand-added *Backward0 classes
# in src/candle/_generated/functions.py still drive grad propagation in 1B-A.
# These tests guard against regressions in that wiring.
# ---------------------------------------------------------------------------

def test_flatten_backward_is_identity_after_fast_path():
    t = torch.randn(3, 4, requires_grad=True)
    u = t.flatten()
    u.sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), np.ones((3, 4), dtype=np.float32))


def test_unflatten_backward_is_identity_after_fast_path():
    t = torch.randn(2, 6, requires_grad=True)
    u = t.unflatten(1, (2, 3))
    u.sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), np.ones((2, 6), dtype=np.float32))


def test_contiguous_backward_is_identity_when_already_contiguous():
    t = torch.randn(3, 4, requires_grad=True)
    u = t.contiguous()
    assert u is t
    u.sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), np.ones((3, 4), dtype=np.float32))
```

- [ ] **Step 2: Run the failing tests**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_view_fastpath.py -v --tb=short
```

Expected before any source changes: most assertions in the storage-sharing tests fail. Specifically:
- `test_contiguous_returns_self_when_already_contiguous` fails because today CPU `contiguous(a)` always returns a fresh `_from_numpy(...)` tensor (line 1364).
- `test_flatten_shares_storage_when_input_is_contiguous` fails (`u.storage() is not t.storage()`).
- `test_unflatten_shares_storage_when_input_is_contiguous` fails (same reason).
- The backward-identity tests should pass already (the existing `*Backward0` classes are independent of forward path).
- The "copies when non-contiguous" tests should pass already.

If the failing tests do NOT include the three storage-share tests, stop and investigate — something about the codebase has shifted and the spec assumptions are stale.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/cpu/test_view_fastpath.py
git commit -m "$(cat <<'EOF'
test(cpu): pin conditional-view fast-path for contiguous/flatten/unflatten

Failing tests for Sub-batch 1B-A. They assert:
- contiguous(a) returns a itself when a is already contiguous
- flatten/unflatten share storage and set _base/_view_meta when contiguous
- non-contiguous input still copies for all three ops
- backward through the fast path is still identity

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: CPU `contiguous` fast-path

**Files:**
- Modify: `src/candle/_backends/cpu/ops.py:1351-1364`

- [ ] **Step 1: Edit `contiguous` to short-circuit on contiguous input**

Replace the current body of `contiguous` (lines 1351-1364):

```python
def contiguous(a, memory_format=None):
    if a.device.type != "cpu":
        raise ValueError("CPU contiguous expects CPU tensors")
    if getattr(memory_format, "_name", None) == "channels_last":
        if len(a.shape) != 4:
            raise RuntimeError("required rank 4 tensor to use channels_last format")
        if _is_channels_last_stride(a.shape, a.stride):
            return a
        arr = np.ascontiguousarray(np.transpose(_to_numpy(a), (0, 2, 3, 1)))
        storage = typed_storage_from_numpy(arr.reshape(-1), a.dtype, device=a.device)
        stride = _channels_last_stride(a.shape)
        return cy_make_tensor_from_storage(storage, a.shape, stride, 0, False)
    arr = np.ascontiguousarray(_to_numpy(a))
    return _from_numpy(arr, a.dtype, a.device)
```

with:

```python
def contiguous(a, memory_format=None):
    if a.device.type != "cpu":
        raise ValueError("CPU contiguous expects CPU tensors")
    if getattr(memory_format, "_name", None) == "channels_last":
        if len(a.shape) != 4:
            raise RuntimeError("required rank 4 tensor to use channels_last format")
        if _is_channels_last_stride(a.shape, a.stride):
            return a
        arr = np.ascontiguousarray(np.transpose(_to_numpy(a), (0, 2, 3, 1)))
        storage = typed_storage_from_numpy(arr.reshape(-1), a.dtype, device=a.device)
        stride = _channels_last_stride(a.shape)
        return cy_make_tensor_from_storage(storage, a.shape, stride, 0, False)
    if a.is_contiguous():
        return a
    arr = np.ascontiguousarray(_to_numpy(a))
    return _from_numpy(arr, a.dtype, a.device)
```

The single new line is `if a.is_contiguous(): return a` immediately before the existing `np.ascontiguousarray` copy. The channels_last branch is unchanged.

- [ ] **Step 2: Run the contiguous tests**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_view_fastpath.py -v --tb=short -k contiguous
```

Expected:
- `test_contiguous_returns_self_when_already_contiguous` PASSES.
- `test_contiguous_copies_when_input_is_non_contiguous` PASSES (was passing before too).
- `test_contiguous_backward_is_identity_when_already_contiguous` PASSES.

- [ ] **Step 3: Run the existing memory_format suite for regression**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/ -v --tb=short -k "memory_format or channels_last"
```

Expected: all green. The channels_last branch was untouched — if these regress, the edit was wrong.

- [ ] **Step 4: Commit**

```bash
git add src/candle/_backends/cpu/ops.py
git commit -m "$(cat <<'EOF'
fix(cpu): contiguous returns input when already contiguous

Match PyTorch's t.contiguous() returning self for already-contiguous inputs
in default memory format. Previously always allocated via np.ascontiguousarray.
channels_last branch unchanged.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: CPU `flatten` and `unflatten` view fast-path

**Files:**
- Modify: `src/candle/_backends/cpu/ops.py:3058-3074`

- [ ] **Step 1: Edit `flatten` and `unflatten`**

Replace lines 3058-3074:

```python
def flatten(a, start_dim=0, end_dim=-1):
    arr = _to_numpy(a)
    ndim = arr.ndim
    start = start_dim if start_dim >= 0 else start_dim + ndim
    end = end_dim if end_dim >= 0 else end_dim + ndim
    new_shape = arr.shape[:start] + (-1,) + arr.shape[end + 1:]
    out = arr.reshape(new_shape)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def unflatten(a, dim, sizes):
    arr = _to_numpy(a)
    ndim = arr.ndim
    d = dim if dim >= 0 else dim + ndim
    new_shape = arr.shape[:d] + tuple(sizes) + arr.shape[d + 1:]
    out = arr.reshape(new_shape)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)
```

with:

```python
def flatten(a, start_dim=0, end_dim=-1):
    ndim = len(a.shape)
    start = start_dim if start_dim >= 0 else start_dim + ndim
    end = end_dim if end_dim >= 0 else end_dim + ndim
    flat_size = 1
    for i in range(start, end + 1):
        flat_size *= a.shape[i]
    new_shape = a.shape[:start] + (flat_size,) + a.shape[end + 1:]
    if a.is_contiguous():
        from ..._backends.common import view as view_backend
        base = view_backend._get_base(a)
        new_stride = view_backend._contiguous_stride(new_shape)
        return view_backend._make_view(
            base, new_shape, new_stride, a.offset, "flatten", source=a
        )
    arr = _to_numpy(a)
    out = arr.reshape(new_shape)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)


def unflatten(a, dim, sizes):
    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    new_shape = a.shape[:d] + tuple(sizes) + a.shape[d + 1:]
    if a.is_contiguous():
        from ..._backends.common import view as view_backend
        base = view_backend._get_base(a)
        new_stride = view_backend._contiguous_stride(new_shape)
        return view_backend._make_view(
            base, new_shape, new_stride, a.offset, "unflatten", source=a
        )
    arr = _to_numpy(a)
    out = arr.reshape(new_shape)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)
```

Key differences from the original:
- Compute `new_shape` from `a.shape` (already a tuple of ints) instead of `arr.shape` — avoids materializing the numpy array on the fast-path.
- `flatten` resolves `-1` itself by computing `flat_size = prod(a.shape[start:end+1])` — the same logic the NPU `flatten_op` uses.
- Fast-path branch goes through `view_backend._make_view` with the standard `_get_base` lookup. `op` argument is the literal string `"flatten"` or `"unflatten"`. `source=a` propagates `creation_kind` from `a._view_meta` if `a` is itself a view.
- `view_func` and `rev_view_func` are deliberately NOT passed — they default to `None`. Sub-batch 1B-B will populate them.
- Non-contiguous fallback is the original two-line copy path.

- [ ] **Step 2: Run flatten/unflatten tests**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_view_fastpath.py -v --tb=short
```

Expected: all 11 tests in `test_view_fastpath.py` PASS.

- [ ] **Step 3: Run the broader cpu tensor-view suite**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/test_tensor_view.py tests/cpu/test_view_dispatch.py \
                   tests/cpu/test_view_tracking_infrastructure.py -v --tb=short
```

Expected: all green. These pin existing view semantics that should be unaffected.

- [ ] **Step 4: Commit**

```bash
git add src/candle/_backends/cpu/ops.py
git commit -m "$(cat <<'EOF'
fix(cpu): flatten/unflatten return a view when input is contiguous

Match PyTorch's conditional-view semantics. When the input is already
contiguous, build a real view via view_backend._make_view instead of
copying through np.ascontiguousarray. Non-contiguous input still copies.

view_func/rev_view_func default to None — autograd routing stays on the
hand-added FlattenBackward0/UnflattenBackward0 classes for now.
Sub-batch 1B-B will wire those callables.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: MPS `flatten` and `unflatten` view fast-path

**Files:**
- Modify: `src/candle/_backends/mps/ops/shape.py:1099-1135`

This task can only be physically run on a host with MPS hardware. CI test-mps runs on macos-14 (M1). Local validation: see CLAUDE.md for the macOS Apple Silicon checklist. If the implementer is on a Linux host (no MPS), they may write the change and rely on test-mps in CI as the safety net — but they MUST visually compare the new MPS code against the CPU implementation in Task 3 to ensure the same semantic invariant.

- [ ] **Step 1: Edit `flatten` and `unflatten` in MPS shape.py**

Replace lines 1099-1135:

```python
def flatten(a, start_dim=0, end_dim=-1):
    ndim = len(a.shape)
    start = start_dim if start_dim >= 0 else start_dim + ndim
    end = end_dim if end_dim >= 0 else end_dim + ndim
    new_shape = a.shape[:start] + (-1,) + a.shape[end + 1:]
    # Compute actual -1 size
    known = 1
    for i, s in enumerate(new_shape):
        if s != -1:
            known *= s
    total = 1
    for s in a.shape:
        total *= s
    new_shape = tuple(s if s != -1 else total // known for s in new_shape)
    if _can_use_gpu(a) and a.is_contiguous():
        from ...._C import _compute_strides
        return _from_metal_buffer(_metal_buf(a), new_shape, _compute_strides(new_shape), a.dtype, a.device)
    # Non-contiguous GPU: make contiguous and retry
    if _can_use_gpu(a):
        return flatten(a.contiguous(), start_dim=start_dim, end_dim=end_dim)
    arr = _to_numpy(a)
    out = arr.reshape(new_shape)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def unflatten(a, dim, sizes):
    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    new_shape = a.shape[:d] + tuple(sizes) + a.shape[d + 1:]
    if _can_use_gpu(a) and a.is_contiguous():
        from ...._C import _compute_strides
        return _from_metal_buffer(_metal_buf(a), new_shape, _compute_strides(new_shape), a.dtype, a.device)
    # Non-contiguous GPU: make contiguous and retry
    if _can_use_gpu(a):
        return unflatten(a.contiguous(), dim, sizes)
    arr = _to_numpy(a)
    out = arr.reshape(new_shape)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)
```

with:

```python
def flatten(a, start_dim=0, end_dim=-1):
    ndim = len(a.shape)
    start = start_dim if start_dim >= 0 else start_dim + ndim
    end = end_dim if end_dim >= 0 else end_dim + ndim
    flat_size = 1
    for i in range(start, end + 1):
        flat_size *= a.shape[i]
    new_shape = a.shape[:start] + (flat_size,) + a.shape[end + 1:]
    if a.is_contiguous():
        from ...common import view as view_backend
        base = view_backend._get_base(a)
        new_stride = view_backend._contiguous_stride(new_shape)
        return view_backend._make_view(
            base, new_shape, new_stride, a.offset, "flatten", source=a
        )
    # Non-contiguous GPU: make contiguous and retry
    if _can_use_gpu(a):
        return flatten(a.contiguous(), start_dim=start_dim, end_dim=end_dim)
    arr = _to_numpy(a)
    out = arr.reshape(new_shape)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def unflatten(a, dim, sizes):
    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    new_shape = a.shape[:d] + tuple(sizes) + a.shape[d + 1:]
    if a.is_contiguous():
        from ...common import view as view_backend
        base = view_backend._get_base(a)
        new_stride = view_backend._contiguous_stride(new_shape)
        return view_backend._make_view(
            base, new_shape, new_stride, a.offset, "unflatten", source=a
        )
    # Non-contiguous GPU: make contiguous and retry
    if _can_use_gpu(a):
        return unflatten(a.contiguous(), dim, sizes)
    arr = _to_numpy(a)
    out = arr.reshape(new_shape)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)
```

Key differences from the original:
- The `_can_use_gpu(a) and a.is_contiguous()` branch becomes simply `a.is_contiguous()` and uses `_make_view` instead of `_from_metal_buffer(_metal_buf(a), ...)`. This produces a same-storage tensor with `_base`, `_view_meta`, etc. — what 1B-B can later wire to view_func.
- The `_compute_strides` import (`from ...._C import _compute_strides`) is replaced with `view_backend._contiguous_stride` for parity with CPU. Both compute the same thing for a contiguous shape; using `view_backend._contiguous_stride` keeps all backends going through the same helper.
- `flatten` resolves `-1` up front via `flat_size = prod(a.shape[start:end+1])` (matches CPU and NPU `flatten_op`). The numpy-based `-1` resolution is gone; that's fine because non-contiguous fallback already operates on `new_shape` with no `-1` left.
- The non-contiguous-GPU branch (`_can_use_gpu(a)` retry) and the CPU fallback path stay unchanged.
- `view_func` / `rev_view_func` not passed (default `None`) — same as CPU.

- [ ] **Step 2: Audit MPS for `_base is None` callers**

Before running the suite, grep for any MPS code that branches on `_base`:

```bash
grep -rn "_base is None\|_base is not None" src/candle/_backends/mps/
```

Expected: only references in shared helpers — no MPS-specific code that would break if a flatten/unflatten output has `_base != None`. If you find new branching that wasn't audited in the spec, stop and surface the finding before continuing.

- [ ] **Step 3: Rebuild Cython if needed**

The `_C/_compute_strides` symbol is no longer imported by `flatten`/`unflatten`. No Cython rebuild required (pure Python edit). To be safe:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python setup.py build_ext --inplace 2>&1 | tail -20
```

Expected: `build_ext` succeeds (or is a no-op since nothing in `_C` changed). If the build fails, the failure is unrelated to this edit — investigate separately.

- [ ] **Step 4: Run MPS tests if hardware is available**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/mps/ -v --tb=short -k "flatten or unflatten or contiguous"
```

If running on a Linux host without MPS hardware, the suite auto-skips per `tests/conftest.py`. Note that you skipped it — do NOT mark this task complete with full confidence; the macOS CI run will be the gate.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/mps/ -v --tb=short
```

Expected when MPS is available: all green. The change preserves storage sharing on the contiguous GPU path (already shared via `_metal_buf`), but now the output also carries view metadata.

- [ ] **Step 5: Commit**

```bash
git add src/candle/_backends/mps/ops/shape.py
git commit -m "$(cat <<'EOF'
fix(mps): flatten/unflatten return a registered view when input is contiguous

Replace the GPU contiguous fast-path that produced a same-storage tensor
via _from_metal_buffer (no _base, no _view_meta) with view_backend._make_view.
Output now has _base, _view_meta, and inherits creation_kind from the input
when it is a view. Non-contiguous fallback unchanged.

view_func/rev_view_func default to None — autograd routing stays on the
hand-added FlattenBackward0/UnflattenBackward0 classes for now.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: NPU `contiguous` fast-path

**Files:**
- Modify: `src/candle/_backends/npu/ops/shape.py:1428-1444`

This task can only be physically run on a host with NPU hardware. The `view-tracking-1b` worktree's host has NPU available (per CLAUDE.md). If the implementer is on a host without NPU, write the change and rely on the existing NPU test suite as the gate.

NPU `flatten_op` (line 1409) and `unflatten_op` (line 1974) already delegate to `view_backend.reshape`, which itself routes to `_make_view` for contiguous inputs. They need NO change in this plan — verified by reading `src/candle/_backends/common/view.py:reshape`.

- [ ] **Step 1: Edit NPU `contiguous`**

Replace lines 1428-1444:

```python
def contiguous(a):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU contiguous expects NPU tensors")

    a_storage = _unwrap_storage(a)
    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    npu_runtime.memcpy_d2d(
        out_ptr,
        out_size,
        a_storage.data_ptr(),
        runtime=runtime,
    )

    storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    return _wrap_tensor(storage, a.shape, npu_runtime._contiguous_stride(a.shape))
```

with:

```python
def contiguous(a):
    if a.device.type != "npu":
        raise ValueError("NPU contiguous expects NPU tensors")
    if a.is_contiguous():
        return a
    runtime = npu_runtime.get_runtime((a.device.index or 0))

    a_storage = _unwrap_storage(a)
    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    npu_runtime.memcpy_d2d(
        out_ptr,
        out_size,
        a_storage.data_ptr(),
        runtime=runtime,
    )

    storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    return _wrap_tensor(storage, a.shape, npu_runtime._contiguous_stride(a.shape))
```

The diff is two lines: hoist the device check above the `runtime` lookup so we don't pay for runtime resolution on the fast-path, and add `if a.is_contiguous(): return a`. Everything else is unchanged.

Note: NPU has no `memory_format` parameter — there's no channels_last branch to preserve.

- [ ] **Step 2: Run NPU tests if hardware is available**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/npu/ -v --tb=short -k "contiguous or flatten or unflatten"
```

Expected when NPU is available: all green. Most tests don't exercise the fast-path explicitly, but several call `.contiguous()` on already-contiguous tensors and should now skip the memcpy.

If NPU hardware is unavailable on this host, the suite auto-skips. Note that.

- [ ] **Step 3: Commit**

```bash
git add src/candle/_backends/npu/ops/shape.py
git commit -m "$(cat <<'EOF'
fix(npu): contiguous returns input when already contiguous

Match PyTorch's t.contiguous() returning self for already-contiguous inputs.
Previously always allocated and ran memcpy_d2d.

NPU flatten_op/unflatten_op already delegate to view_backend.reshape, which
handles the conditional-view semantics correctly — no change needed there.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Pylint and full gate

**Files:** none (validation only)

- [ ] **Step 1: Run pylint on src/candle**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  pylint src/candle/ --rcfile=.github/pylint.conf 2>&1 | tail -40
```

Expected: pylint passes (`Your code has been rated at 10.00/10` or no new violations vs. `view-tracking-1b` baseline). If new violations appear, fix them before continuing. Common false positives:
- `import-outside-toplevel` on the `from ..._backends.common import view as view_backend` lines — already used by `movedim`, `narrow`, etc. in the same file, so the rule should already be disabled. If it fires, add `# pylint: disable=import-outside-toplevel` on the line.

- [ ] **Step 2: Run the cpu + contract gate**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/cpu/ tests/contract/ -v --tb=short 2>&1 | tail -40
```

Expected: same pass/fail/skip count as the `view-tracking-1b` baseline (PR #372 shipped with 3013 passed, 30 skipped, 1 failed — the 1 failure is the codegen-roundtrip drift on 110 unrelated legacy ops). Compare against:

```bash
git stash && \
  conda run -n candle python -m pytest tests/cpu/ tests/contract/ --tb=no -q 2>&1 | tail -3 && \
  git stash pop
```

If the new diff introduces additional failures, investigate and fix before proceeding.

- [ ] **Step 3: Run NPU suite if hardware is available**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle \
  python -m pytest tests/npu/ -v --tb=short 2>&1 | tail -40
```

If NPU is unavailable, the suite auto-skips. Note skipped tests in the PR description.

- [ ] **Step 4: Final commit, if any pylint fixes were needed**

If you had to add a pylint suppression or formatting fix in Step 1:

```bash
git add -A
git commit -m "style: silence pylint false-positive on view_backend import"
```

If no pylint fixes were needed, skip this step.

---

## Task 7: Push and open PR

**Files:** none.

- [ ] **Step 1: Verify the commit log is clean**

```bash
git log --oneline view-tracking-1b ^upstream/main
```

Expected output (5–7 commits):
- `docs(specs): add view-tracking 1B six-ops conditional-view design`
- `docs(specs): tighten 1B-A contiguous semantics — return self, not _make_view`
- `test(cpu): pin conditional-view fast-path for contiguous/flatten/unflatten`
- `fix(cpu): contiguous returns input when already contiguous`
- `fix(cpu): flatten/unflatten return a view when input is contiguous`
- `fix(mps): flatten/unflatten return a registered view when input is contiguous`
- `fix(npu): contiguous returns input when already contiguous`
- (optional) `style: silence pylint false-positive on view_backend import`

If commits are missing or out of order, address that before pushing.

- [ ] **Step 2: Rebase onto upstream/main**

```bash
git fetch upstream main
git rebase upstream/main
```

Expected: clean rebase. If conflicts arise, resolve them before continuing.

- [ ] **Step 3: Push to origin**

```bash
git push -u origin view-tracking-1b
```

Expected: push succeeds. If `gh` CLI behaves oddly, recall that the real CLI lives at `~/anaconda3/envs/candle311/bin/gh` (v2.91.0).

- [ ] **Step 4: Create the PR**

```bash
~/anaconda3/envs/candle311/bin/gh pr create \
  --repo candle-org/candle \
  --head lvyufeng:view-tracking-1b \
  --base main \
  --title "fix(backends): contiguous/flatten/unflatten conditional-view forward (1B-A of view-tracking)" \
  --body "$(cat <<'EOF'
## Summary

Sub-batch 1B-A of view-tracking. Wires three of the six PyTorch `VIEW_FUNCTIONS` ops onto candle's view infrastructure by fixing their forward path:

- `contiguous(a)` returns `a` itself when already contiguous in default format. Was: always allocated a fresh copy.
- `flatten(a, ...)` and `unflatten(a, ...)` return a registered view (`_make_view`) when the input is contiguous. Was: always allocated a copy via `np.ascontiguousarray`.
- For non-contiguous input, all three ops still allocate and copy as before.

Backends touched: cpu, mps, npu. CUDA has no separate impl. NPU `flatten_op`/`unflatten_op` already delegate to `view_backend.reshape`, which handles conditional-view correctly — no change there.

## What is NOT in this PR

- No `view_func` / `rev_view_func` wiring. They stay `None` on every output. The infrastructure landed in PR #372 still has no live consumers.
- No removal of hand-added `ContiguousBackward0` / `FlattenBackward0` / `UnflattenBackward0` classes. Autograd still flows through them.
- No autograd-routing changes for `narrow`, `squeeze`, `movedim` (their forward is already correct — they're 1B-B work).

Sub-batch 1B-B will land both pieces in a separate PR once 1B-A is merged.

## Spec

`docs/superpowers/specs/2026-05-04-view-tracking-1b-six-ops-design.md`

## Test plan

- [x] `tests/cpu/test_view_fastpath.py` — new file, 11 tests covering identity-on-fast-path, storage-sharing, mutation propagation, copy fallback, backward identity.
- [x] `pytest tests/cpu/ tests/contract/` — same baseline as `view-tracking-1b` after PR #372.
- [ ] `pytest tests/mps/` — runs in CI (macos-14).
- [ ] `pytest tests/npu/` — local NPU host.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: gh prints the PR URL. Save the URL for follow-up.

- [ ] **Step 5: Surface the PR URL to the user**

Print the URL and the change summary so the user can review.

---

## Spec coverage check (run after writing the plan)

Cross-walk the spec → tasks:

| Spec section | Implemented in |
|---|---|
| `contiguous` returns `a` when contiguous (CPU) | Task 2 |
| `contiguous` returns `a` when contiguous (MPS) | already correct, spec line 122–124 |
| `contiguous` returns `a` when contiguous (NPU) | Task 5 |
| `flatten` view via `_make_view` when contiguous (CPU) | Task 3 |
| `flatten` view via `_make_view` when contiguous (MPS) | Task 4 |
| `flatten` view (NPU) | already correct via `view_backend.reshape`, spec line 136 |
| `unflatten` view (CPU) | Task 3 |
| `unflatten` view (MPS) | Task 4 |
| `unflatten` view (NPU) | already correct, spec line 137 |
| view_func / rev_view_func stay None | Tasks 3 and 4 explicitly omit them |
| Test file `tests/cpu/test_view_fastpath.py` | Task 1 |
| Pylint + cpu/contract gate | Task 6 |
| MPS regression coverage | Task 6 (CI) |
| NPU regression coverage | Task 6 (local) |
| Open PR | Task 7 |

No spec gaps. `contiguous` MPS and `flatten`/`unflatten` NPU are explicitly no-ops per spec — no task is needed for those, by design.
