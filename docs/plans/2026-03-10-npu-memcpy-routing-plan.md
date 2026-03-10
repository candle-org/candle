# NPU Memcpy Routing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Route all NPU memcpy call-sites through runtime helpers to enforce current-stream semantics and avoid direct ACL usage outside runtime.

**Architecture:** Add runtime `memcpy_h2d/d2h/d2d` helpers that default to `current_stream()` when `stream=None`. Replace all non-runtime `acl.rt.memcpy` usages with these helpers while keeping sync explicit at call-sites.

**Tech Stack:** Python, candle NPU runtime/ops, pytest

---

### Task 1: Add Static Contract Test For Direct ACL Memcpy

**Files:**
- Create: `tests/contract/test_npu_memcpy_routing.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_npu_memcpy_calls_use_runtime_helpers():
    repo_root = Path(__file__).resolve().parents[2]
    disallowed = [
        repo_root / "src/candle/_dispatch/functionalize.py",
        repo_root / "src/candle/_storage.py",
        repo_root / "src/candle/_backends/npu/ops.py",
        repo_root / "src/candle/distributed/_process_group.py",
        repo_root / "src/candle/distributed/__init__.py",
    ]
    for path in disallowed:
        text = path.read_text()
        assert "acl.rt.memcpy" not in text, f"direct acl memcpy in {path}"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/contract/test_npu_memcpy_routing.py -v --tb=short`
Expected: FAIL with message about `acl.rt.memcpy` in one or more files.

**Step 3: Commit (optional checkpoint)**

```bash
git add tests/contract/test_npu_memcpy_routing.py
git commit -m "test: guard against direct acl memcpy"
```

### Task 2: Add Runtime Helper Tests For Stream Selection

**Files:**
- Modify: `tests/npu/test_npu_streams.py`

**Step 1: Write the failing tests**

```python
def test_memcpy_helpers_use_current_stream(monkeypatch):
    import candle._backends.npu.runtime as npu_runtime
    import candle._backends.npu.state as npu_state

    calls = {}

    class FakeRt:
        def memcpy_async(self, dst, dst_size, src, src_size, kind, stream):
            calls["stream"] = stream
            return 0

        def memcpy(self, dst, dst_size, src, src_size, kind):
            calls["stream"] = "sync"
            return 0

    class FakeAcl:
        def __init__(self):
            self.rt = FakeRt()

    class DummyRuntime:
        device_id = 0
        def activate(self):
            return None

    class DummyStream:
        stream = 123

    monkeypatch.setattr(npu_runtime, "acl", FakeAcl())
    monkeypatch.setattr(npu_runtime, "ensure_acl", lambda: npu_runtime.acl)
    monkeypatch.setattr(npu_state, "current_stream", lambda device_id=0: DummyStream())

    npu_runtime.memcpy_d2d(1, 4, 2, runtime=DummyRuntime(), stream=None, non_blocking=True)
    assert calls.get("stream") == 123
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/npu/test_npu_streams.py -v --tb=short`
Expected: FAIL because `memcpy_d2d` helper does not yet exist.

**Step 3: Commit (optional checkpoint)**

```bash
git add tests/npu/test_npu_streams.py
git commit -m "test: cover runtime memcpy stream selection"
```

### Task 3: Implement Runtime Memcpy Helpers

**Files:**
- Modify: `src/candle/_backends/npu/runtime.py`

**Step 1: Implement minimal helpers**

```python
ACL_MEMCPY_DEVICE_TO_DEVICE = 3

def _resolve_stream(runtime, stream):
    if stream is None:
        from . import state as npu_state
        stream = npu_state.current_stream(runtime.device_id)
    return stream.stream if hasattr(stream, "stream") else stream

def memcpy_h2d(dst, size, src_ptr, runtime=None, stream=None, non_blocking=False):
    if runtime is None:
        runtime = get_runtime(0)
    _ensure_acl()
    runtime.activate()
    stream_handle = _resolve_stream(runtime, stream)
    if non_blocking and hasattr(acl.rt, "memcpy_async"):
        ret = acl.rt.memcpy_async(dst, size, src_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE, stream_handle)
    else:
        ret = acl.rt.memcpy(dst, size, src_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.memcpy H2D failed: {ret}")

def memcpy_d2h(dst_ptr, size, src, runtime=None, stream=None, non_blocking=False):
    ...

def memcpy_d2d(dst_ptr, size, src_ptr, runtime=None, stream=None, non_blocking=False):
    ...
```

**Step 2: Run tests**

Run: `PYTHONPATH=src pytest tests/npu/test_npu_streams.py -v --tb=short`
Expected: PASS for the new helper tests.

**Step 3: Commit**

```bash
git add src/candle/_backends/npu/runtime.py
git commit -m "feat: add npu runtime memcpy helpers"
```

### Task 4: Replace Direct ACL Memcpy Call-Sites

**Files:**
- Modify: `src/candle/_dispatch/functionalize.py`
- Modify: `src/candle/_storage.py`
- Modify: `src/candle/_backends/npu/ops.py`
- Modify: `src/candle/distributed/_process_group.py`
- Modify: `src/candle/distributed/__init__.py`

**Step 1: Replace direct calls**

- Use `npu_runtime.memcpy_d2d` for device-to-device copies.
- Use `npu_runtime.memcpy_h2d` for host-to-device copies.
- Use `npu_runtime.memcpy_d2h` for device-to-host copies.
- Keep any explicit `synchronize_stream` calls in place; do not move sync into helpers.

**Step 2: Run contract test**

Run: `pytest tests/contract/test_npu_memcpy_routing.py -v --tb=short`
Expected: PASS (no direct `acl.rt.memcpy` outside runtime).

**Step 3: Run stream tests**

Run: `PYTHONPATH=src pytest tests/npu/test_npu_streams.py -v --tb=short`
Expected: PASS.

**Step 4: Commit**

```bash
git add src/candle/_dispatch/functionalize.py \
        src/candle/_storage.py \
        src/candle/_backends/npu/ops.py \
        src/candle/distributed/_process_group.py \
        src/candle/distributed/__init__.py
git commit -m "refactor: route npu memcpy through runtime"
```

### Task 5: Full Verification Gate

**Step 1: Run required tests**

Run: `pytest tests/contract/ -v --tb=short`
Expected: PASS.

**Step 2: Commit any remaining changes**

```bash
git add docs/plans/2026-03-10-npu-memcpy-routing-design.md \
        docs/plans/2026-03-10-npu-memcpy-routing-plan.md
git commit -m "docs: add npu memcpy routing plan"
```
