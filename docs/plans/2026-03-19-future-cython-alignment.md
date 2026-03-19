# Futures Cython Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move candle's `Future` / `collect_all` implementation onto a compiled Cython fast path while preserving the existing `candle.futures` import surface and compatibility with `distributed.Work.get_future()`.

**Architecture:** Add `src/candle/_cython/_future.pyx` as the compiled primary implementation of `Future` and `collect_all`. Keep `src/candle/futures.py` as a fast-path loader plus pure-Python fallback, following the same pattern already used for stream alignment. Existing callers continue importing from `candle.futures` unchanged.

**Tech Stack:** Cython >= 3.0, Python 3.11, `threading`, `typing`, setuptools extensions, pytest

---

### Task 1: Add failing Future fast-path and compatibility tests

**Files:**
- Modify: `tests/cpu/test_futures.py`
- Create if missing: `tests/cpu/test_futures.py`
- Test: `tests/cpu/test_futures.py`

**Step 1: Write the failing tests**

Create `tests/cpu/test_futures.py` with these tests:

```python
import threading

from candle.futures import Future, collect_all


def test_future_basic_result():
    fut = Future()
    fut.set_result(42)
    assert fut.done() is True
    assert fut.wait() == 42
    assert fut.value() == 42


def test_future_exception_propagates():
    fut = Future()
    fut.set_exception(RuntimeError("boom"))

    try:
        fut.wait()
        assert False, "expected exception"
    except RuntimeError as exc:
        assert str(exc) == "boom"


def test_future_then_chains_result():
    fut = Future()
    chained = fut.then(lambda src: src.wait() * 2)
    fut.set_result(21)
    assert chained.wait() == 42


def test_future_add_done_callback_after_completion_runs_immediately():
    fut = Future()
    seen = []
    fut.set_result("ok")
    fut.add_done_callback(lambda f: seen.append(f.wait()))
    assert seen == ["ok"]


def test_collect_all_returns_original_futures_in_order():
    futures = [Future(), Future(), Future()]
    agg = collect_all(futures)
    futures[0].set_result("a")
    futures[1].set_result("b")
    futures[2].set_result("c")
    assert agg.wait() == futures


def test_collect_all_propagates_first_exception_after_all_done():
    futures = [Future(), Future()]
    agg = collect_all(futures)
    futures[0].set_exception(ValueError("bad"))
    futures[1].set_result("ok")

    try:
        agg.wait()
        assert False, "expected exception"
    except ValueError as exc:
        assert str(exc) == "bad"


def test_future_wait_blocks_until_other_thread_sets_result():
    fut = Future()
    seen = []

    def waiter():
        seen.append(fut.wait())

    thread = threading.Thread(target=waiter)
    thread.start()
    fut.set_result("threaded")
    thread.join(timeout=2)
    assert seen == ["threaded"]
```

**Step 2: Run test to verify it fails only after we add the new compiled-surface check**

Append this additional red test at the bottom:

```python
def test_future_module_reports_compiled_fast_path():
    import candle.futures as futures_mod

    assert hasattr(futures_mod, "_FUTURE_CYTHON")
    assert futures_mod._FUTURE_CYTHON is True
```

Run:
```bash
python -m pytest tests/cpu/test_futures.py::test_future_module_reports_compiled_fast_path -v --tb=short
```

Expected: FAIL because `_FUTURE_CYTHON` does not exist yet.

**Step 3: Commit**

```bash
git add tests/cpu/test_futures.py
git commit -m "test: add futures Cython fast-path checks"
```

---

### Task 2: Add the new Cython extension to `setup.py`

**Files:**
- Modify: `setup.py`
- Test: `python setup.py build_ext --inplace`

**Step 1: Add the extension entry**

In `setup.py`, add this extension near the other `_cython` modules, right after `_stream`:

```python
                Extension(
                    "candle._cython._future",
                    ["src/candle/_cython/_future.pyx"],
                ),
```

The relevant block should become:

```python
                Extension(
                    "candle._cython._stream",
                    ["src/candle/_cython/_stream.pyx"],
                ),
                Extension(
                    "candle._cython._future",
                    ["src/candle/_cython/_future.pyx"],
                ),
                Extension(
                    "candle.distributed._c10d",
                    ["src/candle/distributed/_c10d.pyx"],
                ),
```

**Step 2: Run build to verify target inclusion**

Run:
```bash
python setup.py build_ext --inplace
```

Expected: output includes `building 'candle._cython._future' extension`.

**Step 3: Commit**

```bash
git add setup.py
git commit -m "build: add Cython future extension"
```

---

### Task 3: Create `_cython/_future.pyx`

**Files:**
- Create: `src/candle/_cython/_future.pyx`
- Source reference: `src/candle/futures.py`
- Test: `python -m cython src/candle/_cython/_future.pyx`

**Step 1: Implement the compiled primary Future path**

Create `src/candle/_cython/_future.pyx` with this header:

```cython
# cython: language_level=3, boundscheck=False, wraparound=False
"""Compiled primary implementation of candle Future / collect_all.

This module mirrors the public API exposed by `candle.futures` while serving as
its compiled fast path.
"""
```

Then port the full current implementation from `src/candle/futures.py`, preserving these public names exactly:
- `T = TypeVar("T")`
- `S = TypeVar("S")`
- `class Future(Generic[T])`
- `def collect_all(futures)`

Use the current behavior exactly:
- `Future.__init__(*, devices=None)`
- `_result`, `_exception`, `_done`, `_event`, `_lock`, `_done_callbacks`, `_devices`
- `set_result`, `set_exception`, `wait`, `value`, `done`, `then`, `add_done_callback`
- `collect_all` propagates the first exception after all futures complete

Keep the initial implementation close to the current Python logic; it is acceptable for the first `.pyx` version to remain Python-style internally.

**Step 2: Run syntax verification**

Run:
```bash
python -m cython src/candle/_cython/_future.pyx
```

Expected: no compile errors.

**Step 3: Commit**

```bash
git add src/candle/_cython/_future.pyx
git commit -m "feat(cython): add compiled future implementation"
```

---

### Task 4: Convert `futures.py` into fast-path loader + Python fallback

**Files:**
- Modify: `src/candle/futures.py`
- Test: import smoke test

**Step 1: Add compiled fast path import block**

At the top of `src/candle/futures.py`, add:

```python
try:
    from ._cython._future import Future, collect_all  # pylint: disable=no-name-in-module
    _FUTURE_CYTHON = True
except ModuleNotFoundError as exc:
    if exc.name in {"candle._cython", "candle._cython._future"}:
        _FUTURE_CYTHON = False
    else:
        raise
except ImportError:
    raise
```

Then wrap the existing pure-Python implementation under:

```python
if not _FUTURE_CYTHON:
    # existing current futures.py body, indented under this block
```

Do not leave duplicate top-level definitions active when the compiled module imports successfully.

**Step 2: Run import smoke test**

Run:
```bash
PYTHONPATH="/home/lvyufeng/lvyufeng/candle/.worktrees/feat/future-cython/src" python -c "import candle.futures as f; print(f._FUTURE_CYTHON); print(f.Future); print(f.collect_all)"
```

Expected before build: `False`, then Python classes/functions.

**Step 3: Commit**

```bash
git add src/candle/futures.py
git commit -m "feat: route futures module through Cython fast path with Python fallback"
```

---

### Task 5: Build and verify end-to-end

**Files:**
- Modify if needed: `tests/cpu/test_futures.py`
- Test: build + futures tests + import smoke tests

**Step 1: Build the extension**

Run:
```bash
python setup.py build_ext --inplace
```

Expected: output includes `building 'candle._cython._future' extension`.

**Step 2: Run the focused futures tests**

Run:
```bash
python -m pytest tests/cpu/test_futures.py -v --tb=short
```

Expected: PASS.

**Step 3: Verify compiled fast path is active**

Run:
```bash
PYTHONPATH="/home/lvyufeng/lvyufeng/candle/.worktrees/feat/future-cython/src" python -c "import candle.futures as f; print(f._FUTURE_CYTHON); fut = f.Future(); fut.set_result(1); print(fut.wait())"
```

Expected:
- `_FUTURE_CYTHON == True`
- `fut.wait()` prints `1`

**Step 4: Verify distributed compatibility**

Run:
```bash
PYTHONPATH="/home/lvyufeng/lvyufeng/candle/.worktrees/feat/future-cython/src" python -c "from candle.distributed._work import Work; w = Work(); fut = w.get_future(); print(type(fut).__name__); print(fut.wait())"
```

Expected:
- prints `Future`
- prints `[]`

**Step 5: Run pylint on touched files**

Run:
```bash
pylint src/candle/futures.py src/candle/_cython/_future.pyx --rcfile=.github/pylint.conf
```

Expected: no new pylint errors.

**Step 6: Commit final fixes if any**

```bash
git add src/candle/futures.py src/candle/_cython/_future.pyx setup.py tests/cpu/test_futures.py
git commit -m "test: verify Cython future alignment end-to-end"
```

---

### Task 6: Clarify module comments and fallback semantics

**Files:**
- Modify: `src/candle/futures.py`
- Modify: `src/candle/_cython/_future.pyx`

**Step 1: Update docstrings/comments**

Make sure:
- `src/candle/_cython/_future.pyx` clearly states it is the compiled primary implementation
- `src/candle/futures.py` clearly states it is the public import surface and Python fallback
- the fallback import logic is easy to understand and mirrors the pattern already used in `_stream.py`

**Step 2: Run import smoke test again**

Run:
```bash
PYTHONPATH="/home/lvyufeng/lvyufeng/candle/.worktrees/feat/future-cython/src" python -c "import candle.futures as f; print(f._FUTURE_CYTHON); print(f.Future); print(f.collect_all)"
```

Expected: `True`, then compiled symbols.

**Step 3: Commit**

```bash
git add src/candle/futures.py src/candle/_cython/_future.pyx
git commit -m "docs: clarify Cython future fallback semantics"
```
