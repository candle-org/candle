# Pure Python `torch.futures` Implementation

**Date:** 2026-03-17
**Status:** Approved

## Goal

Replace the minimal synchronous `Future` in `src/candle/futures.py` with a full-featured, thread-safe, generic `Future[T]` class and add `collect_all`, matching the PyTorch `torch.futures` API without any C++ dependencies.

## Background

PyTorch's `torch.futures.Future` wraps `torch._C.Future` (C++ binding) and `torch._C._collect_all`. Candle needs a pure Python equivalent that:

- Supports real async waiting (threads can block on `wait()` until another thread calls `set_result`)
- Is generic (`Future[T]`) for type-checker compatibility
- Provides `collect_all` for aggregating multiple futures
- Remains backward-compatible with existing DDP comm hooks and `Work.get_future()`

## Design

### `Future[T]` Class

```python
class Future(Generic[T]):
    def __init__(self, *, devices=None)
    def set_result(self, result: T) -> None
    def set_exception(self, exception: BaseException) -> None
    def wait(self) -> T
    def value(self) -> T                    # alias for wait()
    def done(self) -> bool
    def then(self, cb: Callable[[Future[T]], S]) -> Future[S]
    def add_done_callback(self, cb: Callable[[Future[T]], None]) -> None
```

Key internals:
- `threading.Event` — `wait()` blocks until `set_result`/`set_exception` signals the event
- `threading.Lock` — protects state mutations and callback registration against races
- `then()` returns a new `Future[S]`; exceptions propagate automatically
- `add_done_callback()` fires immediately if already done, otherwise queued

### `collect_all` Function

```python
def collect_all(futures: List[Future]) -> Future[List[Future]]
```

- Uses `add_done_callback` on each input future (no extra threads)
- Atomic counter tracks remaining futures under a lock
- Resolves the aggregate future when all inputs complete
- Propagates the first exception encountered (after all futures finish)

## Impact on Existing Code

| Caller | Impact |
|--------|--------|
| `Work.get_future()` | None — `set_result`/`set_exception` API unchanged |
| DDP comm hooks | None — `Future()` + `set_result()` pattern unchanged |
| `distributed.py` `.wait()` calls | None — `wait()` semantics unchanged |
| `.value()` | Semantics change from `[result]` to `result`, but zero callers exist |

## Files Changed

| File | Change |
|------|--------|
| `src/candle/futures.py` | Rewrite `Future[T]`, add `collect_all` |

No other files require modification.
