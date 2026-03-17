"""Thread-safe Future[T] and collect_all, pure-Python replacement for torch._C.Future."""

import threading
from typing import TypeVar, Generic, Optional, List, Callable

T = TypeVar("T")
S = TypeVar("S")


class Future(Generic[T]):
    """A thread-safe generic Future compatible with the PyTorch torch.futures.Future API.

    Supports real async waiting: ``wait()`` blocks until another thread calls
    ``set_result`` or ``set_exception``.
    """

    def __init__(self, *, devices=None):
        self._result: Optional[T] = None
        self._exception: Optional[BaseException] = None
        self._done = False
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._done_callbacks: List[Callable] = []
        self._devices = devices

    # ---- producers ----

    def set_result(self, result: T) -> None:
        """Set the result, wake waiters, and fire done-callbacks."""
        with self._lock:
            if self._done:
                raise RuntimeError("Future already completed")
            self._result = result
            self._done = True
            self._event.set()
            callbacks = list(self._done_callbacks)
        for cb in callbacks:
            cb(self)

    def set_exception(self, exception: BaseException) -> None:
        """Set an exception, wake waiters, and fire done-callbacks."""
        with self._lock:
            if self._done:
                raise RuntimeError("Future already completed")
            self._exception = exception
            self._done = True
            self._event.set()
            callbacks = list(self._done_callbacks)
        for cb in callbacks:
            cb(self)

    # ---- consumers ----

    def wait(self) -> T:
        """Block until the future completes and return the result.

        Raises the stored exception if one was set.
        """
        self._event.wait()
        if self._exception is not None:
            raise self._exception
        return self._result  # type: ignore[return-value]

    def value(self) -> T:
        """Alias for ``wait()``."""
        return self.wait()

    def done(self) -> bool:
        """Return ``True`` if the future has completed."""
        return self._done

    # ---- chaining ----

    def then(self, callback: Callable[["Future[T]"], S]) -> "Future[S]":
        """Register *callback* and return a new ``Future[S]`` for its result.

        *callback* receives this future (already completed) and should return
        a value of type *S*.  If *callback* raises, the returned future gets
        the exception instead.
        """
        new_future: "Future[S]" = Future()

        def _run(source: "Future[T]") -> None:
            try:
                new_future.set_result(callback(source))
            except Exception as exc:  # pylint: disable=broad-except
                new_future.set_exception(exc)

        with self._lock:
            if self._done:
                _run(self)
            else:
                self._done_callbacks.append(_run)
        return new_future

    def add_done_callback(self, callback: Callable[["Future[T]"], None]) -> None:
        """Register a callback that fires when this future completes.

        If the future is already done the callback runs immediately (in the
        caller's thread).
        """
        with self._lock:
            if self._done:
                callback(self)
            else:
                self._done_callbacks.append(callback)


def collect_all(futures: List[Future]) -> "Future[List[Future]]":
    """Return a future that completes when every future in *futures* completes.

    The result is the original list of (now-completed) futures.  If any input
    future has an exception, the aggregate future propagates the first one.
    """
    result_future: Future[List[Future]] = Future()

    if not futures:
        result_future.set_result([])
        return result_future

    remaining = len(futures)
    lock = threading.Lock()
    first_exception: List[Optional[BaseException]] = [None]  # mutable cell

    def _on_done(fut: Future) -> None:
        nonlocal remaining
        with lock:
            if fut._exception is not None and first_exception[0] is None:  # pylint: disable=protected-access
                first_exception[0] = fut._exception  # pylint: disable=protected-access
            remaining -= 1
            all_done = remaining == 0
        if all_done:
            if first_exception[0] is not None:
                result_future.set_exception(first_exception[0])
            else:
                result_future.set_result(futures)

    for f in futures:
        f.add_done_callback(_on_done)

    return result_future
