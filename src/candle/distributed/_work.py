# Import Future at module load time so tests can inject a stub via sys.modules
# before this module is exec'd.  A lazy `from ..futures import Future` inside
# get_future() would re-resolve the real compiled Future after the stub is
# cleaned up, breaking the test bootstrap.
try:
    from ..futures import Future as _Future  # normal package import
except ImportError:  # pragma: no cover
    _Future = None  # type: ignore[assignment,misc]


class Work:
    def __init__(self, stream=None, device_id=None, source_rank=-1):
        self._completed = False
        self._stream = stream
        self._device_id = device_id
        self._exception = None
        self._source_rank = source_rank
        self._on_wait = None
        self._future = None  # lazily created by get_future()
        # ACL event recorded immediately after the HCCL kernel was enqueued.
        # When set, wait() syncs the event (not the whole stream) and
        # is_completed() polls via query_event() without blocking.
        self._aclrt_event = None

    def wait(self, timeout=None):
        if self._completed:
            return True
        if self._aclrt_event is not None:
            # Event-based path: synchronize only the recorded event, not the
            # entire stream.  This is the correct async completion semantic for
            # HCCL collectives — the stream may have work enqueued after ours.
            try:
                from candle._backends.npu import runtime as npu_runtime
                dev = self._device_id if self._device_id is not None else 0
                npu_runtime.get_runtime(dev).synchronize_event(self._aclrt_event)
            except Exception as e:
                self._exception = e
                self._resolve_future(exc=e)
                raise
        elif self._stream is not None:
            # Legacy stream-sync fallback (no event recorded).
            try:
                from candle._backends.npu import runtime as npu_runtime
                dev = self._device_id if self._device_id is not None else 0
                npu_runtime.get_runtime(dev).synchronize_stream(self._stream)
            except Exception as e:
                self._exception = e
                self._resolve_future(exc=e)
                raise
        if self._on_wait is not None:
            try:
                self._on_wait()  # pylint: disable=not-callable
            except Exception as e:
                self._exception = e
                self._on_wait = None
                self._completed = True
                self._resolve_future(exc=e)
                raise
            finally:
                self._on_wait = None
        self._completed = True
        self._resolve_future(exc=None)
        return True

    def _resolve_future(self, exc):
        """Resolve the stored future (if any) with result or exception."""
        if self._future is None:
            return
        fut = self._future
        if exc is not None:
            try:
                fut.set_exception(exc)
            except RuntimeError:
                pass  # already resolved
        else:
            try:
                fut.set_result(self.result())
            except RuntimeError:
                pass  # already resolved

    def is_completed(self):
        if self._completed:
            return True
        if self._aclrt_event is not None:
            # Non-blocking poll: query the event without synchronizing.
            try:
                from candle._backends.npu import runtime as npu_runtime
                dev = self._device_id if self._device_id is not None else 0
                done = npu_runtime.get_runtime(dev).query_event(self._aclrt_event)
                if done:
                    self._completed = True
                    self._resolve_future(None)
                return done
            except Exception:  # pylint: disable=broad-except
                pass
        return self._completed

    def is_success(self):
        return self._completed and self._exception is None

    def exception(self):
        return self._exception

    def source_rank(self):
        return self._source_rank

    def result(self):
        return []

    def synchronize(self):
        self.wait()

    def get_future(self):
        if self._future is not None:
            return self._future

        fut = _Future()
        self._future = fut

        # If already completed, resolve immediately
        if self._completed:
            if self._exception is not None:
                try:
                    fut.set_exception(self._exception)
                except RuntimeError:
                    pass
            else:
                try:
                    fut.set_result(self.result())
                except RuntimeError:
                    pass

        return fut
