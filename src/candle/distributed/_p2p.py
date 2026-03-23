from ._work import Work


class P2POp:
    def __init__(self, op, tensor, peer=None, group=None, tag=0, group_peer=None):
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group
        self.tag = tag
        self.group_peer = group_peer


class _BatchWork(Work):
    """Aggregate Work representing a batch of P2P ops.

    PyTorch pipelining treats ``batch_isend_irecv`` as returning a batch-level
    handle whose lifecycle covers all constituent P2P ops.  This wrapper keeps
    the existing per-op ``Work`` objects but exposes a single ``Work``-like
    interface for the whole batch.
    """

    def __init__(self, works):
        super().__init__()
        self._works = list(works)

    def wait(self, timeout=None):
        if self._completed:
            return True
        try:
            for work in self._works:
                work.wait(timeout=timeout)
        except Exception as exc:  # pylint: disable=broad-except
            self._exception = exc
            self._completed = True
            self._resolve_future(exc)
            raise
        self._completed = True
        self._resolve_future(None)
        return True

    def is_completed(self):
        if self._completed:
            return True
        done = all(work.is_completed() for work in self._works)
        if done:
            self._completed = True
            self._resolve_future(None)
        return done

    def is_success(self):
        if not self.is_completed():
            return False
        return all(work.is_success() for work in self._works)

    def exception(self):
        if self._exception is not None:
            return self._exception
        for work in self._works:
            exc = work.exception()
            if exc is not None:
                return exc
        return None

    def result(self):
        return [work.result() for work in self._works]



def batch_isend_irecv(p2p_op_list):
    works = []
    for p2p in p2p_op_list:
        dst_or_src = p2p.peer
        kwargs = {"group": p2p.group, "tag": p2p.tag}
        # isend/irecv accept group_dst/group_src as keyword args
        if dst_or_src is None and p2p.group_peer is not None:
            # Determine the right keyword based on the op name
            op_name = getattr(p2p.op, "__name__", "")
            if "send" in op_name:
                kwargs["group_dst"] = p2p.group_peer
            else:
                kwargs["group_src"] = p2p.group_peer
        work = p2p.op(p2p.tensor, dst_or_src, **kwargs)
        works.append(work)
    return _BatchWork(works)
