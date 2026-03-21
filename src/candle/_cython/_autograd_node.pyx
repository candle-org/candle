# cython: language_level=3, boundscheck=False, wraparound=False
import weakref
_METADATA_BY_NODE = weakref.WeakKeyDictionary()

_RELEASED_SAVED_TENSORS_ERROR = (
    "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). "
    "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). "
    "Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."
)
_RAW_SAVED_FIELD_PREFIX = "_raw_saved_"


cdef class _SavedValue:
    cdef object _value

    def __init__(self, value):
        self._value = value

    def release(self):
        return

    def materialize(self):
        return self._value


class InputMetadata:
    def __init__(self, tensor):
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = tensor.device
        self.is_nested_tensor = False
        self.is_cpp_nested_tensor = False


cdef class SavedTensor:
    cdef object _tensor_ref
    cdef object _saved_version
    cdef bint _released
    cdef object _hooks
    cdef object _global_hooks
    cdef object _packed

    def __init__(self, tensor):
        self._tensor_ref = tensor
        self._saved_version = None if tensor is None else tensor._version_counter.value
        self._released = False
        self._hooks = None
        from candle._cython._hooks_state import get_stack
        stack = get_stack()
        self._global_hooks = stack[len(stack) - 1] if stack else None
        hooks = self._global_hooks
        if hooks is None:
            self._packed = None
        else:
            pack, _ = hooks
            if tensor is None:
                self._packed = None
            else:
                self._packed = pack(tensor)


    def register_hooks(self, *args):
        cdef object pack
        cdef object unpack
        cdef object before_version
        cdef object after_version
        cdef object packed
        if len(args) != 2:
            raise TypeError("incompatible function arguments")
        pack, unpack = args
        if not callable(pack) or not callable(unpack):
            raise TypeError("incompatible function arguments")
        if self._released:
            raise RuntimeError(_RELEASED_SAVED_TENSORS_ERROR)
        if self._hooks is not None:
            raise RuntimeError("SavedTensor hooks have already been set")
        if self._tensor_ref is None:
            raise RuntimeError("None is forbidden")
        before_version = self._tensor_ref._version_counter.value
        packed = pack(self._tensor_ref)
        after_version = self._tensor_ref._version_counter.value
        if before_version != after_version:
            raise RuntimeError("A saved tensor pack hook is modifying its input in place.")
        self._hooks = (pack, unpack)
        self._packed = packed

    def release(self):
        self._released = True

    def materialize(self):
        cdef object hooks
        cdef object unpack
        cdef object result
        cdef object shape
        cdef str tensor_type
        cdef str op
        if self._released:
            raise RuntimeError(_RELEASED_SAVED_TENSORS_ERROR)
        if self._tensor_ref is None:
            return None
        if self._tensor_ref._version_counter.value != self._saved_version:
            shape = "x".join(str(d) for d in getattr(self._tensor_ref, "shape", ()))
            tensor_type = "torch.Tensor"
            op = "AsStridedBackward0"
            raise RuntimeError(
                "one of the variables needed for gradient computation has been modified by an inplace operation: "
                f"[{tensor_type} [{shape}]], which is output 0 of {op}, is at version {self._tensor_ref._version_counter.value}; "
                f"expected version {self._saved_version} instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, "
                "with torch.autograd.set_detect_anomaly(True)."
            )
        hooks = self._hooks or self._global_hooks
        if hooks is None:
            return self._tensor_ref
        _, unpack = hooks
        result = unpack(self._packed)
        from candle._tensor import Tensor
        if not isinstance(result, Tensor):
            raise TypeError("Output of saved tensor unpack_hook expected to be a Tensor")
        return result


cdef int _node_hook_handle_counter = 0


cdef class _NodeHookHandle:
    cdef object _hooks
    cdef public object id
    cdef object __weakref__
    cdef dict __dict__

    def __init__(self, hooks):
        global _node_hook_handle_counter
        self._hooks = hooks
        self.id = _node_hook_handle_counter
        _node_hook_handle_counter += 1

    def remove(self):
        if self._hooks is None:
            return
        self._hooks.pop(self.id, None)
        self._hooks = None


cdef class AccumulateGrad:
    cdef public object tensor
    cdef public dict _hooks
    cdef public dict _prehooks
    cdef public object _metadata
    cdef public str _name
    cdef dict __dict__
    cdef object __weakref__

    def __init__(self, tensor):
        self.tensor = tensor
        self._hooks = {}
        self._prehooks = {}
        self._metadata = None
        self._name = "torch::autograd::AccumulateGrad"

    def register_hook(self, hook):
        cdef _NodeHookHandle handle = _NodeHookHandle(self._hooks)
        self._hooks[handle.id] = hook
        return handle

    def register_prehook(self, hook):
        cdef _NodeHookHandle handle = _NodeHookHandle(self._prehooks)
        self._prehooks[handle.id] = hook
        return handle

    def apply_prehooks(self, grad):
        cdef object grads = (grad,)
        cdef object result
        for hook in self._prehooks.values():
            result = hook(grads)
            if result is not None:
                grads = tuple(result)
        return grads[0]

    def apply_posthooks(self, grad):
        cdef object grad_input = (grad,)
        cdef object grad_output = ()
        for hook in self._hooks.values():
            hook(grad_output, grad_input)

    @property
    def next_functions(self):
        return ()

    @property
    def metadata(self):
        meta = _METADATA_BY_NODE.get(self)
        if meta is None:
            meta = {}
            _METADATA_BY_NODE[self] = meta
        return meta

    def name(self):
        return self._name

    def release_saved_tensors(self):
        return


cdef class Node:
    cdef public object backward
    cdef public tuple inputs
    cdef public list _saved_tensors_list
    cdef public dict _saved_fields
    cdef public dict _hooks
    cdef public dict _prehooks
    cdef public tuple _next_functions_cache
    cdef public object _metadata
    cdef public str _name
    cdef public object _anomaly_trace
    cdef public object _anomaly_parent
    cdef dict __dict__
    cdef object __weakref__

    def __init__(self, backward=None, inputs=(), *, name=None):
        if backward is not None:
            self.backward = backward
        self.inputs = tuple(inputs)
        self._saved_tensors_list = []
        self._saved_fields = {}
        self._hooks = {}
        self._prehooks = {}
        self._metadata = None
        self._name = name or type(self).__name__
        self._anomaly_trace = None
        self._anomaly_parent = None
        self._next_functions_cache = self._freeze_next_functions()

    def save_for_backward(self, *tensors):
        cdef list saved = []
        cdef dict seen = {}
        cdef object t
        cdef object tid
        cdef object cached
        for t in tensors:
            if t is None:
                saved.append(SavedTensor(None))
            elif hasattr(t, "_version_counter"):
                tid = id(t)
                cached = seen.get(tid)
                if cached is None:
                    cached = SavedTensor(t)
                    seen[tid] = cached
                saved.append(cached)
            else:
                saved.append(_SavedValue(t))
        self._saved_tensors_list = saved

    def saved_tensors(self):
        cdef object saved
        cdef dict seen_materialized = {}
        cdef object key
        cdef object materialized
        for saved in self._saved_tensors_list:
            if getattr(saved, "_released", False):
                raise RuntimeError(_RELEASED_SAVED_TENSORS_ERROR)
        out = []
        for saved in self._saved_tensors_list:
            if isinstance(saved, SavedTensor):
                key = id(saved)
                materialized = seen_materialized.get(key)
                if materialized is None:
                    materialized = saved.materialize()
                    seen_materialized[key] = materialized
                out.append(materialized)
            else:
                out.append(saved.materialize())
        return tuple(out)

    def release_saved_tensors(self):
        cdef object saved
        for saved in self._saved_tensors_list:
            saved.release()
        for saved in self._saved_fields.values():
            if isinstance(saved, (list, tuple)):
                for item in saved:
                    item.release()
            else:
                saved.release()

    def register_hook(self, hook):
        cdef _NodeHookHandle handle = _NodeHookHandle(self._hooks)
        self._hooks[handle.id] = hook
        return handle

    def register_prehook(self, hook):
        cdef _NodeHookHandle handle = _NodeHookHandle(self._prehooks)
        self._prehooks[handle.id] = hook
        return handle

    cdef tuple _freeze_next_functions(self):
        cdef list next_functions = []
        cdef object inp
        cdef object fn
        cdef object acc
        for inp in self.inputs:
            fn = getattr(inp, "grad_fn", None)
            if fn is not None:
                next_functions.append((fn, 0))
            elif getattr(inp, "requires_grad", False):
                acc = getattr(inp, "_accumulate_grad_node", None)
                if acc is None:
                    acc = AccumulateGrad(inp)
                    inp._accumulate_grad_node = acc
                next_functions.append((acc, 0))
            else:
                next_functions.append((None, 0))
        return tuple(next_functions)

    @property
    def next_functions(self):
        return self._next_functions_cache

    def __getattr__(self, name):
        cdef object saved_tensors
        cdef str key
        cdef object saved
        if name == "metadata":
            meta = _METADATA_BY_NODE.get(self)
            if meta is None:
                meta = {}
                _METADATA_BY_NODE[self] = meta
            return meta
        if name == "_raw_saved_tensors":
            return tuple(self._saved_tensors_list)
        if name == "_saved_tensors":
            saved_tensors = self._saved_tensors_list
            for saved in saved_tensors:
                if getattr(saved, "_released", False):
                    raise RuntimeError(_RELEASED_SAVED_TENSORS_ERROR)
            seen_materialized = {}
            out = []
            for saved in saved_tensors:
                if isinstance(saved, SavedTensor):
                    saved_id = id(saved)
                    materialized = seen_materialized.get(saved_id)
                    if materialized is None:
                        materialized = saved.materialize()
                        seen_materialized[saved_id] = materialized
                    out.append(materialized)
                else:
                    out.append(saved.materialize())
            return tuple(out)
        if name.startswith(_RAW_SAVED_FIELD_PREFIX):
            key = name[len(_RAW_SAVED_FIELD_PREFIX):]
            if key in self._saved_fields:
                return self._saved_fields[key]
        if name.startswith("_saved_"):
            key = name[len("_saved_"):]
            if key in self._saved_fields:
                saved = self._saved_fields[key]
                if isinstance(saved, (list, tuple)):
                    return tuple(item.materialize() for item in saved)
                return saved.materialize()
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def name(self):
        return self._name


FastNode = Node
