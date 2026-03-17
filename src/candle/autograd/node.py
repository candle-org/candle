from .graph import current_saved_tensors_hooks

# FastNode base class: Cython if available, else no base
try:
    from .._cython._autograd_node import FastNode as _NodeBase
    _HAS_FAST_NODE = True
except ImportError:
    _NodeBase = object
    _HAS_FAST_NODE = False


class _SavedValue:
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


_RAW_SAVED_FIELD_PREFIX = "_raw_saved_"


class SavedTensor:
    def __init__(self, tensor):
        self._tensor_ref = tensor
        self._saved_version = None if tensor is None else tensor._version_counter.value
        self._released = False
        self._hooks = None
        self._global_hooks = current_saved_tensors_hooks()
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
        if len(args) != 2:
            raise TypeError("incompatible function arguments")
        pack, unpack = args
        if not callable(pack) or not callable(unpack):
            raise TypeError("incompatible function arguments")
        if self._released:
            raise RuntimeError(
                "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). "
                "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). "
                "Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."
            )
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
        if self._released:
            raise RuntimeError(
                "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). "
                "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). "
                "Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."
            )
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
        from .._tensor import Tensor
        if not isinstance(result, Tensor):
            raise TypeError("Output of saved tensor unpack_hook expected to be a Tensor")
        return result


class _NodeHookHandle:
    _next_id = 0

    def __init__(self, hooks):
        self._hooks = hooks
        self.id = _NodeHookHandle._next_id
        _NodeHookHandle._next_id += 1

    def remove(self):
        if self._hooks is None:
            return
        self._hooks.pop(self.id, None)
        self._hooks = None


class AccumulateGrad:
    def __init__(self, tensor):
        self.tensor = tensor
        self._hooks = {}
        self._prehooks = {}
        self._metadata = None
        self._name = "torch::autograd::AccumulateGrad"

    def register_hook(self, hook):
        handle = _NodeHookHandle(self._hooks)
        self._hooks[handle.id] = hook
        return handle

    def register_prehook(self, hook):
        handle = _NodeHookHandle(self._prehooks)
        self._prehooks[handle.id] = hook
        return handle

    def apply_prehooks(self, grad):
        grads = (grad,)
        for hook in self._prehooks.values():
            result = hook(grads)
            if result is not None:
                grads = tuple(result)
        return grads[0]

    def apply_posthooks(self, grad):
        grad_input = (grad,)
        grad_output = ()
        for hook in self._hooks.values():
            hook(grad_output, grad_input)

    @property
    def next_functions(self):
        return ()

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = {}
        return self._metadata

    def name(self):
        return self._name


class Node(_NodeBase):
    def __init__(self, backward, inputs, *, name=None):
        if _HAS_FAST_NODE:
            # FastNode.__init__ handles all field setup + _freeze_next_functions
            _NodeBase.__init__(self, backward, inputs, name=name)
        else:
            if backward is not None:
                self.backward = backward
            self.inputs = tuple(inputs)
            self._saved_tensors_list = []
            self._saved_fields = {}
            self._hooks = {}
            self._prehooks = {}
            self._next_functions_cache = self._freeze_next_functions()
            self._metadata = None
            self._name = name or type(self).__name__
            self._anomaly_trace = None
            self._anomaly_parent = None

    def save_for_backward(self, *tensors):
        saved = []
        for t in tensors:
            if t is None:
                saved.append(SavedTensor(None))
            elif hasattr(t, "_version_counter"):
                saved.append(SavedTensor(t))
            else:
                saved.append(_SavedValue(t))
        self._saved_tensors_list = saved

    def saved_tensors(self):
        if any(getattr(saved, "_released", False) for saved in self._saved_tensors_list):
            raise RuntimeError(
                "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). "
                "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). "
                "Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."
            )
        return tuple(saved.materialize() for saved in self._saved_tensors_list)

    def release_saved_tensors(self):
        for saved in self._saved_tensors_list:
            saved.release()

    def register_hook(self, hook):
        handle = _NodeHookHandle(self._hooks)
        self._hooks[handle.id] = hook
        return handle

    def register_prehook(self, hook):
        handle = _NodeHookHandle(self._prehooks)
        self._prehooks[handle.id] = hook
        return handle

    def _freeze_next_functions(self):
        next_functions = []
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
        if name == "metadata":
            if self._metadata is None:
                self._metadata = {}
            return self._metadata
        if name == "_raw_saved_tensors":
            return tuple(self._saved_tensors_list)
        if name == "_saved_tensors":
            saved_tensors = self._saved_tensors_list
            if any(getattr(saved, "_released", False) for saved in saved_tensors):
                raise RuntimeError(
                    "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). "
                    "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). "
                    "Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."
                )
            return tuple(saved.materialize() for saved in saved_tensors)
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
