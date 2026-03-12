from .graph import current_saved_tensors_hooks


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
        self._saved_version = tensor._version_counter.value
        self._released = False
        self._hooks = current_saved_tensors_hooks()
        if self._hooks is None:
            self._packed = None
        else:
            pack, _ = self._hooks
            self._packed = pack(tensor)

    def register_hooks(self, *args):
        if len(args) != 2:
            raise TypeError("incompatible function arguments")
        pack, unpack = args
        if not callable(pack) or not callable(unpack):
            raise TypeError("incompatible function arguments")
        if self._hooks is not None:
            raise RuntimeError("SavedTensor hooks have already been set")
        self._hooks = (pack, unpack)
        self._packed = pack(self._tensor_ref)

    def release(self):
        self._released = True

    def materialize(self):
        if self._released:
            raise RuntimeError(
                "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). "
                "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). "
                "Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."
            )
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
        if self._hooks is None:
            return self._tensor_ref
        _, unpack = self._hooks
        return unpack(self._packed)


class Node:
    def __init__(self, backward, inputs):
        self.backward = backward
        self.inputs = inputs
        self._saved_tensors = []
        self._saved_fields = {}

    def save_for_backward(self, *tensors):
        saved = []
        for t in tensors:
            if hasattr(t, "_version_counter"):
                saved.append(SavedTensor(t))
            else:
                saved.append(_SavedValue(t))
        self._saved_tensors = saved

    def saved_tensors(self):
        return tuple(saved.materialize() for saved in self._saved_tensors)

    def release_saved_tensors(self):
        for saved in self._saved_tensors:
            saved.release()

    def __getattr__(self, name):
        if name.startswith(_RAW_SAVED_FIELD_PREFIX):
            key = name[len(_RAW_SAVED_FIELD_PREFIX):]
            if key in self._saved_fields:
                return self._saved_fields[key]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
