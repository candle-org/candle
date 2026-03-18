import inspect
import numpy as np

from .registry import registry
from .pipeline import current_pipeline
from .keys import DispatchKey, DispatchKeySet, apply_tls_masks
from .functionalize import functionalize_op, is_functionalize_enabled, should_functionalize
import threading

from ..autograd.grad_mode import is_grad_enabled
from ..autograd import forward_ad
from ..amp.state import is_autocast_enabled
from ..amp.policy import apply_autocast_policy
from ..profiler.profiler import is_profiler_enabled, dispatch_op_enter, dispatch_op_exit


_DISPATCH_STATE = threading.local()


def _state_stack():
    stack = getattr(_DISPATCH_STATE, "stack", None)
    if stack is None:
        stack = []
        _DISPATCH_STATE.stack = stack
    return stack


def current_dispatch_keyset():
    stack = _state_stack()
    if not stack:
        return None
    return stack[-1][0]


def current_dispatch_key():
    stack = _state_stack()
    if not stack:
        return None
    return stack[-1][1]


def _push_dispatch_context(keyset, key):
    _state_stack().append((keyset, key))


def _pop_dispatch_context():
    stack = _state_stack()
    if stack:
        stack.pop()


def _accepts_device(func):
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if "device" in params:
        return True
    return False


def _prepare_kwargs(func, kwargs, device):
    if not kwargs:
        kwargs = {}
    filtered = {k: v for k, v in kwargs.items() if k != "device"}
    if "device" in kwargs and _accepts_device(func):
        return kwargs
    if _accepts_device(func):
        merged = dict(filtered)
        merged["device"] = device
        return merged
    return filtered


def _mutating_args(schema_obj, args, kwargs):
    if schema_obj is None:
        return []
    if kwargs is None:
        kwargs = {}
    params = schema_obj.params
    positional = [p for p in params if not p.kw_only]
    bound = {}
    for idx, value in enumerate(args):
        if idx < len(positional):
            bound[positional[idx].name] = value
    for key, value in kwargs.items():
        bound[key] = value
    mutated = []
    for param in params:
        if not param.mutates:
            continue
        if getattr(param, "alias_set", None) in (None, ""):
            continue
        if param.name in bound:
            mutated.append(bound[param.name])
    return mutated


def _bump_versions(schema_obj, args, kwargs):
    if schema_obj is None:
        return
    mutated = _mutating_args(schema_obj, args, kwargs)
    seen = set()
    for target in mutated:
        counter = getattr(target, "_version_counter", None)
        if counter is None:
            continue
        base = getattr(target, "_base", None)
        if base is not None:
            target = base
        if getattr(target, "device", None) is not None and target.device.type == "meta":
            continue
        counter = getattr(target, "_version_counter", None)
        if counter is None:
            continue
        key = id(counter)
        if key in seen:
            continue
        counter.bump()
        seen.add(key)


def _check_inplace_targets(schema_obj, args, kwargs):
    mutated = _mutating_args(schema_obj, args, kwargs)
    seen = set()
    for target in mutated:
        check = getattr(target, "_check_inplace", None)
        if check is None:
            continue
        key = id(target)
        if key in seen:
            continue
        check()
        seen.add(key)


class _PendingOp:
    def __init__(self, impl, args, kwargs, out, keyset, key, schema_obj=None, op_name=None):
        self.impl = impl
        self.args = args
        self.kwargs = kwargs
        self.out = out
        self.keyset = keyset
        self.key = key
        self.schema_obj = schema_obj
        self.op_name = op_name

    def _copy_result(self, pending, result):
        prev_requires_grad = pending.requires_grad
        pending._storage = result.storage()
        pending.shape = result.shape
        pending.stride = result.stride
        pending.offset = result.offset
        result_requires_grad = result.requires_grad
        pending_requires_grad = prev_requires_grad or result_requires_grad
        pending.requires_grad = pending_requires_grad

        result_grad_fn = result.grad_fn
        if result_grad_fn is not None:
            pending.grad_fn = result_grad_fn
        elif not pending_requires_grad:
            pending.grad_fn = None

        # Fast path: most outputs are non-view tensors, so avoid redundant writes.
        result_base = result._base
        result_view_meta = result._view_meta
        if result_base is not None or result_view_meta is not None:
            pending._base = result_base
            pending._view_meta = result_view_meta
        elif pending._base is not None or pending._view_meta is not None:
            pending._base = None
            pending._view_meta = None

        result_version = result._version_value
        if pending._version_value != result_version:
            pending._version_value = result_version
        pending._pending = False

    def _execute_body(self):
        result = self.impl(*self.args, **self.kwargs)
        if isinstance(self.out, (tuple, list)):
            for pending, item in zip(self.out, result):
                self._copy_result(pending, item)
        else:
            self._copy_result(self.out, result)
        _bump_versions(self.schema_obj, self.args, self.kwargs)

    def enter_dispatch_context(self):
        _push_dispatch_context(self.keyset, self.key)

    @staticmethod
    def exit_dispatch_context():
        _pop_dispatch_context()

    def execute_with_active_context(self):
        self._execute_body()

    def execute(self):
        _push_dispatch_context(self.keyset, self.key)
        try:
            self._execute_body()
        finally:
            _pop_dispatch_context()

class _FunctionalizePendingOp:
    def __init__(self, target, thunk, keyset, key, finalize=None, op_name=None, schema_obj=None, args=None, kwargs=None):
        self.target = target
        self.thunk = thunk
        self.keyset = keyset
        self.key = key
        self.finalize = finalize
        self.op_name = op_name
        self.schema_obj = schema_obj
        self.args = args
        self.kwargs = kwargs

    def _execute_body(self):
        result = self.thunk()

        if self.finalize is not None:
            self.finalize(result)
            return

        self.target._storage = result.storage()
        self.target.shape = result.shape
        self.target.stride = result.stride
        self.target.offset = result.offset
        self.target._base = result._base
        self.target._view_meta = result._view_meta
        self.target._pending = False
        target = self.target._base if self.target._base is not None else self.target
        if getattr(target, "device", None) is None or target.device.type != "meta":
            target._version_counter.bump()

    def enter_dispatch_context(self):
        _push_dispatch_context(self.keyset, self.key)

    @staticmethod
    def exit_dispatch_context():
        _pop_dispatch_context()

    def execute_with_active_context(self):
        self._execute_body()

    def execute(self):
        _push_dispatch_context(self.keyset, self.key)
        try:
            self._execute_body()
        finally:
            _pop_dispatch_context()



def _pending_tensor_from_spec(spec, device):
    from .._storage import PendingStorage
    from .._tensor import Tensor

    storage = PendingStorage(spec.shape, spec.dtype, device)
    return Tensor(storage, spec.shape, spec.stride, spec.offset)


def _pending_from_meta(meta, device):
    if isinstance(meta, (tuple, list)):
        return tuple(_pending_tensor_from_spec(spec, device) for spec in meta)
    return _pending_tensor_from_spec(meta, device)


def _kernel_for_entry(entry, key_order):
    for key in key_order:
        if key in entry.fallthrough:
            continue
        kernel = entry.kernels.get(key)
        if kernel is not None:
            return kernel, key
        global_fallthrough = getattr(registry, "_global_fallthrough", set())
        if key in global_fallthrough:
            continue
    return None, None


def _key_order(keyset):
    if isinstance(keyset, set):
        from .keys import DispatchKeySet
        keyset = DispatchKeySet(keyset)
    return list(keyset.iter_keys())


def _extract_tensors(args, kwargs):
    tensors = []

    def _visit(value):
        if hasattr(value, "device") and not isinstance(value, np.ndarray):
            tensors.append(value)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _visit(item)

    for value in list(args) + list(kwargs.values()):
        _visit(value)
    return tensors


def _validate_tensor_devices(tensors):
    if not tensors:
        return
    expected = tensors[0].device
    expected_type = expected.type if hasattr(expected, "type") else expected
    expected_index = expected.index if hasattr(expected, "index") else None
    expected_label = expected_type if expected_index is None else f"{expected_type}:{expected_index}"
    for tensor in tensors[1:]:
        device = tensor.device
        dev_type = device.type if hasattr(device, "type") else device
        dev_index = device.index if hasattr(device, "index") else None
        dev_label = dev_type if dev_index is None else f"{dev_type}:{dev_index}"
        if dev_type != expected_type or dev_index != expected_index:
            raise RuntimeError(
                f"Tensor on device {dev_label} is not on the expected device {expected_label}!"
            )


def _infer_dispatch_device(dispatch_device, tensors, keyset):
    if dispatch_device is not None:
        return dispatch_device
    for tensor in tensors:
        if hasattr(tensor, "device"):
            return tensor.device
    if not tensors:
        from .._device import get_default_device

        return get_default_device()
    if keyset.has(DispatchKey.Meta):
        return "meta"
    if keyset.has(DispatchKey.NPU):
        return "npu"
    if keyset.has(DispatchKey.PrivateUse2):
        return "mps"
    return "cpu"


def _format_dispatch_context(op_name, dispatch_device):
    device_value = getattr(dispatch_device, "type", dispatch_device)
    return f"op={op_name}, device={device_value}"


def _wrap_dispatch_error(exc, op_name, dispatch_device):
    # Keep PyTorch parity for specific ops where tests compare exact messages.
    if op_name in {"sum_to_size"} and isinstance(exc, RuntimeError):
        return exc
    context = _format_dispatch_context(op_name, dispatch_device)
    msg = f"{exc} [{context}]"
    exc_type = type(exc)
    if isinstance(exc, (ValueError, IndexError, TypeError, RuntimeError, NotImplementedError)):
        if isinstance(exc, NotImplementedError):
            return exc
        try:
            return exc_type(msg)
        except Exception:
            pass
    return RuntimeError(msg)


def _collect_torch_dispatch_types(tensors):
    """Collect tensor subclass types with custom __torch_dispatch__."""
    from .._tensor import Tensor as _BaseTensor
    base_td = getattr(_BaseTensor, "__torch_dispatch__", None)
    seen = set()
    types = []
    for tensor in tensors:
        cls = type(tensor)
        if cls in seen:
            continue
        seen.add(cls)
        td = getattr(cls, "__torch_dispatch__", None)
        if td is not None and td is not base_td:
            types.append(cls)
    # Sort by MRO depth (most derived first)
    types.sort(key=lambda c: len(c.__mro__), reverse=True)
    return types


def _dispatch_torch_dispatch(func_name, tensors, args, kwargs):
    """Invoke __torch_dispatch__ on tensor subclasses.

    Returns the result from the first subclass that returns non-NotImplemented,
    or NotImplemented if all subclasses decline.
    """
    types = _collect_torch_dispatch_types(tensors)
    if not types:
        return NotImplemented
    for cls in types:
        result = cls.__torch_dispatch__(func_name, types, args, kwargs)
        if result is not NotImplemented:
            return result
    return NotImplemented


def dispatch_with_keyset(name, keyset, dispatch_device, *args, **kwargs):
    tensors = _extract_tensors(args, kwargs)
    _validate_tensor_devices(tensors)
    keyset = apply_tls_masks(keyset)
    pipe = current_pipeline()
    dispatch_device = _infer_dispatch_device(dispatch_device, tensors, keyset)
    alias_name = name
    try:
        name = registry.resolve(name)
        entry = registry.get(name)
    except Exception as exc:
        raise _wrap_dispatch_error(exc, alias_name, dispatch_device) from exc

    if entry.schema_obj is not None:
        entry.schema_obj.bind(args, kwargs, op_name=alias_name, error_overrides=entry.error_overrides)
        if keyset.has(DispatchKey.ADInplaceOrView):
            _check_inplace_targets(entry.schema_obj, args, kwargs)

    if keyset.has(DispatchKey.Functionalize) and should_functionalize(entry):
        if pipe is not None and keyset.has(DispatchKey.Pipeline):
            pending = functionalize_op(name, alias_name, entry, keyset, args, kwargs, redispatch, pipeline=pipe, dispatch_device=dispatch_device)
            return pending
        return functionalize_op(name, alias_name, entry, keyset, args, kwargs, redispatch, dispatch_device=dispatch_device)

    def _run_kernel():
        kernel, key = _kernel_for_entry(entry, _key_order(keyset))
        if kernel is None:
            exc = RuntimeError(
                f"could not find kernel for op {name} with keys {[k.name for k in _key_order(keyset)]}"
            )
            raise _wrap_dispatch_error(exc, alias_name, dispatch_device) from exc
        impl_kwargs = _prepare_kwargs(kernel, kwargs, dispatch_device)
        token = None
        if is_profiler_enabled():
            token = dispatch_op_enter(alias_name, dispatch_device, args, impl_kwargs)
        _push_dispatch_context(keyset, key)
        try:
            result = kernel(*args, **impl_kwargs)
        except Exception as exc:
            raise _wrap_dispatch_error(exc, alias_name, dispatch_device) from exc
        finally:
            _pop_dispatch_context()
            if token is not None:
                dispatch_op_exit(token)
        _bump_versions(entry.schema_obj, args, impl_kwargs)

        level = forward_ad._current_level()
        if level >= 0:
            tangents = [forward_ad.get_tangent(t, level) for t in tensors]
            if any(t is not None for t in tangents):
                jvp = forward_ad.get_jvp(alias_name)
                if jvp is None:
                    raise RuntimeError(f"no forward-mode rule registered for op {alias_name}")
                inner_keyset = keyset.without({DispatchKey.Autograd, DispatchKey.AutogradCPU, DispatchKey.AutogradCUDA, DispatchKey.AutogradNPU, DispatchKey.AutogradMeta, DispatchKey.AutogradXPU, DispatchKey.AutogradOther})
                jvp_key = key
                if jvp_key in inner_keyset:
                    inner_keyset = inner_keyset.without(jvp_key)

                def _eval_jvp():
                    _push_dispatch_context(inner_keyset, jvp_key)
                    try:
                        return jvp(*args, **impl_kwargs, _tangents=tangents)
                    finally:
                        _pop_dispatch_context()

                try:
                    with forward_ad.temporarily_disable(level):
                        out_tangent = _eval_jvp()
                except Exception:
                    out_tangent = _eval_jvp()
                if isinstance(result, (tuple, list)):
                    for out, t in zip(result, out_tangent):
                        if hasattr(out, "_fw_set"):
                            out._fw_set(level, t)
                else:
                    if hasattr(result, "_fw_set"):
                        result._fw_set(level, out_tangent)

        return result

    # __torch_dispatch__ subclass protocol: fires after autograd, before backend
    if keyset.has(DispatchKey.Python):
        result = _dispatch_torch_dispatch(alias_name, tensors, args, kwargs)
        if result is not NotImplemented:
            return result
        # All subclasses returned NotImplemented; fall through to backend kernel

    if keyset.has(DispatchKey.Autocast):
        device_type = dispatch_device.type if hasattr(dispatch_device, "type") else dispatch_device
        if device_type is None:
            tensors = _extract_tensors(args, kwargs)
            if tensors:
                device_type = getattr(tensors[0].device, "type", None)
        casted_args, casted_kwargs = apply_autocast_policy(alias_name, args, kwargs, device_type)
        return redispatch(alias_name, keyset.without(DispatchKey.Autocast), *casted_args, **casted_kwargs)

    if pipe is not None and keyset.has(DispatchKey.Pipeline):
        if not pipe.should_defer_next():
            return _run_kernel()
        meta = entry.kernels.get(DispatchKey.Meta)
        if meta is None:
            raise RuntimeError(f"pipeline requires meta kernel for op {name}")
        meta_kwargs = _prepare_kwargs(meta, kwargs, dispatch_device)
        spec = meta(*args, **meta_kwargs)
        out = _pending_from_meta(spec, dispatch_device)
        if isinstance(out, (tuple, list)):
            for item in out:
                item._pending = True
        else:
            out._pending = True
        backend_keys = [key for key in _key_order(keyset) if key != DispatchKey.Pipeline]
        impl, impl_key = _kernel_for_entry(entry, backend_keys)
        if impl is None:
            raise RuntimeError(f"pipeline requires backend kernel for op {name}")
        impl_kwargs = _prepare_kwargs(impl, kwargs, dispatch_device)
        pipe.record(
            _PendingOp(
                impl,
                args,
                impl_kwargs,
                out,
                keyset.without(DispatchKey.Pipeline),
                impl_key,
                schema_obj=entry.schema_obj,
                op_name=alias_name,
            ),
            pending=out,
        )
        return out
    if pipe is not None and keyset.has(DispatchKey.Pipeline):
        pipe.flush()
    return _run_kernel()


def dispatch(name, dispatch_device, *args, **kwargs):
    tensors = _extract_tensors(args, kwargs)
    # Infer autocast device from tensors when dispatch_device is not specified
    autocast_device_type = getattr(dispatch_device, "type", dispatch_device)
    if autocast_device_type is None and tensors:
        autocast_device_type = getattr(tensors[0].device, "type", None)
    pipe = current_pipeline()
    keyset = DispatchKeySet.from_tensors(
        tensors,
        grad_enabled=is_grad_enabled(),
        pipeline_enabled=pipe is not None,
        functionalize_enabled=is_functionalize_enabled(),
        device=dispatch_device,
        autocast_enabled=is_autocast_enabled(autocast_device_type),
    )
    return dispatch_with_keyset(name, keyset, dispatch_device, *args, **kwargs)


# Save original Python implementation for fallback reference
_py_dispatch_with_keyset = dispatch_with_keyset


def redispatch(name, keyset, *args, **kwargs):
    return dispatch_with_keyset(name, keyset, None, *args, **kwargs)


# ---------------------------------------------------------------------------
# Cython fast-path: replace dispatch() and helpers if Cython extension is
# available.  These must come AFTER the Python definitions so they override.
# ---------------------------------------------------------------------------
try:
    from .._cython._dispatch import cy_prepare_kwargs as _prepare_kwargs  # noqa: F811
    from .._cython._dispatch import cy_extract_tensors as _extract_tensors  # noqa: F811
except ImportError:
    pass  # keep existing Python versions

# Full dispatcher core: single-function dispatch (replaces both dispatch and
# dispatch_with_keyset) — aligned with PyTorch Dispatcher::call architecture.
try:
    from .._cython._dispatcher_core import cy_dispatch_full as dispatch  # noqa: F811
    from .._cython._dispatcher_core import cy_dispatch_with_keyset_fast as dispatch_with_keyset  # noqa: F811
except ImportError:
    pass  # keep existing Python versions
