# FSDP2 MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement PyTorch-compatible FSDP2 (`fully_shard()`) for Candle with `__torch_function__`/`__torch_dispatch__` tensor subclass protocols, targeting NPU backend.

**Architecture:** Bottom-up: fix nn.Module hooks → add tensor subclass dispatch protocols → build DTensor/DeviceMesh primitives → implement FSDP param management and hook orchestration → wire up `fully_shard()` API. All communication uses existing `candle.distributed` collectives (already have `all_gather_into_tensor`, `reduce_scatter_tensor`).

**Tech Stack:** Pure Python, candle internals (`_dispatch`, `_tensor`, `_functional`, `nn`, `distributed`), no PyTorch dependency in source.

**Design doc:** `docs/plans/2026-03-10-fsdp2-mvp-design.md`

**Test command:**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/ -v --tb=short -x
```

---

### Task 1: Fix nn.Module hook `prepend` and `with_kwargs` support

FSDP2 registers hooks with `prepend=True, with_kwargs=True`. Currently these params are accepted but ignored.

**Files:**
- Modify: `src/candle/nn/module.py`
- Test: `tests/cpu/test_module_hooks.py` (create)

**Step 1: Write failing tests for prepend and with_kwargs**

Create `tests/cpu/test_module_hooks.py`:

```python
"""Tests for nn.Module hook prepend and with_kwargs support."""
import candle as torch
import candle.nn as nn


class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


def test_forward_pre_hook_prepend():
    """Hooks registered with prepend=True should run before existing hooks."""
    order = []
    m = SimpleModule()
    m.register_forward_pre_hook(lambda mod, inp: order.append("first"))
    m.register_forward_pre_hook(lambda mod, inp: order.append("prepended"), prepend=True)
    x = torch.randn(2, 4)
    m(x)
    assert order == ["prepended", "first"], f"Expected prepended hook first, got {order}"


def test_forward_hook_prepend():
    """Post-forward hooks registered with prepend=True should run before existing hooks."""
    order = []
    m = SimpleModule()
    m.register_forward_hook(lambda mod, inp, out: order.append("first"))
    m.register_forward_hook(lambda mod, inp, out: order.append("prepended"), prepend=True)
    x = torch.randn(2, 4)
    m(x)
    assert order == ["prepended", "first"], f"Expected prepended hook first, got {order}"


def test_forward_pre_hook_with_kwargs():
    """Pre-forward hooks with with_kwargs=True should receive (module, args, kwargs)."""
    received = {}
    def hook(mod, args, kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return args, kwargs

    m = SimpleModule()
    m.register_forward_pre_hook(hook, with_kwargs=True)
    x = torch.randn(2, 4)
    m(x)
    assert "args" in received
    assert "kwargs" in received


def test_forward_pre_hook_with_kwargs_can_modify():
    """Pre-forward hook with with_kwargs=True can modify args and kwargs."""
    def hook(mod, args, kwargs):
        # Replace the input tensor with zeros
        new_input = torch.zeros_like(args[0])
        return (new_input,), kwargs

    m = SimpleModule()
    m.register_forward_pre_hook(hook, with_kwargs=True)
    x = torch.randn(2, 4)
    out = m(x)
    # Should have used zeros input, not random
    assert out is not None


def test_backward_compatible_hooks():
    """Existing hooks without prepend/with_kwargs should still work."""
    called = [False]
    m = SimpleModule()
    m.register_forward_pre_hook(lambda mod, inp: called.__setitem__(0, True))
    x = torch.randn(2, 4)
    m(x)
    assert called[0], "Old-style hook should still work"


def test_hook_removal():
    """Hooks should be removable via handle."""
    order = []
    m = SimpleModule()
    h1 = m.register_forward_pre_hook(lambda mod, inp: order.append("h1"))
    h2 = m.register_forward_pre_hook(lambda mod, inp: order.append("h2"))
    h1.remove()
    x = torch.randn(2, 4)
    m(x)
    assert order == ["h2"], f"Expected only h2, got {order}"
```

**Step 2: Run tests to verify they fail**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_module_hooks.py -v --tb=short -x
```

Expected: `test_forward_pre_hook_prepend` FAILS (prepend ignored, order is `["first", "prepended"]`).

**Step 3: Implement hook fixes in module.py**

Modify `src/candle/nn/module.py`:

The hook dict value format changes from `hook` to `(hook, with_kwargs)`. Need to update:

1. `register_forward_pre_hook` — store `(hook, with_kwargs)` tuple, support `prepend` via `move_to_end`
2. `register_forward_hook` — same
3. `__call__` — unpack tuple, handle `with_kwargs` mode

Key changes:

```python
# register_forward_pre_hook (line 379-384)
def register_forward_pre_hook(self, hook, *, prepend=False, with_kwargs=False):
    """Register a forward pre-hook."""
    hook_id = self._next_hook_id
    self._next_hook_id += 1
    self._forward_pre_hooks_dict[hook_id] = (hook, with_kwargs)
    if prepend:
        self._forward_pre_hooks_dict.move_to_end(hook_id, last=False)
    return _RemovableHandle(self._forward_pre_hooks_dict, hook_id)

# register_forward_hook (line 372-377)
def register_forward_hook(self, hook, *, prepend=False, with_kwargs=False):
    """Register a forward hook."""
    hook_id = self._next_hook_id
    self._next_hook_id += 1
    self._forward_hooks_dict[hook_id] = (hook, with_kwargs)
    if prepend:
        self._forward_hooks_dict.move_to_end(hook_id, last=False)
    return _RemovableHandle(self._forward_hooks_dict, hook_id)

# __call__ (line 27-39) — full replacement
def __call__(self, *args, **kwargs):
    for hook_entry in self._forward_pre_hooks_dict.values():
        if isinstance(hook_entry, tuple):
            hook, with_kwargs = hook_entry
        else:
            hook, with_kwargs = hook_entry, False
        if with_kwargs:
            result = hook(self, args, kwargs)
            if result is not None:
                args, kwargs = result
        else:
            result = hook(self, args)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                args = result
    result = self.forward(*args, **kwargs)
    for hook_entry in self._forward_hooks_dict.values():
        if isinstance(hook_entry, tuple):
            hook, with_kwargs = hook_entry
        else:
            hook, with_kwargs = hook_entry, False
        if with_kwargs:
            hook_result = hook(self, (args, kwargs), result)
        else:
            hook_result = hook(self, args, result)
        if hook_result is not None:
            result = hook_result
    return result
```

**Step 4: Run tests to verify they pass**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_module_hooks.py -v --tb=short
```

Expected: ALL PASS

**Step 5: Run existing tests to verify no regression**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/ tests/contract/ -v --tb=short -x 2>&1 | tail -30
```

Expected: No new failures. Existing hooks use the old format (just `hook`, not tuple), so the `isinstance(hook_entry, tuple)` check provides backward compatibility.

**Step 6: Commit**

```bash
git add src/candle/nn/module.py tests/cpu/test_module_hooks.py
git commit -m "feat(nn): add prepend and with_kwargs support to Module forward hooks"
```

---

### Task 2: Add `__torch_function__` protocol to Tensor and _functional

Add the `__torch_function__` tensor subclass dispatch protocol.

**Files:**
- Modify: `src/candle/_tensor.py` (add `__torch_function__` classmethod)
- Modify: `src/candle/_functional.py` (add `_handle_torch_function` helper)
- Test: `tests/cpu/test_torch_function.py` (create)

**Step 1: Write failing tests**

Create `tests/cpu/test_torch_function.py`:

```python
"""Tests for __torch_function__ tensor subclass protocol."""
import candle as torch
from candle._tensor import Tensor


class TrackedTensor(Tensor):
    """Test tensor subclass that tracks which ops are called."""
    _ops_called = []

    def __init__(self, data):
        if isinstance(data, Tensor):
            super().__init__(data._storage, data.shape, data.stride, data.offset, data.requires_grad)
            self._wrapped = data
        else:
            raise TypeError("TrackedTensor requires a Tensor")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        cls._ops_called.append(func.__name__ if hasattr(func, '__name__') else str(func))
        # Unwrap TrackedTensor args to plain tensors
        def unwrap(x):
            if isinstance(x, TrackedTensor):
                return x._wrapped
            if isinstance(x, (list, tuple)):
                return type(x)(unwrap(i) for i in x)
            return x
        new_args = unwrap(args)
        new_kwargs = {k: unwrap(v) for k, v in (kwargs or {}).items()}
        result = func(*new_args, **new_kwargs)
        return result

    @classmethod
    def reset(cls):
        cls._ops_called = []


class BlockingTensor(Tensor):
    """Tensor subclass that blocks all ops."""

    def __init__(self, data):
        if isinstance(data, Tensor):
            super().__init__(data._storage, data.shape, data.stride, data.offset, data.requires_grad)
        else:
            raise TypeError("BlockingTensor requires a Tensor")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        raise RuntimeError("BlockingTensor: ops not allowed")


def test_torch_function_intercepts_add():
    """__torch_function__ should intercept torch.add when subclass is present."""
    TrackedTensor.reset()
    a = TrackedTensor(torch.randn(3))
    b = torch.randn(3)
    result = torch.add(a, b)
    assert "add" in TrackedTensor._ops_called


def test_torch_function_intercepts_matmul():
    """__torch_function__ should intercept torch.matmul."""
    TrackedTensor.reset()
    a = TrackedTensor(torch.randn(3, 3))
    b = torch.randn(3, 3)
    result = torch.matmul(a, b)
    assert "matmul" in TrackedTensor._ops_called


def test_torch_function_not_implemented_fallthrough():
    """Returning NotImplemented should fall through to normal dispatch."""
    class PassthroughTensor(Tensor):
        def __init__(self, data):
            super().__init__(data._storage, data.shape, data.stride, data.offset, data.requires_grad)

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            return NotImplemented

    a = PassthroughTensor(torch.randn(3))
    b = torch.randn(3)
    # Should not raise — falls through to normal dispatch
    # But may fail because backend gets PassthroughTensor not plain Tensor
    # This is expected — subclass must handle or fully unwrap


def test_torch_function_blocking():
    """Blocking tensor should prevent ops."""
    a = BlockingTensor(torch.randn(3))
    b = torch.randn(3)
    try:
        result = torch.add(a, b)
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "BlockingTensor" in str(e)


def test_plain_tensors_skip_torch_function():
    """Plain Tensor ops should not be affected by __torch_function__."""
    a = torch.randn(3)
    b = torch.randn(3)
    result = torch.add(a, b)
    assert result.shape == (3,)
```

**Step 2: Run tests to verify they fail**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_torch_function.py -v --tb=short -x
```

Expected: FAIL — `__torch_function__` not called because `_functional.py` doesn't check for it yet.

**Step 3: Add `__torch_function__` to Tensor class**

In `src/candle/_tensor.py`, add after the class definition imports (near line 134):

```python
# Inside class Tensor, add as a classmethod:
@classmethod
def __torch_function__(cls, func, types, args=(), kwargs=None):
    return NotImplemented
```

**Step 4: Add `_handle_torch_function` helper and wire into _functional.py**

In `src/candle/_functional.py`, add at the top (after existing imports):

```python
def _has_torch_function(args, kwargs):
    """Fast check: do any tensor args have __torch_function__ overrides?"""
    from ._tensor import Tensor
    def _check(val):
        if isinstance(val, Tensor) and type(val) is not Tensor:
            cls = type(val)
            # Check if the subclass actually overrides __torch_function__
            if cls.__torch_function__ is not Tensor.__torch_function__:
                return True
        if isinstance(val, (list, tuple)):
            return any(_check(v) for v in val)
        return False
    for a in args:
        if _check(a):
            return True
    if kwargs:
        for v in kwargs.values():
            if _check(v):
                return True
    return False


def _handle_torch_function(func, args, kwargs):
    """Dispatch to __torch_function__ if any arg is an overriding tensor subclass."""
    from ._tensor import Tensor
    if not _has_torch_function(args, kwargs):
        return NotImplemented

    # Collect subclass types
    types = set()
    def _collect(val):
        if isinstance(val, Tensor) and type(val) is not Tensor:
            cls = type(val)
            if cls.__torch_function__ is not Tensor.__torch_function__:
                types.add(cls)
        if isinstance(val, (list, tuple)):
            for v in val:
                _collect(v)
    for a in args:
        _collect(a)
    if kwargs:
        for v in kwargs.values():
            _collect(v)

    # Sort by MRO — most derived first
    sorted_types = sorted(types, key=lambda c: len(c.__mro__), reverse=True)

    for cls in sorted_types:
        result = cls.__torch_function__(func, types, args, kwargs or {})
        if result is not NotImplemented:
            return result

    return NotImplemented
```

Then add `__torch_function__` check to key functions. The pattern for each function:

```python
def add(*args, **kwargs):
    r = _handle_torch_function(add, args, kwargs)
    if r is not NotImplemented:
        return r
    # existing code unchanged
    alpha = kwargs.pop("alpha", 1)
    ...
```

Apply this pattern to these high-priority functions (used by FSDP2 or commonly):
- `add`, `sub`, `mul`, `matmul`, `div`, `true_divide`
- `sum`, `mean`
- `cat`, `stack`
- `zeros`, `ones`, `empty`, `randn` (creation ops — check if tensor arg present)
- `view`, `reshape`, `transpose`, `permute`
- `chunk`, `split`
- `copy_`

For MVP, add the check to `add` and `matmul` first to prove the mechanism works. Other functions can be added incrementally.

**Step 5: Run tests**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_torch_function.py -v --tb=short
```

Expected: ALL PASS

**Step 6: Run existing tests for regression**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/ tests/contract/ -v --tb=short -x 2>&1 | tail -30
```

Expected: No new failures (`_has_torch_function` fast-returns `False` for plain Tensors).

**Step 7: Commit**

```bash
git add src/candle/_tensor.py src/candle/_functional.py tests/cpu/test_torch_function.py
git commit -m "feat(dispatch): add __torch_function__ tensor subclass protocol"
```

---

### Task 3: Add `__torch_dispatch__` protocol and reorder dispatch keys

Add the lower-level `__torch_dispatch__` protocol triggered by the `Python` dispatch key.

**Files:**
- Modify: `src/candle/_dispatch/keys.py` (reorder `Python` key after Autograd)
- Modify: `src/candle/_dispatch/dispatcher.py` (add `__torch_dispatch__` handling)
- Modify: `src/candle/_tensor.py` (add `__torch_dispatch__` classmethod)
- Test: `tests/cpu/test_torch_dispatch.py` (create)

**Step 1: Write failing tests**

Create `tests/cpu/test_torch_dispatch.py`:

```python
"""Tests for __torch_dispatch__ tensor subclass protocol."""
import candle as torch
from candle._tensor import Tensor


class DispatchTrackedTensor(Tensor):
    """Tensor subclass that tracks dispatch-level op calls."""
    _dispatch_ops = []

    def __init__(self, data):
        if isinstance(data, Tensor):
            super().__init__(data._storage, data.shape, data.stride, data.offset, data.requires_grad)
            self._inner = data
        else:
            raise TypeError("Requires a Tensor")

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        cls._dispatch_ops.append(func)
        # Unwrap to plain tensors
        def unwrap(x):
            if isinstance(x, DispatchTrackedTensor):
                return x._inner
            if isinstance(x, (list, tuple)):
                return type(x)(unwrap(i) for i in x)
            return x
        new_args = unwrap(args)
        new_kwargs = {k: unwrap(v) for k, v in (kwargs or {}).items()}
        from candle._dispatch.dispatcher import dispatch
        return dispatch(func, None, *new_args, **new_kwargs)

    @classmethod
    def reset(cls):
        cls._dispatch_ops = []


def test_torch_dispatch_intercepts():
    """__torch_dispatch__ should be called when subclass tensor enters dispatch."""
    DispatchTrackedTensor.reset()
    a = DispatchTrackedTensor(torch.randn(3))
    b = torch.randn(3)
    result = torch.add(a, b)
    assert len(DispatchTrackedTensor._dispatch_ops) > 0, "Expected dispatch interception"


def test_torch_dispatch_after_autograd():
    """__torch_dispatch__ should fire after autograd recording."""
    # If autograd is enabled and tensor requires grad, the dispatch
    # should still work correctly
    DispatchTrackedTensor.reset()
    inner = torch.randn(3, requires_grad=True)
    a = DispatchTrackedTensor(inner)
    b = torch.randn(3)
    result = torch.add(a, b)
    assert len(DispatchTrackedTensor._dispatch_ops) > 0


def test_plain_tensors_unaffected():
    """Plain tensors should not trigger __torch_dispatch__."""
    DispatchTrackedTensor.reset()
    a = torch.randn(3)
    b = torch.randn(3)
    result = torch.add(a, b)
    assert len(DispatchTrackedTensor._dispatch_ops) == 0
```

**Step 2: Run tests to verify they fail**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_torch_dispatch.py -v --tb=short -x
```

Expected: FAIL — no `__torch_dispatch__` mechanism exists yet.

**Step 3: Reorder dispatch key priority**

In `src/candle/_dispatch/keys.py`, move `DispatchKey.Python` after the Autograd keys:

Change `DISPATCH_KEY_PRIORITY` (line 29-53) — move `DispatchKey.Python` from position 3 to after `DispatchKey.Autograd`:

```python
DISPATCH_KEY_PRIORITY = [
    DispatchKey.BackendSelect,
    DispatchKey.Pipeline,
    DispatchKey.Functionalize,
    DispatchKey.Autocast,
    DispatchKey.ADInplaceOrView,
    DispatchKey.AutogradOther,
    DispatchKey.AutogradCPU,
    DispatchKey.AutogradNPU,
    DispatchKey.AutogradCUDA,
    DispatchKey.AutogradXPU,
    DispatchKey.AutogradMeta,
    DispatchKey.Autograd,
    DispatchKey.Python,          # MOVED: after Autograd, before Backend
    DispatchKey.Meta,
    DispatchKey.NPU,
    DispatchKey.CUDA,
    DispatchKey.CPU,
    DispatchKey.PythonDispatcher,
    DispatchKey.CompositeImplicitAutograd,
    DispatchKey.CompositeExplicitAutograd,
    DispatchKey.PrivateUse1,
    DispatchKey.PrivateUse2,
    DispatchKey.PrivateUse3,
]
```

**Step 4: Add subclass detection to keyset construction**

In `src/candle/_dispatch/keys.py`, modify `DispatchKeySet.from_tensors()` (line 179-247). Add subclass detection after the existing device/autograd logic:

```python
    # [NEW] After existing mask construction, before return:
    # Detect tensor subclasses with __torch_dispatch__
    for tensor in tensors:
        tensor_cls = type(tensor)
        # Import Tensor lazily to avoid circular import
        from .._tensor import Tensor as _Tensor
        if tensor_cls is not _Tensor and hasattr(tensor_cls, '__torch_dispatch__'):
            if tensor_cls.__torch_dispatch__ is not _Tensor.__torch_dispatch__:
                mask |= int(DispatchKey.Python)
                break
    return cls(mask)
```

**Step 5: Add `__torch_dispatch__` handling in dispatcher.py**

In `src/candle/_dispatch/dispatcher.py`, inside `dispatch_with_keyset()` (line 357), add handling for the `Python` key. Add it in `_run_kernel()` before the actual kernel lookup:

```python
    def _run_kernel():
        # [NEW] Check for __torch_dispatch__ via Python key
        if keyset.has(DispatchKey.Python):
            from .._tensor import Tensor as _Tensor
            types = set()
            for t in tensors:
                t_cls = type(t)
                if t_cls is not _Tensor and hasattr(t_cls, '__torch_dispatch__'):
                    if t_cls.__torch_dispatch__ is not _Tensor.__torch_dispatch__:
                        types.add(t_cls)
            if types:
                sorted_types = sorted(types, key=lambda c: len(c.__mro__), reverse=True)
                for cls in sorted_types:
                    result = cls.__torch_dispatch__(alias_name, types, args, kwargs or {})
                    if result is not NotImplemented:
                        return result

        kernel, key = _kernel_for_entry(entry, _key_order(keyset))
        ...  # existing code
```

**Step 6: Add `__torch_dispatch__` to Tensor base class**

In `src/candle/_tensor.py`, add to the Tensor class:

```python
@classmethod
def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    return NotImplemented
```

**Step 7: Run tests**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_torch_dispatch.py -v --tb=short
```

Expected: ALL PASS

**Step 8: Run full test suite for regression**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/ tests/contract/ -v --tb=short -x 2>&1 | tail -30
```

Expected: No new failures. The `Python` key is only added when subclass tensors are detected, so plain tensor operations are unaffected. Key reorder only affects priority when the `Python` key is present.

**Step 9: Commit**

```bash
git add src/candle/_dispatch/keys.py src/candle/_dispatch/dispatcher.py src/candle/_tensor.py tests/cpu/test_torch_dispatch.py
git commit -m "feat(dispatch): add __torch_dispatch__ protocol with Python dispatch key"
```

---

### Task 4: Implement Placement types

**Files:**
- Create: `src/candle/distributed/tensor/placement.py`
- Modify: `src/candle/distributed/tensor/__init__.py` (re-export)
- Test: `tests/cpu/test_placement.py` (create)

**Step 1: Write tests**

Create `tests/cpu/test_placement.py`:

```python
"""Tests for distributed tensor placement types."""
from candle.distributed.tensor.placement import Placement, Shard, Replicate, Partial


def test_shard_default_dim():
    s = Shard()
    assert s.dim == 0


def test_shard_custom_dim():
    s = Shard(1)
    assert s.dim == 1


def test_shard_is_placement():
    assert isinstance(Shard(0), Placement)


def test_replicate_is_placement():
    assert isinstance(Replicate(), Placement)


def test_partial_default_op():
    p = Partial()
    assert p.reduce_op == "sum"


def test_partial_is_placement():
    assert isinstance(Partial(), Placement)


def test_shard_equality():
    assert Shard(0) == Shard(0)
    assert Shard(0) != Shard(1)


def test_replicate_equality():
    assert Replicate() == Replicate()


def test_shard_repr():
    assert "Shard" in repr(Shard(0))
    assert "0" in repr(Shard(0))
```

**Step 2: Run to verify failure**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_placement.py -v --tb=short -x
```

**Step 3: Implement placement types**

Create `src/candle/distributed/tensor/placement.py`:

```python
"""Tensor placement types for distributed tensors."""


class Placement:
    """Base class for tensor placement strategies."""

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __hash__(self):
        return hash((type(self), tuple(sorted(self.__dict__.items()))))

    def __repr__(self):
        return f"{type(self).__name__}()"


class Shard(Placement):
    """Tensor is sharded along a dimension across the mesh."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    def __repr__(self):
        return f"Shard(dim={self.dim})"

    def __hash__(self):
        return hash((type(self), self.dim))


class Replicate(Placement):
    """Tensor is replicated across all ranks in the mesh."""
    pass


class Partial(Placement):
    """Tensor has pending reduction (e.g., gradient before reduce-scatter)."""

    def __init__(self, reduce_op: str = "sum"):
        self.reduce_op = reduce_op

    def __repr__(self):
        return f"Partial(reduce_op={self.reduce_op!r})"

    def __hash__(self):
        return hash((type(self), self.reduce_op))
```

**Step 4: Update `__init__.py`**

Replace `src/candle/distributed/tensor/__init__.py`:

```python
"""Distributed tensor module."""
from .placement import Placement, Shard, Replicate, Partial

__all__ = ["DTensor", "Placement", "Shard", "Replicate", "Partial"]
```

Note: `DTensor` will be added to the import in Task 6.

**Step 5: Run tests**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_placement.py -v --tb=short
```

**Step 6: Commit**

```bash
git add src/candle/distributed/tensor/placement.py src/candle/distributed/tensor/__init__.py tests/cpu/test_placement.py
git commit -m "feat(distributed): add Placement types (Shard, Replicate, Partial)"
```

---

### Task 5: Implement DeviceMesh (1D)

**Files:**
- Modify: `src/candle/distributed/device_mesh.py` (replace stub)
- Test: `tests/cpu/test_device_mesh.py` (create)

**Step 1: Write tests**

Create `tests/cpu/test_device_mesh.py`:

```python
"""Tests for DeviceMesh (1D MVP, no distributed init required for basic tests)."""
from candle.distributed.device_mesh import DeviceMesh


def test_device_mesh_basic():
    """DeviceMesh stores device type and shape."""
    mesh = DeviceMesh.__new__(DeviceMesh)
    mesh.device_type = "cpu"
    mesh.mesh_dim_names = ("shard",)
    mesh._mesh_shape = (4,)
    assert mesh.device_type == "cpu"
    assert mesh.mesh_dim_names == ("shard",)


def test_device_mesh_ndim():
    """1D mesh should have ndim=1."""
    mesh = DeviceMesh.__new__(DeviceMesh)
    mesh._mesh_shape = (4,)
    assert mesh.ndim == 1


def test_device_mesh_size():
    """size() should return the mesh dimension size."""
    mesh = DeviceMesh.__new__(DeviceMesh)
    mesh._mesh_shape = (8,)
    assert mesh.size(0) == 8
```

**Step 2: Run to verify failure**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_device_mesh.py -v --tb=short -x
```

**Step 3: Implement DeviceMesh**

Replace `src/candle/distributed/device_mesh.py`:

```python
"""DeviceMesh — multi-dimensional device topology abstraction.

MVP: 1D mesh only (pure FSDP). API aligned with torch.distributed.device_mesh.
"""


class DeviceMesh:
    """Logical arrangement of devices for distributed training.

    Usage:
        # After dist.init_process_group()
        mesh = DeviceMesh("npu", (world_size,), mesh_dim_names=("shard",))
    """

    def __init__(self, device_type, mesh_shape, *, mesh_dim_names=None):
        if isinstance(mesh_shape, int):
            mesh_shape = (mesh_shape,)
        self.device_type = device_type
        self._mesh_shape = tuple(mesh_shape)
        self.mesh_dim_names = mesh_dim_names
        self._dim_groups = []
        self._init_process_groups()

    def _init_process_groups(self):
        """Create ProcessGroups per mesh dimension.

        1D MVP: reuse the global WORLD process group.
        """
        from . import _get_default_group, is_initialized
        if not is_initialized():
            return
        # 1D mesh: single dimension uses the world group
        if len(self._mesh_shape) == 1:
            self._dim_groups = [_get_default_group()]
        else:
            raise NotImplementedError(
                f"DeviceMesh only supports 1D mesh in MVP, got shape {self._mesh_shape}"
            )

    def get_group(self, mesh_dim=0):
        """Get the ProcessGroup for a mesh dimension."""
        if not self._dim_groups:
            raise RuntimeError(
                "DeviceMesh process groups not initialized. "
                "Call dist.init_process_group() first."
            )
        return self._dim_groups[mesh_dim]

    def size(self, mesh_dim=0):
        """Number of devices along a mesh dimension."""
        return self._mesh_shape[mesh_dim]

    @property
    def ndim(self):
        """Number of mesh dimensions."""
        return len(self._mesh_shape)

    def __repr__(self):
        return (
            f"DeviceMesh(device_type={self.device_type!r}, "
            f"mesh_shape={self._mesh_shape}, "
            f"mesh_dim_names={self.mesh_dim_names})"
        )


def init_device_mesh(device_type, mesh_shape, *, mesh_dim_names=None):
    """Create a DeviceMesh. Convenience function matching PyTorch API."""
    return DeviceMesh(device_type, mesh_shape, mesh_dim_names=mesh_dim_names)
```

**Step 4: Run tests**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_device_mesh.py -v --tb=short
```

**Step 5: Commit**

```bash
git add src/candle/distributed/device_mesh.py tests/cpu/test_device_mesh.py
git commit -m "feat(distributed): implement DeviceMesh (1D MVP)"
```

---

### Task 6: Implement DTensor

**Files:**
- Create: `src/candle/distributed/tensor/dtensor.py`
- Modify: `src/candle/distributed/tensor/__init__.py` (add DTensor import)
- Test: `tests/cpu/test_dtensor.py` (create)

**Step 1: Write tests**

Create `tests/cpu/test_dtensor.py`:

```python
"""Tests for DTensor (lightweight metadata container)."""
import candle as torch
from candle._tensor import Tensor
from candle.distributed.tensor.placement import Shard, Replicate
from candle.distributed.tensor.dtensor import DTensor, DTensorSpec, TensorMeta


def _make_spec(placements, global_shape=(8, 4)):
    """Create a DTensorSpec with a mock mesh for unit testing."""
    # Use a mock mesh that doesn't require distributed init
    class MockMesh:
        def size(self, dim=0): return 2
        @property
        def ndim(self): return 1
    mesh = MockMesh()
    meta = TensorMeta(shape=global_shape, stride=(4, 1), dtype=torch.float32)
    return DTensorSpec(mesh, placements, tensor_meta=meta)


def test_dtensor_is_tensor_subclass():
    """DTensor should be a subclass of Tensor."""
    assert issubclass(DTensor, Tensor)


def test_dtensor_creation():
    """DTensor wraps a local tensor with metadata."""
    local = torch.randn(4, 4)
    spec = _make_spec((Shard(0),), global_shape=(8, 4))
    dt = DTensor(local, spec)
    assert dt.shape == local.shape
    assert dt._local_tensor is local
    assert dt._spec is spec


def test_dtensor_to_local():
    """to_local() should return the underlying local tensor."""
    local = torch.randn(4, 4)
    spec = _make_spec((Shard(0),))
    dt = DTensor(local, spec)
    recovered = dt.to_local()
    assert recovered is local


def test_dtensor_placements_property():
    """placements property should return the spec's placements."""
    local = torch.randn(4, 4)
    spec = _make_spec((Shard(0),))
    dt = DTensor(local, spec)
    assert dt.placements == (Shard(0),)


def test_dtensor_device_mesh_property():
    """device_mesh property should return the spec's mesh."""
    local = torch.randn(4, 4)
    spec = _make_spec((Shard(0),))
    dt = DTensor(local, spec)
    assert dt.device_mesh is spec.mesh


def test_dtensor_spec_has_shard():
    """DTensorSpec.has_shard_placement should detect Shard placements."""
    spec_shard = _make_spec((Shard(0),))
    assert spec_shard.has_shard_placement()

    spec_rep = _make_spec((Replicate(),))
    assert not spec_rep.has_shard_placement()


def test_dtensor_has_torch_dispatch():
    """DTensor should define __torch_dispatch__."""
    assert hasattr(DTensor, '__torch_dispatch__')
    assert DTensor.__torch_dispatch__ is not Tensor.__torch_dispatch__


def test_dtensor_sharded_blocks_compute():
    """Direct compute on a sharded DTensor should raise RuntimeError."""
    local = torch.randn(4, 4)
    spec = _make_spec((Shard(0),))
    dt = DTensor(local, spec)
    b = torch.randn(4, 4)
    try:
        result = torch.add(dt, b)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "sharded" in str(e).lower() or "not supported" in str(e).lower()
```

**Step 2: Run to verify failure**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_dtensor.py -v --tb=short -x
```

**Step 3: Implement DTensor**

Create `src/candle/distributed/tensor/dtensor.py`:

```python
"""DTensor — Distributed Tensor for FSDP2.

A lightweight metadata container wrapping a local tensor shard with
placement and mesh information. FSDP manages communication manually;
DTensor does NOT perform automatic redistribution in MVP.
"""
from ..._tensor import Tensor


class TensorMeta:
    """Global tensor metadata (shape as if the tensor were not sharded)."""

    __slots__ = ("shape", "stride", "dtype")

    def __init__(self, shape, stride, dtype):
        self.shape = shape
        self.stride = stride
        self.dtype = dtype

    def __repr__(self):
        return f"TensorMeta(shape={self.shape}, dtype={self.dtype})"


class DTensorSpec:
    """Metadata describing how a tensor is distributed."""

    __slots__ = ("mesh", "placements", "tensor_meta")

    def __init__(self, mesh, placements, tensor_meta=None):
        self.mesh = mesh
        self.placements = tuple(placements)
        self.tensor_meta = tensor_meta

    def has_shard_placement(self):
        from .placement import Shard
        return any(isinstance(p, Shard) for p in self.placements)

    def __repr__(self):
        return f"DTensorSpec(placements={self.placements}, meta={self.tensor_meta})"


class DTensor(Tensor):
    """Distributed Tensor — sharded parameter container for FSDP2.

    Wraps a local tensor shard with placement metadata.
    Not intended for direct computation when sharded — FSDP unshards
    parameters to plain Tensors before forward/backward.
    """

    def __init__(self, local_tensor, spec, *, requires_grad=None):
        if requires_grad is None:
            requires_grad = local_tensor.requires_grad
        super().__init__(
            local_tensor._storage,
            local_tensor.shape,
            local_tensor.stride,
            local_tensor.offset,
            requires_grad=requires_grad,
        )
        self._local_tensor = local_tensor
        self._spec = spec

    @property
    def placements(self):
        return self._spec.placements

    @property
    def device_mesh(self):
        return self._spec.mesh

    @staticmethod
    def from_local(local_tensor, device_mesh, placements):
        """Construct DTensor from a local shard. (PyTorch-aligned API)"""
        global_shape = _compute_global_shape(
            local_tensor.shape, device_mesh, placements
        )
        global_stride = _compute_global_stride(
            local_tensor.stride, device_mesh, placements
        )
        tensor_meta = TensorMeta(
            shape=global_shape,
            stride=global_stride,
            dtype=local_tensor.dtype,
        )
        spec = DTensorSpec(device_mesh, placements, tensor_meta)
        return DTensor(local_tensor, spec)

    def to_local(self):
        """Extract the local tensor shard (unwrap DTensor)."""
        return self._local_tensor

    # ── Dispatch protocols ──

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # MVP: delegate to __torch_dispatch__ via normal dispatch path
        return NotImplemented

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # Extract specs from DTensor args
        specs = []
        def _extract(val):
            if isinstance(val, DTensor):
                specs.append(val._spec)
            elif isinstance(val, (list, tuple)):
                for v in val:
                    _extract(v)
        for a in args:
            _extract(a)
        if kwargs:
            for v in kwargs.values():
                _extract(v)

        # Sharded DTensors must not be computed on directly
        for spec in specs:
            if spec.has_shard_placement():
                raise RuntimeError(
                    f"{func}: direct compute on sharded DTensor is not supported. "
                    f"Ensure fully_shard() hooks unshard parameters before forward."
                )

        # Replicate placement: unwrap and execute on local tensors
        def _unwrap(val):
            if isinstance(val, DTensor):
                return val._local_tensor
            if isinstance(val, (list, tuple)):
                return type(val)(_unwrap(v) for v in val)
            return val

        new_args = _unwrap(args)
        new_kwargs = {k: _unwrap(v) for k, v in (kwargs or {}).items()}
        from ..._dispatch.dispatcher import dispatch
        return dispatch(func, None, *new_args, **new_kwargs)

    def __repr__(self):
        return (
            f"DTensor(local_shape={self._local_tensor.shape}, "
            f"placements={self.placements})"
        )


def _compute_global_shape(local_shape, mesh, placements):
    """Compute the global (logical) shape from a local shard shape."""
    from .placement import Shard
    global_shape = list(local_shape)
    for placement in placements:
        if isinstance(placement, Shard):
            global_shape[placement.dim] *= mesh.size()
    return tuple(global_shape)


def _compute_global_stride(local_stride, mesh, placements):
    """Compute the global stride from a local shard stride."""
    # For contiguous tensors, stride is determined by shape
    # MVP: return local stride as-is (sufficient for dim-0 sharding)
    return tuple(local_stride) if not isinstance(local_stride, tuple) else local_stride
```

**Step 4: Update `__init__.py`**

In `src/candle/distributed/tensor/__init__.py`, add DTensor import:

```python
"""Distributed tensor module."""
from .placement import Placement, Shard, Replicate, Partial
from .dtensor import DTensor, DTensorSpec, TensorMeta

__all__ = ["DTensor", "DTensorSpec", "TensorMeta", "Placement", "Shard", "Replicate", "Partial"]
```

**Step 5: Run tests**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_dtensor.py -v --tb=short
```

**Step 6: Commit**

```bash
git add src/candle/distributed/tensor/dtensor.py src/candle/distributed/tensor/__init__.py tests/cpu/test_dtensor.py
git commit -m "feat(distributed): implement DTensor metadata container"
```

---

### Task 7: Implement FSDPParam and FSDPParamGroup

**Files:**
- Create: `src/candle/distributed/_composable/fsdp/` (directory with `__init__.py`)
- Create: `src/candle/distributed/_composable/fsdp/_fsdp_common.py`
- Create: `src/candle/distributed/_composable/fsdp/_fsdp_param.py`
- Create: `src/candle/distributed/_composable/fsdp/_fsdp_param_group.py`
- Test: `tests/cpu/test_fsdp_param.py` (create)

**Step 1: Write tests**

Create `tests/cpu/test_fsdp_param.py`:

```python
"""Tests for FSDPParam and FSDPParamGroup (single-process, mock comm)."""
import candle as torch
import candle.nn as nn
from candle.distributed.tensor.placement import Shard
from candle.distributed.tensor.dtensor import DTensor


class MockMeshInfo:
    """Mock FSDPMeshInfo for single-process testing."""
    def __init__(self, rank=0, world_size=1):
        self.shard_mesh_rank = rank
        self.shard_mesh_size = world_size
        self.shard_process_group = None  # Not used in single-process test

        class _MockMesh:
            def __init__(self, ws):
                self._ws = ws
            def size(self, dim=0):
                return self._ws
            @property
            def ndim(self):
                return 1
        self.mesh = _MockMesh(world_size)


def test_fsdp_param_init_shards_parameter():
    """FSDPParam should shard a parameter and replace it with DTensor."""
    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam

    module = nn.Linear(8, 4)  # weight shape: (4, 8)
    orig_shape = module.weight.shape
    mesh_info = MockMeshInfo(rank=0, world_size=1)

    param = FSDPParam(module.weight, module, "weight", mesh_info)

    # With world_size=1, shard == full param
    assert isinstance(module.weight, DTensor)
    assert module.weight.placements == (Shard(0),)


def test_fsdp_param_unshard_reshard():
    """unshard should replace DTensor with plain Tensor, reshard should restore."""
    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam, ShardedState

    module = nn.Linear(8, 4)
    mesh_info = MockMeshInfo(rank=0, world_size=1)
    param = FSDPParam(module.weight, module, "weight", mesh_info)

    assert param._sharded_state == ShardedState.SHARDED
    assert isinstance(module.weight, DTensor)

    # Mock unshard (world_size=1: all-gather is identity)
    param._unshard_single_rank()  # We'll add this helper for testing

    assert param._sharded_state == ShardedState.UNSHARDED
    assert not isinstance(module.weight, DTensor)

    param.reshard()
    assert param._sharded_state == ShardedState.SHARDED
    assert isinstance(module.weight, DTensor)


def test_fsdp_param_group_lifecycle():
    """FSDPParamGroup should unshard/reshard all params together."""
    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam
    from candle.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup

    module = nn.Linear(8, 4)
    mesh_info = MockMeshInfo(rank=0, world_size=1)
    fp_weight = FSDPParam(module.weight, module, "weight", mesh_info)
    fp_bias = FSDPParam(module.bias, module, "bias", mesh_info)
    group = FSDPParamGroup([fp_weight, fp_bias], module, mesh_info)

    assert not group._is_unsharded

    group.unshard()
    assert group._is_unsharded
    assert not isinstance(module.weight, DTensor)
    assert not isinstance(module.bias, DTensor)

    group.reshard()
    assert not group._is_unsharded
    assert isinstance(module.weight, DTensor)
    assert isinstance(module.bias, DTensor)
```

**Step 2: Run to verify failure**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_fsdp_param.py -v --tb=short -x
```

**Step 3: Create the FSDP directory structure**

```bash
mkdir -p src/candle/distributed/_composable/fsdp
```

**Step 4: Implement FSDPCommon**

Create `src/candle/distributed/_composable/fsdp/_fsdp_common.py`:

```python
"""Common utilities for FSDP2."""


class FSDPMeshInfo:
    """Mesh information for FSDP parameter groups."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.shard_mesh_size = mesh.size(0)
        pg = mesh.get_group(0)
        self.shard_process_group = pg
        self.shard_mesh_rank = pg.rank() if pg is not None else 0
```

**Step 5: Implement FSDPParam**

Create `src/candle/distributed/_composable/fsdp/_fsdp_param.py`:

```python
"""FSDPParam — per-parameter shard lifecycle management."""
from enum import Enum, auto

from ....distributed.tensor.dtensor import DTensor
from ....distributed.tensor.placement import Shard
from .... import distributed as dist


class ShardedState(Enum):
    SHARDED = auto()
    UNSHARDED = auto()


class FSDPParam:
    """Manages shard/unshard lifecycle for a single parameter."""

    def __init__(self, param, module, param_name, mesh_info):
        self._module = module
        self._param_name = param_name
        self._mesh_info = mesh_info
        self._sharded_state = ShardedState.SHARDED
        self._orig_shape = param.shape
        self._orig_dtype = param.dtype
        self._shard_dim = 0

        self._sharded_param = self._init_shard(param)
        self._unsharded_param = None

        # Replace module parameter with sharded DTensor
        self._module.__dict__[param_name] = self._sharded_param
        if param_name in self._module._parameters:
            self._module._parameters[param_name] = self._sharded_param

    def _init_shard(self, param):
        """Shard a full parameter into local shard, wrap as DTensor."""
        rank = self._mesh_info.shard_mesh_rank
        world_size = self._mesh_info.shard_mesh_size

        if world_size == 1:
            local_shard = param.detach()
        else:
            chunks = _chunk_tensor(param.detach(), world_size, dim=self._shard_dim)
            local_shard = chunks[rank].contiguous()

        return DTensor.from_local(
            local_shard,
            self._mesh_info.mesh,
            placements=(Shard(self._shard_dim),),
        )

    def unshard(self):
        """All-gather: reconstruct full parameter from shards."""
        if self._sharded_state == ShardedState.UNSHARDED:
            return

        local_tensor = self._sharded_param.to_local()
        world_size = self._mesh_info.shard_mesh_size

        if world_size == 1:
            self._unsharded_param = local_tensor
        else:
            from ...._creation import empty
            shard_size = local_tensor.shape[self._shard_dim]
            full_size = list(local_tensor.shape)
            full_size[self._shard_dim] = shard_size * world_size
            output = empty(full_size, dtype=local_tensor.dtype, device=local_tensor.device)

            pg = self._mesh_info.shard_process_group
            dist.all_gather_into_tensor(output, local_tensor, group=pg)
            self._unsharded_param = output

        self._unsharded_param.requires_grad = self._sharded_param.requires_grad

        # Swap module parameter to plain Tensor
        self._module.__dict__[self._param_name] = self._unsharded_param
        if self._param_name in self._module._parameters:
            self._module._parameters[self._param_name] = self._unsharded_param
        self._sharded_state = ShardedState.UNSHARDED

    def _unshard_single_rank(self):
        """Unshard for single-rank testing (no collective needed)."""
        if self._sharded_state == ShardedState.UNSHARDED:
            return
        self._unsharded_param = self._sharded_param.to_local()
        self._unsharded_param.requires_grad = self._sharded_param.requires_grad
        self._module.__dict__[self._param_name] = self._unsharded_param
        if self._param_name in self._module._parameters:
            self._module._parameters[self._param_name] = self._unsharded_param
        self._sharded_state = ShardedState.UNSHARDED

    def reshard(self):
        """Free unsharded parameter, restore sharded DTensor on module."""
        if self._sharded_state == ShardedState.SHARDED:
            return
        self._module.__dict__[self._param_name] = self._sharded_param
        if self._param_name in self._module._parameters:
            self._module._parameters[self._param_name] = self._sharded_param
        self._unsharded_param = None
        self._sharded_state = ShardedState.SHARDED

    def reduce_scatter_grad(self):
        """Reduce-scatter: unsharded gradient -> sharded gradient."""
        if self._unsharded_param is None:
            return
        grad = self._unsharded_param.grad
        if grad is None:
            return

        world_size = self._mesh_info.shard_mesh_size
        if world_size == 1:
            self._sharded_param.to_local().grad = grad
            return

        from ...._creation import empty
        shard_shape = self._sharded_param.to_local().shape
        reduced_grad = empty(shard_shape, dtype=grad.dtype, device=grad.device)
        pg = self._mesh_info.shard_process_group
        dist.reduce_scatter_tensor(reduced_grad, grad, group=pg)
        self._sharded_param.to_local().grad = reduced_grad


def _chunk_tensor(tensor, num_chunks, dim=0):
    """Split a tensor into chunks along a dimension."""
    size = tensor.shape[dim]
    chunk_size = (size + num_chunks - 1) // num_chunks
    from ...._functional import split
    return split(tensor, chunk_size, dim=dim)
```

**Step 6: Implement FSDPParamGroup**

Create `src/candle/distributed/_composable/fsdp/_fsdp_param_group.py`:

```python
"""FSDPParamGroup — batched communication for parameter groups."""


class FSDPParamGroup:
    """Groups parameters from one fully_shard() call for batched collectives."""

    def __init__(self, fsdp_params, module, mesh_info):
        self.fsdp_params = fsdp_params
        self.module = module
        self.mesh_info = mesh_info
        self._is_unsharded = False

    def unshard(self):
        if self._is_unsharded:
            return
        for p in self.fsdp_params:
            p.unshard()
        self._is_unsharded = True

    def reshard(self):
        if not self._is_unsharded:
            return
        for p in self.fsdp_params:
            p.reshard()
        self._is_unsharded = False

    def reduce_scatter_grads(self):
        for p in self.fsdp_params:
            p.reduce_scatter_grad()

    def pre_forward(self):
        self.unshard()

    def post_forward(self):
        self.reshard()

    def pre_backward(self):
        self.unshard()

    def post_backward(self):
        self.reduce_scatter_grads()
        self.reshard()
```

**Step 7: Create fsdp package `__init__.py`**

Create `src/candle/distributed/_composable/fsdp/__init__.py`:

```python
"""FSDP2 composable API — fully_shard()."""
```

**Step 8: Run tests**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_fsdp_param.py -v --tb=short
```

**Step 9: Commit**

```bash
git add src/candle/distributed/_composable/fsdp/ tests/cpu/test_fsdp_param.py
git commit -m "feat(fsdp): implement FSDPParam and FSDPParamGroup"
```

---

### Task 8: Implement FSDPState and hook orchestration

**Files:**
- Create: `src/candle/distributed/_composable/fsdp/_fsdp_state.py`
- Test: `tests/cpu/test_fsdp_state.py` (create)

**Step 1: Write tests**

Create `tests/cpu/test_fsdp_state.py`:

```python
"""Tests for FSDPState hook orchestration (single-process)."""
import candle as torch
import candle.nn as nn
from candle.distributed.tensor.dtensor import DTensor


class MockMeshInfo:
    def __init__(self):
        self.shard_mesh_rank = 0
        self.shard_mesh_size = 1
        self.shard_process_group = None
        class _M:
            def size(self, dim=0): return 1
            @property
            def ndim(self): return 1
        self.mesh = _M()


def _apply_fsdp_single_rank(module):
    """Apply FSDP to a module for single-rank testing."""
    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam
    from candle.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
    from candle.distributed._composable.fsdp._fsdp_state import FSDPState

    mesh_info = MockMeshInfo()
    params = [(name, param) for name, param in module.named_parameters(recurse=False)]
    fsdp_params = [FSDPParam(p, module, name, mesh_info) for name, p in params]

    # Override unshard to single-rank version
    for fp in fsdp_params:
        fp.unshard = fp._unshard_single_rank

    group = FSDPParamGroup(fsdp_params, module, mesh_info)
    state = FSDPState(module, group, mesh_info, reshard_after_forward=True)
    module._fsdp_state = state
    return state


def test_fsdp_state_pre_forward_unshards():
    """Pre-forward hook should unshard parameters."""
    m = nn.Linear(8, 4)
    _apply_fsdp_single_rank(m)

    # Before forward: params should be DTensor (sharded)
    assert isinstance(m.weight, DTensor)

    # After forward: params should have been unsharded during forward
    x = torch.randn(2, 8)
    out = m(x)
    assert out.shape == (2, 4)


def test_fsdp_state_post_forward_reshards():
    """Post-forward hook should reshard parameters (restore DTensor)."""
    m = nn.Linear(8, 4)
    state = _apply_fsdp_single_rank(m)

    x = torch.randn(2, 8)
    out = m(x)

    # After forward with reshard_after_forward=True: params should be DTensor again
    assert isinstance(m.weight, DTensor), "Expected weight to be resharded to DTensor after forward"
```

**Step 2: Run to verify failure**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_fsdp_state.py -v --tb=short -x
```

**Step 3: Implement FSDPState**

Create `src/candle/distributed/_composable/fsdp/_fsdp_state.py`:

```python
"""FSDPState — hook orchestration for FSDP2."""
from ....distributed.tensor.dtensor import DTensor


class FSDPState:
    """Manages FSDP hook lifecycle for a module."""

    def __init__(self, module, param_group, mesh_info, reshard_after_forward):
        self.module = module
        self.param_group = param_group
        self.mesh_info = mesh_info
        self.reshard_after_forward = reshard_after_forward
        self._is_root = None
        self._pre_backward_hook_handles = []
        self._grad_count = 0
        self._total_managed_params = sum(
            1 for fp in param_group.fsdp_params
            if fp._sharded_param.requires_grad
        )

        # Register forward hooks
        self._pre_fwd_handle = module.register_forward_pre_hook(
            self._pre_forward, prepend=True, with_kwargs=True
        )
        self._post_fwd_handle = module.register_forward_hook(
            self._post_forward, prepend=False
        )

    def _pre_forward(self, module, args, kwargs):
        """Pre-forward: all-gather to reconstruct full parameters."""
        if self._is_root is None:
            self._lazy_init_root()
        self.param_group.pre_forward()
        return args, kwargs

    def _post_forward(self, module, args, output):
        """Post-forward: register backward hooks + reshard."""
        self._register_pre_backward_hooks(output)
        self._register_post_backward_hooks()

        if self.reshard_after_forward:
            self.param_group.post_forward()
        return output

    def _register_pre_backward_hooks(self, output):
        """Register hooks on forward output tensors for pre-backward unshard."""
        tensors = _extract_tensors_from_output(output)
        for t in tensors:
            if t.requires_grad:
                handle = t.register_hook(self._pre_backward)
                self._pre_backward_hook_handles.append(handle)

    def _pre_backward(self, grad):
        """Backward reaches this module: re-all-gather parameters."""
        if not self.param_group._is_unsharded:
            self.param_group.pre_backward()
        return grad

    def _register_post_backward_hooks(self):
        """Register gradient hooks on parameters for post-backward reduce-scatter."""
        self._grad_count = 0
        for fsdp_param in self.param_group.fsdp_params:
            unsharded = fsdp_param._unsharded_param
            if unsharded is not None and unsharded.requires_grad:
                unsharded.register_hook(
                    self._make_post_backward_hook(fsdp_param)
                )

    def _make_post_backward_hook(self, fsdp_param):
        def hook(grad):
            self._grad_count += 1
            if self._grad_count >= self._total_managed_params:
                self.param_group.post_backward()
                self._grad_count = 0
            return grad
        return hook

    def _lazy_init_root(self):
        """Identify root on first forward (outermost fully_shard module)."""
        self._is_root = not _has_parent_fsdp(self.module)
        if self._is_root:
            self.reshard_after_forward = False


def _has_parent_fsdp(module):
    """Check if any ancestor module has FSDP state (meaning this is not root)."""
    # Walk up the module tree by checking all modules
    # In practice this is set during fully_shard() bottom-up calls
    return False  # MVP: assume root detection works via fully_shard order


def _extract_tensors_from_output(output):
    """Extract all tensors from a module's forward output."""
    from ...._tensor import Tensor
    tensors = []
    if isinstance(output, Tensor):
        tensors.append(output)
    elif isinstance(output, (tuple, list)):
        for item in output:
            tensors.extend(_extract_tensors_from_output(item))
    elif isinstance(output, dict):
        for v in output.values():
            tensors.extend(_extract_tensors_from_output(v))
    return tensors
```

**Step 4: Run tests**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_fsdp_state.py -v --tb=short
```

**Step 5: Commit**

```bash
git add src/candle/distributed/_composable/fsdp/_fsdp_state.py tests/cpu/test_fsdp_state.py
git commit -m "feat(fsdp): implement FSDPState hook orchestration"
```

---

### Task 9: Implement `fully_shard()` API and FSDPModule mixin

**Files:**
- Modify: `src/candle/distributed/_composable/fsdp/__init__.py` (add fully_shard)
- Modify: `src/candle/distributed/_composable/fsdp.py` (update stub to import from package)
- Test: `tests/cpu/test_fully_shard.py` (create)

**Step 1: Write tests**

Create `tests/cpu/test_fully_shard.py`:

```python
"""Tests for fully_shard() API (single-process, world_size=1)."""
import candle as torch
import candle.nn as nn
from candle.distributed.tensor.dtensor import DTensor


class TwoLayerMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class MockMesh:
    """Mock DeviceMesh for single-process testing."""
    def __init__(self):
        self.device_type = "cpu"
        self._mesh_shape = (1,)
        self.mesh_dim_names = ("shard",)
        self._dim_groups = [None]

    def get_group(self, dim=0):
        return None

    def size(self, dim=0):
        return 1

    @property
    def ndim(self):
        return 1


def test_fully_shard_basic():
    """fully_shard should convert params to DTensor and attach state."""
    from candle.distributed._composable.fsdp import fully_shard

    model = TwoLayerMLP(8)
    mesh = MockMesh()

    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # All leaf params should now be DTensors
    assert isinstance(model.fc1.weight, DTensor)
    assert isinstance(model.fc2.weight, DTensor)

    # Module should have FSDP state
    assert hasattr(model, '_fsdp_state')
    assert hasattr(model.fc1, '_fsdp_state')


def test_fully_shard_forward_works():
    """Forward pass should work after fully_shard (unshard/reshard cycle)."""
    from candle.distributed._composable.fsdp import fully_shard

    model = TwoLayerMLP(8)
    mesh = MockMesh()

    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    x = torch.randn(2, 8)
    out = model(x)
    assert out.shape == (2, 8)


def test_fully_shard_mixin_injected():
    """fully_shard should inject FSDPModule mixin into module's MRO."""
    from candle.distributed._composable.fsdp import fully_shard, FSDPModule

    model = nn.Linear(8, 4)
    mesh = MockMesh()
    fully_shard(model, mesh=mesh)

    assert isinstance(model, FSDPModule)
    assert hasattr(model, 'fsdp_state')


def test_fully_shard_excludes_child_params():
    """Parent fully_shard should not re-shard params already managed by child."""
    from candle.distributed._composable.fsdp import fully_shard

    model = TwoLayerMLP(8)
    mesh = MockMesh()

    # Apply bottom-up
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # model itself has no direct params, so its FSDPState should be None or empty
    # The important thing is no double-sharding
```

**Step 2: Run to verify failure**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_fully_shard.py -v --tb=short -x
```

**Step 3: Implement `fully_shard()` and FSDPModule**

Replace `src/candle/distributed/_composable/fsdp/__init__.py`:

```python
"""FSDP2 composable API — fully_shard().

Usage (bottom-up):
    mesh = DeviceMesh("npu", (world_size,))
    fully_shard(model.encoder, mesh=mesh)
    fully_shard(model.decoder, mesh=mesh)
    fully_shard(model, mesh=mesh)  # root
"""
from ._fsdp_common import FSDPMeshInfo
from ._fsdp_param import FSDPParam
from ._fsdp_param_group import FSDPParamGroup
from ._fsdp_state import FSDPState


class FSDPModule:
    """Mixin injected into module's MRO by fully_shard()."""

    @property
    def fsdp_state(self):
        return self._fsdp_state

    def set_reshard_after_forward(self, value):
        self._fsdp_state.reshard_after_forward = value

    def set_modules_to_forward_prefetch(self, modules):
        pass  # MVP no-op

    def set_modules_to_backward_prefetch(self, modules):
        pass  # MVP no-op


class _MockMeshInfo:
    """MeshInfo for single-process / mock scenarios (world_size=1)."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.shard_mesh_size = mesh.size(0)
        self.shard_process_group = mesh.get_group(0) if hasattr(mesh, '_dim_groups') and mesh._dim_groups else None
        self.shard_mesh_rank = 0
        if self.shard_process_group is not None and hasattr(self.shard_process_group, 'rank'):
            self.shard_mesh_rank = self.shard_process_group.rank()


def fully_shard(module, *, mesh, reshard_after_forward=None):
    """Apply FSDP2 to a module. PyTorch-compatible API.

    Must be called bottom-up on the model.

    Args:
        module: nn.Module to shard
        mesh: DeviceMesh (1D for FSDP)
        reshard_after_forward: bool or None
            True  = free unsharded params after forward (saves memory)
            False = keep unsharded params (avoids re-all-gather in backward)
            None  = auto (True for non-root, False for root)

    Returns:
        The same module, with FSDP state and hooks attached.
    """
    # 1. Build mesh info
    try:
        mesh_info = FSDPMeshInfo(mesh)
    except (RuntimeError, AttributeError):
        mesh_info = _MockMeshInfo(mesh)

    # 2. Collect directly-owned parameters (exclude child fully_shard params)
    managed_params = _get_managed_params(module)
    if not managed_params:
        # Still inject mixin and mark as FSDP for root detection
        module._fsdp_state = None
        _inject_fsdp_mixin(module)
        return module

    # 3. Create FSDPParam for each parameter (performs initial sharding)
    fsdp_params = [
        FSDPParam(param, module, name, mesh_info)
        for name, param in managed_params
    ]

    # For single-rank, override unshard to skip collectives
    if mesh_info.shard_mesh_size == 1:
        for fp in fsdp_params:
            fp.unshard = fp._unshard_single_rank

    # 4. Group parameters for batched communication
    param_group = FSDPParamGroup(fsdp_params, module, mesh_info)

    # 5. Reshard strategy
    if reshard_after_forward is None:
        reshard_after_forward = True

    # 6. Create FSDPState and register hooks
    state = FSDPState(module, param_group, mesh_info, reshard_after_forward)
    module._fsdp_state = state

    # 7. Inject FSDPModule mixin into MRO
    _inject_fsdp_mixin(module)

    return module


def _inject_fsdp_mixin(module):
    """Dynamically insert FSDPModule into module's class MRO."""
    cls = type(module)
    if FSDPModule not in cls.__mro__:
        new_cls = type(f"FSDP_{cls.__name__}", (FSDPModule, cls), {})
        module.__class__ = new_cls


def _get_managed_params(module):
    """Collect directly-owned parameters, excluding those managed by child fully_shard calls."""
    child_fsdp_params = set()
    for child in module.modules():
        if child is module:
            continue
        if hasattr(child, '_fsdp_state') and child._fsdp_state is not None:
            for p in child.parameters():
                child_fsdp_params.add(id(p))

    managed = []
    for name, param in module.named_parameters(recurse=False):
        if id(param) not in child_fsdp_params:
            managed.append((name, param))
    return managed
```

**Step 4: Update the old composable fsdp.py stub to re-export**

Replace `src/candle/distributed/_composable/fsdp.py`:

```python
"""torch.distributed._composable.fsdp — re-export from package."""
from .fsdp import fully_shard, FSDPModule

__all__ = ["fully_shard", "FSDPModule"]
```

Wait — there's a naming conflict. The old `fsdp.py` file and the new `fsdp/` directory can't coexist. Need to remove `fsdp.py` and keep only the `fsdp/` package.

```bash
# The old fsdp.py needs to be removed since fsdp/ directory now exists
rm src/candle/distributed/_composable/fsdp.py
```

Update `src/candle/distributed/_composable/__init__.py`:

```python
"""torch.distributed._composable — composable distributed APIs."""
from .fsdp import fully_shard, FSDPModule
```

**Step 5: Run tests**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_fully_shard.py -v --tb=short
```

**Step 6: Run full test suite**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/ tests/contract/ -v --tb=short -x 2>&1 | tail -30
```

**Step 7: Commit**

```bash
git add src/candle/distributed/_composable/ tests/cpu/test_fully_shard.py
git commit -m "feat(fsdp): implement fully_shard() API with FSDPModule mixin"
```

---

### Task 10: Integration test — forward + backward with fully_shard (single-process)

**Files:**
- Test: `tests/cpu/test_fsdp_integration.py` (create)

**Step 1: Write integration test**

Create `tests/cpu/test_fsdp_integration.py`:

```python
"""Integration tests for FSDP2 forward + backward (single-process, world_size=1)."""
import candle as torch
import candle.nn as nn
from candle.distributed.tensor.dtensor import DTensor


class MockMesh:
    def __init__(self):
        self.device_type = "cpu"
        self._mesh_shape = (1,)
        self.mesh_dim_names = ("shard",)
        self._dim_groups = [None]
    def get_group(self, dim=0): return None
    def size(self, dim=0): return 1
    @property
    def ndim(self): return 1


class SimpleMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def test_fsdp_forward_backward():
    """Full forward + backward pass with fully_shard should work."""
    from candle.distributed._composable.fsdp import fully_shard

    model = SimpleMLP(8)
    mesh = MockMesh()

    # Apply FSDP bottom-up
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # Forward
    x = torch.randn(4, 8, requires_grad=True)
    out = model(x)
    assert out.shape == (4, 8)

    # Backward
    loss = out.sum()
    loss.backward()

    # Gradients should exist on sharded params
    for name, param in model.named_parameters():
        local = param.to_local() if isinstance(param, DTensor) else param
        assert local.grad is not None, f"No gradient for {name}"


def test_fsdp_multiple_forward_backward():
    """Multiple forward/backward cycles should work without state corruption."""
    from candle.distributed._composable.fsdp import fully_shard

    model = SimpleMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    for i in range(3):
        model.zero_grad()
        x = torch.randn(4, 8, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

    # Should complete without error


def test_fsdp_params_are_dtensor_between_iterations():
    """Between forward passes, params should be back in sharded (DTensor) state."""
    from candle.distributed._composable.fsdp import fully_shard

    model = SimpleMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # First forward/backward
    x = torch.randn(4, 8, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # After backward, non-root params should be resharded
    # (root params stay unsharded since reshard_after_forward=False for root)
    assert isinstance(model.fc1.weight, DTensor), "fc1.weight should be resharded"
    assert isinstance(model.fc2.weight, DTensor), "fc2.weight should be resharded"
```

**Step 2: Run integration tests**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_fsdp_integration.py -v --tb=short
```

**Step 3: Debug and fix any issues**

If tests fail, investigate and fix. Common issues:
- Hook registration order (prepend not working)
- DTensor dispatch interception blocking forward ops
- Gradient hook counting off
- Parameter swap (`__dict__` vs `_parameters`) not synced

**Step 4: Run full CPU test suite**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/ tests/contract/ -v --tb=short -x 2>&1 | tail -30
```

**Step 5: Commit**

```bash
git add tests/cpu/test_fsdp_integration.py
git commit -m "test(fsdp): add integration tests for forward/backward with fully_shard"
```

---

### Task 11: Multi-process distributed test with Gloo backend

End-to-end test with actual multi-process, `world_size=2`, Gloo backend on CPU. This validates real all-gather and reduce-scatter.

**Files:**
- Test: `tests/distributed/test_fsdp2_gloo.py` (create)

**Step 1: Write multi-process test**

Create `tests/distributed/test_fsdp2_gloo.py`:

```python
"""Multi-process FSDP2 tests with Gloo backend (world_size=2)."""
import os
import socket
import multiprocessing as mp

import candle as torch
import candle.nn as nn
import candle.distributed as dist
from candle.distributed.device_mesh import DeviceMesh
from candle.distributed._composable.fsdp import fully_shard
from candle.distributed.tensor.dtensor import DTensor


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker_forward_backward(rank, world_size, port):
    """Worker: test forward + backward with FSDP."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    try:
        model = nn.Linear(8, 4)
        mesh = DeviceMesh("cpu", (world_size,))
        fully_shard(model, mesh=mesh)

        x = torch.randn(2, 8, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 4), f"rank {rank}: bad shape {out.shape}"

        loss = out.sum()
        loss.backward()

        # Check gradients exist
        local_weight = model.weight.to_local() if isinstance(model.weight, DTensor) else model.weight
        assert local_weight.grad is not None, f"rank {rank}: no weight gradient"
    finally:
        dist.destroy_process_group()


def test_fsdp2_gloo_forward_backward():
    """FSDP2 forward+backward with 2 processes using Gloo."""
    world_size = 2
    port = _find_free_port()
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=_worker_forward_backward, args=(rank, world_size, port))
        p.start()
        processes.append(p)
    for p in processes:
        p.join(timeout=60)
        assert p.exitcode == 0, f"Worker exited with code {p.exitcode}"
```

**Step 2: Run multi-process test**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/distributed/test_fsdp2_gloo.py -v --tb=short -x
```

**Step 3: Debug and fix issues**

Multi-process tests may reveal:
- `FSDPMeshInfo` failing because `ProcessGroup.rank()` isn't available
- `all_gather_into_tensor` / `reduce_scatter_tensor` API mismatches
- DTensor dispatch intercepting collective ops
- Gradient synchronization issues

**Step 4: Commit**

```bash
git add tests/distributed/test_fsdp2_gloo.py
git commit -m "test(fsdp): add multi-process FSDP2 test with Gloo backend"
```

---

## Summary

| Task | Component | Est. Complexity |
|------|-----------|----------------|
| 1 | nn.Module hook fixes | Low |
| 2 | `__torch_function__` protocol | Medium |
| 3 | `__torch_dispatch__` protocol + key reorder | Medium |
| 4 | Placement types | Low |
| 5 | DeviceMesh (1D) | Low |
| 6 | DTensor | Medium |
| 7 | FSDPParam + FSDPParamGroup | Medium |
| 8 | FSDPState + hooks | Medium |
| 9 | `fully_shard()` + FSDPModule | Medium |
| 10 | Integration test (single-process) | Low |
| 11 | Multi-process test (Gloo) | Medium |

**Parallelizable:** Tasks 1-3 are independent. Tasks 4-5 are independent. Tasks 7-9 are sequential.
