# Phase 2: Replace Python redispatch with Direct Cython Kernel Calls

**Date**: 2026-06-16  
**Status**: Planning  
**Related**: [Phase 1 Plan](2026-06-15-npu-autograd-full-cython-stack-phase1.md), [Design Spec](../specs/2026-06-15-npu-autograd-full-cython-stack-design.md)

## Goal

Replace Python `redispatch()` calls in Cython backward formulas with direct Cython kernel calls to eliminate the 27ms backward overhead (41% of Qwen2 step time).

**Architecture**: Cython Node → Cython backward formula → **Cython kernel (direct call, no dispatch)**

## Current State (Post-Phase 1)

Phase 1 completed:
- ✅ 181 real Cython backward nodes (was 503 subclass-fallbacks)
- ✅ Nodes execute in Cython space
- ❌ But formulas still call Python `redispatch()` → 27ms overhead remains

**Performance**: 3.15ms per simple MLP step (baseline: 2.98ms) — 5.7% slower due to Python boundary crossing.

## Problem Analysis

### Where is the 27ms?

From Qwen2 profiling (memory: `project_npu_qwen2_eager_levers.md`):
- Backward redispatch: **~27ms per step** (41% of total)
- Top backward ops by call count:
  1. `zeros()` - 14 calls
  2. `__setitem__` - 13 calls  
  3. `reshape` - 12 calls
  4. `expand` - 6 calls
  5. `pow` - 5 calls

### Root Cause

Cython formulas call Python functions:
```python
# In _functions_cy.pyx
def backward(self, grad):
    result = _redispatch("mul", keyset, grad, other)  # Python call!
```

`_redispatch()` is a Python wrapper that:
1. Crosses Python/Cython boundary
2. Does dispatch key lookup (Python dict)
3. Calls back into Cython kernel
4. Returns through Python boundary

Each backward op calls `_redispatch()` 1-4 times → massive overhead.

## Phase 2 Approach

### Strategy

Replace `_redispatch()` with direct Cython kernel calls from generated formulas:

**Before**:
```python
result = _redispatch("mul", keyset, grad, other)
```

**After**:
```cython
from candle._C._npu_ops cimport npu_mul
result = npu_mul(grad, other)
```

### Implementation Plan

#### Step 1: Create Cython Kernel Registry

Create `src/candle/_C/_kernel_registry.pxd`:
```cython
# cdef declarations for all NPU kernels
from candle._C._tensor_impl cimport Tensor

cdef Tensor npu_mul(Tensor a, Tensor b)
cdef Tensor npu_add(Tensor a, Tensor b, object alpha=*)
cdef Tensor npu_sub(Tensor a, Tensor b, object alpha=*)
cdef Tensor npu_div(Tensor a, Tensor b)
cdef Tensor npu_pow(Tensor base, object exponent)
cdef Tensor npu_neg(Tensor t)
cdef Tensor npu_zeros_like(Tensor t)
cdef Tensor npu_sum_to_size(Tensor t, tuple shape)
cdef Tensor npu_expand(Tensor t, tuple shape)
cdef Tensor npu_reshape(Tensor t, tuple shape)
# ... etc for all backward hot ops
```

Create `src/candle/_C/_kernel_registry.pyx`:
```cython
# Import implementations
from candle._C._npu_ops cimport (
    _npu_mul_impl,
    _npu_add_impl,
    # ... etc
)

cdef Tensor npu_mul(Tensor a, Tensor b):
    return _npu_mul_impl(a, b)

# ... wrappers for all ops
```

#### Step 2: Update Formula Transpiler

Modify `tools/autograd/formula_transpiler.py` to emit direct Cython calls:

```python
# Add mapping: op_name -> cython kernel function
CYTHON_KERNEL_MAP = {
    "mul": "npu_mul",
    "add": "npu_add",
    "sub": "npu_sub",
    "div": "npu_div",
    "pow": "npu_pow",
    "neg": "npu_neg",
    # ... etc
}

def transpile_for_cython(formula: str, op_name: str) -> str:
    """Replace _redispatch calls with direct Cython kernel calls."""
    if op_name in CYTHON_KERNEL_MAP:
        pattern = r'_redispatch\("' + op_name + r'", keyset, (.+?)\)'
        replacement = CYTHON_KERNEL_MAP[op_name] + r'(\1)'
        formula = re.sub(pattern, replacement, formula)
    return formula
```

#### Step 3: Update Generator

Modify `tools/autograd/gen_functions.py`:

1. Add cimport at top of `_functions_cy.pyx`:
   ```cython
   from candle._C._kernel_registry cimport (
       npu_mul, npu_add, npu_sub, npu_div, npu_pow,
       npu_neg, npu_zeros_like, npu_sum_to_size,
       npu_expand, npu_reshape
   )
   ```

2. In `_gen_one_node_pyx()`, transpile formulas before emission:
   ```python
   formula = transpile_for_cython(formula, derivative.op_name)
   ```

#### Step 4: Validation

**RED test** (`tests/npu/cython/test_phase2_direct_kernel_calls.py`):
```python
def test_backward_uses_direct_cython_kernel_not_redispatch(npu_device, monkeypatch):
    """Backward formulas should call Cython kernels directly, not redispatch."""
    import candle as torch
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey
    
    calls = {"redispatch": 0}
    
    def fail_redispatch(*args, **kwargs):
        calls["redispatch"] += 1
        raise AssertionError("Backward should use direct Cython kernel, not Python redispatch")
    
    # Monkey-patch backward hot ops
    monkeypatch.setitem(registry.get("mul").kernels, DispatchKey.NPU, fail_redispatch)
    monkeypatch.setitem(registry.get("add").kernels, DispatchKey.NPU, fail_redispatch)
    
    x = torch.tensor([2.0, 3.0], device=npu_device, requires_grad=True)
    y = torch.tensor([4.0, 5.0], device=npu_device, requires_grad=True)
    
    z = x * y + x
    loss = z.sum()
    loss.backward()
    
    assert calls["redispatch"] == 0, f"Backward went through Python redispatch {calls['redispatch']} times"
```

**Performance test**:
```python
def test_phase2_backward_faster_than_phase1():
    # Same MLP test as before
    # Expected: < 2.5ms per step (Phase 1 was 3.15ms, baseline 2.98ms)
```

#### Step 5: Incremental Rollout

Start with **top 5 hot ops** from Qwen2 profiling:
1. `zeros` / `zeros_like`
2. `__setitem__` (tensor assignment)
3. `reshape`
4. `expand`
5. `pow`

Then expand to all common backward ops:
- `mul`, `add`, `sub`, `div`, `neg`
- `sum`, `sum_to_size`
- `transpose`, `permute`
- etc.

## Success Criteria

1. **Routing test passes**: Backward does NOT call Python `redispatch()`
2. **Correctness**: All 3577 CPU + 611 NPU tests pass
3. **Performance**: 
   - Simple MLP: < 2.5ms (Phase 1: 3.15ms, baseline: 2.98ms)
   - Qwen2 backward: < 10ms (was 37ms post-Phase 1, 27ms is redispatch overhead)
   - **Target: ~8x backward speedup**

## Risks & Mitigation

**Risk 1**: Cython kernel signatures might not match formula args exactly
- **Mitigation**: Add wrapper functions in `_kernel_registry.pyx` to handle arg normalization

**Risk 2**: Some ops don't have direct Cython kernels yet
- **Mitigation**: Keep `_redispatch()` fallback for unimplemented ops, gradually expand coverage

**Risk 3**: Device-specific kernels (NPU vs CPU vs MPS)
- **Mitigation**: Phase 2 targets NPU only; other backends keep using redispatch

## Timeline

- Step 1 (Kernel registry): 1 hour
- Step 2 (Transpiler): 1 hour  
- Step 3 (Generator): 1 hour
- Step 4 (Tests): 1 hour
- Step 5 (Rollout): 2 hours

**Total**: ~6 hours

## Next Phase (Phase 3)

After Phase 2, if needed:
- Profile again to find remaining bottlenecks
- Consider fused backward kernels for common patterns
- Optimize autograd engine overhead (node allocation, graph traversal)
