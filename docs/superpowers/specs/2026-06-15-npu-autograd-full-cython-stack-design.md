# NPU Autograd Full-Cython Stack — Design Spec

**Date:** 2026-06-15
**Worktree:** `npu-autograd-cython`
**Goal:** Eliminate ~24ms/iter of Python `redispatch` + schema overhead in the Qwen2 backward path by migrating backward formulas to a full Cython stack: Cython Node → Cython backward formula → Cython direct kernel call (no dispatch).

## Motivation

Profile of the tiny 2-layer Qwen2 training step on NPU (post-PR #566, worktree `npu-autograd-cython`, 10 backward iters):

- **Backward: median 36.4ms/iter** (full step 63ms)
- Python `redispatch`: **24.1ms cumtime/iter**, 980 calls
- `schema.bind` + `_validate_types`: **11.0ms/iter**, 1073 calls (runs once per redispatch)
- `_getitem_backward` → zeros + setitem: 9.7ms/iter, 130 calls
- `_pow_backward_helper`: 2.5ms/iter
- `reshape` view: 2.8ms/iter

**~35ms of the 36ms backward is Python overhead.** The kernels themselves are fast; the cost is crossing the Python↔Cython boundary ~1000 times per backward via `redispatch` and schema validation.

## Architecture Reality (confirmed by exploration)

The "Cython autograd" already in the tree is partly illusory:

1. The generated `_functions_cy.pyx` defines node classes with `def apply(self, grad)`, but the autograd engine calls `node.backward(grad)`. The Cython nodes are therefore **not wired into the engine** — they define a method the engine never calls.
2. `_variable_type_cy.pyx` lazy-imports the **Python** `functions.py` as `_F` and instantiates Python node classes. Even the "Cython" variable-type path uses Python nodes.
3. Python `functions.py` nodes call `_redispatch(...)` → full Python dispatch → schema validation → Cython kernel. This is where the 24ms + 11ms lives.
4. The only working Cython backward nodes are **21 hand-written** `_Npu*Backward` cdef classes in `_functional_ops.pyx` and `_tensor_api.pyx` (add/mul/sub/div/reshape/transpose/sum/gelu/silu/rsqrt/linear/layer_norm/sdpa/addmm/etc.). These already run direct kernels and are the reason single-op Candle is *faster* than torch_npu. They are attached during forward via `attach_npu_*_grad` helpers and `_attach_*_grad` helpers.

The gap: every backward formula that is **not** one of those 21 hand-written nodes runs entirely in Python.

## Target Architecture

Per the user directive: **Cython Node → Cython backward formula → Cython direct kernel call (无 dispatch).**

```
Forward:  Cython kernel  →  attach Cython backward node
                          (generated cdef class with def backward,
                           OR existing hand-written _Npu*Backward)

Backward: engine calls node.backward(grad)
            → cdef/def backward(self, grad)              [Cython, no Python frame for formula]
                → for each op in the formula:
                    cy_backward_<op>(...)                [routing: direct NPU kernel if available]
                    else _redispatch("<op>", keyset, ...) [correctness fallback for long-tail ops]
                → create_graph=True forces ALL routing through _redispatch
                   (gradient must itself be differentiable)
```

Two coordinated changes achieve this:

- **Generator fix (Option B scope):** make the generator emit real `cdef class X(Node)` nodes with `def backward(self, grad)`, and switch `_variable_type_cy.pyx` to use `_functions_cy` instead of Python `functions`. Every generated formula then runs in Cython.
- **Per-op routing (Option A scope):** replace `_redispatch("<op>", ...)` in generated formulas with `cy_backward_<op>(...)`, which calls a direct NPU kernel when one exists and `create_graph` is off, otherwise falls back to `_redispatch`. The transpiler remains the single source of PyTorch-alignment truth.

## Components

### Component 1 — Generator fix (correctness, makes every node Cython)

Modify `tools/autograd/gen_functions.py`:

- `_gen_one_node_pyx`: emit `cdef class {cls}(Node)` instead of `class {cls}(_Node)`, and `def backward(self, grad)` instead of `def apply(self, grad)`. Mirror the body of the existing `_Npu*Backward` pattern: manual field init in `__init__` (no `super().__init__` cost when not needed), saved-tensor retrieval, formula body.
- Keep the existing skip set (`pyx_backward_skip_ops`) — those ops keep their hand-written / composite Python implementations.
- `_variable_type_cy.pyx` generation (`gen_variable_type.py`): the `_F` lazy import should resolve to `_functions_cy` (the compiled module) rather than the Python `functions` module. The generated node classes must expose the same `_save` / `saved_tensors` / `next_functions` surface the engine and variable-type code already use.

Acceptance: after regenerating, `import candle; candle._generated._functions_cy` provides `def backward` node classes, `_variable_type_cy` instantiates them, and the full NPU + CPU + contract test suites pass.

### Component 2 — `cy_backward_*` routing layer (performance)

- In `formula_transpiler.py`, the transpiler currently emits `_redispatch("<op>", keyset, ...)`. Change the `.pyx`-target emission to `cy_backward_<op>(...)`. The Python-target emission (for `functions.py`) stays as `redispatch(...)`.
- Define `cy_backward_<op>` helpers in the `_functions_cy.pyx` preamble (generated) that:
  1. Check `is_create_graph_enabled()` (existing helper in `_autograd_engine`). If true → `_redispatch("<op>", keyset, ...)` (differentiability required).
  2. Else, if a direct NPU kernel exists and inputs match the fast-path guards → call it.
  3. Else → `_redispatch("<op>", keyset, ...)` fallback.
- Coverage of `cy_backward_*` grows over phases. Ops without a direct kernel just always take the fallback — still Cython-wrapped (no `redispatch` Python wrapper frame for the call dispatch itself, though the fallback re-enters Python dispatch).

### Component 3 — Six missing fast nodes + kernels (the gap)

| Op | Backward formula | Work |
|---|---|---|
| `neg` | `_NpuNegBackward` | `fast_neg` direct, same-shape pass-through |
| `pow` (tensor exponent) | `_NpuPowBackward` | `fast_pow` + `fast_mul`; scalar-tensor path already covered by #566 |
| `expand` | `_NpuExpandBackward` | pure view → `_sum_to` reduction on-device, no copy |
| `zeros` | `cy_backward_zeros` kernel (for getitem/slice) | direct alloc + `inplace_zero`, ported from `npu/creation.py:zeros_create` |
| `setitem` | `cy_backward_setitem` kernel (for getitem/slice) | scatter-add / copy, ported from `npu/shape.py:setitem` |
| `matmul` | `_NpuMatmulBackward` | `fast_mm_mat1_backward` / `fast_mm_mat2_backward` already exist |

The `getitem` / `slice` backward composites (the 9.7ms `_getitem_backward` block) call `zeros` + `setitem`; with those two as `cy_backward_*` direct kernels, the composite runs without dispatch.

## Phasing (each phase ships independently, each is a PR)

### Phase 1 — Generator fix + `_F` switch (correctness, no perf change expected)

Deliverable: every generated backward formula runs in a `cdef class` with `def backward`. Full test suites pass. Backward median stays ≤ baseline (36.4ms).

### Phase 2 — `neg`, `pow` (tensor), `expand`, `matmul` fast nodes

Expected: ~6–8ms recovered. These are same-shape element-wise / simple reductions.

### Phase 3 — `zeros` / `setitem` Cython kernels + getitem/slice composite

Expected: ~8–10ms recovered (the single largest block, `_getitem_backward`).

### Phase 4 — `cy_backward_*` routing through the transpiler

Replace remaining `_redispatch` calls in generated `.pyx` formulas with routed `cy_backward_*`, removing the per-call Python `redispatch` wrapper frame and schema cost for the covered ops.

## Testing Strategy

1. **Correctness (primary safety net):** the existing `tests/npu/` gradient-accuracy suite must stay green. Add targeted gradient tests (vs torch_npu tolerances) for each new fast node: `neg`, `pow`, `expand`, `matmul`, plus `zeros`/`setitem` exercised through `getitem`/`slice`.
2. **Routing:** tests asserting Cython formulas call the direct kernel when `create_graph=False`, and fall back to `_redispatch` when `create_graph=True`. Locks the performance property and the create_graph correctness.
3. **Performance:** commit the Qwen2 backward benchmark to `benchmarks/`. Run after each build. Regression gate: backward median must stay ≤ current baseline; target < 15ms.

## Data Flow (end-to-end, after all phases)

```
forward mul:    fast_mul_exact → attach _NpuMulBackward (cdef)       [already works]
backward:       engine → _NpuMulBackward.backward(grad)              [Cython]
                  → _cy_fast_npu_mul(grad, other)                    [direct kernel, no dispatch]

forward getitem: dispatch → attach generated GetitemBackward0        [Cython after Phase 1]
backward:       engine → GetitemBackward0.backward(grad)             [Cython]
                  → cy_backward_zeros(...) + cy_backward_setitem(...) [direct kernels, Phase 3]

forward pow:    dispatch → attach generated PowBackward0             [Cython after Phase 1]
backward:       engine → PowBackward0.backward(grad)                 [Cython]
                  → cy_backward_pow / cy_backward_mul / cy_backward_log [routing, Phase 4]
```

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Generator fix breaks hundreds of generated nodes | Phase 1 ships correctness only; full NPU + CPU + contract suites run before merge. The `cdef` conversion is mechanical since the generator already mirrors the Python node structure. |
| `cdef class` nodes cannot dynamically set `backward` (engine expects `.backward()`) | Generate `def backward(self, grad)` as a real method. Engine confirmed to call `node.backward(grad)`. |
| `create_graph` fallback not triggered → double-backward wrong | Routing tests cover `create_graph=True`; the pattern in `_NpuMulBackward` (line ~1283) is the template. |
| Bypassing schema validation masks real bugs | Bypass applies only when called from backward formulas; the forward path keeps full validation. Backward inputs are gradients already produced by validated ops. |
| `setitem`/`zeros` fast kernels have alias/version bugs | Start from the validated `npu/creation.py:zeros_create` and `shape.py:setitem` — port existing Python kernels to Cython, no new logic. |

## Constraints Honored

- No torch import in `src/candle/` (tests only compare against torch_npu).
- No CPU fallback for NPU ops; all kernels stay on-device (views are pure stride logic; zeros/setitem/matmul use ACLNN).
- No schema-validation bypass on the public forward path.
- No application-specific or benchmark-only branches; the generator change applies to all generated formulas generically.

## Acceptance Criteria

- `tests/npu/`, `tests/cpu/`, `tests/contract/` all green.
- Pylint green.
- Backward median < 15ms on the tiny Qwen2 benchmark (from 36.4ms baseline).
- `create_graph=True` paths verified correct (routing tests + higher-order grad tests).
