# Candle torch.compile Feasibility Analysis Report

> Date: 2026-03-11
> Goal: Fully replicate torch.compile end-to-end in candle, targeting Inductor/Triton (CUDA), GE/torchair (Ascend NPU), and other backends

---

## 1. torch.compile Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Python Code (user model)                               │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  TorchDynamo  (CPython PEP 523 bytecode interception)    │
│  - frame evaluation hook (C extension)                   │
│  - symbolic bytecode interpreter (SymbolicConvert)       │
│  - guard generation + compilation cache                  │
│  - graph break handling + resume bytecode generation     │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  FX Graph IR  (flat node list)                           │
│  - 6 opcodes: placeholder / get_attr / call_function     │
│                call_method / call_module / output         │
│  - Node metadata: shape, dtype, stride, device           │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  AOTAutograd                                             │
│  - FakeTensor-driven joint forward+backward graph capture│
│  - Functionalization (in-place → functional)             │
│  - Decomposition (~2000 ops → ~250 core ATen ops)        │
│  - Partitioning (min-cut rematerialization)              │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Backend Interface                                       │
│  CompilerFn(GraphModule, example_inputs) → CompiledFn    │
├──────────────┬───────────────┬───────────────────────────┤
│  Inductor    │  torchair/GE  │  custom backends          │
│  (Triton/C++)│  (Ascend NPU) │                           │
└──────────────┴───────────────┴───────────────────────────┘
```

---

## 2. Candle Existing Infrastructure

| Component | Status | Reusability |
|-----------|--------|-------------|
| **Dispatch system** | 23 DispatchKeys, priority-based routing, ATen-like | High — op interception point |
| **Op schema registry** | 315+ ops with param types, mutation markers, alias info | High — op metadata for graph IR |
| **Functionalize** | `add_` → `add` + writeback, traceable mutations | High — required for graph normalization |
| **Autograd engine** | Node-based computation graph, backward DFS, SavedTensor | Medium — backward graph only, no forward capture |
| **Meta backend** | Shape-only inference, no computation | Medium — basis for FakeTensor |
| **`__torch_dispatch__`** | Implemented in DTensor | Medium — reference for op interception |
| **Pipeline system** | Deferred execution + batching infrastructure | Low — different design goals |
| **nn.Module** | Full hooks, state_dict, named_parameters | High — meets fx Module requirements |
| **torch.compile / jit** | Pure stubs, compile returns original model | None — build from scratch |

---

## 3. Layer-by-Layer Gap Analysis

### Layer 1: TorchDynamo — Bytecode Capture Engine

**Requirements**:
- C extension: CPython PEP 523 frame evaluation hook (~2k lines C)
- Symbolic bytecode interpreter: interprets Python bytecode instruction-by-instruction, records tensor ops into FX Graph (~80k lines Python)
- Variable Tracker: tracks symbolic state of each local variable (TensorVariable, ConstantVariable, ListVariable...)
- Guard system: generates runtime check conditions (shape, dtype, Python object identity), caches compiled results
- Graph break: detects untraceable code, splits into subgraphs, generates resume bytecode

**Candle status**: Completely blank. `torch.jit` / `torch.compile` are stubs.

**Key challenges**:
- Dynamo is **tightly coupled to CPython versions** — bytecode format changes across 3.10 / 3.11 / 3.12 / 3.13
- PyTorch's `torch/_dynamo/` directory is ~**150k+ lines of code**, the largest and most complex part of torch.compile
- Involves deep CPython internals: `dis`, code objects, frame objects, `MAKE_FUNCTION`, `LOAD_CLOSURE`, etc.

### Layer 2: FX Graph IR

**Requirements**:
- `Node`: op, target, args, kwargs, name, users, meta
- `Graph`: ordered node linked list, codegen (`python_code()`), graph transformation API
- `GraphModule(nn.Module)`: wraps Graph + Module into executable Module
- `Proxy` + `Tracer`: symbolic tracing
- `Interpreter`: per-node executor with customizable behavior

**Candle status**: Completely blank. However, the dispatch system's Schema registry provides op metadata foundation.

**Complexity**: Medium, ~10k lines, clean design, independently implementable.

### Layer 3: AOTAutograd

**Requirements**:
- `FakeTensor`: `__torch_dispatch__` subclass, propagates metadata only (shape/dtype/stride), no computation
- `FakeTensorMode`: context manager routing all tensor ops to meta implementations
- Joint graph capture: trace forward + backward with FakeTensors, produce unified FX Graph
- Functionalization: candle has partial implementation (`functionalize_context()`)
- Decomposition table: decompose high-level ops into core op set
- Partitioning: split joint graph into forward + backward graphs (min-cut algorithm determining which intermediates to save vs. recompute)

**Candle status**:
- Functionalize: exists
- Meta backend: exists (incomplete op coverage)
- `__torch_dispatch__` protocol: implemented in DTensor
- FakeTensor / FakeTensorMode: does not exist
- Joint graph capture: does not exist
- Decomposition table: does not exist
- Graph partitioning: does not exist

### Layer 4: Backend Interface + Bridge

**Inductor/Triton (CUDA)**:
- Option A: candle FX Graph → convert to `torch.fx.GraphModule` → feed to `torch._inductor.compile_fx`
  - Pros: fully reuse Inductor, zero codegen work
  - Cost: PyTorch becomes a compile-time dependency
- Option B: generate Triton kernels directly
  - Requires implementing Inductor's scheduling + codegen logic (massive effort)

**GE/torchair (Ascend NPU)**:
- Option A: candle FX Graph → convert to `torch.fx.GraphModule` → feed to torchair
  - torchair already has FX → GE bridging logic
  - Also requires PyTorch as compile-time dependency
- Option B: candle FX Graph → construct GE Graph directly (via GE C API / protobuf)
  - Requires binding GE's `ge::Graph`, `ge::Operator`, `ge::TensorDesc` C++ APIs
  - Or use ONNX as intermediate: candle graph → ONNX → GE

**Candle NPU backend status**:
- Comprehensive ACLNN ctypes bindings (16k+ lines, 100+ ops)
- ACL runtime bindings (device/stream/event/memory)
- No ACL Graph / GE API bindings
- Currently pure eager-mode execution

---

## 4. Key Architectural Decisions

### Decision 1: Dynamo — Build vs Fork vs Skip

| Option | Pros | Cons |
|--------|------|------|
| **A: Fork PyTorch Dynamo** | Most complete Python coverage; community-maintained | Must replace all torch references; massive adaptation; CPython version coupling |
| **B: Build Dynamo from scratch** | Full control; can simplify design | Massive effort (150k+ lines); must track CPython evolution |
| **C: FX symbolic trace only** | Much simpler; covers ~80% of models | No data-dependent control flow; fails entirely on graph break |

**Recommendation**: Start with C, then move to A. Use symbolic tracing in Phase 1 to validate the full pipeline, then fork Dynamo for bytecode-level tracing in Phase 5.

### Decision 2: Backend Integration Strategy

| Option | Pros | Cons |
|--------|------|------|
| **A: PyTorch as compile-time dep** | Directly reuse Inductor/torchair; zero backend dev | Compile-time PyTorch dependency |
| **B: Pure self-built IR bridge** | Fully independent | Must implement IR bridge per backend |
| **C: Hybrid** — own backend protocol + optional torch.fx bridge | Flexible; pluggable backends | Must maintain conversion layer |

**Recommendation**: C. Candle defines its own `CompilerFn` protocol and FX IR, provides a `candle.fx.Graph → torch.fx.Graph` converter as an optional bridge.

- Inductor users: `pip install torch` as compile-time dep → convert → Inductor
- GE users: candle's own GE bridge (ctypes-bound GE C API), no PyTorch dependency
- Future custom backends: consume candle FX IR directly

### Decision 3: Core Op Set Definition

Inductor expects ~250 core ATen ops. Candle has 315 op schemas. Required:
- Define candle's core op set (~100-150 ops)
- All non-core ops decompose to core ops via decomposition table
- Core op set must have clear mapping to ATen core ops (for Inductor/GE interop)

---

## 5. Implementation Roadmap

```
Phase 1: FX Graph IR                    Phase 2: Tracing + FakeTensor
┌─────────────────────┐                ┌─────────────────────────┐
│ candle.fx.Node      │                │ candle.fx.Proxy         │
│ candle.fx.Graph     │                │ candle.fx.Tracer        │
│ candle.fx.GraphModule│───────────────▶│ FakeTensor + Mode       │
│ candle.fx.Interpreter│               │ symbolic_trace()        │
│ IR definition + serde│                │ Meta op completion      │
└─────────────────────┘                └────────────┬────────────┘
                                                    │
Phase 3: AOTAutograd                                ▼
┌─────────────────────────┐            Phase 4: Backend Bridge
│ Joint fwd+bwd capture   │            ┌─────────────────────────┐
│ Decomposition table     │            │ CompilerFn protocol     │
│ Graph partitioning      │───────────▶│ candle.fx → torch.fx    │
│ Functionalize refinement│            │   → Inductor/Triton     │
│ Core op set definition  │            │ candle.fx → GE Graph    │
└─────────────────────────┘            │   → Ascend NPU          │
                                       └────────────┬────────────┘
                                                    │
                                       Phase 5: Dynamo
                                       ┌─────────────────────────┐
                                       │ Fork torch._dynamo      │
                                       │ Adapt for candle tensor │
                                       │ Guard + cache           │
                                       │ Graph break handling    │
                                       └─────────────────────────┘
```

### Phase 1: FX Graph IR (Foundation)

**Goal**: Establish core graph IR data structures.

**Deliverables**:
- `candle.fx.Node` — graph node: op, target, args, kwargs, name, users, meta
- `candle.fx.Graph` — ordered node list + graph transformation API + `python_code()` codegen
- `candle.fx.GraphModule` — executable Module wrapping Graph + Module
- `candle.fx.Interpreter` — per-node executor
- Serialization / deserialization

**Estimated size**: ~5-8k lines

### Phase 2: Tracing + FakeTensor

**Goal**: `candle.fx.symbolic_trace(model)` captures model graph.

**Deliverables**:
- `candle.fx.Proxy` — proxy object intercepting operators + `__torch_function__`
- `candle.fx.Tracer` — trace(module) → Graph
  - Handle nn.Module hierarchy (call_module vs recursive trace)
  - Handle parameters/buffers (get_attr nodes)
  - Handle placeholder/output
- `FakeTensor` — `__torch_dispatch__` subclass, propagates shape/dtype only
- `FakeTensorMode` — context manager
- Complete meta (shape inference) implementations for all 315 ops

**Estimated size**: ~10-15k lines

### Phase 3: AOTAutograd

**Goal**: Joint forward+backward graph capture + op decomposition.

**Deliverables**:
- Joint forward+backward graph capture
- Decomposition table (define candle core op set + decomposition rules)
- Graph partitioning (min-cut rematerialization)
- Functionalization layer refinement

**Estimated size**: ~15-20k lines

### Phase 4: Backend Bridge

**Goal**: `candle.compile()` is functional with real backends.

**Deliverables**:
- `CompilerFn` protocol definition
- `candle.fx.Graph → torch.fx.GraphModule` converter (for Inductor/torchair)
- `candle.fx.Graph → GE Graph` direct bridge (via ctypes-bound GE C API)
- Eager backend (execute GraphModule directly, for debugging)

**Estimated size**: ~8-12k lines

### Phase 5: Dynamo

**Goal**: Bytecode-level tracing with data-dependent control flow support.

**Deliverables**:
- Fork `torch._dynamo`
- Adapt for candle tensor/dispatch
- Guard system + compilation cache
- Graph break + resume bytecode
- CPython version adaptation (3.11 / 3.12)

**Estimated size**: ~30-50k lines (including fork adaptation)

---

## 6. Feasibility Assessment

| Dimension | Assessment |
|-----------|-----------|
| **Technical feasibility** | **High** — candle's dispatch system, schema registry, functionalize, and meta backend provide a strong foundation. Core gaps (Graph IR + Proxy) are engineering work with no theoretical barriers |
| **Op coverage** | **Medium-High** — 315+ ops registered, but meta implementation completeness needs per-op verification |
| **nn.Module compatibility** | **High** — complete Module hierarchy, hooks, state_dict meet fx tracing requirements |
| **Control flow handling** | **Low (Phase 1-4) / High (Phase 5)** — symbolic tracing cannot handle data-dependent control flow; Dynamo can |
| **Backend integration** | **High** — Inductor/torchair have mature interfaces; GE accessible via C API |
| **Maintenance cost** | **Medium-High** — Graph IR is stable; Dynamo must track CPython evolution; backend bridges must track Inductor/GE versions |

---

## 7. Risk Matrix

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Dynamo CPython coupling** | Requires adaptation per Python version | Use symbolic tracing first; lock Dynamo to Python 3.11/3.12 |
| **Inductor internal API instability** | candle.fx → torch.fx converter may break frequently | Pin PyTorch version; add version adaptation in converter |
| **Insufficient GE C API documentation** | GE bridge development blocked | Reverse-engineer from torchair source; ONNX intermediate path as fallback |
| **Op coverage long tail** | Some models fail to trace | Define core op set + decomposition fallback; unsupported ops trigger graph break |
| **Performance regression** | Compiled code slower than eager | Provide eager fallback; compile defaults to off |

---

## 8. Conclusion and Recommendations

**Fully replicating torch.compile is technically feasible.** Candle's dispatch system, schema registry, functionalize, and meta backend provide a better starting point than most frameworks.

**Core recommendations**:

1. **Phase 1-3: Build in-house** (FX IR + Tracing + AOTAutograd) — these are core capabilities candle must own
2. **Phase 4: Hybrid approach** — define own backend protocol while providing torch.fx bridge to reuse Inductor/torchair
3. **Phase 5: Fork Dynamo** — after Phase 1-4 validates the full pipeline

**The critical first step**: implement `candle.fx` (Graph IR + Proxy + Tracer) — this is the keystone of the entire compilation stack.
