# Candle 实现 torch.compile 可行性分析报告

> 日期: 2026-03-11
> 目标: 在 candle 中完全复刻 torch.compile 全链路，对接 Inductor/Triton (CUDA)、GE/torchair (Ascend NPU) 等后端

---

## 一、torch.compile 全链路架构

```
┌─────────────────────────────────────────────────────────┐
│  Python Code (用户模型)                                  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  TorchDynamo  (CPython PEP 523 字节码拦截)                │
│  - frame evaluation hook (C extension)                   │
│  - 符号字节码解释器 (SymbolicConvert)                      │
│  - Guard 生成 + 编译缓存                                  │
│  - Graph Break 处理 + resume bytecode 生成                │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  FX Graph IR  (扁平 node 列表)                            │
│  - 6 opcodes: placeholder / get_attr / call_function     │
│                call_method / call_module / output         │
│  - Node metadata: shape, dtype, stride, device           │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  AOTAutograd                                             │
│  - FakeTensor 驱动的 joint forward+backward graph capture │
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

## 二、Candle 现有基础盘点

| 组件 | 现状 | 可复用度 |
|------|------|----------|
| **Dispatch 系统** | 23 个 DispatchKey，优先级路由，类似 ATen dispatcher | 高 — 可作为 op 拦截点 |
| **Op Schema 注册** | 315+ ops，含参数类型、mutation 标记、alias 信息 | 高 — 图 IR 的 op 元数据来源 |
| **Functionalize** | `add_` → `add` + writeback，mutation 可追踪 | 高 — 图规范化必需 |
| **Autograd 引擎** | Node-based 计算图、backward DFS、SavedTensor | 中 — 只有反向图，缺前向图 |
| **Meta Backend** | shape-only 推导，无实际计算 | 中 — 可作为 FakeTensor 基础 |
| **`__torch_dispatch__`** | DTensor 已实现此协议 | 中 — 可参考做 op 拦截 |
| **Pipeline 系统** | 延迟执行 + 批处理基础设施 | 低 — 设计目标不同 |
| **nn.Module** | 完整的 hooks、state_dict、named_parameters | 高 — 符合 fx 对 Module 的需求 |
| **torch.compile / jit** | 纯 stub，compile 返回原模型 | 无 — 需从零构建 |

---

## 三、逐层 Gap 分析

### Layer 1: TorchDynamo — 字节码捕获引擎

**需要什么**:
- C extension: CPython PEP 523 frame evaluation hook (~2k 行 C)
- 符号字节码解释器: 逐条解释 Python bytecode，将 tensor op 记录到 FX Graph (~80k 行 Python)
- Variable Tracker: 跟踪每个局部变量的符号状态 (TensorVariable, ConstantVariable, ListVariable...)
- Guard 系统: 生成运行时检查条件 (shape, dtype, Python 对象 identity)，缓存编译结果
- Graph Break: 检测无法 trace 的代码，切分子图，生成 resume bytecode

**Candle 现状**: 完全空白。`torch.jit` / `torch.compile` 都是 stub。

**核心难点**:
- Dynamo 与 CPython 版本**强绑定** — bytecode format 在 3.10 / 3.11 / 3.12 / 3.13 之间都有变化
- PyTorch 的 `torch/_dynamo/` 目录约 **150k+ 行代码**，是 torch.compile 中最大最复杂的部分
- 涉及大量 Python 内部机制: `dis`, `code object`, `frame object`, `MAKE_FUNCTION`, `LOAD_CLOSURE` 等

### Layer 2: FX Graph IR

**需要什么**:
- `Node`: op, target, args, kwargs, name, users, meta
- `Graph`: 有序 node 链表，代码生成 (`python_code()`)，图变换 API
- `GraphModule(nn.Module)`: 将 Graph 包装为可执行 Module
- `Proxy` + `Tracer`: symbolic tracing
- `Interpreter`: 逐 node 执行器，支持自定义 per-node 行为

**Candle 现状**: 完全空白。但 dispatch 系统的 Schema 注册提供了 op 元数据基础。

**复杂度**: 中等，~10k 行，设计清晰，可独立实现。

### Layer 3: AOTAutograd

**需要什么**:
- `FakeTensor`: `__torch_dispatch__` 子类，只传播 metadata (shape/dtype/stride)，不做计算
- `FakeTensorMode`: context manager，将所有 tensor 操作路由到 meta 实现
- Joint graph capture: 用 FakeTensor trace forward + backward，产出统一 FX Graph
- Functionalization: candle 已部分实现 (`functionalize_context()`)
- Decomposition table: 将高级 op 分解为 core op 集合
- Partitioning: 将 joint graph 切分为 forward graph + backward graph (min-cut 算法)

**Candle 现状**:
- Functionalize 已有
- Meta backend 已有 (但 op 覆盖不全)
- `__torch_dispatch__` 协议在 DTensor 中已实现
- FakeTensor / FakeTensorMode 不存在
- Joint graph capture 不存在
- Decomposition table 不存在
- Graph partitioning 不存在

### Layer 4: Backend Interface + Bridge

**对接 Inductor/Triton (CUDA)**:
- 方案 A: candle FX Graph → 转换为 `torch.fx.GraphModule` → 喂给 `torch._inductor.compile_fx`
  - 优点: 完全复用 Inductor，零 codegen 工作
  - 代价: PyTorch 成为编译期依赖
- 方案 B: 直接生成 Triton kernel
  - 需要实现 Inductor 的 scheduling + codegen 逻辑 (极大工作量)

**对接 GE/torchair (Ascend NPU)**:
- 方案 A: candle FX Graph → 转换为 `torch.fx.GraphModule` → 喂给 torchair
  - torchair 已有 FX → GE 的桥接逻辑
  - 同样需要 PyTorch 作为编译期依赖
- 方案 B: candle FX Graph → 直接构造 GE Graph (通过 GE C API / protobuf)
  - 需要绑定 GE 的 `ge::Graph`, `ge::Operator`, `ge::TensorDesc` 等 C++ API
  - 或通过 ONNX 作为中间格式: candle graph → ONNX → GE

**Candle NPU 后端现状**:
- 已有完善的 ACLNN ctypes 绑定 (16k+ 行，100+ 算子)
- 已有 ACL runtime 绑定 (device/stream/event/memory)
- 没有任何 ACL Graph / GE API 绑定
- 当前纯 eager 模式执行

---

## 四、关键架构决策

### 决策 1: Dynamo — 自研 vs Fork vs 跳过

| 选项 | 优点 | 缺点 |
|------|------|------|
| **A: Fork PyTorch Dynamo** | 最完整的 Python 覆盖率；社区持续维护 | 需替换所有 torch 引用；巨大适配工作量；CPython 版本强绑定 |
| **B: 全自研 Dynamo** | 完全控制；可简化设计 | 工作量极大 (150k+ 行)；需跟随 CPython 版本演进 |
| **C: 先只做 FX symbolic trace** | 简单得多；覆盖 80% 场景 | 不支持数据依赖控制流；graph break 时整体失败 |

**建议**: 先 C 后 A。Phase 1 用 symbolic tracing 验证全链路，Phase 2 fork Dynamo 做字节码级别。

### 决策 2: 后端对接策略

| 选项 | 优点 | 缺点 |
|------|------|------|
| **A: PyTorch 作为编译期依赖** | 直接复用 Inductor/torchair；后端零开发 | 编译期依赖 PyTorch |
| **B: 纯自研 IR bridge** | 完全独立 | 需要为每个后端实现 IR bridge |
| **C: 混合** — 自有 backend protocol + 可选 torch.fx bridge | 灵活；后端可插拔 | 需维护转换层 |

**建议**: C。candle 定义自己的 `CompilerFn` 协议和 FX IR，提供 `candle.fx.Graph → torch.fx.Graph` 转换器作为可选 bridge。

- Inductor 用户: `pip install torch` 作为编译期依赖 → 转换 → Inductor
- GE 用户: 通过 candle 自研的 GE bridge (ctypes 绑定 GE C API)，不依赖 PyTorch
- 未来自研后端: 直接消费 candle FX IR

### 决策 3: Core Op Set 定义

Inductor 期望 ~250 个 core ATen ops。Candle 有 315 个 op schema。需要:
- 定义 candle 的 core op set (~100-150 个)
- 所有非 core op 通过 decomposition table 分解为 core ops
- core op set 必须与 ATen core ops 有明确映射关系 (方便对接 Inductor/GE)

---

## 五、实施路线图

```
Phase 1: FX Graph IR                    Phase 2: Tracing + FakeTensor
┌─────────────────────┐                ┌─────────────────────────┐
│ candle.fx.Node      │                │ candle.fx.Proxy         │
│ candle.fx.Graph     │                │ candle.fx.Tracer        │
│ candle.fx.GraphModule│───────────────▶│ FakeTensor + Mode       │
│ candle.fx.Interpreter│               │ symbolic_trace()        │
│ IR 定义 + 序列化     │                │ Meta op 补全            │
└─────────────────────┘                └────────────┬────────────┘
                                                    │
Phase 3: AOTAutograd                                ▼
┌─────────────────────────┐            Phase 4: Backend Bridge
│ Joint fwd+bwd capture   │            ┌─────────────────────────┐
│ Decomposition table     │            │ CompilerFn protocol     │
│ Graph partitioning      │───────────▶│ candle.fx → torch.fx    │
│ Functionalize 完善      │            │   → Inductor/Triton     │
│ Core op set 定义        │            │ candle.fx → GE Graph    │
└─────────────────────────┘            │   → Ascend NPU          │
                                       └────────────┬────────────┘
                                                    │
                                       Phase 5: Dynamo
                                       ┌─────────────────────────┐
                                       │ Fork torch._dynamo      │
                                       │ 适配 candle tensor      │
                                       │ Guard + cache           │
                                       │ Graph break handling    │
                                       └─────────────────────────┘
```

### Phase 1: FX Graph IR (基础)

**目标**: 建立图 IR 核心数据结构

**内容**:
- `candle.fx.Node` — 图节点: op, target, args, kwargs, name, users, meta
- `candle.fx.Graph` — 有序节点列表 + 图变换 API + `python_code()` 代码生成
- `candle.fx.GraphModule` — 包装 Graph + Module 的可执行模块
- `candle.fx.Interpreter` — 逐节点执行器
- 序列化 / 反序列化

**估算代码量**: ~5-8k 行

### Phase 2: Tracing + FakeTensor

**目标**: 能够 `candle.fx.symbolic_trace(model)` 捕获模型图

**内容**:
- `candle.fx.Proxy` — 代理对象，拦截运算符 + `__torch_function__`
- `candle.fx.Tracer` — trace(module) → Graph
  - 处理 nn.Module 层级 (call_module vs 递归 trace)
  - 处理 parameters/buffers (get_attr nodes)
  - 处理 placeholder/output
- `FakeTensor` — `__torch_dispatch__` 子类，只传播 shape/dtype
- `FakeTensorMode` — context manager
- 补全 315 个 op 的 meta (shape inference) 实现

**估算代码量**: ~10-15k 行

### Phase 3: AOTAutograd

**目标**: 联合前向+反向图捕获 + op 分解

**内容**:
- Joint forward+backward graph capture
- Decomposition table (定义 candle core op set + 分解规则)
- Graph partitioning (min-cut rematerialization)
- 完善 Functionalization 层

**估算代码量**: ~15-20k 行

### Phase 4: Backend Bridge

**目标**: `candle.compile()` 真正可用，对接真实后端

**内容**:
- `CompilerFn` 协议定义
- `candle.fx.Graph → torch.fx.GraphModule` 转换器 (对接 Inductor/torchair)
- `candle.fx.Graph → GE Graph` 直接桥接 (通过 ctypes 绑定 GE C API)
- Eager backend (直接执行 GraphModule，用于调试)

**估算代码量**: ~8-12k 行

### Phase 5: Dynamo

**目标**: 字节码级别 tracing，支持数据依赖控制流

**内容**:
- Fork `torch._dynamo`
- 适配 candle tensor/dispatch
- Guard 系统 + 编译缓存
- Graph break + resume bytecode
- CPython 版本适配 (3.11 / 3.12)

**估算代码量**: ~30-50k 行 (含 fork 适配)

---

## 六、可行性评估

| 维度 | 评估 |
|------|------|
| **技术可行性** | **高** — candle 的 dispatch 系统、schema 注册、functionalize、meta backend 提供了良好基础。核心缺失 (Graph IR + Proxy) 是工程工作，不存在理论障碍 |
| **Op 覆盖** | **中高** — 315+ ops 已注册，但 meta 实现完整度需逐个验证 |
| **nn.Module 兼容性** | **高** — 完整的 Module 层级、hooks、state_dict，满足 fx tracing 需求 |
| **控制流处理** | **低 (Phase 1-4) / 高 (Phase 5)** — Symbolic tracing 无法处理数据依赖控制流，Dynamo 可以 |
| **后端对接** | **高** — Inductor/torchair 已有成熟接口；GE 可通过 C API 直接绑定 |
| **维护成本** | **中高** — Graph IR 稳定；Dynamo 需跟随 CPython 演进；后端 bridge 需跟随 Inductor/GE 版本 |

---

## 七、风险矩阵

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **Dynamo 与 CPython 强耦合** | 每个 Python 版本都需适配 | 先用 symbolic tracing；Dynamo 阶段锁定 Python 3.11/3.12 |
| **Inductor 内部 API 不稳定** | candle.fx → torch.fx 转换层可能频繁 break | 锁定 PyTorch 版本；转换层做版本适配 |
| **GE C API 文档不足** | GE bridge 开发受阻 | 参考 torchair 源码逆向；走 ONNX 中间路径作为 fallback |
| **Op 覆盖长尾** | 某些模型 trace 失败 | 定义 core op set + decomposition 兜底；不支持的 op 触发 graph break |
| **性能回归** | 编译后反而变慢 | 提供 eager fallback；compile 默认 off |

---

## 八、结论与建议

**完全复刻 torch.compile 技术上可行**。candle 的 dispatch 系统、schema 注册、functionalize、meta backend 提供了比大多数框架更好的起点。

**核心建议**:

1. **Phase 1-3 全自研** (FX IR + Tracing + AOTAutograd) — 这些是 candle 必须拥有的核心能力
2. **Phase 4 走混合路线** — 定义自己的 backend protocol，同时提供 torch.fx bridge 复用 Inductor/torchair
3. **Phase 5 Dynamo fork** — 等 Phase 1-4 验证全链路后再做

**最关键的第一步**: 实现 `candle.fx` (Graph IR + Proxy + Tracer)，这是整个编译栈的基石。
