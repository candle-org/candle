# Candle → TorchTitan (NPU, No TorchTitan Modifications) Readiness Plan

> Date: 2026-03-12
> Goal: Run TorchTitan main on Ascend NPU with **no TorchTitan code changes** and **no external packages installed**.
> Assumptions: TorchTitan expects PyTorch/FSDP/DTensor/compile ecosystems; Candle must emulate these on top of NPU/HCCL.

---

## 1. Blocking Gaps (Import & Initialization)

These must be fixed before TorchTitan can even start:

- **FSDP availability**
  - `torch.distributed.fsdp` is currently a stub (`src/candle/distributed/fsdp/__init__.py`).
  - TorchTitan depends on FSDP (and often FSDP2/composable APIs). Without a real implementation, import fails.

- **DTensor / tensor parallel API**
  - `torch.distributed.tensor.parallel` and `torch.distributed._tensor` are stubs (`src/candle/distributed/tensor/parallel.py`, `src/candle/distributed/_tensor/__init__.py`).
  - TorchTitan imports these modules and expects behavior (not stubs).

- **torch.compile / torch._dynamo**
  - Candle’s `_dynamo` is a stub and `compile` is a no‑op (`src/candle/_dynamo.py`, `src/candle/compiler.py`).
  - TorchTitan often uses compile paths; stubs are incompatible.

---

## 2. Distributed & Parallelism (Full Training)

TorchTitan requires a full distributed surface beyond basic DDP:

- **FSDP2 / composable FSDP**
  - Candle has a **partial FSDP2 MVP** under `src/candle/distributed/_composable/fsdp/`, but it lacks full features (reshard strategies, state dict parity, mixed precision, etc.).

- **DeviceMesh**
  - Candle’s `DeviceMesh` is MVP‑only (1D mesh) in `src/candle/distributed/device_mesh.py`. TorchTitan may require multi‑dim meshes.

- **Collectives**
  - TorchTitan needs reliable `all_reduce`, `all_gather`, `reduce_scatter`, etc.
  - Candle has HCCL/Gloo, but lacks NCCL‑compatible behavior and integration with FSDP2/DTensor paths.

---

## 3. Checkpointing & State Dicts

- `torch.distributed.checkpoint.stateful` is stubbed (`src/candle/distributed/checkpoint/stateful.py`).
  - TorchTitan expects distributed checkpoint save/load, especially for multi‑node training.

---

## 4. Kernel / Op Coverage (Training Path)

TorchTitan’s transformer training path requires:

- `layer_norm` / `rms_norm`
- `softmax` / `log_softmax`
- `dropout`
- `gelu` / `silu`
- `matmul` / `bmm`
- `view` / `reshape` / `transpose`
- attention stack ops (scaled dot‑product attention, rotary embeddings)

Candle’s functional/autograd gaps (see prior gap report) will break TorchTitan forward/backward.

---

## 5. External Dependency Shims (No Packages Installed)

TorchTitan imports these by default. Candle must provide import shims:

- `torch.distributed.fsdp` (real implementation, not stub)
- `torch.distributed.tensor.parallel`
- `torch.distributed._tensor`
- Optional: `torch._dynamo` / `torch.compile` must be non‑stub or safely bypassed

---

## 6. Verification Gates

- **Import gate**: TorchTitan imports without errors.
- **Init gate**: FSDP/DeviceMesh init completes without stub errors.
- **Training gate**: one iteration forward/backward/optimizer step runs.
- **Checkpoint gate**: distributed save + reload works.
- **Multi‑node gate**: 2‑node pretrain can run without deadlock.

---

## 7. Phase Plan (P0 / P1 / P2)

### P0 (Make TorchTitan start)
- Replace `torch.distributed.fsdp` stub with a working FSDP2 path.
- Replace DTensor/tensor.parallel stubs with minimal but real APIs.
- Provide compile stubs that TorchTitan can bypass safely.

### P1 (Full training path)
- Complete FSDP2 features needed by TorchTitan (state dicts, reshard, mixed precision).
- Multi‑dim DeviceMesh support.
- Autograd completeness for transformer ops.

### P2 (Performance & Stability)
- Kernel fusion and host‑round‑trip elimination.
- Benchmark parity vs torch+npu.
- Long‑run stability for multi‑node training.

---

## Summary

TorchTitan main cannot run on Candle today because of **FSDP stubs, DTensor stubs, and compile stubs**. The plan above identifies the minimal set of APIs and kernel coverage required for full training with no TorchTitan modifications.
