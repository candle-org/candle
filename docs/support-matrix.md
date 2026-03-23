# Candle 0.1 Support Matrix (NPU-First)

This matrix defines the effective support scope for Candle `0.1.x`.

For NPU installation and runtime prerequisites, see [install-npu.md](install-npu.md).

## GA

- Ascend 910B single-card training core path on a host where the newest installed CANN toolkit is exposed through `/usr/local/Ascend/ascend-toolkit/latest` and sourced into the shell.
  - tensor creation,
  - forward ops used by baseline training,
  - autograd backward,
  - optimizer step,
  - checkpoint save/load on the critical path.
- CPU backend for development and CI baseline.

## Experimental

- Additional NPU ops outside the baseline training path.
- Transformers compatibility runner (`tests/run_test.py`) and related patches.
- Distributed/HCCL collectives: all core collectives implemented; multi-card
  scenarios validated via the `tests/distributed/test_hccl_*_multicard.py` suite
  (requires 2+ Ascend cards).
- **Parallel training stack** (single-process contracts fully tested; multi-card
  requires HCCL hardware):
  - `DeviceMesh` / `init_device_mesh`
  - `DTensor` with `Shard` / `Replicate` / `Partial` placements;
    Cython fastpath active when the compiled extension is present.
  - Tensor Parallel: `parallelize_module`, `ColwiseParallel`, `RowwiseParallel`
    for `nn.Linear`; `SequenceParallel` deferred.
  - Composable FSDP2: `fully_shard` with shard bookkeeping in Cython.
  - DDP: bucket reducer with Cython fastpath and comm-hook support.
  - Pipeline parallel: `PipelineStage`, `ScheduleGPipe`, `TensorChunkSpec`,
    `split_microbatches`; minimal manual schedule only.
  - Shard-aware distributed checkpoint (`save` / `load` with per-rank shards).

## Not Supported In 0.1 Scope

- CUDA / NCCL backends.
- `SequenceParallel`, `PrepareModuleInput`, `PrepareModuleOutput`.
- `FullyShardedDataParallel` (legacy FSDP1 wrapper).
- `transformer_auto_wrap_policy`.
- 2D mesh tensor parallel.
- True JIT/compile acceleration backends.
- Full ONNX export compatibility guarantees.

## Validation Gates

0.1 release quality is evaluated by:

- local Ascend 910B NPU gate set:
  - `tests/npu/test_npu_golden_training_loop.py`,
  - `tests/npu/test_npu_training_checkpoint_continuity.py`,
  - `tests/npu/test_mul_scalar_regression_npu.py`,
  - `tests/npu/test_no_cpu_fallback_npu.py`.
- CPU + contract CI baseline.
- MPS CI baseline for cross-backend regression visibility.

## Runtime Rule

- NPU execution paths must remain on NPU.
- Runtime fallback from NPU kernels to CPU is not allowed.
