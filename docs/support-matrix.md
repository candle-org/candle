# Candle 0.1 Support Matrix (NPU-First)

This matrix defines the effective support scope for Candle `0.1.x`.

## GA

- Ascend 910B single-card training core path:
  - tensor creation,
  - forward ops used by baseline training,
  - autograd backward,
  - optimizer step,
  - checkpoint save/load on critical path.
- CPU fallback path for development and CI baseline.

## Experimental

- Additional NPU ops outside the baseline training path.
- Transformers compatibility runner (`tests/run_test.py`) and related patches.
- Partial distributed collectives where behavior is validated only in limited scenarios.

## Not Supported In 0.1 Scope

- Distributed high-level features:
  - DeviceMesh,
  - Tensor Parallel,
  - FSDP and composable FSDP.
- True JIT/compile acceleration backends.
- Full ONNX export compatibility guarantees.

## Validation Gates

0.1 release quality is evaluated by:

- deterministic NPU golden training loop on Ascend 910B,
- checkpoint continuity on NPU (save/load then continue training),
- CPU fallback CI test baseline.
