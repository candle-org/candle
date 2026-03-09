---
name: gpu-kernel-fallback-policy
description: "Mandatory policy for all GPU/NPU backend work: never fall back to CPU, use on-device composite workarounds, preserve native kernel entry points, and document issues in docs/known-kernel-issues.md."
---

# GPU/NPU Kernel Fallback Policy

## Scope

This policy applies to ALL changes under:
- `src/candle/_backends/mps/`
- `src/candle/_backends/cuda/`
- `src/candle/_backends/npu/`

## Mandatory Rules

### Rule 1: NEVER Fall Back to CPU

- **NEVER** move computation from GPU/NPU to CPU (numpy) to work around a kernel bug or missing op.
- CPU fallback hides real problems, breaks device-placement guarantees, and will not be accepted in review.
- This includes: converting tensors to numpy for computation, calling CPU ops from GPU kernels, or using `.cpu()` round-trips inside op implementations.

### Rule 2: Composite On-Device Workarounds ARE Allowed

When a native kernel (Metal shader, ACLNN large kernel, CUDA kernel) has a bug:

1. You MAY reimplement the op as a **composite of smaller on-device ops** that already work correctly.
2. Every op in the composite must execute on the **same device** — no CPU round-trips.
3. Example: if `aten.baddbmm` crashes on MPS Metal, reimplement using `bmm` + `add` + `mul` — all staying on MPS.

### Rule 3: Preserve Native Kernel Entry Points

- Do NOT delete broken native kernel code.
- Keep it in the codebase behind a clear guard so it can be re-enabled and tested when the underlying platform is updated.
- Mark with comment: `# TODO: re-enable native kernel when <platform> fixes <issue>`
- This enables automated regression testing after CANN SDK / CUDA toolkit / macOS updates.

### Rule 4: Document Every Known Kernel Issue

Record every known kernel issue in `docs/known-kernel-issues.md` with:

| Field | Description |
|-------|-------------|
| Op name | e.g., `aten.baddbmm` |
| Backend | `mps` / `cuda` / `npu` |
| Error description | What happens when the native kernel runs |
| Composite workaround | What smaller ops replace it |
| Platform version | e.g., `macOS 15.3 / Metal 3`, `CANN 8.0` |
| Status | `open` (workaround active) or `resolved` (native kernel re-enabled) |

This document is the checklist for regression testing after platform upgrades.

## Verification Checklist

Before opening a PR that touches GPU/NPU backend code:

- [ ] No CPU fallback introduced (grep for `.numpy()`, `.cpu()`, `np.` in device backend code)
- [ ] Broken native kernels are preserved, not deleted
- [ ] `docs/known-kernel-issues.md` updated if a new workaround was added
- [ ] All composite ops run on the same device as the original op
