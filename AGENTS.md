# Candle Agent Rules

This file defines mandatory development order and verification gates for all contributors and coding agents working on candle.

## Scope

- Applies to all changes under `src/candle/` and `tests/`.
- Priority: mechanism alignment with PyTorch behavior over adding new operator count.

## Non-Negotiable Order

For any new operator or API path in candle, follow this order:

1. Register schema first in `src/candle/_dispatch/schemas.py`.
2. Add or update contract tests in `tests/contract/`.
3. Register backend kernels (CPU/MPS/NPU/Autograd/Functionalize).
4. Add or update functional/tensor API exports.

Do not register a kernel before schema registration.

## Hard Invariant

`OpRegistry.register_kernel` enforces schema-first registration.

If schema is missing, registration must fail with:
- `schema must be registered before kernel registration for op ...`

Treat this as a design guardrail, not a temporary check.

## Required Tests Before PR

Every PR touching `src/candle/` must pass:

```bash
pytest tests/contract/ -v --tb=short
```

Recommended full gate:

```bash
pytest tests/cpu/ tests/contract/ -v --tb=short
```

On macOS with Apple Silicon, also run:

```bash
pytest tests/mps/ -v --tb=short
```

## PR Scope Rule

- Keep PRs mechanism-focused and small.
- Do not mix unrelated features in one PR.
- If you add a new operator family, include only required schema/tests/registration/API for that family.

## PyTorch Alignment Rule

- Match PyTorch dispatch semantics first (schema binding, error class, dispatch path), then optimize implementation.
- Error message wording can differ slightly unless a contract test requires exact match.

## Branch Rule

- Always develop on a feature branch from latest `main`.
- Rebase before opening PR to avoid conflicts.

```bash
git checkout main && git pull upstream main
git checkout -b feat/<name>
```

## GPU/NPU Kernel Fallback Policy (Mandatory)

This rule applies to all MPS, CUDA, and NPU backend code. Violations will block PR merge.

### NEVER fall back to CPU

- **NEVER** move computation from GPU/NPU to CPU (numpy) to work around a kernel bug or missing op.
- CPU fallback hides real problems, breaks device-placement guarantees, and will not be accepted in review.

### Composite workarounds ARE allowed

- When a native kernel (Metal shader, ACLNN large kernel, CUDA kernel) has a bug, you MAY reimplement the op as a **composite of smaller on-device ops** that already work.
- Every op in the composite must execute on the **same device** — no CPU round-trips.

### Preserve native kernel entry points

- Do NOT delete broken native kernel code. Keep it guarded so it can be re-enabled when the underlying platform is updated (CANN SDK, CUDA toolkit, macOS/Metal).
- Mark with: `# TODO: re-enable native kernel when <platform> fixes <issue>`

### Document every known issue

- Record every known kernel issue in `docs/known-kernel-issues.md`.
- Each entry must include: op name, backend, error description, composite workaround used, and the platform version that exhibits the bug.
- This document is the checklist for regression testing after platform upgrades.
