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
