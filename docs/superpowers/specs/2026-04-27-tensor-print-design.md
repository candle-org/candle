# Tensor print / repr deep alignment design

Date: 2026-04-27

## Context

Candle has already aligned most of the Tensor and Storage Python surface closely with local torch, and the current branch now passes the focused contract suites for tensor API behavior, storage behavior, and printing basics. However, the tensor printing system is still structurally far from torch:

- `src/candle/_tensor_str.py` is currently only a thin wrapper around `_printing.format_tensor()`.
- local `torch/_tensor_str.py` is a large, self-contained implementation that owns print options, formatting policy, summarization, scalar/vector formatting, and most repr-path decisions.
- Candle currently has two printing-related modules (`_tensor_str.py` and `_printing.py`) sharing responsibility in a way that drifts from torch and makes future alignment harder.

The user explicitly chose to push this area toward a near-complete torch-style alignment rather than continuing with incremental formatting patches.

## Goal

Make Candle’s tensor printing system structurally match `torch/_tensor_str.py` as closely as practical, with dense/main-path behavior aligned as tightly as possible and controlled fallbacks for runtime features Candle does not yet fully support.

This batch should improve long-term maintainability by making `_tensor_str.py` the canonical print implementation instead of leaving the main logic split across two files.

## Non-goals

This batch should **not** introduce unrelated runtime features just to satisfy obscure repr branches. If Candle lacks full runtime support for a special tensor category, the design should preserve stability through a local fallback in the printing path instead of widening implementation scope.

This batch should also **not** attempt to bundle unrelated Tensor or Storage surface work.

## Recommended approach

### 1. Make `src/candle/_tensor_str.py` the primary implementation file

`src/candle/_tensor_str.py` should become the canonical owner of tensor-printing behavior, mirroring the role of `torch/_tensor_str.py`.

It should absorb the torch-style main components:

- `PRINT_OPTS`
- `set_printoptions`
- `get_printoptions`
- `printoptions`
- `_Formatter`
- `_scalar_str`
- `_vector_str`
- `_tensor_str_with_formatter`
- `_tensor_str`
- `_str`

This is the key structural change. The goal is not just matching visible output, but matching the ownership model of the printing subsystem so future alignment work stays local to `_tensor_str.py`.

### 2. Reduce `_printing.py` to a compatibility layer

`src/candle/_printing.py` should no longer contain the primary formatting policy.

Instead, it should be reduced to a thin compatibility layer or wrapper that delegates into `_tensor_str.py`, so that:

- internal import sites do not break immediately,
- public print-option state exists in only one place,
- future repr changes do not require keeping `_printing.py` and `_tensor_str.py` in sync manually.

This avoids the current split-brain design where formatting state and formatting execution live in separate modules without matching torch’s structure.

### 3. Prioritize dense/main-path fidelity first, inside the full torch-style structure

The user asked for as much completeness as practical, but the safest way to achieve that is to keep the full torch-style structure while prioritizing the dense/main path first.

This means the first implementation should make the following categories as close to local torch as possible:

- 0-D, 1-D, and N-D dense tensors
- bool, int, float, and complex tensors
- summarization behavior (`threshold`, `edgeitems`)
- wrapping behavior (`linewidth`)
- precision and scientific notation (`precision`, `sci_mode`)
- suffixes for `dtype`, `device`, `requires_grad`, and `grad_fn`
- meta tensor printing where Candle already has a meaningful path

This is the highest-value, highest-coverage path and should be treated as the success baseline for the batch.

### 4. Adapt torch internals to Candle runtime deliberately

When porting torch’s `_tensor_str.py`, places that call directly into torch runtime helpers should be mapped deliberately onto Candle equivalents instead of left as raw copies.

The expected adaptation points include:

- `torch.no_grad()` → Candle’s equivalent grad-mode context
- `torch.masked_select`, `torch.isfinite`, `torch.ceil`, `torch.abs`, `torch.min`, `torch.max`, etc. → Candle equivalents where available
- Tensor suffix fields (`dtype`, `device`, `requires_grad`, `grad_fn`) → Candle’s existing tensor attributes
- CPU/device conversion helpers used only for formatting → Candle-safe equivalents

The main principle is to preserve torch’s formatting algorithm while only changing the runtime hook points needed to make it valid in Candle.

### 5. Use controlled fallbacks for unsupported special branches

Special printing branches should be handled in three tiers:

1. **Fully supported by Candle** — keep or port the torch branch directly.
2. **Partially supported by Candle** — preserve the torch branch structure but adapt the implementation to Candle’s current runtime.
3. **Not meaningfully supported yet** — use a local fallback only in that branch, ensuring printing remains stable and non-crashing.

This is important because trying to “fully support everything” inside the print rewrite could accidentally turn a printing task into a sparse/runtime/platform feature project. The printing system should degrade gracefully rather than forcing unrelated runtime work into scope.

## Expected user-visible outcome

After this change, Candle users should see behavior much closer to torch for:

- `repr(tensor)`
- `str(tensor)`
- `torch.set_printoptions(...)`
- `torch.get_printoptions()`
- `with torch.printoptions(...): ...`

In practice, dense tensor formatting should become far more torch-like in both output and corner-case handling, and the code should be organized around a single torch-style implementation file.

## Critical files

### Primary implementation
- `src/candle/_tensor_str.py`

### Compatibility / delegation layer
- `src/candle/_printing.py`

### Likely public export touchpoints
- `src/candle/__init__.py`

### Tests
- `tests/cpu/test_tensor_print.py`
- potentially targeted additions near other tensor contract tests if needed

## Verification plan

### Primary behavioral tests
Expand `tests/cpu/test_tensor_print.py` so that Candle output is compared directly against local torch output for representative dense cases:

- scalar tensors
- vectors
- matrices
- 3D+ tensors
- summarized tensors (`threshold` / `edgeitems`)
- line wrapping (`linewidth`)
- precision changes
- `sci_mode=True/False/None`
- bool / int / float / complex tensors
- dtype/device suffix cases
- `requires_grad=True`
- `grad_fn` suffix presence where applicable
- meta tensors

### Stability tests for special branches
For special tensor categories that Candle does not yet fully model like torch, the expectation in this batch is:

- printing should not crash,
- output should be deterministic,
- fallback behavior should be localized and explicit.

### Regression suites
After implementation, run at least:

- `conda run -n candle311 python -m pytest tests/cpu/test_tensor_print.py -q --tb=short`
- `conda run -n candle311 python -m pytest tests/common/test_c_core.py -q --tb=short`
- `conda run -n candle311 python -m pytest tests/cpu/test_tensor_api_contract.py -q --tb=short`
- `conda run -n candle311 python -m pytest tests/contract/test_storage_contract.py -q --tb=short`

These keep printing changes honest while also guarding against accidental regressions in the broader aligned Tensor surface.

## Risks and mitigation

### Risk: the port becomes a hidden runtime project
Mitigation: keep unsupported branches on controlled local fallbacks instead of expanding the task into sparse/runtime feature work.

### Risk: two print-option states remain in circulation
Mitigation: make `_tensor_str.py` the only source of truth and reduce `_printing.py` to delegation.

### Risk: output changes become hard to audit
Mitigation: compare Candle and local torch output directly in focused print tests, not just broad “contains substring” assertions.

## Recommendation

Proceed with a torch-structured rewrite of `src/candle/_tensor_str.py`, using dense/main-path fidelity as the execution priority and compatibility fallbacks for branches Candle cannot yet fully model. This gives the best long-term alignment outcome while keeping the batch bounded to the printing subsystem.