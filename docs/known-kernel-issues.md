# Known Kernel Issues

This document tracks native kernel bugs and their on-device composite workarounds across GPU/NPU backends. It serves as the regression testing checklist after platform upgrades (CANN SDK, CUDA toolkit, macOS/Metal).

## How to Use This Document

- **After a platform upgrade**: Test each `open` entry by re-enabling the native kernel and running the associated test. If the native kernel works, mark the entry as `resolved` and remove the composite workaround.
- **When adding a workaround**: Add a new entry here with all required fields.

## Issue Table

| Op | Backend | Error Description | Composite Workaround | Platform Version | Status |
|----|---------|-------------------|----------------------|------------------|--------|
| — | — | No known issues yet | — | — | — |

<!--
Entry template:
| `aten.op_name` | mps/cuda/npu | Brief error description | `op_a` + `op_b` composite | macOS XX.X / CANN X.X / CUDA XX.X | open/resolved |
-->
