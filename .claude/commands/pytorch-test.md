Run PyTorch official tests against candle and report results.

## Usage
/pytorch-test                     # run tier1 mechanism tests
/pytorch-test test_tensor.py      # run single file
/pytorch-test --tier gpu:1        # CUDA/NPU tests

## Instructions
1. Parse $ARGUMENTS:
   - Empty -> use `--tier mechanism:1`
   - Starts with `test_` -> use `--file <arg>`
   - Starts with `--` -> pass through directly
2. Run: `USE_CANDLE=1 python compat/pytorch/run.py $ARGS --json-report /tmp/pt-report.json -v --tb=short`
3. Summarize: `python compat/pytorch/run.py --summarize /tmp/pt-report.json`
4. For each collection error or failure:
   - Identify the root cause (missing module, missing attribute, wrong result)
   - Map to the relevant candle source file
   - Suggest which label to use
5. Present a summary table and list of actionable issues

## Issue Labels
- `pytorch-compat/import-error` — missing torch.* module/attribute in candle
- `pytorch-compat/missing-op` — operator not implemented
- `pytorch-compat/wrong-result` — computation produces wrong result
- `pytorch-compat/testing-infra` — candle.testing._internal missing API
- `pytorch-compat/device-mapping` — NPU device mapping issue
