Run transformers compatibility tests against candle and report results.

## Usage

```
/compat              # run all tier-1 models
/compat bert         # run a specific model
/compat --tier 2     # run tier 1+2
```

## Instructions

1. Parse arguments from `$ARGUMENTS`:
   - If empty → `--tier 1`
   - If a model name (e.g. `bert`, `gpt2`) → `--model <name>`
   - Otherwise pass through as-is

2. Run the compat test suite:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && \
conda run -n candle env USE_CANDLE=1 \
  python compat/transformers/run.py $PARSED_ARGS \
  --json-report /tmp/compat-report.json \
  -v --tb=short
```

3. Summarize the results:

```bash
python compat/transformers/run.py --summarize /tmp/compat-report.json
```

4. For each **collection error**, report:
   - The file that failed to import
   - The missing module / attribute
   - Suggest which candle source file needs the fix

5. For each **test failure** (not xfail), report:
   - Test name and error type
   - Root cause category: missing op, wrong result, missing attribute, etc.
   - Which candle source file is likely responsible

6. Suggest next steps:
   - New entries for `compat/transformers/xfail.yaml` (with reason)
   - Issues to file (title + label)

## Labels for Issues

- `compat/import-error` — missing torch.* module or attribute
- `compat/test-failure` — test runs but produces wrong result
- `compat/stub-needed` — candle needs a new stub or API surface
