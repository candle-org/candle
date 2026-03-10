# Transformers Compatibility Testing

Run HuggingFace transformers' official unit tests against candle to verify
`import candle as torch` drop-in compatibility.

## Quick Start

```bash
# 1. Install candle + compat deps
pip install -e .
pip install -r compat/transformers/requirements.txt

# 2. Run bert smoke test
USE_CANDLE=1 python compat/transformers/run.py --model bert -v --tb=short

# 3. Run all tier-1 models
USE_CANDLE=1 python compat/transformers/run.py --tier 1

# 4. Generate JSON report
USE_CANDLE=1 python compat/transformers/run.py --model bert --json-report /tmp/report.json

# 5. Print summary from existing report
python compat/transformers/run.py --summarize /tmp/report.json
```

## How It Works

```
compat/transformers/
├── run.py           # Entry script — clone, pytest, report
├── conftest.py      # Compatibility patches (version spoof, stubs, etc.)
├── models.yaml      # Model tiers + deselect/skip rules
├── xfail.yaml       # Known failures (pattern → reason)
└── requirements.txt # Compat-only dependencies
```

1. `run.py` shallow-clones transformers (pinned tag) to `compat/transformers/_transformers/` (gitignored)
2. It generates a bridge `conftest.py` inside the clone's `tests/` dir
3. That bridge loads `compat/transformers/conftest.py` which patches candle to look like torch
4. pytest runs with `USE_CANDLE=1` and `--continue-on-collection-errors`
5. Results are collected via `pytest-json-report` and summarized

### Patches Applied (conftest.py)

| Patch | Purpose |
|---|---|
| Version spoof | `candle.__version__` → `"2.5.0"` so `is_torch_available()` passes |
| Meta path finder | Any `import torch.X.Y` resolves to `candle.X.Y` or a lenient stub |
| Module stubs | `torch.backends.cuda`, `torch._dynamo`, `torch.compiler`, `torch.hub`, etc. |
| Dep check bypass | Prevents transformers from rejecting mismatched dep versions |
| Safetensors patch | Pure-Python safetensors loader (no C extension dep on torch) |
| torch_npu shim | Fake `torch_npu` module for NPU detection |
| xfail injection | Marks known failures from `xfail.yaml` as `xfail(strict=False)` |

## CLI Reference

```
python compat/transformers/run.py [options] [-- pytest-args]

Options:
  --model MODEL        Single model (e.g. bert, gpt2, t5)
  --tier N             Run tier 1..N models (default: 1)
  --setup-only         Clone transformers without running tests
  --json-report PATH   Write pytest-json-report output
  --summarize PATH     Print human-readable summary and exit
  --transformers-ref   Override git ref (default: from models.yaml)

Extra args after the options are forwarded to pytest:
  python compat/transformers/run.py --model bert -v --tb=short -x
```

## Model Tiers

Defined in `models.yaml`:

| Tier | Models | Purpose |
|---|---|---|
| 1 | bert, gpt2, t5 | Must-pass, release blocker |
| 2 | llama, roberta, distilbert, bart | Important models |
| 3 | opt, bloom, mistral, gemma, qwen2 | Extended coverage |

## Tracking Known Failures (xfail.yaml)

```yaml
_global:                          # applies to all models
  - pattern: "test_.*sdpa.*"
    reason: "SDPA optimization paths not implemented"

bert:                             # model-specific
  - pattern: "test_some_specific_test"
    reason: "candle issue #42"
```

### Workflow

1. Run tests → see failures in the report
2. For each failure, either fix candle or add to `xfail.yaml` with a reason
3. When a fix lands, the xfail becomes XPASS → remove from `xfail.yaml`
4. Over time, `xfail.yaml` shrinks = compatibility improves

## Interpreting the Report

```
=== Candle x Transformers Compatibility ===
Transformers: v4.47.0 | Candle: 0.1.0

Model          Total  Pass  Fail XFail XPass  Skip  Pass%
---------------------------------------------------------
bert              45    28     3    12     0     2  62.2%
---------------------------------------------------------
TOTAL             45    28     3    12     0     2  62.2%

Collection errors: 0
```

| Column | Meaning |
|---|---|
| Pass | Tests that passed |
| Fail | Tests that failed (not in xfail.yaml) — these need attention |
| XFail | Tests expected to fail (in xfail.yaml) and did fail |
| XPass | Tests expected to fail but passed — remove from xfail.yaml |
| Skip | Tests skipped by markers |
| Collection errors | Files that couldn't be imported at all |

## Filing Issues

When a compat test run reveals failures:

1. **Collection errors** → file an issue with label `compat/import-error`
   - These indicate missing torch API stubs in candle
2. **Test failures** → file an issue with label `compat/test-failure`
   - Include: model name, test name, error message, traceback
3. **Add to xfail.yaml** with the issue number as reason

Use the `/compat` slash command in Claude Code to automate this:
```
/compat bert     # run bert, summarize, suggest issues
```

## CI

The `compat-transformers.yaml` workflow runs:
- **Weekly** (Monday 04:00 UTC)
- **On manual dispatch** (with model/tier/ref inputs)

It does NOT block the main CI pipeline. Results are uploaded as artifacts.

## Development Tips

- **First run** clones transformers (~30s shallow clone). Subsequent runs reuse the clone.
- **`--setup-only`** is useful to pre-warm the clone before running tests.
- **Collection errors** often mean candle is missing a `torch.*` submodule or attribute.
  The meta path finder in `conftest.py` stubs most of these, but some need real
  implementations (e.g., `torch.library.Library`).
- **Don't modify candle source** to fix compat issues in conftest.py — the conftest
  is a shim layer. Real fixes should go into `src/candle/`.
