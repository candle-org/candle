## Agent tool

- **Tool:**
- **Model / environment** (optional):

## Linked issue(s)

- Closes #
- Related to #

## Change summary

<!-- What changed and why — 2-3 bullets max. -->

-

## Files changed / affected areas

-

## Validation evidence

### Commands run

<!-- Paste the exact commands you ran. Do not edit or abbreviate. -->

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest ...
pylint src/candle/ --rcfile=.github/pylint.conf
```

### Key output

<!-- Paste the relevant terminal output — do not paraphrase. -->

```text
...
```

## Test plan

- [ ] CPU tests
- [ ] MPS tests (if relevant)
- [ ] NPU tests (if relevant)
- [ ] Pylint clean
- [ ] Not run — explain why below

## Reviewer notes / remaining risks

-

## Checklist

- [ ] I disclosed the agent tool used
- [ ] I linked the relevant issue(s)
- [ ] I included a concise change summary
- [ ] I pasted actual validation output (not paraphrased)
- [ ] I did **not** bypass schema validation to make tests pass
- [ ] I did **not** introduce CPU fallback for GPU/NPU behavior
- [ ] I did **not** add unrelated changes
- [ ] I updated docs/comments only where needed
