## Agent tool

- Tool:
- Model / environment (optional):

## Linked issue(s)

- Closes #
- Related to #

## Change summary

-

## Files changed / affected areas

-

## Validation evidence

### Commands run

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n mindnlp python -m pytest ...
pylint src/candle/ --rcfile=.github/pylint.conf
```

### Key output

```text
Paste the relevant passing output here.
```

## Test plan

- [ ] CPU tests
- [ ] MPS tests (if relevant)
- [ ] NPU tests (if relevant)
- [ ] Pylint
- [ ] Not run (explain why)

## Reviewer notes / remaining risks

-

## Checklist

- [ ] I disclosed the agent tool used for this submission
- [ ] I linked the relevant issue(s)
- [ ] I included a concise change summary
- [ ] I included validation evidence and commands run
- [ ] I did not add unrelated changes
- [ ] I did not bypass schema validation to make tests pass
- [ ] I did not introduce CPU fallback for GPU/NPU behavior
- [ ] I only updated docs/comments where needed
