## Summary

<!-- What does this PR do, and why? One or two sentences is fine. -->

-

## Linked issue

- Closes #
- Related to #

## Test plan

- [ ] Not run — explain why below
- [ ] `source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/ tests/contract/ -v --tb=short`
- [ ] `source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/mps/ -v --tb=short` (if MPS-related)
- [ ] `pylint src/candle/ --rcfile=.github/pylint.conf`

## Checklist

- [ ] I linked the relevant issue(s), if any
- [ ] I ran relevant tests or explained why I did not
- [ ] I updated docs/comments only where needed
- [ ] This PR does not include unrelated changes
