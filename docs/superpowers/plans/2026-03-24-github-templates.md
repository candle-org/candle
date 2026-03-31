# GitHub Templates Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GitHub-native issue and PR templates that distinguish human contributors from AI agents while keeping human submissions low-friction and AI submissions evidence-driven.

**Architecture:** Create issue forms under `.github/ISSUE_TEMPLATE/` and separate PR templates under `.github/PULL_REQUEST_TEMPLATE/`. Human templates stay concise and contributor-friendly; AI templates require explicit tool provenance, linked issues, change summaries, and validation evidence.

**Tech Stack:** GitHub issue forms (YAML), Markdown PR templates, git worktrees, gh CLI, Python stdlib for lightweight local validation

---

## File Map

### New files
- `.github/ISSUE_TEMPLATE/config.yml` — issue chooser behavior and blank-issue policy
- `.github/ISSUE_TEMPLATE/bug-report.yml` — human bug report form
- `.github/ISSUE_TEMPLATE/feature-request.yml` — human feature request form
- `.github/ISSUE_TEMPLATE/task.yml` — human roadmap/sub-issue task form
- `.github/ISSUE_TEMPLATE/question.yml` — human question form
- `.github/ISSUE_TEMPLATE/ai-bug-report.yml` — AI bug report form
- `.github/ISSUE_TEMPLATE/ai-feature-request.yml` — AI feature request form
- `.github/ISSUE_TEMPLATE/ai-task.yml` — AI task form
- `.github/PULL_REQUEST_TEMPLATE/human.md` — concise human PR template
- `.github/PULL_REQUEST_TEMPLATE/ai.md` — structured AI PR template

### Existing files to inspect during implementation
- `CLAUDE.md` — repository constraints that AI PR checklist should reinforce
- `.github/workflows/ci.yaml` — current CI jobs to reference in test-plan wording if needed
- `docs/superpowers/specs/2026-03-24-github-templates-design.md` — approved design spec

---

## Chunk 1: Issue template infrastructure and human templates

### Task 1: Create issue chooser config and core human forms

**Files:**
- Create: `.github/ISSUE_TEMPLATE/config.yml`
- Create: `.github/ISSUE_TEMPLATE/bug-report.yml`
- Create: `.github/ISSUE_TEMPLATE/feature-request.yml`
- Test: `.github/ISSUE_TEMPLATE/config.yml`
- Test: `.github/ISSUE_TEMPLATE/bug-report.yml`
- Test: `.github/ISSUE_TEMPLATE/feature-request.yml`

- [ ] **Step 1: Confirm the template directories do not already exist**

Run:
```bash
ls .github/ISSUE_TEMPLATE .github/PULL_REQUEST_TEMPLATE
```
Expected: `No such file or directory` for both paths on a fresh checkout.

- [ ] **Step 2: Create the template directories**

Run:
```bash
mkdir -p .github/ISSUE_TEMPLATE .github/PULL_REQUEST_TEMPLATE
```

- [ ] **Step 3: Write `.github/ISSUE_TEMPLATE/config.yml`**

```yml
blank_issues_enabled: false
```

- [ ] **Step 4: Write `.github/ISSUE_TEMPLATE/bug-report.yml`**

```yml
name: Human: Bug Report
description: Report a reproducible bug in candle
title: "[Bug]: "
body:
  - type: markdown
    attributes:
      value: |
        Thanks for filing a bug report. Please share the smallest reproducible case you can.

  - type: textarea
    id: summary
    attributes:
      label: What happened?
      description: Describe the incorrect behavior.
      placeholder: candle returned X, but expected Y
    validations:
      required: true

  - type: textarea
    id: reproduce
    attributes:
      label: How can we reproduce it?
      description: Include exact code, commands, or inputs.
      placeholder: |
        1. Run ...
        2. Call ...
        3. Observe ...
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: What did you expect to happen?
      placeholder: Describe the correct or PyTorch-compatible behavior
    validations:
      required: true

  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: OS, Python version, backend/device, and candle commit if known.
      placeholder: |
        - OS:
        - Python:
        - Device/backend:
        - candle commit:
    validations:
      required: true

  - type: textarea
    id: logs
    attributes:
      label: Logs, tracebacks, or screenshots
      description: Paste any relevant output.
      render: shell
    validations:
      required: false
```

- [ ] **Step 5: Write `.github/ISSUE_TEMPLATE/feature-request.yml`**

```yml
name: Human: Feature Request
description: Propose a new feature or API improvement
title: "[Feature]: "
body:
  - type: markdown
    attributes:
      value: |
        Please focus on the user problem first and implementation details second.

  - type: textarea
    id: problem
    attributes:
      label: What problem are you trying to solve?
      placeholder: I need candle to support ... because ...
    validations:
      required: true

  - type: textarea
    id: proposal
    attributes:
      label: What would you like to see?
      placeholder: Describe the API, workflow, or behavior you want
    validations:
      required: true

  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives considered
      description: Optional. Share workarounds or other ideas you considered.
    validations:
      required: false

  - type: textarea
    id: context
    attributes:
      label: Additional context
      description: Links to prior issues, benchmarks, docs, or model compatibility needs.
    validations:
      required: false
```

- [ ] **Step 6: Run lightweight structural validation for the three files**

Run:
```bash
python - <<'PY'
from pathlib import Path
paths = [
    Path('.github/ISSUE_TEMPLATE/config.yml'),
    Path('.github/ISSUE_TEMPLATE/bug-report.yml'),
    Path('.github/ISSUE_TEMPLATE/feature-request.yml'),
]
for path in paths:
    text = path.read_text()
    assert text.strip(), f"{path} is empty"
    if path.name == 'config.yml':
        assert 'blank_issues_enabled:' in text
    else:
        assert text.startswith('name:'), f"{path} missing name header"
        assert 'body:' in text, f"{path} missing body section"
print('validated', len(paths), 'files')
PY
```
Expected: `validated 3 files`

- [ ] **Step 7: Commit the first issue template batch**

```bash
git add .github/ISSUE_TEMPLATE/config.yml .github/ISSUE_TEMPLATE/bug-report.yml .github/ISSUE_TEMPLATE/feature-request.yml
git commit -m "docs(github): add human bug and feature templates"
```

### Task 2: Add human task and question forms

**Files:**
- Create: `.github/ISSUE_TEMPLATE/task.yml`
- Create: `.github/ISSUE_TEMPLATE/question.yml`
- Test: `.github/ISSUE_TEMPLATE/task.yml`
- Test: `.github/ISSUE_TEMPLATE/question.yml`

- [ ] **Step 1: Write `.github/ISSUE_TEMPLATE/task.yml`**

```yml
name: Human: Task (Sub-issue)
description: Break a roadmap item or larger issue into a concrete task
title: "[Task]: "
body:
  - type: markdown
    attributes:
      value: |
        Use this template for concrete work items that hang off a larger roadmap issue or parent task.

  - type: input
    id: parent
    attributes:
      label: Parent issue or roadmap item
      description: Enter an issue number or full GitHub URL.
      placeholder: "#204"
    validations:
      required: true

  - type: textarea
    id: objective
    attributes:
      label: Task objective
      placeholder: Implement ... so that ...
    validations:
      required: true

  - type: textarea
    id: scope
    attributes:
      label: Scope boundaries
      description: Describe what is in scope and what should explicitly stay out of scope.
      placeholder: |
        In scope:
        - ...

        Out of scope:
        - ...
    validations:
      required: true

  - type: textarea
    id: acceptance
    attributes:
      label: Acceptance criteria
      placeholder: |
        - [ ] ...
        - [ ] ...
    validations:
      required: true
```

- [ ] **Step 2: Write `.github/ISSUE_TEMPLATE/question.yml`**

```yml
name: Human: Question
description: Ask a usage or project question
title: "[Question]: "
body:
  - type: markdown
    attributes:
      value: |
        Please share enough context for someone else to answer without guessing.

  - type: textarea
    id: question
    attributes:
      label: What is your question?
      placeholder: I'm trying to understand ...
    validations:
      required: true

  - type: textarea
    id: context
    attributes:
      label: Relevant context
      description: Share the API, backend, model, or workflow involved.
    validations:
      required: true

  - type: textarea
    id: tried
    attributes:
      label: What have you already tried?
      description: Optional. Mention docs, code, or experiments you checked first.
    validations:
      required: false
```

- [ ] **Step 3: Run structural validation for the new human issue forms**

Run:
```bash
python - <<'PY'
from pathlib import Path
paths = [
    Path('.github/ISSUE_TEMPLATE/task.yml'),
    Path('.github/ISSUE_TEMPLATE/question.yml'),
]
for path in paths:
    text = path.read_text()
    assert text.startswith('name:'), f"{path} missing name header"
    assert 'body:' in text, f"{path} missing body section"
    assert 'required: true' in text, f"{path} missing required fields"
print('validated', len(paths), 'files')
PY
```
Expected: `validated 2 files`

- [ ] **Step 4: Commit the remaining human issue forms**

```bash
git add .github/ISSUE_TEMPLATE/task.yml .github/ISSUE_TEMPLATE/question.yml
git commit -m "docs(github): add human task and question templates"
```

## Chunk 2: AI issue forms and PR templates

### Task 3: Add AI issue forms with structured provenance and linkage

**Files:**
- Create: `.github/ISSUE_TEMPLATE/ai-bug-report.yml`
- Create: `.github/ISSUE_TEMPLATE/ai-feature-request.yml`
- Create: `.github/ISSUE_TEMPLATE/ai-task.yml`
- Test: `.github/ISSUE_TEMPLATE/ai-bug-report.yml`
- Test: `.github/ISSUE_TEMPLATE/ai-feature-request.yml`
- Test: `.github/ISSUE_TEMPLATE/ai-task.yml`

- [ ] **Step 1: Write `.github/ISSUE_TEMPLATE/ai-bug-report.yml`**

```yml
name: AI: Bug Report
description: Structured bug report generated or filed by an AI agent
title: "[AI Bug]: "
body:
  - type: dropdown
    id: tool
    attributes:
      label: Agent tool
      options:
        - Claude Code
        - Cursor
        - GitHub Copilot
        - Other
    validations:
      required: true

  - type: input
    id: linked
    attributes:
      label: Related issue or roadmap item
      placeholder: "#205"
    validations:
      required: false

  - type: textarea
    id: summary
    attributes:
      label: Structured summary
      placeholder: Describe the bug in 2-5 sentences
    validations:
      required: true

  - type: textarea
    id: files
    attributes:
      label: Relevant files or subsystems
      placeholder: |
        - src/candle/...
        - tests/...
    validations:
      required: true

  - type: textarea
    id: reproduce
    attributes:
      label: Exact reproduction commands
      render: shell
      placeholder: |
        python -m pytest ...
    validations:
      required: true

  - type: textarea
    id: observed
    attributes:
      label: Observed output
      render: shell
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: Expected behavior
    validations:
      required: true

  - type: textarea
    id: root_cause
    attributes:
      label: Likely root cause
      description: Optional but encouraged.
    validations:
      required: false
```

- [ ] **Step 2: Write `.github/ISSUE_TEMPLATE/ai-feature-request.yml`**

```yml
name: AI: Feature Request
description: Structured feature proposal generated or filed by an AI agent
title: "[AI Feature]: "
body:
  - type: dropdown
    id: tool
    attributes:
      label: Agent tool
      options:
        - Claude Code
        - Cursor
        - GitHub Copilot
        - Other
    validations:
      required: true

  - type: input
    id: linked
    attributes:
      label: Linked roadmap issue
      placeholder: "#204"
    validations:
      required: true

  - type: textarea
    id: motivation
    attributes:
      label: Motivation
      placeholder: Explain the problem this proposal solves
    validations:
      required: true

  - type: textarea
    id: proposal
    attributes:
      label: Proposed implementation direction
      placeholder: Summarize the intended approach without pasting full code
    validations:
      required: true

  - type: textarea
    id: areas
    attributes:
      label: Affected files or subsystems
      placeholder: |
        - src/candle/...
        - tests/...
        - compat/...
    validations:
      required: true

  - type: textarea
    id: success
    attributes:
      label: Success criteria
      placeholder: |
        - [ ] ...
        - [ ] ...
    validations:
      required: true
```

- [ ] **Step 3: Write `.github/ISSUE_TEMPLATE/ai-task.yml`**

```yml
name: AI: Task
description: Structured implementation task created by an AI agent
title: "[AI Task]: "
body:
  - type: dropdown
    id: tool
    attributes:
      label: Agent tool
      options:
        - Claude Code
        - Cursor
        - GitHub Copilot
        - Other
    validations:
      required: true

  - type: input
    id: parent
    attributes:
      label: Parent issue or roadmap item
      placeholder: "#206"
    validations:
      required: true

  - type: textarea
    id: objective
    attributes:
      label: Task objective
      placeholder: Describe the concrete deliverable
    validations:
      required: true

  - type: textarea
    id: scope
    attributes:
      label: Scope boundaries
      placeholder: |
        In scope:
        - ...

        Out of scope:
        - ...
    validations:
      required: true

  - type: textarea
    id: acceptance
    attributes:
      label: Acceptance criteria
      placeholder: |
        - [ ] ...
        - [ ] ...
    validations:
      required: true

  - type: textarea
    id: files
    attributes:
      label: Expected touched files
      placeholder: |
        - .github/...
        - src/candle/...
    validations:
      required: false
```

- [ ] **Step 4: Validate the AI issue form files**

Run:
```bash
python - <<'PY'
from pathlib import Path
paths = [
    Path('.github/ISSUE_TEMPLATE/ai-bug-report.yml'),
    Path('.github/ISSUE_TEMPLATE/ai-feature-request.yml'),
    Path('.github/ISSUE_TEMPLATE/ai-task.yml'),
]
for path in paths:
    text = path.read_text()
    assert text.startswith('name:'), f"{path} missing name header"
    assert 'Agent tool' in text, f"{path} missing agent disclosure"
    assert 'body:' in text, f"{path} missing body section"
print('validated', len(paths), 'files')
PY
```
Expected: `validated 3 files`

- [ ] **Step 5: Commit the AI issue forms**

```bash
git add .github/ISSUE_TEMPLATE/ai-bug-report.yml .github/ISSUE_TEMPLATE/ai-feature-request.yml .github/ISSUE_TEMPLATE/ai-task.yml
git commit -m "docs(github): add AI issue templates"
```

### Task 4: Add separate human and AI PR templates

**Files:**
- Create: `.github/PULL_REQUEST_TEMPLATE/human.md`
- Create: `.github/PULL_REQUEST_TEMPLATE/ai.md`
- Test: `.github/PULL_REQUEST_TEMPLATE/human.md`
- Test: `.github/PULL_REQUEST_TEMPLATE/ai.md`

- [ ] **Step 1: Write `.github/PULL_REQUEST_TEMPLATE/human.md`**

```md
## Summary

-

## Linked issue

- Closes #
- Related to #

## Test plan

- [ ] Not run (explain why)
- [ ] `source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest ...`
- [ ] `pylint src/candle/ --rcfile=.github/pylint.conf`

## Checklist

- [ ] I linked the relevant issue(s), if any
- [ ] I ran relevant tests or explained why I did not
- [ ] I updated docs/comments only where needed
- [ ] This PR does not include unrelated changes
```

- [ ] **Step 2: Write `.github/PULL_REQUEST_TEMPLATE/ai.md`**

```md
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
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest ...
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
```

- [ ] **Step 3: Validate the PR templates contain the required sections**

Run:
```bash
python - <<'PY'
from pathlib import Path
required = {
    Path('.github/PULL_REQUEST_TEMPLATE/human.md'): [
        '## Summary',
        '## Linked issue',
        '## Test plan',
        '## Checklist',
    ],
    Path('.github/PULL_REQUEST_TEMPLATE/ai.md'): [
        '## Agent tool',
        '## Linked issue(s)',
        '## Change summary',
        '## Validation evidence',
        '## Checklist',
    ],
}
for path, sections in required.items():
    text = path.read_text()
    for section in sections:
        assert section in text, f"{path} missing {section}"
print('validated', len(required), 'files')
PY
```
Expected: `validated 2 files`

- [ ] **Step 4: Commit the PR templates**

```bash
git add .github/PULL_REQUEST_TEMPLATE/human.md .github/PULL_REQUEST_TEMPLATE/ai.md
git commit -m "docs(github): add human and AI PR templates"
```

## Chunk 3: Final validation and polish

### Task 5: Validate the full template set and polish wording

**Files:**
- Modify: `.github/ISSUE_TEMPLATE/config.yml` (only if chooser wording needs a final tweak)
- Modify: `.github/ISSUE_TEMPLATE/*.yml` (only if validation or review reveals wording problems)
- Modify: `.github/PULL_REQUEST_TEMPLATE/*.md` (only if validation or review reveals wording problems)
- Test: `.github/ISSUE_TEMPLATE/*.yml`
- Test: `.github/PULL_REQUEST_TEMPLATE/*.md`

- [ ] **Step 1: Verify only the expected template files were added under `.github/`**

Run:
```bash
git status --short .github
```
Expected: only the new template files appear.

- [ ] **Step 2: Run whitespace and formatting validation**

Run:
```bash
git diff --check
```
Expected: no output.

- [ ] **Step 3: Run a full structural validation pass across all templates**

Run:
```bash
python - <<'PY'
from pathlib import Path
issue_files = sorted(Path('.github/ISSUE_TEMPLATE').glob('*.yml'))
pr_files = sorted(Path('.github/PULL_REQUEST_TEMPLATE').glob('*.md'))
assert len(issue_files) == 7, issue_files
assert len(pr_files) == 2, pr_files
for path in issue_files:
    text = path.read_text()
    assert text.strip(), f"{path} is empty"
    if path.name == 'config.yml':
        assert 'blank_issues_enabled:' in text
    else:
        assert 'name:' in text and 'body:' in text, f"{path} missing issue-form structure"
for path in pr_files:
    text = path.read_text()
    assert '## Checklist' in text, f"{path} missing checklist"
print('validated', len(issue_files), 'issue files and', len(pr_files), 'PR files')
PY
```
Expected: `validated 7 issue files and 2 PR files`

- [ ] **Step 4: Do a manual content pass focused on UX differences**

Check manually:
- human forms are clearly shorter than AI forms
- AI forms require agent disclosure and linked issue context
- AI PR template requires validation evidence
- Task templates are suitable for roadmap decomposition

- [ ] **Step 5: Make any final wording-only edits revealed by the validation pass**

Apply only wording and required/optional-field tweaks. Do not add automation, CI, or extra template families.

- [ ] **Step 6: Commit the final polish**

```bash
git add .github/ISSUE_TEMPLATE .github/PULL_REQUEST_TEMPLATE
git commit -m "docs(github): polish contribution templates"
```

---

## Notes for the implementing agent

- Keep the scope limited to template files only.
- Do not add GitHub Actions enforcement, bots, or auto-labeling.
- Do not add `CONTRIBUTING.md` in this implementation.
- Follow the approved design spec exactly unless a GitHub platform limitation forces a small compatibility adjustment; if that happens, document the reason in the PR description.
- If GitHub PR-template selection behavior needs a fallback after manual validation, surface that as a follow-up issue instead of silently expanding scope.
