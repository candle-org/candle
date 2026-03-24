# GitHub Issue and PR Templates Design Spec

**Date:** 2026-03-24
**Status:** Draft

## Goal

Add a first-class GitHub contribution intake system for candle with:

1. structured **issue templates** for common contributor workflows
2. differentiated **human** vs **AI agent** submission paths
3. structured **PR templates** optimized for review quality and verification
4. alignment with candle's existing engineering expectations (roadmap-driven work, conventional-commit style, verification evidence, and backend safety constraints)

The design should improve issue triage quality and PR review quality without adding unnecessary process burden for human contributors.

---

## Background

The repository currently has no issue templates and no PR templates under `.github/`. As a result:

- issue quality depends entirely on contributor initiative
- maintainers do not get consistent reproduction info, environment details, or acceptance criteria
- AI-authored submissions are not explicitly distinguished from human-authored ones
- review expectations such as linked issue, validation commands, and changed-file summary are not consistently requested

At the same time, candle now has larger roadmap issues for Hugging Face ecosystem support, vLLM support, and Megatron compatibility. That makes it useful to add a **Task / sub-issue** template that can hang concrete implementation work off those umbrella roadmaps.

---

## Design Principles

1. **Separate contributor experiences**
   Human contributors should see short, friendly forms. AI agents should see more structured, machine-oriented forms with required evidence fields.

2. **Use GitHub-native mechanisms**
   Prefer GitHub issue forms (`.yml`) and built-in PR template selection rather than custom bots or CI parsing.

3. **Minimize burden on humans**
   Human templates should ask only for information that materially improves triage or review.

4. **Require stronger structure from AI submissions**
   AI-authored issues and PRs should explicitly disclose the agent/tool used, linked issue context, test evidence, and concise change summary.

5. **Support roadmap decomposition**
   Add a dedicated Task issue type so maintainers can turn large roadmap items into smaller tracked work units.

6. **Keep the system simple to maintain**
   Avoid dynamic automation or template logic that would require GitHub Apps, Actions, or external services.

---

## Approaches Considered

### Approach A — Single shared template with a source field

Use one issue template family and one PR template, with a dropdown for `Human` vs `AI Agent`.

**Pros**
- fewer files
- easier to maintain
- one canonical structure

**Cons**
- weaker separation between human and AI expectations
- human contributors still see AI-oriented fields or instructions
- harder to make AI templates significantly more structured

### Approach B — Separate human and AI templates

Create distinct issue templates for human contributors and AI contributors, plus separate PR templates for human and AI submissions.

**Pros**
- clear contributor-specific UX
- human forms stay concise
- AI forms can require structured fields without burdening humans
- easier for maintainers to reason about submission quality expectations

**Cons**
- more files to maintain
- some overlap in wording

### Approach C — Shared human-facing templates plus automation-based AI labeling

Keep one contributor-facing template set and rely on bot/CI logic or conventions to detect AI submissions.

**Pros**
- minimal template count
- less visible complexity in the GitHub UI

**Cons**
- detection is unreliable
- review quality still depends on submitter discipline
- requires more automation later

### Recommendation

Use **Approach B: Separate human and AI templates**.

This best matches the repository's needs: humans get a lighter-weight contribution path, while AI agents are held to a higher bar for structured context and verification evidence.

---

## Final Architecture

### Directory layout

```text
.github/
├── ISSUE_TEMPLATE/
│   ├── config.yml
│   ├── bug-report.yml
│   ├── feature-request.yml
│   ├── task.yml
│   ├── question.yml
│   ├── ai-bug-report.yml
│   ├── ai-feature-request.yml
│   └── ai-task.yml
└── PULL_REQUEST_TEMPLATE/
    ├── human.md
    └── ai.md
```

### Why this structure

- `ISSUE_TEMPLATE/config.yml` controls template chooser behavior and contributor guidance.
- Human issue forms cover the four requested public contribution modes:
  - Bug Report
  - Feature Request
  - Task (sub-issue)
  - Question
- AI issue forms focus on structured, high-signal reporting for the issue types most likely to be opened by agents:
  - AI Bug Report
  - AI Feature Request
  - AI Task
- PR templates are split into `human.md` and `ai.md` so reviewers can choose the right review contract for the submission source.

---

## Issue Template Design

### Human issue templates

#### 1. Bug Report

**Purpose:** collect enough information to reproduce and triage a defect.

**Fields**
- summary / what happened
- reproduction steps
- expected behavior
- environment
- logs or screenshots (optional)

**Tone:** natural language, low friction.

#### 2. Feature Request

**Purpose:** capture user value before implementation detail.

**Fields**
- problem or motivation
- proposed solution
- alternatives considered (optional)
- additional context

**Tone:** product/problem oriented.

#### 3. Task (sub-issue)

**Purpose:** break roadmap items or larger work into concrete, actionable tasks.

**Fields**
- parent issue / roadmap link
- task description
- scope boundaries
- acceptance criteria

**Tone:** actionable and maintainable.

#### 4. Question

**Purpose:** support usage questions without forcing them into bug/feature categories.

**Fields**
- question
- context
- what has already been tried

**Tone:** conversational and lightweight.

### AI issue templates

AI issue templates should explicitly optimize for maintainability and reviewability.

#### Common AI issue fields

- **Agent tool declaration**
  Example: Claude Code / Cursor / Copilot / other
- **Linked issue or roadmap item**
- **Structured summary**
- **Relevant files or subsystems**
- **Evidence** (commands, logs, reproduction details)

#### AI Bug Report

Adds:
- exact reproduction commands
- observed output / stack trace
- expected result
- likely root cause (optional but encouraged)

#### AI Feature Request

Adds:
- motivation
- proposed implementation direction
- affected modules
- linked roadmap issue

#### AI Task

Adds:
- parent issue
- task objective
- implementation boundaries
- acceptance criteria
- expected touched files (optional)

### Template chooser behavior

`config.yml` should group templates clearly so contributors understand whether they are choosing a human or AI path. It should also include a discussions/support link if available in the future.

---

## PR Template Design

### Human PR template

The human PR template should optimize for concise reviewer context.

**Sections**
- Summary
- Linked issue
- Test plan
- Checklist

**Checklist items**
- relevant tests were run
- linked issue exists if applicable
- docs updated if needed
- no unrelated changes included

This keeps the bar reasonable for human contributors while still improving review quality.

### AI PR template

The AI PR template should require stronger structure.

**Required sections**
- Agent tool declaration
- Linked issue(s)
- Change summary
- Files changed / affected areas
- Validation evidence
- Test plan
- Reviewer notes / remaining risks

**Validation evidence should explicitly request**
- command(s) run
- key output
- whether `pylint` was run
- whether relevant `pytest` commands were run

**AI checklist items**
- linked issue(s) included
- verification evidence included
- no unrelated changes included
- no schema validation bypass added
- no CPU fallback introduced for GPU/NPU backends
- comments/docs updated only where needed

This aligns the PR template with candle's repository constraints and reduces the chance of low-context AI submissions.

---

## Human vs AI Optimization Strategy

### Human optimization

The human path should optimize for:
- low friction
- discoverability
- clear wording
- fewer mandatory fields

Human contributors are more likely to provide nuance in freeform text, so the templates should not feel bureaucratic.

### AI optimization

The AI path should optimize for:
- structured fields
- explicit provenance
- verification evidence
- issue linkage
- concise impact summary

AI submissions often benefit from forcing key review context into explicit sections instead of assuming the PR description will be strong by default.

---

## Data Flow / Maintainer Workflow

### Issue intake

1. Contributor opens GitHub issue chooser.
2. Contributor selects either a human template or AI template.
3. Submitted issue arrives with consistent structure.
4. Maintainers can triage more quickly because the issue type is obvious and the required information is normalized.

### PR intake

1. Contributor opens a PR.
2. Contributor selects either the human or AI PR template.
3. Reviewers immediately see either the concise human format or the evidence-heavy AI format.
4. Review quality improves because linked issues, validation commands, and scope summary are consistently requested.

---

## Error Handling / Edge Cases

1. **AI contributors using human templates**
   This cannot be perfectly prevented with GitHub-native templates alone. The design accepts this limitation and optimizes for the happy path.

2. **Human contributors using AI templates**
   Harmless; they simply provide more detail.

3. **AI-generated PRs without explicit AI disclosure**
   Also cannot be perfectly prevented without automation. The design makes the expected path explicit but does not rely on enforcement.

4. **Too many template files**
   The selected set is intentionally small. Only three AI issue templates are added, not AI versions of every possible issue type.

5. **Question issues opened as bug reports**
   This is normal GitHub behavior. Adding a dedicated Question template reduces, but does not eliminate, misclassification.

---

## Testing Strategy

Testing here is structural rather than code-heavy.

### Manual validation

- verify GitHub recognizes all issue form files under `.github/ISSUE_TEMPLATE/`
- verify template chooser ordering and names are understandable
- verify `config.yml` renders properly
- verify PR template selection works with:
  - default `pull_request_template.md` behavior avoided
  - explicit `?template=human.md`
  - explicit `?template=ai.md`
- verify markdown rendering for checklists and fenced commands is clean

### Content validation

- ensure human forms are materially shorter than AI forms
- ensure AI PR template explicitly requests:
  - agent tool declaration
  - linked issue(s)
  - change summary
  - verification evidence
- ensure the Task issue template supports roadmap decomposition well

---

## Non-Goals

This design does **not** include:

- GitHub Actions enforcement of template completion
- automatic labeling based on human vs AI detection
- bots that reject incomplete issues or PRs
- issue forms for every niche scenario
- repository-wide CONTRIBUTING.md authoring

These can be added later if the template system proves valuable.

---

## Rollout Plan

### Phase 1

Add the template files only:
- issue forms
- PR templates
- chooser configuration

### Phase 2

Observe real usage and adjust:
- required vs optional fields
- wording
- whether AI and human split is working in practice
- whether labels or automation are worth adding later

---

## Success Criteria

The design is successful if:

1. maintainers receive more reproducible bug reports
2. roadmap work is more easily decomposed into Task issues
3. human contributors are not burdened by AI-oriented required fields
4. AI PRs consistently include linked issues, verification evidence, and concise change summaries
5. reviewers can understand PR intent faster without searching comments or commit history
