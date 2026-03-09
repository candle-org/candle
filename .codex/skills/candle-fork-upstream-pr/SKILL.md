---
name: candle-fork-upstream-pr
description: "Enforce Candle contribution flow: never modify main directly, always create a worktree branch, push to fork lvyufeng/candle, and open PR to upstream candle-org/candle. Use for any coding, bugfix, refactor, or release-prep work in this repository."
---

# Candle Fork -> Upstream PR Workflow

## Required Outcome

- All implementation happens on a non-`main` branch in an isolated worktree.
- Branch is pushed to fork: `origin` (`lvyufeng/candle`).
- Pull request is opened against upstream: `candle-org/candle:main`.

## Mandatory Rules

- Never modify `main` directly.
- Never open the final PR to `lvyufeng/candle`.
- Always run verification tests before push/PR.

## Step 1: Safety Gate (Before Any Edit)

Run:

```bash
git rev-parse --abbrev-ref HEAD
git status --short
```

If branch is `main` and working tree is dirty:

1. Stop implementation.
2. Create a worktree branch immediately (Step 2).
3. Move in-progress edits into worktree.
4. Restore `main` to clean state for those files.

## Step 2: Create Isolated Worktree Branch

Preferred location in this repo: `.worktrees/`.

```bash
git worktree add .worktrees/<branch-name> -b <branch-name>
cd .worktrees/<branch-name>
```

Verify:

```bash
git rev-parse --abbrev-ref HEAD
```

Must not be `main`.

## Step 3: Implement and Verify

Use TDD discipline where feasible:

1. Add/adjust tests first.
2. Run targeted tests (expect fail).
3. Implement minimal fix.
4. Re-run targeted tests (expect pass).
5. Run affected regression suite.

Minimum release gate before PR:

- Targeted tests for changed behavior pass.
- Focused regression tests for adjacent modules pass.

## Step 4: Commit on Feature Branch

```bash
git add <files>
git commit -m "<type>: <summary>"
```

Use small, logical commits.

## Step 5: Push to Fork (origin)

```bash
git push -u origin <branch-name>
```

`origin` for this repo must be `https://github.com/lvyufeng/candle`.

## Step 6: Create PR to Upstream

Always create PR to upstream repo explicitly:

```bash
gh pr create \
  -R candle-org/candle \
  --base main \
  --head lvyufeng:<branch-name> \
  --title "<title>" \
  --body-file <body.md>
```

Do not omit `-R candle-org/candle`.

## Step 7: Validate PR Target and Link

Verify PR exists in upstream:

```bash
gh pr list -R candle-org/candle --head lvyufeng:<branch-name> --state open
```

Report upstream PR URL.

## Step 8: Optional Cleanup of Accidental Fork PR

If an accidental PR was opened in fork repo:

```bash
gh pr list -R lvyufeng/candle --head <branch-name> --state open
gh pr close -R lvyufeng/candle <pr-number>
```

## Red Flags

- Running code edits while on `main`.
- Pushing commits only to local branch without fork push.
- Creating PR without explicit upstream repo (`-R candle-org/candle`).
- Claiming completion without test output.

## Completion Checklist

- [ ] Branch is non-`main` and in `.worktrees/`.
- [ ] `main` has no accidental edits from this task.
- [ ] Tests for changed behavior pass.
- [ ] Branch pushed to `origin` (`lvyufeng/candle`).
- [ ] PR opened to `candle-org/candle:main`.
- [ ] Upstream PR URL confirmed.
