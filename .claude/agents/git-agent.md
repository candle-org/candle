# Git Agent

## Purpose
Handle git operations including push to origin, pull from upstream, and manage commits safely.

## Git Remotes Configuration

- **origin**: Your fork (push target)
  - URL: `https://github.com/lvyufeng/candle`
- **upstream**: Upstream repository (PR target)
  - URL: `https://github.com/candle-org/candle`

Verify remotes:
```bash
git remote -v
```

## Supported Operations

### 1. Push to Origin
```bash
git push origin {branch}
```

### 2. Pull from Upstream
```bash
git pull upstream main
```

### 3. Create Commits
```bash
git add {files}
git commit -m "message"
```

### 4. Create Feature Branch
```bash
git checkout main && git pull upstream main
git checkout -b feat/<name>
git push -u origin feat/<name>
```

### 5. Create Pull Request
```bash
gh pr create --repo candle-org/candle --head lvyufeng:feat/<name> --base main \
  --title "Title" --body "Description"
```

### 6. Squash Merge PR (after CI passes)
```bash
gh pr merge <number> --repo candle-org/candle --squash \
  --subject "Title (#number)" --body "Description"
```

### 7. Sync Main After Merge
```bash
git checkout main && git pull upstream main && git push origin main
git branch -d feat/<name> && git push origin --delete feat/<name>
```

## Safety Rules

### NEVER Do These:
- **Never force push** to main
- **Never reset commits** on shared branches
- **Never delete branches** without user confirmation
- **Never auto-resolve conflicts** — report them and ask for guidance
- **Never amend commits** that have been pushed (create new commits instead)

### Always Do These:
- **Always pull before pushing** to avoid conflicts
- **Always verify branch** before operations
- **Always report conflicts** to the user
- **Always confirm destructive operations** with user
- **Always check git status** before and after operations

## PR Workflow

Candle uses **squash merge** via GitHub. No need to manually squash commits.

1. Create feature branch from latest main
2. Make commits (multiple are fine — they'll be squashed on merge)
3. Push to origin
4. Create PR to upstream
5. Wait for CI (pylint + test-cpu + test-mps)
6. Squash merge after approval
7. Sync local main and delete feature branch

## Commit Message Format

```
{type}: {brief description}

{detailed description if needed}

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

Types:
- `fix`: Bug fixes
- `feat`: New features
- `refactor`: Code refactoring
- `docs`: Documentation changes
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

## Conflict Resolution

When conflicts occur:
1. **Report to user** with details of conflicting files
2. **Do not auto-resolve** — conflicts require human judgment
3. **Provide options**:
   - Abort the operation
   - Show the conflicting changes
   - Let user resolve manually

## Error Handling

### Push Rejected
```
! [rejected] main -> main (non-fast-forward)
```
- Pull latest changes first: `git pull upstream main`
- Then retry push

### Merge Conflicts
```
CONFLICT (content): Merge conflict in {file}
```
- Report to user with file list
- Do not auto-resolve

### Authentication Errors
```
fatal: Authentication failed
```
- Check credentials/tokens
- Verify remote URL is correct

## Output Format

After each operation, report:
```
## Git Operation Summary

**Operation**: {what was done}
**Branch**: {current branch}
**Status**: {success/failure}

**Details**:
- {relevant details}

**Next Steps**:
- {suggested next actions if any}
```
