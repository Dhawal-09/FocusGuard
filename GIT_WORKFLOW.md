# FocusGuard Git Workflow

This document defines the Git branching, commit, pull request, merge, release, and debugging practices for FocusGuard.

The goal is twofold:

1. Maintain a clean, production-style repository.
2. Give the developer hands-on experience with professional Git workflows.

This document is the source of truth for how Git is used on this project. `AGENTS.md` requires Claude Code and other coding agents to follow it.

---

## 1. Principles

- `main` is always the stable branch.
- No unfinished features should be merged into `main`.
- No direct development should normally happen on `main`.
- Every meaningful change should happen on an appropriate branch.
- Features should be independently reviewable.
- Commits should be small and logically grouped.
- Tests should accompany implementation changes.
- PRs should be reviewed before merging.
- Do not commit secrets.
- Do not commit generated artifacts.
- Do not commit `.venv`.
- Do not commit model weights.
- Do not commit logs or caches unless explicitly required.
- History should remain understandable.
- Prefer reversible changes.
- Never rewrite shared `main` history.

---

## 2. Branch Strategy

### `main`
Purpose: stable, reviewed, working code.

### `feature/*`
For new functionality.

Examples:
- `feature/phase-1-camera-manager`
- `feature/phase-2-yolo-detection`
- `feature/phone-detection`
- `feature/focus-state-engine`

### `fix/*`
For normal bug fixes.

Examples:
- `fix/camera-initialization`
- `fix/phone-detection-threshold`

### `refactor/*`
For restructuring code without changing intended behavior.

Examples:
- `refactor/camera-manager`
- `refactor/state-machine`

### `docs/*`
For documentation-only changes.

Examples:
- `docs/update-readme`
- `docs/git-workflow`

### `test/*`
For test-focused changes.

Examples:
- `test/camera-edge-cases`
- `test/state-manager`

### `chore/*`
For tooling, dependencies, configuration, CI, formatting, etc.

Examples:
- `chore/update-dependencies`
- `chore/setup-ci`

### `hotfix/*`
For urgent production-style fixes.

Examples:
- `hotfix/crash-on-startup`

---

## 3. Branch Naming Convention

Use:

```
<type>/<short-kebab-case-description>
```

Rules:
- lowercase
- kebab-case
- concise
- descriptive
- no spaces
- avoid vague names

Good: `feature/phase-1-camera-manager`

Bad: `feature/my-new-feature`

Bad: `feature/DhawalStuff`

---

## 4. Phase Development Strategy

Every FocusGuard phase after Phase 0 follows:

```
main
  |
  +--> feature/phase-X-name
            |
            +--> implementation
            +--> tests
            +--> debugging
            +--> commits
            |
            +--> Pull Request
                    |
                    +--> review
                    +--> tests
                    +--> diff inspection
                    |
                    +--> merge to main
                            |
                            +--> release tag
```

The next phase starts only after the previous phase is merged into `main`.

Example:

```
main
 |
 +-- v0.1.0 Phase 0
 |
 +-- feature/phase-1-camera-manager
          |
          +-- PR
          |
          +-- merge
 |
 +-- v0.2.0 Phase 1
 |
 +-- feature/phase-2-detection
          |
          +-- PR
          |
          +-- merge
 |
 +-- v0.3.0 Phase 2
```

---

## 5. Phase 0 Exception

Phase 0 is the initial repository foundation. It may be committed directly to `main` as the initial baseline — this has already happened.

After Phase 0, no normal feature development should happen directly on `main`. The workflow becomes:

```
main -> feature branch -> PR -> review -> main
```

---

## 6. Starting a New Phase

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/phase-1-camera-manager
```

Then:

```bash
git push -u origin feature/phase-1-camera-manager
```

- `git switch main` moves to the stable branch.
- `git pull --ff-only origin main` updates local `main` with the latest remote history, refusing to create a merge commit if history has diverged (a safety check).
- `git switch -c <branch>` creates and checks out a new branch from the current `main`.
- `git push -u origin <branch>` publishes the branch and sets it to track the remote branch of the same name.

Claude Code must verify the current branch before making implementation changes. Recommended:

```bash
git branch --show-current
git status
```

---

## 7. Claude Code Branch Safety

This section is extremely important.

Claude Code must:

- Check current branch before implementation.
- Never assume the current branch is correct.
- Never develop Phase 1+ work on `main`.
- Never automatically push to `main`.
- Never force-push.
- Never rewrite `main` history.
- Never commit unless explicitly instructed by the developer.
- Never push unless explicitly instructed.
- Never merge unless explicitly instructed.
- Never delete branches unless explicitly instructed.
- Show relevant Git status/diff before major commits.
- Respect `GIT_WORKFLOW.md`.
- Keep implementation changes isolated to the current task.
- Avoid unrelated modifications.

Before starting a phase, Claude should report:

```
Current branch:
Working tree:
Target phase:
Expected branch:
```

---

## 8. Commit Convention

Use Conventional Commits:

```
<type>(<scope>): <description>
```

Allowed common types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `build`, `ci`.

Examples:

```
feat(camera): implement webcam initialization
feat(camera): add frame capture
test(camera): add camera failure tests
fix(camera): handle unavailable camera gracefully
feat(detection): add YOLO phone detection
test(detection): add phone detection tests
refactor(face): separate landmark extraction
docs(readme): document local setup
chore(deps): update OpenCV version
perf(detection): reduce redundant inference calls
```

Use `feat` for new functionality, `fix` for bug fixes, `refactor` for behavior-preserving restructuring, `test` for test-only changes, `docs` for documentation-only changes, `chore` for tooling/config/dependency work, `perf` for performance improvements, `build`/`ci` for build system or CI changes.

---

## 9. Commit Quality

Prefer small, logical commits.

Do **not** create giant commits such as:

```
feat: implement entire FocusGuard
```

Instead:

```
feat(camera): add camera manager
test(camera): add camera manager tests
fix(camera): handle camera disconnect
docs(camera): document camera configuration
```

Rules:
- One logical purpose per commit.
- Avoid mixing unrelated changes.
- Commit working increments when practical.
- Tests should accompany behavior changes.
- Commit messages should describe what changed.
- Do not use meaningless messages like `update`, `changes`, `final`, `stuff`, `fixes`, `test`.

---

## 10. Git History

```bash
git log --oneline
git log --oneline --graph --decorate --all
git show <commit>
git diff
git diff --staged
```

These commands let you review history at a glance (`log --oneline`), visualize branching and merges (`--graph --decorate --all`), inspect the full contents of a specific commit (`show`), and review unstaged (`diff`) or staged (`diff --staged`) changes before committing.

---

## 11. Working Tree Safety

```bash
git status
git add <file>
git restore <file>
git restore --staged <file>
git diff
git diff --staged
git stash
```

- `git status` — always check this first to see what's modified, staged, or untracked.
- `git add <file>` — stage a specific file (avoid `git add -A`/`.` blindly).
- `git restore --staged <file>` — unstage a file accidentally added, without losing the edit.
- `git restore <file>` — discard local (unstaged) modifications to a file.
- `git diff` / `git diff --staged` — review exactly what will be committed before committing.
- `git stash` — temporarily shelve uncommitted work (e.g., to switch branches), and `git stash pop` to bring it back.

Example: if you accidentally `git add` a file that shouldn't be staged, run `git restore --staged <file>` to unstage it without discarding the edits.

---

## 12. Remote Synchronization

```bash
git fetch origin
git pull --ff-only origin main
git push
git push -u origin <branch>
```

`git fetch origin` downloads remote history without touching your working tree or current branch — safe to run anytime to see what changed remotely. `git pull` is effectively `fetch` + `merge` (or `rebase`) into the current branch; we prefer `--ff-only` on `main` so a pull never silently creates a merge commit or masks divergence. Explicit synchronization (fetch first, decide what to do next) is preferred over blind `pull` because it keeps history predictable and avoids surprise merge commits.

---

## 13. Keeping a Feature Branch Updated

Use rebase for local feature branch synchronization when appropriate:

```bash
git fetch origin
git rebase origin/main
```

Rebasing a local/unshared feature branch onto the latest `main` keeps history linear and clean.

**Important:**
- Do not rebase shared `main`.
- Do not force-push `main`.
- If a feature branch has already been pushed and then rebased, `git push --force-with-lease` may be required — but only when explicitly appropriate, and never casually. Prefer avoiding force pushes when unnecessary.

---

## 14. Merge Strategy

- Feature branch is reviewed through a PR.
- Tests should pass.
- Diff is reviewed.
- Branch is merged into `main`.
- Feature branch is deleted after merge.

Options:
- **Fast-forward merge** — no merge commit, linear history, only possible if `main` hasn't moved.
- **Merge commit** — preserves the full branch history and an explicit merge point.
- **Squash merge** — collapses all branch commits into one commit on `main`.
- **Rebase** — replays branch commits onto `main` individually, linear history, no merge commit.

**Recommendation for FocusGuard:** prefer **squash merge** for PRs that contain many small implementation/debugging commits, so `main` remains clean and each merged feature becomes one meaningful unit of history. Educational Git exercises may intentionally preserve merge commits when practicing branching/conflict resolution.

---

## 15. Pull Request Process

Every Phase 1+ feature should normally use a PR.

**PR template:**

Title:
```
feat: implement Phase 1 camera manager
```

Body:
```markdown
## Summary
- Implemented CameraManager
- Added webcam initialization
- Added frame capture
- Added graceful camera failure handling

## Tests
- X tests added
- X tests passing

## Validation
- Tested integrated webcam
- Tested unavailable camera index

## Related Phase
Phase 1 — Camera Manager
```

Include: summary, implementation details, tests, edge cases, validation, known limitations, related phase/issue.

---

## 16. PR Checklist

Before merging:

- [ ] Correct branch
- [ ] Working tree understood
- [ ] No secrets
- [ ] No generated files
- [ ] Tests pass
- [ ] Relevant edge cases tested
- [ ] Diff reviewed
- [ ] No unrelated changes
- [ ] README/docs updated if needed
- [ ] Requirements updated if dependencies changed
- [ ] PR description complete
- [ ] Branch is synchronized with `main`
- [ ] No merge conflicts
- [ ] No force-push required on `main`

---

## 17. Merge Process

1. Review PR.
2. Run tests.
3. Inspect changed files.
4. Verify no secrets/generated files.
5. Verify branch is correct.
6. Merge PR.
7. Delete feature branch.
8. Switch local repository back to `main`.
9. Pull latest `main`.
10. Verify clean working tree.

```bash
git switch main
git pull --ff-only origin main
```

---

## 18. Branch Cleanup

After a successful merge:

```bash
git branch -d feature/phase-1-camera-manager
git push origin --delete feature/phase-1-camera-manager
```

`git branch -d` deletes the local branch reference. `git push origin --delete` deletes the branch on the remote. Do not delete a branch if its work has not been safely merged.

---

## 19. Release Tagging

Use semantic versioning: `MAJOR.MINOR.PATCH`.

For FocusGuard:
- `v0.1.0` = Phase 0 foundation
- `v0.2.0` = Phase 1
- `v0.3.0` = Phase 2
- ...
- `v1.0.0` = FocusGuard V1 MVP

- **MAJOR** = breaking changes
- **MINOR** = new functionality
- **PATCH** = bug fixes / backwards-compatible changes

```bash
git tag -a v0.1.0 -m "Phase 0: Project Foundation"
git push origin v0.1.0
```

Inspect tags:

```bash
git tag
git show v0.1.0
```

---

## 20. Release History

```
main
 |
 o v0.1.0 Phase 0
 |
 o v0.2.0 Phase 1
 |
 o v0.3.0 Phase 2
 |
 o v0.4.0 Phase 3
 |
 ...
 |
 o v1.0.0 FocusGuard V1
```

Tags make the portfolio project easy to walk through phase-by-phase for a reviewer or interviewer, and give clean rollback points.

---

## 21. Git Debugging Practice

FocusGuard should intentionally provide opportunities to practice real Git debugging:

- `git bisect` — binary-search commit history to find a regression.
- `git revert` — undo a commit by creating a new inverse commit (safe for shared history).
- `git reset` — move the branch pointer / unstage / discard changes (safe only for local, unpublished work).
- `git stash` — shelve uncommitted work temporarily.
- `git reflog` — recover from mistakes by finding "lost" commits.
- `git diff`, `git log`, `git show` — inspect changes and history.
- `git blame` — find which commit/author last changed a line.
- Merge conflict resolution.

**Safe:** `revert`, `stash`, `diff`, `log`, `show`, `blame`, `bisect`.

**Dangerous:** `reset --hard` and `push --force` (and any other history-rewriting command) can destroy work. Prefer `git revert` over rewriting history when undoing changes already shared on `main`.

---

## 22. Bug Investigation with Git Bisect

```bash
git bisect start
git bisect bad
git bisect good <known-good-commit>
```

Then test each revision Git checks out, marking it with `git bisect good` or `git bisect bad` until the offending commit is found.

```bash
git bisect reset
```

This restores the original branch/HEAD once done. `bisect` identifies exactly which commit introduced a regression via binary search, which is far faster than manual inspection on any history of meaningful size.

---

## 23. Revert vs Reset

- **`git revert`** creates a new commit that undoes a previous commit's changes. History is preserved and the undo is itself visible and shareable.
- **`git reset`** moves the branch pointer (and optionally the index/working tree) backward, effectively erasing commits from the branch's history.

**Policy:**
- For commits already pushed/shared on `main`: prefer `git revert`.
- For local, unpublished work: `reset` may be appropriate.
- Never casually rewrite shared `main` history.

---

## 24. Merge Conflict Practice

Conflicts occur when the same lines (or related changes) are modified differently on two branches being combined.

```bash
git fetch origin
git rebase origin/main
```

(or during a `git merge`.)

Workflow:
1. Open conflicted files.
2. Understand both changes.
3. Resolve manually.
4. Run tests.
5. `git add <resolved-files>`
6. Continue the merge/rebase (`git rebase --continue` or complete the merge commit).
7. Verify history.

Do not blindly accept "ours" or "theirs" — understand what each side actually changed before resolving.

---

## 25. .gitignore Policy

Must not be committed:

```
.venv/
__pycache__/
.pytest_cache/
*.pyc
model weights
logs
generated artifacts
OS-specific files
IDE caches
secrets
.env
API keys
credentials
```

`.gitignore` is a safety layer, not a substitute for checking `git status` before committing.

---

## 26. Secrets Policy

Never commit: API keys, tokens, passwords, private keys, SSH private keys, `.env` secrets, cloud credentials.

If a secret is accidentally committed:
1. Do **not** simply delete it from the working tree — the secret remains in history.
2. Treat it as compromised — rotate/revoke the secret first.
3. Then clean history if necessary (e.g., with tooling designed for this, done deliberately and not as a routine operation).

---

## 27. Dependency Changes

Dependency changes should use:

```
chore(deps): ...
```

Examples:
```
chore(deps): update opencv
chore(deps): pin ultralytics version
```

Dependency changes should include: a `requirements.txt` update, test verification, compatibility verification, and documentation if required.

---

## 28. Database / Config / Model Changes

For future phases:

- **Configuration changes** — commit them normally; keep reproducible configuration in version control (e.g., `config/config.yaml`).
- **Database migrations** (if ever introduced) — commit migration scripts, not database files.
- **Model files / weights** — do not commit large binary model weights unless explicitly decided; prefer documenting how to (re)download them.
- **Environment configuration** — commit templates (e.g., `.env.example`), never actual secrets.

---

## 29. Generated Files

Do not commit: logs, caches, temporary files, Python bytecode, virtual environments, downloaded ML model weights, runtime output, build artifacts — unless explicitly required by the project.

---

## 30. Git Command Cheat Sheet

| Purpose | Command |
|---|---|
| Current branch | `git branch --show-current` |
| Status | `git status` |
| Create branch | `git switch -c feature/name` |
| Switch branch | `git switch branch` |
| Fetch | `git fetch origin` |
| Update main | `git switch main` then `git pull --ff-only origin main` |
| Stage | `git add <file>` |
| Unstage | `git restore --staged <file>` |
| Discard local modification | `git restore <file>` |
| Commit | `git commit -m "type(scope): message"` |
| Push new branch | `git push -u origin <branch>` |
| Show history | `git log --oneline --graph --decorate --all` |
| Show diff | `git diff` |
| Show staged diff | `git diff --staged` |
| Rebase feature | `git fetch origin` then `git rebase origin/main` |
| Revert | `git revert <commit>` |
| Stash | `git stash` |
| Restore stash | `git stash pop` |
| List branches | `git branch -a` |
| Delete local branch | `git branch -d <branch>` |
| Delete remote branch | `git push origin --delete <branch>` |

---

## 31. Standard Phase Workflow

1. Finish current phase.
2. Verify: `git status`.
3. Run tests.
4. Commit approved baseline.
5. Push.
6. Tag release.
7. Start next phase from updated `main`.
8. Create feature branch.
9. Claude implements phase.
10. Run tests continuously.
11. Review diff.
12. Push feature branch.
13. Create PR.
14. Review PR.
15. Merge.
16. Delete feature branch.
17. Update local `main`.
18. Create release tag.

---

## 32. Example: Phase 1

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/phase-1-camera-manager
```

Claude works on this branch. Example commits:

```
feat(camera): implement camera manager
test(camera): add camera manager tests
fix(camera): handle camera disconnect
docs(camera): document camera configuration
```

Then:

```bash
git push -u origin feature/phase-1-camera-manager
```

Create PR → review → merge → delete branch → tag `v0.2.0`.

---

## 33. Example: Bug Fix

```bash
git switch main
git pull --ff-only origin main
git switch -c fix/camera-crash
```

Implement fix, add a regression test, commit:

```
fix(camera): prevent crash when frame capture fails
test(camera): cover frame capture failure
```

Create PR → merge → tag a patch release if appropriate.

---

## 34. Solo Developer Policy

Even though FocusGuard is currently a solo project, follow professional practices.

The developer acts as: **Developer**, **Reviewer**, **Release manager**.

Claude acts as: **Coding agent**.

Claude should not be treated as an autonomous repository administrator. Human approval remains required for:

- commits when not explicitly requested
- pushes
- merges
- branch deletion
- release tags
- destructive Git operations

---

## 35. Production-Style Main Protection

Recommended GitHub repository settings for `main` (to be enabled when the repo has a remote and the developer wants this discipline):

- PR required
- status checks required
- force pushes disabled
- deletion disabled
- direct pushes restricted/discouraged
- branch must be up-to-date before merge when practical

These settings provide production-like discipline even for a solo-maintained repository.

---

## 36. Why We Are Using This Strategy

This strategy is intentionally lightweight. We are **not** using `main` + `develop` + `release/*` + `feature/*` + `hotfix/*` + `support/*` etc. — that's unnecessary complexity for a solo project.

Instead:

```
main
feature/*
fix/*
refactor/*
docs/*
test/*
chore/*
hotfix/*
```

This gives enough structure to practice professional Git without introducing unnecessary ceremony.

---

## 37. Portfolio Value

This workflow demonstrates: Git proficiency, branching strategy, clean commit history, PR workflow, release management, debugging with Git, conflict resolution, code review discipline, testing discipline, and production engineering practices.

---

## 38. Non-Negotiable Rules

1. Never develop normal features directly on `main`.
2. Never force-push `main`.
3. Never commit secrets.
4. Never commit `.venv`.
5. Never commit generated logs/caches.
6. Never commit model weights unless explicitly required.
7. Every Phase 1+ feature gets its own branch.
8. Every meaningful feature should go through a PR.
9. Tests must pass before merging.
10. Review the diff before merging.
11. Use Conventional Commits.
12. Keep commits logically focused.
13. Use revert instead of rewriting shared `main` history.
14. Claude must respect the current branch.
15. Claude must not push/merge/delete branches without explicit permission.
16. Do not start the next phase until the previous phase is merged and tagged.

---

## 39. Final Workflow Diagram

```
                ┌──────────────────┐
                │      main        │
                │ Stable / Release │
                └────────┬─────────┘
                         │
                         ▼
              feature/phase-X-name
                         │
                 ┌───────┴────────┐
                 │ Implementation │
                 │ Tests          │
                 │ Debugging      │
                 │ Commits        │
                 └───────┬────────┘
                         │
                         ▼
                    Push Branch
                         │
                         ▼
                    Pull Request
                         │
                 ┌───────┴────────┐
                 │ Review         │
                 │ Tests          │
                 │ Diff           │
                 └───────┬────────┘
                         │
                         ▼
                    Merge to main
                         │
                         ▼
                       Tag
                         │
                         ▼
                   Next Phase
```
