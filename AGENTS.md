\# FocusGuard — Coding Agent Instructions



\## Project



FocusGuard is a local real-time computer-vision productivity assistant.



Read `FOCUSGUARD\_PRD.md` completely before making implementation decisions.



\## Development Rules



1\. Work strictly phase-by-phase according to the PRD.

2\. Never implement future phases automatically.

3\. Before starting a phase, inspect the current repository state.

4\. Preserve existing working functionality.

5\. Keep the architecture simple and appropriate for a 1–2 day MVP.

6\. Do not add backend, database, authentication, cloud services, mobile application, web application, or custom ML training.

7\. All configurable thresholds must live in configuration.

8\. Do not use one noisy frame as an event.

9\. Use temporal filtering for event confirmation.

10\. Missing facial landmarks must never automatically mean eyes are closed.

11\. Never upload webcam frames.

12\. Never automatically save webcam images.

13\. Do not implement face recognition.

14\. Do not introduce multiprocessing unless profiling proves it is necessary.

15\. Prefer simple, maintainable Python.

16\. Use current compatible versions/APIs of dependencies.

17\. If a dependency or API is incompatible with the PRD, explain the issue and choose the smallest compatible alternative.



\## Phase Workflow



Implement only the requested phase.



After completing a phase:



1\. Run relevant tests.

2\. Run the application or verification command where applicable.

3\. Check for obvious regressions.

4\. Report files created/modified.

5\. Report tests performed.

6\. Report known issues.

7\. Provide exact verification instructions.

8\. STOP.



Do not start the next phase until the user explicitly tells you to.



\## Git



Create clean commits after successfully verified phases.



Commit format:



\- `docs:` documentation

\- `chore:` setup/configuration

\- `feat:` new functionality

\- `fix:` bug fixes

\- `test:` tests

\- `perf:` performance improvements

\- `refactor:` refactoring



Keep commits focused.



Do not commit:



\- `.env`

\- virtual environments

\- cache files

\- logs

\- model weights unless explicitly requested

\- generated temporary files

\- personal webcam data



\## Git Workflow



FocusGuard follows the repository Git workflow documented in `GIT\_WORKFLOW.md`. Coding agents must follow that policy.



Before implementation:



\- inspect git status

\- inspect current branch

\- understand the requested phase

\- ensure the branch is appropriate



For Phase 1+ work:



\- do not implement on main

\- use the appropriate feature/fix/refactor/etc. branch

\- keep changes scoped

\- run relevant tests

\- do not push unless explicitly instructed

\- do not merge unless explicitly instructed

\- do not delete branches unless explicitly instructed

\- never force-push main

\- never commit secrets

\- never modify unrelated files



When asked to commit, use Conventional Commits.



When asked to create a PR, provide a clear summary, test results, validation, and known limitations.



If the current branch conflicts with the requested workflow, STOP and report the conflict instead of silently switching branches.



\## Quality



The objective is not maximum feature count.



The objective is:



Camera

→ Detection

→ Face Analysis

→ Temporal Filtering

→ State Machine

→ Events

→ Audio/UI

→ Analytics



The implementation must be reliable, understandable, testable, and easy to explain during an interview.

