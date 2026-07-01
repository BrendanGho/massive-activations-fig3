---
description: Autonomously implement the current SPEC.md against its tests
argument-hint: [optional: which part of the spec]
---
Implement `SPEC.md` $ARGUMENTS.

1. Read `SPEC.md`. If it is missing, stop and tell me to run `/spec` first.
2. Propose a short plan, then proceed.
3. Work in small steps. After each change the post-edit hook runs the tests — fix failures before continuing.
4. Commit after each acceptance criterion passes, with a clear message (`feat: ...`, `test: ...`, etc.).
5. If genuinely blocked, append the blocker (with context and what you tried) to `BLOCKERS.md`, then
   move to the next independent criterion.
6. Stop when all acceptance criteria pass or everything remaining is blocked. Summarize what's done
   and anything left in `BLOCKERS.md`.

Do not refactor unrelated code or expand scope beyond the spec.
