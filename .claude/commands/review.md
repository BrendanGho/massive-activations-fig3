---
description: Review the current changes for bugs, security, and missed acceptance criteria
argument-hint: [optional base ref, default main]
---
Review the changes on this branch against `${ARGUMENTS:-main}` — do not fix anything, just report.

1. Run `git diff ${ARGUMENTS:-main}...HEAD` and read the full diff.
2. If `SPEC.md` exists, verify every acceptance criterion is met.
3. Flag: correctness bugs, missing edge cases, security issues (injection, secrets, missing
   authz/validation), and anything that will break tests.
4. Note style/maintainability concerns separately as non-blocking.

Output a concise list grouped by severity: blocking / should-fix / nit. Be specific (file:line + why).
