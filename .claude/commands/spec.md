---
description: Draft a tight SPEC.md (with testable acceptance criteria) for a feature or project
argument-hint: [what to build]
---
Create or update `SPEC.md` for: **$ARGUMENTS**

Use the template at `.claude/templates/SPEC.template.md`. Keep scope tight and explicit.

Include:
- A one-sentence goal describing what "done" looks like.
- Explicit in-scope and out-of-scope lists.
- Acceptance criteria written as concrete, testable statements (each should map to a test).
- Constraints: stack, libraries, conventions, and any limits.

Ask me at most 2 clarifying questions, and only if the goal is genuinely ambiguous; otherwise make
reasonable assumptions and record them under "Assumptions".
