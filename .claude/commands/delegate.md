---
description: Hand a bounded, testable task to the free local/cloud worker (free-claude-code)
argument-hint: <detailed task: what, which files, how to test>
---
Delegate this to the worker instead of coding it yourself: **$ARGUMENTS**

1. Make the task precise first: which files, expected behavior, and how it will be tested. If
   $ARGUMENTS is vague, tighten it into an exact instruction before delegating.
2. Ensure we are on a feature branch (the worker edits files freely). If on `main`, create one.
3. Run: `scripts/worker.sh "<precise instruction — include: edit the files, run the tests, and reply
   with only a short summary>"`
4. When it returns, run `git diff` to review what changed and confirm the tests pass.
5. If the output is wrong or incomplete, fix it yourself or re-delegate with a tighter spec.

Only delegate large, mechanical, independently testable work. Do hard logic and debugging yourself.
