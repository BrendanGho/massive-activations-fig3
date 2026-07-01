# CLAUDE.md

Guidance for Claude Code (and any agent) working in this repo. This repo is configured for
**supervised autonomy**: you can be handed a goal and run with minimal intervention, *as long as*
you follow the workflow below.

## Golden rules
- Work on a feature branch, never directly on `main`. Commit in small, logical steps.
- Tests are the source of truth. Don't claim a task is done until the relevant tests pass.
- When genuinely blocked, append the question (with context and what you tried) to `BLOCKERS.md`
  and move on to the next independent task instead of stopping.
- Prefer editing existing files over adding new ones. Keep changes minimal and scoped to the task.
- Never commit secrets. Never run destructive commands (force-push, history rewrite, `rm -rf`,
  dropping data/migrations) without explicit approval.

## Commands
- **Install:** `uv sync`
- **Run (pipeline):** `uv run harness run examples/todo_api.yaml -w ./output`
- **Run (MCP store):** `uv run harness-mcp` (usually launched by Claude Code via `.mcp.json`)
- **Test:** `uv run pytest -q`  ← the post-edit hook runs this automatically after edits
- **Lint/format:** `uv run ruff format && uv run ruff check`

## The autonomy workflow
1. **Spec** — before non-trivial work, write or read `SPEC.md` (run `/spec <goal>`). It must include
   acceptance criteria written as testable statements. Tight scope is what makes autonomy safe.
2. **Plan** — in plan mode, propose an implementation plan against the spec. Wait for approval on
   anything non-trivial.
3. **Implement** — run `/implement`. Work the plan in small steps; after each change the post-edit
   hook runs the tests, so fix failures before moving on.
4. **Commit at checkpoints** — commit after each acceptance criterion passes, with a clear message.
   This keeps progress reviewable and reversible.
5. **Review** — when the spec is satisfied, run `/review` to self-check the diff for bugs, security,
   and missed criteria before merging.

## Tools available in this repo
- **Slash commands** (`.claude/commands/`): `/spec`, `/implement`, `/review`, `/delegate`,
  `/orchestrate` (blackboard research/build loop).
- **Subagents** (`.claude/agents/`): `test-writer`, `code-reviewer`. Use the Task tool to delegate
  bounded sub-work so the main thread stays focused and context stays small.
- **Knowledge store (blackboard)** — the `harness` MCP server (`.mcp.json`, code in
  `src/harness/`). Coordinate through it instead of passing raw context around: `add_finding`,
  `query_findings` (full-text), `upsert_entity`/`get_entity`/`list_entities`, `link_entities`,
  `related`. Backed by SQLite at `.harness/kb.sqlite` (gitignored, per-machine). Workers write
  short findings; the orchestrator queries rather than re-reading sources.
- **Hooks** (`.claude/settings.json`): a post-edit hook auto-formats and runs the tests after every
  edit; a `Stop` hook (`harness-snapshot`) writes a resumable checkpoint to
  `.harness/checkpoints/latest.md` so a fresh session can pick up without replaying the transcript.
- **Worker** (`scripts/worker.sh`): hands a bounded, well-specified task to a free local/cloud model
  via free-claude-code. See "Delegation & budget".

## Delegation & budget
This setup assumes a limited premium budget plus free fallback models. Spend the smart model on
judgment; offload volume.
- **Do yourself** (premium model): architecture, hard logic, debugging, review — anything needing judgment.
- **Delegate** (`/delegate` or `scripts/worker.sh`): large, mechanical, independently testable chunks —
  scaffolding, boilerplate, repetitive edits, writing many similar tests.
- Only delegate work that tests can verify. Always review the resulting `git diff` and confirm tests
  pass before accepting.
- Tell the worker to edit files, run tests, and report a **short summary** — not paste code back — to
  keep context (and token cost) small.

## Guardrails for unattended runs
- Stay within the current task/spec; don't refactor unrelated code or expand scope.
- If a change would touch more than ~10 files, or deletes data/migrations, pause and ask.
- Keep secrets in `.env` (gitignored); never hardcode them.
