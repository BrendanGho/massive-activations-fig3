---
description: Run a research/build goal as an orchestrator that dispatches subagents and coordinates through the shared knowledge store (blackboard).
argument-hint: <goal, e.g. "reproduce Table 2 of the ViT paper">
---

# /orchestrate — blackboard orchestration loop

You are the **orchestrator**. Your job is to drive `$ARGUMENTS` to done while keeping
*your own* context small. You do this by delegating expensive work to subagents and
coordinating through the `harness` MCP store — never by holding raw sources in your context.

## The one rule that makes this token-lean
**A subagent is a compression boundary.** A worker may burn 50k tokens reading a paper,
running code, or searching a repo, but it returns a ~200-token distillation *and writes
its findings to the store*. That expensive context never enters your window. Coordinate by
reading/writing structured state, not by passing transcripts around.

## Loop

1. **Orient (cheap).** Query the blackboard before doing anything, so you build on prior work
   instead of redoing it:
   - `query_findings(query=<goal keywords>)` and `list_entities()` to see what's known.
     Search (`query=`) returns short **snippets** for cheap scanning; to read a hit in
     full, re-query filtered by its `task=` or entity (that path returns full `content`).
   - If resuming, also read `.harness/checkpoints/latest.md`.

2. **Decompose.** Break the goal into *bounded, independently verifiable* subtasks. For paper
   reproduction, typical nodes are: locate source/code → extract each claim → set up env →
   run each experiment → compare result to the paper → record discrepancies. Model these as
   entities: `paper`, `claim`, `experiment`, `result`, `discrepancy`, linked with
   `link_entities` (e.g. `paper -[makes_claim]-> claim`, `claim -[tested_by]-> experiment`).

3. **Dispatch.** For each independent subtask, launch a subagent with the **Task tool**
   (`general-purpose` for build/run, `Explore` for search-only). In its prompt, instruct it to:
   - **First** `query_findings`/`get_entity` for context — do not re-read what's already recorded.
   - Do the bounded work.
   - **Write results back**: `add_finding(content=<short>, task=<goal>, source=<file/url>,
     confidence=<0-1>, entity_type=..., entity_name=...)`, and `upsert_entity`/`link_entities`
     for structure.
   - **Return only** a short summary + the ids/names it wrote. Never paste raw code or sources back.
   Dispatch independent subtasks in parallel; research is breadth-first and parallelizes well.

4. **Synthesize (cheap).** Read the blackboard (`query_findings(task=<goal>)`, `related(...)`)
   to integrate — not the raw sources. Decide what's done and what's next.

5. **Checkpoint.** After each acceptance criterion, commit (per CLAUDE.md). The Stop hook also
   snapshots the blackboard to `.harness/checkpoints/` so a fresh session can resume.

6. Repeat until the goal's acceptance criteria are met, then run `/review`.

## Guardrails
- Stay on a feature branch; small commits (CLAUDE.md). Don't refactor unrelated code.
- Prefer many small findings over few large ones — they search better and keep coordination cheap.
- If blocked, `add_finding(..., task="BLOCKER")` and append to `BLOCKERS.md`, then move to the
  next independent subtask instead of stalling.
- Do the judgment yourself (decomposition, comparing results, deciding done); delegate the volume.
