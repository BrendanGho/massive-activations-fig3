"""Phase 2 — Execute loop.

Cheap model, called per stage. The core feedback loop:

    inject [frozen contract + plan slice + upstream files + last error]
    → generate
    → deterministic gate(s)
    → on pass: scope-guard, checkpoint, tag
    → on fail: revert, distill error, retry (up to N), then escalate

Key design decisions:
- The frozen contract is re-injected every call (isolation + coherence).
- Error distillation extracts the minimal diagnostic before retry.
- Revert happens BEFORE retry, so each attempt starts from known-good state.
- Escalation path: bump to strong model, then human.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from harness.checkpoints.manager import CheckpointManager
from harness.gates.base import Gate, GateResult
from harness.models.events import Event, EventKind, EventLog
from harness.models.spec import Contract, Stage, StageDag, StageStatus
from harness.providers.base import GenerateRequest, Message, ModelTier, Provider

logger = logging.getLogger(__name__)

EXECUTE_SYSTEM = """\
You are implementing one stage of a larger system. You must:
1. Follow the contracts EXACTLY — they are frozen and non-negotiable.
2. Only create/modify files listed in your owned_files scope.
3. Meet every acceptance criterion.
4. Output valid, parseable code.

Return a JSON object mapping file paths to their complete contents:
{{"path/to/file.py": "file content...", ...}}
"""


def _distill_error(gate_results: list[GateResult]) -> str:
    """Extract the minimal, actionable diagnostic from gate failures.

    This is the error distillation step: instead of passing raw stderr
    to the model, we localize the failure to specific lines/symbols.
    """
    parts: list[str] = []
    for r in gate_results:
        if not r.passed:
            parts.append(f"[{r.gate_name}] {r.message}")
            if r.details:
                # Pick the most useful detail fields
                for key in ("violations", "failures", "output_tail", "error", "missing_symbol"):
                    if key in r.details:
                        val = r.details[key]
                        if isinstance(val, list) and len(val) > 5:
                            val = val[:5]  # Truncate to avoid prompt bloat
                        parts.append(f"  {key}: {json.dumps(val, indent=2)}")
    return "\n".join(parts)


def _build_stage_prompt(
    stage: Stage,
    dag: StageDag,
    upstream_outputs: dict[str, str],
    last_error: str | None = None,
) -> str:
    """Build the prompt for one stage execution.

    Injects: frozen contracts, stage plan, upstream file contents, error diagnostic.
    """
    sections: list[str] = []

    # 1. Frozen contracts (re-injected every call)
    relevant_contracts = [dag.contracts[c] for c in stage.contracts if c in dag.contracts]
    if relevant_contracts:
        sections.append("## Contracts (FROZEN — implement these exactly)")
        for c in relevant_contracts:
            sections.append(f"### {c.name}\n```\n{c.body}\n```")

    # 2. This stage's plan slice
    sections.append(f"## Stage: {stage.name}")
    sections.append(f"Description: {stage.description}")
    sections.append(f"Owned files: {', '.join(stage.owned_files)}")
    sections.append("Acceptance criteria:")
    for ac in stage.acceptance_criteria:
        sections.append(f"  - {ac}")

    # 3. Upstream dependency files (only what this stage needs)
    if upstream_outputs:
        sections.append("## Upstream files (read-only context, do not modify)")
        for path, content in upstream_outputs.items():
            sections.append(f"### {path}\n```\n{content}\n```")

    # 4. Last attempt's error (if retrying)
    if last_error:
        sections.append(f"## Previous attempt FAILED. Fix these specific issues:\n{last_error}")
        sections.append("Do NOT repeat the same mistakes. Address each issue directly.")

    sections.append(
        "\nReturn a JSON object mapping file paths to their complete file contents. "
        "Only include files in your owned_files scope."
    )

    return "\n\n".join(sections)


def _gather_upstream_outputs(stage: Stage, dag: StageDag) -> dict[str, str]:
    """Collect outputs from this stage's dependencies."""
    upstream: dict[str, str] = {}
    for dep_id in stage.depends_on:
        dep = dag.stages.get(dep_id)
        if dep and dep.outputs:
            upstream.update(dep.outputs)
    return upstream


async def execute_stage(
    stage: Stage,
    dag: StageDag,
    provider: Provider,
    gates: list[Gate],
    checkpoint_mgr: CheckpointManager,
    event_log: EventLog,
    workspace: Path,
) -> bool:
    """Execute a single stage with the feedback loop.

    Returns True if the stage passed, False if it exhausted retries.
    """
    stage.status = StageStatus.RUNNING
    event_log.emit(Event(kind=EventKind.STAGE_START, stage_id=stage.id, message=stage.name))

    upstream = _gather_upstream_outputs(stage, dag)
    last_error: str | None = None
    tier = ModelTier.CHEAP

    for attempt in range(stage.max_retries + 1):
        stage.attempts = attempt + 1

        # Build prompt with frozen contracts + error if retrying
        prompt = _build_stage_prompt(stage, dag, upstream, last_error)

        request = GenerateRequest(
            messages=[Message(role="user", content=prompt)],
            system=EXECUTE_SYSTEM,
            tier=tier,
            output_schema={
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Map of file_path -> file_content",
            },
        )

        response = await provider.generate(request)

        # Parse outputs
        outputs = response.parsed
        if not outputs:
            try:
                outputs = json.loads(response.content)
            except json.JSONDecodeError:
                last_error = f"Output was not valid JSON. Raw start: {response.content[:200]}"
                event_log.emit(
                    Event(
                        kind=EventKind.STAGE_RETRY,
                        stage_id=stage.id,
                        message=f"Attempt {attempt + 1}: invalid JSON output",
                    )
                )
                continue

        if not isinstance(outputs, dict):
            last_error = (
                f"Expected a JSON object mapping paths to contents, got {type(outputs).__name__}"
            )
            continue

        stage.outputs = outputs

        # Run all gates
        gate_context: dict[str, Any] = {
            "contracts": dag.contracts,
            "stage_contracts": stage.contracts,
            "owned_files": stage.owned_files,
            "workspace_path": str(workspace),
        }

        gate_results: list[GateResult] = []
        all_passed = True
        for gate in gates:
            result = await gate.check(outputs, gate_context)
            gate_results.append(result)
            event_log.emit(
                Event(
                    kind=EventKind.GATE_PASS if result.passed else EventKind.GATE_FAIL,
                    stage_id=stage.id,
                    message=f"{gate.name}: {result.message}",
                    data=result.details,
                )
            )
            if not result.passed:
                all_passed = False

        if all_passed:
            # SUCCESS — checkpoint and tag
            cp = await checkpoint_mgr.create(stage.id, outputs)
            stage.status = StageStatus.PASSED
            event_log.emit(
                Event(
                    kind=EventKind.STAGE_PASS,
                    stage_id=stage.id,
                    message=f"Passed on attempt {attempt + 1}",
                    data={"checkpoint": cp.tag},
                )
            )
            return True

        # FAIL — revert, distill error, retry
        await checkpoint_mgr.revert(stage.id)
        last_error = _distill_error(gate_results)
        event_log.emit(
            Event(
                kind=EventKind.STAGE_RETRY if attempt < stage.max_retries else EventKind.STAGE_FAIL,
                stage_id=stage.id,
                message=f"Attempt {attempt + 1} failed, {stage.max_retries - attempt} retries left",
                data={"distilled_error": last_error},
            )
        )
        event_log.emit(
            Event(
                kind=EventKind.ERROR_DISTILL,
                stage_id=stage.id,
                message=last_error,
            )
        )

        # Escalate to strong model on last retry
        if attempt == stage.max_retries - 1:
            tier = ModelTier.STRONG
            logger.info(f"Stage {stage.id}: escalating to strong model for final attempt")

    # Exhausted retries
    stage.status = StageStatus.FAILED
    event_log.emit(
        Event(
            kind=EventKind.STAGE_ESCALATE,
            stage_id=stage.id,
            message=f"Failed after {stage.max_retries + 1} attempts",
        )
    )
    return False


async def execute_all(
    dag: StageDag,
    provider: Provider,
    gates: list[Gate],
    checkpoint_mgr: CheckpointManager,
    event_log: EventLog,
    workspace: Path,
) -> bool:
    """Execute all stages in dependency order.

    Processes stages level by level (all ready stages, then next level).
    Returns True if all stages passed.
    """
    event_log.emit(Event(kind=EventKind.PHASE_START, phase="execute"))

    order = dag.topological_order()
    all_passed = True

    for stage in order:
        if stage.status != StageStatus.PENDING:
            continue

        passed = await execute_stage(
            stage, dag, provider, gates, checkpoint_mgr, event_log, workspace
        )
        if not passed:
            all_passed = False
            # Check if any downstream stages depend on this one — mark them failed too
            for other in dag.stages.values():
                if stage.id in other.depends_on and other.status == StageStatus.PENDING:
                    other.status = StageStatus.FAILED
                    other.last_error = f"Upstream stage {stage.id} failed"

    event_log.emit(
        Event(
            kind=EventKind.PHASE_END,
            phase="execute",
            message=f"{'All passed' if all_passed else 'Some stages failed'}",
        )
    )
    return all_passed
