"""Phase 1 — Plan.

Strong model, called once. Turns a Spec into three frozen artifacts:
1. A stage DAG (units + dependencies + file ownership)
2. Contracts (every shared interface, pinned with signatures not prose)
3. Per-stage acceptance criteria

Optionally validates the plan with a dry-run stub check before committing.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from harness.models.events import Event, EventKind, EventLog
from harness.models.spec import Contract, Spec, Stage, StageDag
from harness.providers.base import GenerateRequest, Message, ModelTier, Provider

logger = logging.getLogger(__name__)

# JSON schema for the structured plan output
PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["stages", "contracts"],
    "properties": {
        "stages": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "description", "owned_files", "depends_on",
                             "acceptance_criteria", "contracts"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "owned_files": {"type": "array", "items": {"type": "string"}},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                    "contracts": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "contracts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "body", "producers", "consumers"],
                "properties": {
                    "name": {"type": "string"},
                    "body": {"type": "string", "description": "Code or structured text defining the interface. Signatures, types, schemas — not prose."},
                    "producers": {"type": "array", "items": {"type": "string"}},
                    "consumers": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

PLAN_SYSTEM = """\
You are a software architect planning a multi-stage implementation.

Your job: decompose a spec into a DAG of stages with explicit contracts.

Rules:
- Each stage owns specific files and produces them. No two stages may own the same file.
- Contracts define shared interfaces between stages — function signatures, schemas, types.
  Write contracts as CODE (class/function defs, type aliases, JSON schemas), not prose.
- Each stage has testable acceptance criteria.
- Dependencies form a DAG (no cycles).
- Keep stages small and independently verifiable.
- Anything byte-critical (manifests, configs, schemas) should be in its own stage.
"""


async def plan(
    spec: Spec,
    provider: Provider,
    event_log: EventLog,
    extra_context: str = "",
) -> StageDag:
    """Generate a stage DAG + contracts from a spec using the strong model."""

    event_log.emit(Event(kind=EventKind.PHASE_START, phase="plan"))

    user_msg = f"""## Goal
{spec.goal}

## Context
{spec.context}

## Acceptance Criteria
{chr(10).join(f'- {c}' for c in spec.acceptance_criteria)}

## Constraints
{chr(10).join(f'- {k}: {v}' for k, v in spec.constraints.items())}

{f'## Additional Context{chr(10)}{extra_context}' if extra_context else ''}

Produce a JSON plan with `stages` and `contracts` arrays.
Each stage: id, name, description, owned_files (glob patterns OK), depends_on (stage IDs),
acceptance_criteria (testable statements), contracts (contract names it must satisfy).
Each contract: name, body (CODE defining the interface), producers (stage IDs), consumers (stage IDs).
"""

    request = GenerateRequest(
        messages=[Message(role="user", content=user_msg)],
        system=PLAN_SYSTEM,
        tier=ModelTier.STRONG,
        output_schema=PLAN_SCHEMA,
        temperature=0.0,
    )

    response = await provider.generate(request)

    # Parse the plan
    data = response.parsed
    if not data:
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            raise ValueError(f"Plan generation failed to produce valid JSON:\n{response.content[:500]}")

    dag = StageDag()

    # Build contracts first
    for c in data.get("contracts", []):
        dag.add_contract(Contract(
            name=c["name"],
            body=c["body"],
            producers=c.get("producers", []),
            consumers=c.get("consumers", []),
        ))

    # Build stages
    for s in data.get("stages", []):
        dag.add_stage(Stage(
            id=s["id"],
            name=s["name"],
            description=s["description"],
            owned_files=s.get("owned_files", []),
            depends_on=s.get("depends_on", []),
            acceptance_criteria=s.get("acceptance_criteria", []),
            contracts=s.get("contracts", []),
        ))

    # Validate: no cycles, all dependencies exist
    dag.topological_order()

    event_log.emit(Event(
        kind=EventKind.PHASE_END,
        phase="plan",
        message=f"{len(dag.stages)} stages, {len(dag.contracts)} contracts",
        data={"stages": list(dag.stages.keys()), "contracts": list(dag.contracts.keys())},
    ))

    return dag


async def validate_plan(
    dag: StageDag,
    provider: Provider,
    event_log: EventLog,
) -> list[str]:
    """Lightweight plan validation: have the cheap model check for interface mismatches.

    Returns a list of warnings (empty = plan looks good).
    """
    contracts_text = "\n\n".join(
        f"### {c.name}\n```\n{c.body}\n```\nProducers: {c.producers}\nConsumers: {c.consumers}"
        for c in dag.contracts.values()
    )
    stages_text = "\n\n".join(
        f"### {s.name} ({s.id})\nOwns: {s.owned_files}\nDepends on: {s.depends_on}\n"
        f"Contracts: {s.contracts}\nCriteria: {s.acceptance_criteria}"
        for s in dag.stages.values()
    )

    request = GenerateRequest(
        messages=[Message(role="user", content=f"""Review this plan for issues:

## Contracts
{contracts_text}

## Stages
{stages_text}

Check for:
1. Contract interfaces that don't match between producers and consumers
2. Missing dependencies (stage uses an output but doesn't depend on the producer)
3. Overlapping file ownership
4. Acceptance criteria that can't be tested mechanically

Return a JSON array of warning strings. Return [] if the plan looks correct.
""")],
        system="You are a plan reviewer. Be precise and concise. Only flag real issues.",
        tier=ModelTier.CHEAP,
        output_schema={"type": "array", "items": {"type": "string"}},
    )

    response = await provider.generate(request)
    warnings = response.parsed or []
    if isinstance(warnings, list):
        return [str(w) for w in warnings]
    return []
