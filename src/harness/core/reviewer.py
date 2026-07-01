"""Phase 4 — Review.

A separate critic agent reads the assembled output against the original spec.
It's a DIFFERENT prompt/persona than the one that generated the code, so it
provides an independent check rather than rubber-stamping its own work.

Checks for:
- Spec drift (requirements not met)
- Missing pieces (files referenced but not created)
- Over-engineering (anything not in the spec)
- Security/quality concerns
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from harness.models.events import Event, EventKind, EventLog
from harness.models.spec import Spec, StageDag
from harness.providers.base import GenerateRequest, Message, ModelTier, Provider

logger = logging.getLogger(__name__)


@dataclass
class ReviewFinding:
    severity: str  # "blocking" | "should_fix" | "nit"
    category: str  # "drift" | "missing" | "over_engineering" | "security" | "quality"
    file: str
    message: str
    line: int | None = None


@dataclass
class ReviewReport:
    passed: bool
    findings: list[ReviewFinding] = field(default_factory=list)
    summary: str = ""

    @property
    def blocking(self) -> list[ReviewFinding]:
        return [f for f in self.findings if f.severity == "blocking"]


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["passed", "findings", "summary"],
    "properties": {
        "passed": {"type": "boolean", "description": "True if no blocking issues"},
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["severity", "category", "file", "message"],
                "properties": {
                    "severity": {"type": "string", "enum": ["blocking", "should_fix", "nit"]},
                    "category": {
                        "type": "string",
                        "enum": ["drift", "missing", "over_engineering", "security", "quality"],
                    },
                    "file": {"type": "string"},
                    "message": {"type": "string"},
                    "line": {"type": ["integer", "null"]},
                },
            },
        },
    },
}

REVIEW_SYSTEM = """\
You are a code critic reviewing a generated codebase against its specification.
You did NOT write this code. Your job is to find problems the author missed.

Be specific: cite file names and line numbers. Group by severity.

Categories:
- drift: a spec requirement is not met or is implemented incorrectly
- missing: something referenced but not present (file, function, import)
- over_engineering: code/files/features that are NOT in the spec (flag these — the model tends to add things)
- security: injection, secrets, missing validation, etc.
- quality: maintainability, naming, dead code, etc.

Be concise. Skip praise. Focus on what needs fixing.
"""


async def review(
    spec: Spec,
    dag: StageDag,
    event_log: EventLog,
    provider: Provider,
) -> ReviewReport:
    """Have a separate critic agent review the output against the spec."""

    event_log.emit(Event(kind=EventKind.PHASE_START, phase="review"))

    # Gather all outputs
    all_outputs: dict[str, str] = {}
    for stage in dag.stages.values():
        all_outputs.update(stage.outputs)

    # Build the review prompt
    files_text = "\n\n".join(
        f"### {path}\n```\n{content}\n```" for path, content in sorted(all_outputs.items())
    )

    contracts_text = "\n\n".join(
        f"### {c.name}\n```\n{c.body}\n```" for c in dag.contracts.values()
    )

    user_msg = f"""## Original Spec
Goal: {spec.goal}

Acceptance Criteria:
{chr(10).join(f"- {c}" for c in spec.acceptance_criteria)}

Constraints:
{chr(10).join(f"- {k}: {v}" for k, v in spec.constraints.items())}

## Contracts
{contracts_text}

## Generated Files
{files_text}

Review the generated files against the spec and contracts. Return findings as JSON.
Flag anything NOT in the spec under "over_engineering".
"""

    request = GenerateRequest(
        messages=[Message(role="user", content=user_msg)],
        system=REVIEW_SYSTEM,
        tier=ModelTier.STRONG,
        output_schema=REVIEW_SCHEMA,
    )

    response = await provider.generate(request)

    data = response.parsed
    if not data:
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            return ReviewReport(
                passed=False,
                summary=f"Review failed to produce valid output: {response.content[:200]}",
            )

    findings = [
        ReviewFinding(
            severity=f.get("severity", "nit"),
            category=f.get("category", "quality"),
            file=f.get("file", "unknown"),
            message=f.get("message", ""),
            line=f.get("line"),
        )
        for f in data.get("findings", [])
    ]

    for finding in findings:
        event_log.emit(
            Event(
                kind=EventKind.REVIEW_FINDING,
                phase="review",
                message=f"[{finding.severity}] {finding.category}: {finding.file} — {finding.message}",
            )
        )

    report = ReviewReport(
        passed=data.get("passed", len([f for f in findings if f.severity == "blocking"]) == 0),
        findings=findings,
        summary=data.get("summary", ""),
    )

    event_log.emit(
        Event(
            kind=EventKind.PHASE_END,
            phase="review",
            message=f"{'Passed' if report.passed else 'Blocking issues found'}: {report.summary}",
            data={"blocking": len(report.blocking), "total_findings": len(findings)},
        )
    )

    return report
