"""Scope guard gate — ensures a stage only touches files it owns.

This prevents model drift where a stage "helpfully" modifies files
belonging to a different stage, breaking isolation.
"""

from __future__ import annotations

import fnmatch
from typing import Any

from harness.gates.base import Gate, GateResult


class ScopeGate(Gate):
    """Reject outputs that write to files outside the stage's owned_files."""

    name = "scope"

    async def check(self, stage_outputs: dict[str, str], context: dict[str, Any]) -> GateResult:
        owned: list[str] = context.get("owned_files", [])

        if not owned:
            # No scope restriction defined — pass
            return GateResult(passed=True, gate_name="scope", message="No scope restriction")

        violations: list[str] = []
        for path in stage_outputs:
            if not any(fnmatch.fnmatch(path, pattern) for pattern in owned):
                violations.append(path)

        if violations:
            return GateResult(
                passed=False,
                gate_name="scope",
                message=f"Stage wrote {len(violations)} file(s) outside its scope",
                details={"violations": violations, "allowed": owned},
            )
        return GateResult(passed=True, gate_name="scope", message="All files within scope")
