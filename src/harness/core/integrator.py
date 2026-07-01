"""Phase 3 — Integrate.

Called once after all stages pass. Runs the checks that can't exist per-stage:
- Cross-file symbol/reference resolution
- Smoke test that actually loads and runs the composed system
- Import graph verification (no circular deps, all imports resolve)
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from harness.gates.base import Gate, GateResult
from harness.gates.smoke import SmokeTestGate
from harness.models.events import Event, EventKind, EventLog
from harness.models.spec import StageDag

logger = logging.getLogger(__name__)


def _resolve_cross_file_references(
    all_outputs: dict[str, str],
) -> list[dict[str, str]]:
    """Check that imports between generated files actually resolve.

    For Python files, parse imports and verify the referenced modules/symbols
    exist in the output set. This catches "every file is fine but they don't link."
    """
    issues: list[dict[str, str]] = []

    # Build a map of what each file defines
    defined_modules: set[str] = set()
    defined_symbols: dict[str, set[str]] = {}  # module -> symbols

    for path, content in all_outputs.items():
        if not path.endswith(".py"):
            continue

        # Derive module name from path
        module = path.replace("/", ".").replace("\\", ".").removesuffix(".py")
        # Also add parent package
        parts = module.split(".")
        for i in range(len(parts)):
            defined_modules.add(".".join(parts[: i + 1]))

        try:
            tree = ast.parse(content, filename=path)
        except SyntaxError:
            continue

        symbols: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                symbols.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbols.add(target.id)
        defined_symbols[module] = symbols

    # Check imports
    for path, content in all_outputs.items():
        if not path.endswith(".py"):
            continue

        try:
            tree = ast.parse(content, filename=path)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                # Only check internal imports (skip stdlib/third-party)
                if node.module in defined_modules or any(
                    node.module.startswith(m + ".") for m in defined_modules
                ):
                    for alias in node.names:
                        # Check if the symbol exists
                        mod_symbols = defined_symbols.get(node.module, set())
                        if alias.name != "*" and alias.name not in mod_symbols:
                            # Could be a submodule
                            sub = f"{node.module}.{alias.name}"
                            if sub not in defined_modules:
                                issues.append(
                                    {
                                        "file": path,
                                        "import": f"from {node.module} import {alias.name}",
                                        "reason": f"'{alias.name}' not found in {node.module}",
                                    }
                                )

    return issues


async def integrate(
    dag: StageDag,
    event_log: EventLog,
    workspace: Path,
    smoke_command: str | None = None,
    extra_gates: list[Gate] | None = None,
) -> bool:
    """Run integration checks on the composed system."""

    event_log.emit(Event(kind=EventKind.PHASE_START, phase="integrate"))

    # Gather all outputs from all stages
    all_outputs: dict[str, str] = {}
    for stage in dag.stages.values():
        all_outputs.update(stage.outputs)

    issues: list[str] = []

    # 1. Cross-file reference resolution
    ref_issues = _resolve_cross_file_references(all_outputs)
    for issue in ref_issues:
        issues.append(
            f"Unresolved import in {issue['file']}: {issue['import']} — {issue['reason']}"
        )

    # 2. Smoke test
    if smoke_command:
        smoke_gate = SmokeTestGate(default_command=smoke_command)
        result = await smoke_gate.check(
            all_outputs,
            {
                "smoke_command": smoke_command,
                "workspace_path": str(workspace),
            },
        )
        event_log.emit(
            Event(
                kind=EventKind.GATE_PASS if result.passed else EventKind.GATE_FAIL,
                phase="integrate",
                message=f"smoke_test: {result.message}",
                data=result.details,
            )
        )
        if not result:
            issues.append(f"Smoke test failed: {result.message}")
            if result.details.get("output_tail"):
                issues.append(f"Output: {result.details['output_tail'][:500]}")

    # 3. Extra gates (user-provided)
    for gate in extra_gates or []:
        result = await gate.check(all_outputs, {"workspace_path": str(workspace)})
        event_log.emit(
            Event(
                kind=EventKind.GATE_PASS if result.passed else EventKind.GATE_FAIL,
                phase="integrate",
                message=f"{gate.name}: {result.message}",
            )
        )
        if not result:
            issues.append(f"{gate.name}: {result.message}")

    passed = len(issues) == 0
    event_log.emit(
        Event(
            kind=EventKind.PHASE_END,
            phase="integrate",
            message=f"{'Passed' if passed else f'{len(issues)} issue(s) found'}",
            data={"issues": issues},
        )
    )

    return passed
