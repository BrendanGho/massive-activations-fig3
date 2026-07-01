"""Smoke test gate — runs a real command and checks it succeeds.

This is the expensive global gate used at integration time. It doesn't
check strings; it checks whether the composed system actually loads/runs.
"""

from __future__ import annotations

import asyncio
from typing import Any

from harness.gates.base import Gate, GateResult


class SmokeTestGate(Gate):
    """Run a shell command against the workspace and check the exit code.

    The command is provided in context["smoke_command"]. Examples:
    - "pytest -q"
    - "python -c 'from myapp import main'"
    - "npm test"
    - "cargo check"
    """

    name = "smoke_test"

    def __init__(self, default_command: str | None = None, timeout: int = 120):
        self._default = default_command
        self._timeout = timeout

    async def check(self, stage_outputs: dict[str, str], context: dict[str, Any]) -> GateResult:
        cmd = context.get("smoke_command", self._default)
        cwd = context.get("workspace_path", ".")

        if not cmd:
            return GateResult(
                passed=True, gate_name="smoke_test", message="No smoke command configured"
            )

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except asyncio.TimeoutError:
            return GateResult(
                passed=False,
                gate_name="smoke_test",
                message=f"Smoke test timed out after {self._timeout}s",
                details={"command": cmd, "timeout": self._timeout},
            )
        except Exception as e:
            return GateResult(
                passed=False,
                gate_name="smoke_test",
                message=f"Smoke test error: {e}",
                details={"command": cmd, "error": str(e)},
            )

        if proc.returncode == 0:
            return GateResult(
                passed=True,
                gate_name="smoke_test",
                message=f"Smoke test passed: {cmd}",
            )

        # Distill the error: take last 30 lines of combined output
        output = (stdout.decode(errors="replace") + stderr.decode(errors="replace")).strip()
        distilled = "\n".join(output.splitlines()[-30:])

        return GateResult(
            passed=False,
            gate_name="smoke_test",
            message=f"Smoke test failed (exit {proc.returncode})",
            details={
                "command": cmd,
                "exit_code": proc.returncode,
                "output_tail": distilled,
            },
        )
