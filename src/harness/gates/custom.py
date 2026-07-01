"""Custom gate — run any shell command as a per-stage gate.

Lets users plug in linters, type-checkers, or domain-specific validators
without writing Python.
"""

from __future__ import annotations

import asyncio
from typing import Any

from harness.gates.base import Gate, GateResult


class CustomCommandGate(Gate):
    """Run an arbitrary command. Passes if exit code is 0."""

    def __init__(self, name: str, command: str, timeout: int = 60):
        self.name = name
        self._command = command
        self._timeout = timeout

    async def check(self, stage_outputs: dict[str, str], context: dict[str, Any]) -> GateResult:
        cwd = context.get("workspace_path", ".")
        try:
            proc = await asyncio.create_subprocess_shell(
                self._command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except asyncio.TimeoutError:
            return GateResult(
                passed=False,
                gate_name=self.name,
                message=f"Timed out after {self._timeout}s",
                details={"command": self._command},
            )

        if proc.returncode == 0:
            return GateResult(passed=True, gate_name=self.name, message="Passed")

        output = (stdout.decode(errors="replace") + stderr.decode(errors="replace")).strip()
        return GateResult(
            passed=False,
            gate_name=self.name,
            message=f"Failed (exit {proc.returncode})",
            details={
                "command": self._command,
                "exit_code": proc.returncode,
                "output_tail": "\n".join(output.splitlines()[-20:]),
            },
        )
