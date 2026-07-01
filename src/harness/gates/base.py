"""Gate interface — deterministic checks the model can't talk its way past."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GateResult:
    passed: bool
    gate_name: str = ""
    message: str = ""
    # Structured diagnostic for error distillation (not raw stderr)
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


class Gate(abc.ABC):
    """A deterministic verification step.

    Gates check meaning, not strings. They should invoke real tools
    (compilers, linters, test runners, import resolvers) rather than grep.
    """

    name: str = "gate"

    @abc.abstractmethod
    async def check(self, stage_outputs: dict[str, str], context: dict[str, Any]) -> GateResult:
        """Run the gate against stage outputs.

        Args:
            stage_outputs: file_path -> file_content for everything the stage produced.
            context: additional info — contracts, workspace path, stage metadata, etc.

        Returns:
            GateResult with pass/fail and a diagnostic if failed.
        """
        ...
