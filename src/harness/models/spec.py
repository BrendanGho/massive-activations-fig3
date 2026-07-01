"""Core data models for the agent harness."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any


class StageStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ESCALATED = "escalated"


@dataclass
class Contract:
    """A frozen interface contract shared between stages.

    Contracts are the coordination mechanism: they're generated once in the Plan phase,
    then re-injected verbatim into every Execute prompt so the model can't drift.
    """

    name: str
    # The contract body — schema definitions, function signatures, type aliases, etc.
    # This is injected as-is into prompts; it should be code or structured text, not prose.
    body: str
    # Optional: which stages produce / consume this contract
    producers: list[str] = field(default_factory=list)
    consumers: list[str] = field(default_factory=list)

    def signature(self) -> str:
        """Stable hash-like identifier for drift detection."""
        import hashlib

        return hashlib.sha256(self.body.encode()).hexdigest()[:16]


@dataclass
class Stage:
    """A single unit of work in the execution DAG."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    description: str = ""
    # Which files this stage is responsible for (scope guard checks this)
    owned_files: list[str] = field(default_factory=list)
    # Stage IDs that must complete before this one
    depends_on: list[str] = field(default_factory=list)
    # Acceptance criteria — each is a testable statement
    acceptance_criteria: list[str] = field(default_factory=list)
    # Which contracts this stage must conform to
    contracts: list[str] = field(default_factory=list)
    # Execution state
    status: StageStatus = StageStatus.PENDING
    attempts: int = 0
    max_retries: int = 3
    # The generated output (file contents, keyed by path)
    outputs: dict[str, str] = field(default_factory=dict)
    # Last error diagnostic (distilled, not raw stderr)
    last_error: str | None = None


@dataclass
class StageDag:
    """Directed acyclic graph of stages with dependency ordering."""

    stages: dict[str, Stage] = field(default_factory=dict)
    contracts: dict[str, Contract] = field(default_factory=dict)

    def add_stage(self, stage: Stage) -> None:
        self.stages[stage.id] = stage

    def add_contract(self, contract: Contract) -> None:
        self.contracts[contract.name] = contract

    def topological_order(self) -> list[Stage]:
        """Return stages in dependency order. Raises on cycles."""
        visited: set[str] = set()
        temp: set[str] = set()
        order: list[str] = []

        def visit(sid: str) -> None:
            if sid in temp:
                raise ValueError(f"Cycle detected involving stage {sid}")
            if sid in visited:
                return
            temp.add(sid)
            stage = self.stages[sid]
            for dep in stage.depends_on:
                if dep not in self.stages:
                    raise ValueError(f"Stage {sid} depends on unknown stage {dep}")
                visit(dep)
            temp.remove(sid)
            visited.add(sid)
            order.append(sid)

        for sid in self.stages:
            visit(sid)
        return [self.stages[sid] for sid in order]

    def ready_stages(self) -> list[Stage]:
        """Stages whose dependencies are all PASSED and that are still PENDING."""
        return [
            s
            for s in self.stages.values()
            if s.status == StageStatus.PENDING
            and all(
                self.stages[d].status == StageStatus.PASSED
                for d in s.depends_on
                if d in self.stages
            )
        ]


@dataclass
class Spec:
    """Top-level specification that drives the entire pipeline."""

    goal: str
    context: str = ""
    # Raw acceptance criteria before they're decomposed into stages
    acceptance_criteria: list[str] = field(default_factory=list)
    constraints: dict[str, str] = field(default_factory=dict)
    # Populated by Phase 1 (Plan)
    dag: StageDag | None = None
    # Arbitrary metadata
    metadata: dict[str, Any] = field(default_factory=dict)
