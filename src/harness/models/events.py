"""Structured event log for observability."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class EventKind(Enum):
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    STAGE_START = "stage_start"
    STAGE_PASS = "stage_pass"
    STAGE_FAIL = "stage_fail"
    STAGE_RETRY = "stage_retry"
    STAGE_ESCALATE = "stage_escalate"
    GATE_RUN = "gate_run"
    GATE_PASS = "gate_pass"
    GATE_FAIL = "gate_fail"
    CHECKPOINT_CREATE = "checkpoint_create"
    CHECKPOINT_REVERT = "checkpoint_revert"
    ERROR_DISTILL = "error_distill"
    REVIEW_FINDING = "review_finding"


@dataclass
class Event:
    kind: EventKind
    stage_id: str | None = None
    phase: str | None = None
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventLog:
    """Append-only structured event log. Writes JSONL to disk and keeps in-memory."""

    def __init__(self, path: Path | None = None):
        self._events: list[Event] = []
        self._path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Event) -> None:
        self._events.append(event)
        if self._path:
            with open(self._path, "a") as f:
                d = asdict(event)
                d["kind"] = event.kind.value
                f.write(json.dumps(d) + "\n")

    def events(self, kind: EventKind | None = None, stage_id: str | None = None) -> list[Event]:
        out = self._events
        if kind:
            out = [e for e in out if e.kind == kind]
        if stage_id:
            out = [e for e in out if e.stage_id == stage_id]
        return out

    def retry_count(self, stage_id: str) -> int:
        return len(self.events(EventKind.STAGE_RETRY, stage_id))

    def summary(self) -> dict[str, Any]:
        """Quick stats for the pipeline run."""
        return {
            "total_events": len(self._events),
            "stages_passed": len(self.events(EventKind.STAGE_PASS)),
            "stages_failed": len(self.events(EventKind.STAGE_FAIL)),
            "stages_escalated": len(self.events(EventKind.STAGE_ESCALATE)),
            "total_retries": len(self.events(EventKind.STAGE_RETRY)),
            "gate_failures": len(self.events(EventKind.GATE_FAIL)),
        }
