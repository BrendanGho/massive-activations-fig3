"""Pipeline — the top-level orchestrator that wires phases 1–4 together.

Usage:
    spec = Spec(goal="...", acceptance_criteria=[...])
    pipeline = Pipeline(provider=AnthropicProvider(), workspace=Path("./output"))
    report = await pipeline.run(spec)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.checkpoints.manager import CheckpointManager
from harness.core.executor import execute_all
from harness.core.integrator import integrate
from harness.core.planner import plan, validate_plan
from harness.core.reviewer import ReviewReport, review
from harness.gates.base import Gate
from harness.gates.contract import ContractGate
from harness.gates.scope import ScopeGate
from harness.gates.syntax import SyntaxGate
from harness.models.events import Event, EventKind, EventLog
from harness.models.spec import Spec, StageDag
from harness.providers.base import Provider

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Tunable pipeline parameters."""

    # Phase 1
    validate_plan: bool = True

    # Phase 2
    default_max_retries: int = 3
    stage_gates: list[Gate] = field(
        default_factory=lambda: [
            SyntaxGate(),
            ContractGate(),
            ScopeGate(),
        ]
    )

    # Phase 3
    smoke_command: str | None = None
    integration_gates: list[Gate] = field(default_factory=list)

    # Phase 4
    skip_review: bool = False

    # Logging
    event_log_path: str | None = None  # JSONL file path


@dataclass
class PipelineResult:
    """Everything you need to know about a pipeline run."""

    success: bool
    dag: StageDag | None = None
    review_report: ReviewReport | None = None
    event_log: EventLog = field(default_factory=EventLog)
    elapsed_seconds: float = 0.0
    plan_warnings: list[str] = field(default_factory=list)
    integration_passed: bool = False

    @property
    def outputs(self) -> dict[str, str]:
        """All generated files across all stages."""
        if not self.dag:
            return {}
        out: dict[str, str] = {}
        for stage in self.dag.stages.values():
            out.update(stage.outputs)
        return out


class Pipeline:
    """The main orchestrator. Runs Plan → Execute → Integrate → Review."""

    def __init__(
        self,
        provider: Provider,
        workspace: Path,
        config: PipelineConfig | None = None,
    ):
        self._provider = provider
        self._workspace = Path(workspace)
        self._config = config or PipelineConfig()

    async def run(self, spec: Spec, extra_context: str = "") -> PipelineResult:
        """Run the full pipeline against a spec."""

        start = time.time()

        log_path = Path(self._config.event_log_path) if self._config.event_log_path else None
        event_log = EventLog(path=log_path)
        event_log.emit(Event(kind=EventKind.PIPELINE_START, message=spec.goal))

        result = PipelineResult(success=False, event_log=event_log)

        try:
            # ── Phase 1: Plan ──────────────────────────────────
            logger.info("Phase 1: Planning...")
            dag = await plan(spec, self._provider, event_log, extra_context)
            spec.dag = dag
            result.dag = dag

            # Optional plan validation (cheap model dry-run)
            if self._config.validate_plan:
                warnings = await validate_plan(dag, self._provider, event_log)
                result.plan_warnings = warnings
                if warnings:
                    logger.warning(f"Plan validation warnings: {warnings}")

            # Apply default max_retries to stages
            for stage in dag.stages.values():
                if stage.max_retries == 3:  # only override default
                    stage.max_retries = self._config.default_max_retries

            # ── Phase 2: Execute ───────────────────────────────
            logger.info("Phase 2: Executing stages...")
            self._workspace.mkdir(parents=True, exist_ok=True)
            checkpoint_mgr = CheckpointManager(self._workspace)
            await checkpoint_mgr.snapshot_baseline()

            exec_passed = await execute_all(
                dag=dag,
                provider=self._provider,
                gates=self._config.stage_gates,
                checkpoint_mgr=checkpoint_mgr,
                event_log=event_log,
                workspace=self._workspace,
            )

            if not exec_passed:
                logger.error("Phase 2 failed: some stages did not pass")
                result.success = False
                return result

            # ── Phase 3: Integrate ─────────────────────────────
            logger.info("Phase 3: Integration checks...")
            int_passed = await integrate(
                dag=dag,
                event_log=event_log,
                workspace=self._workspace,
                smoke_command=self._config.smoke_command,
                extra_gates=self._config.integration_gates,
            )
            result.integration_passed = int_passed

            if not int_passed:
                logger.error("Phase 3 failed: integration checks did not pass")
                result.success = False
                return result

            # ── Phase 4: Review ────────────────────────────────
            if not self._config.skip_review:
                logger.info("Phase 4: Review...")
                report = await review(spec, dag, event_log, self._provider)
                result.review_report = report
                result.success = report.passed
            else:
                result.success = True

        except Exception as e:
            logger.exception(f"Pipeline failed with exception: {e}")
            event_log.emit(
                Event(
                    kind=EventKind.PIPELINE_END,
                    message=f"Failed: {e}",
                )
            )
            result.success = False
            raise
        finally:
            result.elapsed_seconds = time.time() - start
            event_log.emit(
                Event(
                    kind=EventKind.PIPELINE_END,
                    message=f"{'Success' if result.success else 'Failed'} in {result.elapsed_seconds:.1f}s",
                    data=event_log.summary(),
                )
            )

        return result
