"""CLI entry point — `harness run spec.yaml`"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import yaml

from harness.core.pipeline import Pipeline, PipelineConfig
from harness.gates.custom import CustomCommandGate
from harness.models.spec import Spec


def _load_spec(path: Path) -> tuple[Spec, dict]:
    """Load a spec from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)

    spec = Spec(
        goal=data["goal"],
        context=data.get("context", ""),
        acceptance_criteria=data.get("acceptance_criteria", []),
        constraints=data.get("constraints", {}),
        metadata=data.get("metadata", {}),
    )
    return spec, data


def _build_config(data: dict, args: argparse.Namespace) -> PipelineConfig:
    """Build pipeline config from spec YAML + CLI overrides."""
    config = PipelineConfig()

    harness_cfg = data.get("harness", {})
    config.smoke_command = harness_cfg.get("smoke_command") or args.smoke_command
    config.skip_review = harness_cfg.get("skip_review", args.skip_review)
    config.default_max_retries = harness_cfg.get("max_retries", args.max_retries)
    config.validate_plan = not args.skip_plan_validation

    if args.event_log:
        config.event_log_path = args.event_log

    # Custom gates from config
    for gate_def in harness_cfg.get("gates", []):
        config.stage_gates.append(
            CustomCommandGate(name=gate_def["name"], command=gate_def["command"])
        )

    return config


def _build_provider(args: argparse.Namespace):
    """Instantiate the right provider based on CLI flags."""
    if args.provider == "cli":
        from harness.providers.claude_cli import ClaudeCliProvider

        return ClaudeCliProvider(
            strong_model=args.strong_model or "opus",
            cheap_model=args.cheap_model or "haiku",
        )
    else:
        from harness.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            strong_model=args.strong_model or "claude-sonnet-4-20250514",
            cheap_model=args.cheap_model or "claude-haiku-4-5-20251001",
        )


async def _run(args: argparse.Namespace) -> int:
    spec, data = _load_spec(Path(args.spec))
    config = _build_config(data, args)
    provider = _build_provider(args)
    workspace = Path(args.workspace)

    pipeline = Pipeline(provider=provider, workspace=workspace, config=config)
    result = await pipeline.run(spec)

    # Print summary
    summary = result.event_log.summary()
    print(f"\n{'=' * 60}")
    print(f"Pipeline: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Elapsed: {result.elapsed_seconds:.1f}s")
    print(f"Stages passed: {summary['stages_passed']}")
    print(f"Stages failed: {summary['stages_failed']}")
    print(f"Total retries: {summary['total_retries']}")

    if result.plan_warnings:
        print(f"\nPlan warnings:")
        for w in result.plan_warnings:
            print(f"  - {w}")

    if result.review_report:
        print(f"\nReview: {'PASSED' if result.review_report.passed else 'BLOCKING ISSUES'}")
        print(f"  {result.review_report.summary}")
        for f in result.review_report.blocking:
            print(f"  [BLOCKING] {f.file}: {f.message}")

    if result.outputs:
        print(f"\nGenerated files:")
        for path in sorted(result.outputs):
            print(f"  {path}")

    print(f"{'=' * 60}")

    return 0 if result.success else 1


def main():
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Agentic pipeline: plan contracts, execute in gated loops, integrate, review.",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run the pipeline against a spec file")
    run_p.add_argument("spec", help="Path to spec YAML file")
    run_p.add_argument("-w", "--workspace", default="./output", help="Output workspace directory")
    run_p.add_argument("--provider", choices=["api", "cli"], default="api", help="LLM provider")
    run_p.add_argument("--strong-model", help="Model for planning/review")
    run_p.add_argument("--cheap-model", help="Model for execution")
    run_p.add_argument("--smoke-command", help="Shell command for smoke testing")
    run_p.add_argument("--max-retries", type=int, default=3, help="Max retries per stage")
    run_p.add_argument("--skip-review", action="store_true", help="Skip Phase 4 review")
    run_p.add_argument("--skip-plan-validation", action="store_true")
    run_p.add_argument("--event-log", help="Path for JSONL event log")
    run_p.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
