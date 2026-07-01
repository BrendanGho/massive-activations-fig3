"""Integration test — full pipeline with a mock provider."""

import json
import pytest
from pathlib import Path

from harness.core.pipeline import Pipeline, PipelineConfig
from harness.models.spec import Spec
from harness.providers.base import (
    GenerateRequest,
    GenerateResponse,
    ModelTier,
    Provider,
)


class MockProvider(Provider):
    """Deterministic mock that returns pre-scripted responses by call index."""

    def __init__(self, responses: list[dict]):
        self._responses = responses
        self._call_index = 0

    def tier_model(self, tier: ModelTier) -> str:
        return "mock-strong" if tier == ModelTier.STRONG else "mock-cheap"

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        if self._call_index >= len(self._responses):
            raise RuntimeError(f"MockProvider exhausted after {self._call_index} calls")

        resp_data = self._responses[self._call_index]
        self._call_index += 1

        return GenerateResponse(
            content=json.dumps(resp_data),
            model=self.tier_model(request.tier),
            parsed=resp_data,
        )


# Pre-scripted responses for a simple 2-stage pipeline
PLAN_RESPONSE = {
    "stages": [
        {
            "id": "models",
            "name": "Data Models",
            "description": "Define the User model",
            "owned_files": ["models.py"],
            "depends_on": [],
            "acceptance_criteria": ["User class exists with name and email fields"],
            "contracts": ["user_contract"],
        },
        {
            "id": "api",
            "name": "API Layer",
            "description": "HTTP endpoint for users",
            "owned_files": ["api.py"],
            "depends_on": ["models"],
            "acceptance_criteria": ["get_user function exists and returns a User"],
            "contracts": ["user_contract"],
        },
    ],
    "contracts": [
        {
            "name": "user_contract",
            "body": "class User:\n    name: str\n    email: str",
            "producers": ["models"],
            "consumers": ["api"],
        }
    ],
}

PLAN_VALIDATION_RESPONSE = []  # No warnings

STAGE_1_RESPONSE = {
    "models.py": "class User:\n    def __init__(self, name: str, email: str):\n        self.name = name\n        self.email = email\n"
}

STAGE_2_RESPONSE = {
    "api.py": "from models import User\n\ndef get_user(user_id: int) -> User:\n    return User(name='test', email='test@example.com')\n"
}

REVIEW_RESPONSE = {
    "passed": True,
    "summary": "All acceptance criteria met. No blocking issues.",
    "findings": [
        {
            "severity": "nit",
            "category": "quality",
            "file": "api.py",
            "message": "Consider adding error handling for invalid user_id",
            "line": 4,
        }
    ],
}


@pytest.mark.asyncio
async def test_full_pipeline(tmp_path: Path):
    """End-to-end test: plan → execute → integrate → review with mock provider."""

    provider = MockProvider(
        [
            PLAN_RESPONSE,  # Phase 1: plan
            PLAN_VALIDATION_RESPONSE,  # Phase 1: validate
            STAGE_1_RESPONSE,  # Phase 2: stage 1
            STAGE_2_RESPONSE,  # Phase 2: stage 2
            REVIEW_RESPONSE,  # Phase 4: review
        ]
    )

    spec = Spec(
        goal="Build a user API",
        acceptance_criteria=[
            "User class with name and email",
            "get_user endpoint returns a User",
        ],
    )

    config = PipelineConfig(
        validate_plan=True,
        skip_review=False,
        smoke_command=None,  # Skip smoke test in unit test
    )

    pipeline = Pipeline(provider=provider, workspace=tmp_path, config=config)
    result = await pipeline.run(spec)

    # Verify structure
    assert result.dag is not None
    assert len(result.dag.stages) == 2
    assert len(result.dag.contracts) == 1

    # Verify outputs exist
    assert "models.py" in result.outputs
    assert "api.py" in result.outputs
    assert "class User" in result.outputs["models.py"]

    # Verify review ran
    assert result.review_report is not None
    assert result.review_report.passed
    assert len(result.review_report.findings) == 1

    # Verify event log
    summary = result.event_log.summary()
    assert summary["stages_passed"] == 2
    assert summary["stages_failed"] == 0


@pytest.mark.asyncio
async def test_pipeline_skip_review(tmp_path: Path):
    provider = MockProvider(
        [
            PLAN_RESPONSE,
            PLAN_VALIDATION_RESPONSE,
            STAGE_1_RESPONSE,
            STAGE_2_RESPONSE,
            # No review response needed
        ]
    )

    spec = Spec(goal="Build a user API", acceptance_criteria=["User class exists"])
    config = PipelineConfig(skip_review=True, validate_plan=True)

    pipeline = Pipeline(provider=provider, workspace=tmp_path, config=config)
    result = await pipeline.run(spec)

    assert result.success
    assert result.review_report is None
