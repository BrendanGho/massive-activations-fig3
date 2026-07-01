"""Configuration loading from YAML spec files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from harness.models.spec import Spec


def load_spec_from_yaml(path: str | Path) -> tuple[Spec, dict[str, Any]]:
    """Load a Spec and raw config from a YAML file.

    Expected format:
        goal: "Build a REST API for user management"
        context: "We're using FastAPI with SQLAlchemy..."
        acceptance_criteria:
          - "GET /users returns a list of users"
          - "POST /users creates a new user"
        constraints:
          stack: "Python 3.11, FastAPI, SQLAlchemy"
          style: "Google docstrings, type hints everywhere"
        harness:
          smoke_command: "pytest -q"
          max_retries: 3
          skip_review: false
          gates:
            - name: typecheck
              command: "mypy src/"
            - name: lint
              command: "ruff check src/"
    """
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
