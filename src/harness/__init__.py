"""Agent Harness — a reusable agentic pipeline plus a queryable knowledge substrate.

Two complementary halves of one package:

- **Pipeline** (`core/`, `gates/`, `models/`, `providers/`): plan contracts, execute in a
  gated feedback loop, integrate, review.
- **Blackboard** (`store`, `mcp_server`, `snapshot`): a SQLite knowledge substrate agents
  query and coordinate through instead of passing raw context around.
"""

from harness.core.pipeline import Pipeline
from harness.models.spec import Contract, Spec, Stage, StageDag
from harness.store import Store

__all__ = ["Pipeline", "Spec", "Stage", "Contract", "StageDag", "Store"]
