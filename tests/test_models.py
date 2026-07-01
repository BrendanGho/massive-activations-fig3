"""Tests for data models — DAG, contracts, stages."""

import pytest

from harness.models.spec import Contract, Spec, Stage, StageDag, StageStatus


class TestContract:
    def test_signature_deterministic(self):
        c = Contract(name="api", body="class User:\n    name: str")
        assert c.signature() == c.signature()

    def test_signature_changes_with_body(self):
        c1 = Contract(name="api", body="class User:\n    name: str")
        c2 = Contract(name="api", body="class User:\n    name: str\n    email: str")
        assert c1.signature() != c2.signature()


class TestStageDag:
    def _simple_dag(self) -> StageDag:
        dag = StageDag()
        dag.add_stage(Stage(id="a", name="Stage A"))
        dag.add_stage(Stage(id="b", name="Stage B", depends_on=["a"]))
        dag.add_stage(Stage(id="c", name="Stage C", depends_on=["a"]))
        dag.add_stage(Stage(id="d", name="Stage D", depends_on=["b", "c"]))
        return dag

    def test_topological_order(self):
        dag = self._simple_dag()
        order = dag.topological_order()
        ids = [s.id for s in order]
        assert ids.index("a") < ids.index("b")
        assert ids.index("a") < ids.index("c")
        assert ids.index("b") < ids.index("d")
        assert ids.index("c") < ids.index("d")

    def test_cycle_detection(self):
        dag = StageDag()
        dag.add_stage(Stage(id="x", depends_on=["y"]))
        dag.add_stage(Stage(id="y", depends_on=["x"]))
        with pytest.raises(ValueError, match="Cycle"):
            dag.topological_order()

    def test_missing_dependency(self):
        dag = StageDag()
        dag.add_stage(Stage(id="a", depends_on=["nonexistent"]))
        with pytest.raises(ValueError, match="unknown stage"):
            dag.topological_order()

    def test_ready_stages(self):
        dag = self._simple_dag()
        ready = dag.ready_stages()
        assert len(ready) == 1
        assert ready[0].id == "a"

    def test_ready_stages_after_pass(self):
        dag = self._simple_dag()
        dag.stages["a"].status = StageStatus.PASSED
        ready = dag.ready_stages()
        ids = {s.id for s in ready}
        assert ids == {"b", "c"}

    def test_ready_stages_all_done(self):
        dag = self._simple_dag()
        for s in dag.stages.values():
            s.status = StageStatus.PASSED
        assert dag.ready_stages() == []


class TestSpec:
    def test_basic_construction(self):
        spec = Spec(
            goal="Build a CLI tool",
            acceptance_criteria=["It prints hello"],
            constraints={"lang": "Python"},
        )
        assert spec.goal == "Build a CLI tool"
        assert spec.dag is None
