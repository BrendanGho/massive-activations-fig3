"""Tests for the resumable checkpoint snapshot."""

from harness.snapshot import render
from harness.store import Store


def test_render_reports_findings_and_entities(tmp_path):
    db = tmp_path / "kb.sqlite"
    s = Store(db)
    eid = s.upsert_entity("paper", "ViT")
    s.add_finding("accuracy within 0.2% of Table 2", task="reproduce", entity_id=eid)
    s.close()

    md, summary = render(str(db))
    assert "accuracy within 0.2%" in md
    assert "[reproduce]" in md
    assert "1 entities" in summary


def test_render_handles_missing_db(tmp_path):
    md, summary = render(str(tmp_path / "does-not-exist.sqlite"))
    assert "none recorded yet" in md
    assert "0 entities" in summary
