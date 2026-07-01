"""Smoke tests for the MCP tool layer.

These verify the marshalling in ``mcp_server`` (entity resolution, return shapes)
on top of an in-memory store — not the stdio transport, which Claude Code drives.
"""

import pytest

from harness.store import Store


@pytest.fixture
def srv(monkeypatch):
    import harness.mcp_server as srv

    monkeypatch.setattr(srv, "store", Store(":memory:"))
    return srv


def test_add_finding_autocreates_entity(srv):
    out = srv.add_finding(
        "loss curve matches figure 3",
        task="reproduce",
        entity_type="paper",
        entity_name="ViT",
    )
    assert out["entity_id"] is not None
    assert srv.get_entity("paper", "ViT") is not None


def test_query_findings_full_text(srv):
    srv.add_finding("gradient exploded at step 400")
    srv.add_finding("training was stable")
    hits = srv.query_findings("exploded")
    assert len(hits) == 1
    assert "exploded" in hits[0]["snippet"]


def test_query_findings_unknown_entity_returns_empty(srv):
    assert srv.query_findings(entity_type="paper", entity_name="ghost") == []


def test_link_and_traverse_graph(srv):
    srv.link_entities("paper", "P", "claim", "C", "makes_claim")
    rel = srv.related("paper", "P")
    assert [r["name"] for r in rel] == ["C"]


def test_upsert_entity_returns_record(srv):
    rec = srv.upsert_entity("module", "auth", {"lang": "py"})
    assert rec["name"] == "auth"
    assert rec["data"] == {"lang": "py"}
