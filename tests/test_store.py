"""Tests for the knowledge store — the source of truth for the blackboard."""

import sqlite3

import pytest

from harness.store import Store


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


# -- entities -------------------------------------------------------------


def test_upsert_entity_is_idempotent_on_type_name(store):
    a = store.upsert_entity("paper", "Attention Is All You Need")
    b = store.upsert_entity("paper", "Attention Is All You Need")
    assert a == b
    assert len(store.list_entities("paper")) == 1


def test_upsert_entity_merges_data(store):
    eid = store.upsert_entity("paper", "P", {"year": 2017})
    store.upsert_entity("paper", "P", {"venue": "NeurIPS"})
    e = store.get_entity(id=eid)
    assert e["data"] == {"year": 2017, "venue": "NeurIPS"}


def test_get_entity_by_type_name_and_missing(store):
    store.upsert_entity("module", "auth")
    assert store.get_entity(type="module", name="auth") is not None
    assert store.get_entity(type="module", name="nope") is None


def test_get_entity_requires_id_or_type_name(store):
    with pytest.raises(ValueError):
        store.get_entity(type="paper")


def test_list_entities_filters_by_type(store):
    store.upsert_entity("paper", "P1")
    store.upsert_entity("module", "M1")
    assert {e["name"] for e in store.list_entities("paper")} == {"P1"}
    assert len(store.list_entities()) == 2


def test_entity_read_drops_timestamps_and_empty_data(store):
    store.upsert_entity("paper", "P")  # no data
    e = store.get_entity(type="paper", name="P")
    assert "created" not in e and "updated" not in e  # epoch floats are noise
    assert "data" not in e  # empty {} omitted
    assert e["name"] == "P"


# -- findings -------------------------------------------------------------


def test_add_and_query_finding_by_task(store):
    store.add_finding("transformer beats RNN on BLEU", task="reproduce-1")
    store.add_finding("unrelated note", task="other")
    rows = store.query_findings(task="reproduce-1")
    assert len(rows) == 1
    assert rows[0]["content"].startswith("transformer")


def test_query_findings_full_text_search(store):
    store.add_finding("the migration failed under concurrent writes")
    store.add_finding("everything is fine here")
    hits = store.query_findings("migration")
    assert len(hits) == 1
    # search returns a snippet, not full content
    assert "migration" in hits[0]["snippet"]


def test_query_findings_sanitizes_special_chars(store):
    # Natural research queries that raise FTS5 syntax errors without sanitizing.
    store.add_finding("ViT-L/16 reached 85% top-1 accuracy")
    for q in ["ViT-L/16", "accuracy (top-1)", "gradient AND", '"unterminated']:
        hits = store.query_findings(q)  # must not raise
        assert isinstance(hits, list)
    assert len(store.query_findings("ViT-L/16")) == 1


def test_query_findings_snippet_windows_long_content(store):
    long = "intro padding. " * 40 + "the SIGNAL token is here. " + "tail padding. " * 40
    store.add_finding(long)
    hit = store.query_findings("SIGNAL")[0]
    assert "SIGNAL" in hit["snippet"]
    assert len(hit["snippet"]) < len(long)  # windowed, not the whole finding
    assert "…" in hit["snippet"]


def test_query_findings_drops_timestamps_and_nulls(store):
    store.add_finding("bare finding")  # no task/source/confidence/entity
    row = store.query_findings(task=None)[0]
    assert "created" not in row  # epoch float is noise
    assert "source" not in row and "confidence" not in row  # nulls omitted
    assert row["content"] == "bare finding"


def test_query_findings_fts_survives_delete_and_reindex(store):
    fid = store.add_finding("ephemeral concurrent finding")
    store._conn.execute("DELETE FROM findings WHERE id = ?", (fid,))
    store._conn.commit()
    assert store.query_findings("ephemeral") == []


def test_finding_links_to_entity(store):
    eid = store.upsert_entity("paper", "P")
    store.add_finding("claim X holds", entity_id=eid, confidence=0.9)
    rows = store.query_findings(entity_id=eid)
    assert len(rows) == 1
    assert rows[0]["confidence"] == 0.9


# -- dedup / accumulation -------------------------------------------------


def test_add_finding_dedups_identical_and_counts_seen(store):
    a = store.add_finding("loss matches Table 2", task="reproduce")
    b = store.add_finding("loss matches Table 2", task="reproduce")
    assert a == b  # same row, no duplicate appended
    rows = store.query_findings(task="reproduce")
    assert len(rows) == 1
    assert rows[0]["seen"] == 2


def test_dedup_distinguishes_task_and_entity(store):
    eid = store.upsert_entity("paper", "P")
    store.add_finding("same text", task="t1")
    store.add_finding("same text", task="t2")
    store.add_finding("same text", entity_id=eid)
    assert len(store.query_findings()) == 3  # different key -> distinct rows


def test_dedup_handles_untagged_null_key(store):
    # NULL task/entity must still dedup (uses `IS`, not `=`).
    store.add_finding("bare observation")
    store.add_finding("bare observation")
    rows = store.query_findings()
    assert len(rows) == 1
    assert rows[0]["seen"] == 2


def test_seen_hidden_until_corroborated(store):
    store.add_finding("once", task="t")
    assert "seen" not in store.query_findings(task="t")[0]  # token-free when seen==1
    store.add_finding("once", task="t")
    assert store.query_findings(task="t")[0]["seen"] == 2


def test_dedup_refreshes_recency(store):
    store.add_finding("older", task="t")
    store.add_finding("newer", task="t")
    store.add_finding("older", task="t")  # re-observe -> bumps to front
    order = [r["content"] for r in store.query_findings(task="t")]
    assert order == ["older", "newer"]


def test_migration_adds_seen_to_legacy_db(tmp_path):
    db = tmp_path / "legacy.sqlite"
    raw = sqlite3.connect(db)
    raw.execute(
        "CREATE TABLE findings (id INTEGER PRIMARY KEY, content TEXT NOT NULL, "
        "task TEXT, source TEXT, confidence REAL, entity_id INTEGER, created REAL NOT NULL)"
    )
    raw.commit()
    raw.close()

    s = Store(db)  # __init__ should ALTER TABLE to add `seen`
    cols = {r["name"] for r in s._conn.execute("PRAGMA table_info(findings)")}
    assert "seen" in cols
    s.add_finding("post-migration", task="t")
    s.add_finding("post-migration", task="t")
    assert s.query_findings(task="t")[0]["seen"] == 2
    s.close()


# -- relations ------------------------------------------------------------


def test_link_entities_is_idempotent_and_traversable(store):
    paper = store.upsert_entity("paper", "P")
    claim = store.upsert_entity("claim", "C")
    store.link_entities(paper, claim, "makes_claim")
    store.link_entities(paper, claim, "makes_claim")  # dup ignored
    rel = store.related(paper)
    assert len(rel) == 1
    assert rel[0]["name"] == "C"
    assert rel[0]["rel_type"] == "makes_claim"


def test_related_filters_by_rel_type(store):
    p = store.upsert_entity("paper", "P")
    c = store.upsert_entity("claim", "C")
    e = store.upsert_entity("experiment", "E")
    store.link_entities(p, c, "makes_claim")
    store.link_entities(p, e, "ran_experiment")
    assert len(store.related(p, "makes_claim")) == 1
    assert len(store.related(p)) == 2


def test_cascade_delete_removes_relations(store):
    p = store.upsert_entity("paper", "P")
    c = store.upsert_entity("claim", "C")
    store.link_entities(p, c, "makes_claim")
    store._conn.execute("DELETE FROM entities WHERE id = ?", (p,))
    store._conn.commit()
    assert store.related(p) == []


def test_persists_to_disk(tmp_path):
    db = tmp_path / "nested" / "kb.sqlite"
    s1 = Store(db)
    eid = s1.upsert_entity("paper", "Durable")
    s1.add_finding("survives restart", entity_id=eid)
    s1.close()
    s2 = Store(db)
    assert s2.get_entity(type="paper", name="Durable") is not None
    assert len(s2.query_findings("survives")) == 1
    s2.close()
