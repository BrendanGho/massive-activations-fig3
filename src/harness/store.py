"""SQLite-backed knowledge store — the harness "blackboard".

This is the substrate that makes agent context *queryable* instead of *re-read*.
It is deliberately a plain library with no MCP or agent dependencies so it can be
unit-tested directly; ``mcp_server.py`` is a thin wrapper that exposes it as tools.

Three primitives, designed to grow into a graph without a schema rewrite:

- **entities**  — durable nodes (a paper, a module, a person), unique by ``(type, name)``.
- **findings**  — append-only observations written by workers; full-text searchable.
- **relations** — typed edges between entities, so the blackboard is a graph.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id      INTEGER PRIMARY KEY,
    type    TEXT NOT NULL,
    name    TEXT NOT NULL,
    data    TEXT NOT NULL DEFAULT '{}',
    created REAL NOT NULL,
    updated REAL NOT NULL,
    UNIQUE(type, name)
);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);

CREATE TABLE IF NOT EXISTS findings (
    id         INTEGER PRIMARY KEY,
    content    TEXT NOT NULL,
    task       TEXT,
    source     TEXT,
    confidence REAL,
    entity_id  INTEGER REFERENCES entities(id) ON DELETE SET NULL,
    seen       INTEGER NOT NULL DEFAULT 1,
    created    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_findings_task   ON findings(task);
CREATE INDEX IF NOT EXISTS idx_findings_entity ON findings(entity_id);
-- Backs O(log n) exact-duplicate detection on write (content + task + entity).
CREATE INDEX IF NOT EXISTS idx_findings_dedup  ON findings(content, task, entity_id);

CREATE TABLE IF NOT EXISTS relations (
    src_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    dst_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    rel_type TEXT NOT NULL,
    created  REAL NOT NULL,
    PRIMARY KEY (src_id, dst_id, rel_type)
);
CREATE INDEX IF NOT EXISTS idx_relations_dst ON relations(dst_id);

-- Full-text index over finding content, kept in sync via triggers below.
CREATE VIRTUAL TABLE IF NOT EXISTS findings_fts USING fts5(
    content,
    content='findings',
    content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS findings_ai AFTER INSERT ON findings BEGIN
    INSERT INTO findings_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS findings_ad AFTER DELETE ON findings BEGIN
    INSERT INTO findings_fts(findings_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS findings_au AFTER UPDATE ON findings BEGIN
    INSERT INTO findings_fts(findings_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO findings_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


def _now() -> float:
    return time.time()


def _fts_query(raw: str) -> str:
    """Make an arbitrary string safe for an FTS5 ``MATCH``.

    Each whitespace-separated token is wrapped in double quotes (embedded quotes
    doubled), so punctuation and reserved words are matched *literally* instead of
    parsed as query syntax. Without this, natural research queries like ``ViT-L/16``
    or ``accuracy (top-1)`` raise ``OperationalError`` — which, in the MCP path,
    becomes a wasted round-trip that punishes the query-before-reading behavior we want.
    """
    tokens = raw.split()
    return " ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def _compact(d: dict[str, Any]) -> dict[str, Any]:
    """Drop keys that carry no signal, so every returned row costs fewer tokens.

    Removes ``None`` values and empty containers/strings; keeps ``0``/``0.0``/``False``
    (a confidence of ``0.0`` is meaningful). Applied to every row the store returns.
    """
    return {
        k: v for k, v in d.items() if v is not None and v != {} and v != [] and v != ""
    }


class Store:
    """A blackboard backed by a single SQLite file (or ``:memory:`` for tests)."""

    def __init__(
        self, path: str | Path = ":memory:", *, check_same_thread: bool = True
    ) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # MCP runs sync tools in a threadpool; the server passes check_same_thread=False
        # so one Store can serve calls arriving on different worker threads.
        self._conn = sqlite3.connect(self.path, check_same_thread=check_same_thread)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Additive migrations for DBs created by an earlier schema version.

        ``CREATE TABLE IF NOT EXISTS`` never adds columns to an existing table, so a
        pre-``seen`` blackboard needs an explicit ``ALTER``. Kept idempotent and cheap.
        """
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(findings)")}
        if "seen" not in cols:
            self._conn.execute(
                "ALTER TABLE findings ADD COLUMN seen INTEGER NOT NULL DEFAULT 1"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_findings_dedup "
                "ON findings(content, task, entity_id)"
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- entities ---------------------------------------------------------

    def upsert_entity(
        self, type: str, name: str, data: dict[str, Any] | None = None
    ) -> int:
        """Create an entity, or merge ``data`` into an existing ``(type, name)``.

        Returns the entity id. Idempotent on ``(type, name)`` so concurrent
        workers converge on one node instead of duplicating it.
        """
        now = _now()
        existing = self.get_entity(type=type, name=name)
        if existing is None:
            cur = self._conn.execute(
                "INSERT INTO entities(type, name, data, created, updated) "
                "VALUES (?, ?, ?, ?, ?)",
                (type, name, json.dumps(data or {}), now, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)
        merged = {**existing.get("data", {}), **(data or {})}
        self._conn.execute(
            "UPDATE entities SET data = ?, updated = ? WHERE id = ?",
            (json.dumps(merged), now, existing["id"]),
        )
        self._conn.commit()
        return int(existing["id"])

    def get_entity(
        self,
        *,
        id: int | None = None,
        type: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch one entity by id, or by ``(type, name)``."""
        if id is not None:
            row = self._conn.execute(
                "SELECT * FROM entities WHERE id = ?", (id,)
            ).fetchone()
        elif type is not None and name is not None:
            row = self._conn.execute(
                "SELECT * FROM entities WHERE type = ? AND name = ?", (type, name)
            ).fetchone()
        else:
            raise ValueError("get_entity requires id, or both type and name")
        return self._entity_row(row)

    def list_entities(
        self, type: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        if type is None:
            rows = self._conn.execute(
                "SELECT * FROM entities ORDER BY updated DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM entities WHERE type = ? ORDER BY updated DESC LIMIT ?",
                (type, limit),
            ).fetchall()
        return [self._entity_row(r) for r in rows if r]  # type: ignore[misc]

    # -- findings ---------------------------------------------------------

    def add_finding(
        self,
        content: str,
        *,
        task: str | None = None,
        source: str | None = None,
        confidence: float | None = None,
        entity_id: int | None = None,
    ) -> int:
        """Record a worker observation — the primary coordination write.

        **Idempotent on ``(content, task, entity_id)``.** Re-observing an identical
        finding does not append a duplicate row; it bumps that finding's ``seen``
        counter and refreshes its recency, then returns the existing id. This keeps
        the blackboard from bloating across re-runs (so every later ``query_findings``
        stays lean) and turns repetition into a corroboration signal. ``IS`` is used
        so untagged findings (``task``/``entity_id`` NULL) dedupe correctly too.

        Only byte-identical content collapses; reworded findings are kept as distinct
        (semantic dedup would need embeddings and is intentionally out of scope).
        """
        now = _now()
        existing = self._conn.execute(
            "SELECT id, seen FROM findings "
            "WHERE content = ? AND task IS ? AND entity_id IS ?",
            (content, task, entity_id),
        ).fetchone()
        if existing is not None:
            self._conn.execute(
                "UPDATE findings SET seen = seen + 1, created = ? WHERE id = ?",
                (now, existing["id"]),
            )
            self._conn.commit()
            return int(existing["id"])
        cur = self._conn.execute(
            "INSERT INTO findings(content, task, source, confidence, entity_id, created) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (content, task, source, confidence, entity_id, now),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def query_findings(
        self,
        query: str | None = None,
        *,
        task: str | None = None,
        entity_id: int | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Read findings, returning only signal-bearing fields (no timestamps, nulls dropped).

        Two modes, tuned for token cost:

        - **search** (``query`` given): relevance-ranked FTS5 match. Returns a
          ``snippet`` — the matched fragment, full text for short findings and a
          ~64-token window for long ones — so scanning many hits stays cheap.
        - **filter** (``task``/``entity_id``, no ``query``): newest first, returns
          full ``content``. This is the path to read a finding in full after a search.
        """
        params: list[Any] = []
        fts = _fts_query(query) if query else ""
        if fts:
            sql = (
                "SELECT f.id, snippet(findings_fts, 0, '', '', '…', 64) AS snippet, "
                "f.task, f.source, f.confidence, f.entity_id, f.seen "
                "FROM findings_fts fts JOIN findings f ON f.id = fts.rowid "
                "WHERE findings_fts MATCH ?"
            )
            params.append(fts)
            if task is not None:
                sql += " AND f.task = ?"
                params.append(task)
            if entity_id is not None:
                sql += " AND f.entity_id = ?"
                params.append(entity_id)
            sql += " ORDER BY fts.rank LIMIT ?"
        else:
            sql = (
                "SELECT id, content, task, source, confidence, entity_id, seen "
                "FROM findings f WHERE 1=1"
            )
            if task is not None:
                sql += " AND f.task = ?"
                params.append(task)
            if entity_id is not None:
                sql += " AND f.entity_id = ?"
                params.append(entity_id)
            sql += " ORDER BY f.created DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._finding_row(dict(r)) for r in rows]

    @staticmethod
    def _finding_row(d: dict[str, Any]) -> dict[str, Any]:
        """Compact a finding row; hide ``seen`` unless the finding was corroborated
        (>1), so the common case stays token-free and repetition shows only when it matters.
        """
        d = _compact(d)
        if d.get("seen") == 1:
            d.pop("seen", None)
        return d

    # -- relations --------------------------------------------------------

    def link_entities(self, src_id: int, dst_id: int, rel_type: str) -> None:
        """Add a typed edge ``src -[rel_type]-> dst`` (idempotent)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO relations(src_id, dst_id, rel_type, created) "
            "VALUES (?, ?, ?, ?)",
            (src_id, dst_id, rel_type, _now()),
        )
        self._conn.commit()

    def related(
        self, entity_id: int, rel_type: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Entities reachable from ``entity_id`` via outgoing edges."""
        sql = (
            "SELECT e.*, r.rel_type AS rel_type FROM relations r "
            "JOIN entities e ON e.id = r.dst_id WHERE r.src_id = ?"
        )
        params: list[Any] = [entity_id]
        if rel_type is not None:
            sql += " AND r.rel_type = ?"
            params.append(rel_type)
        sql += " LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = self._entity_row(r)
            if d is not None:
                d["rel_type"] = r["rel_type"]
                out.append(d)
        return out

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _entity_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        """Project an entity to signal-bearing fields: id, type, name, and non-empty
        ``data``. Drops ``created``/``updated`` epoch floats — they cost tokens on
        every read and are rarely what an agent needs.
        """
        if row is None:
            return None
        d = dict(row)
        d["data"] = json.loads(d["data"]) if d.get("data") else {}
        d.pop("created", None)
        d.pop("updated", None)
        return _compact(d)
