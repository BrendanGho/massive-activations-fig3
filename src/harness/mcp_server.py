"""MCP server exposing the knowledge store as tools.

This is the thin boundary that lets Claude Code agents *query* the blackboard
instead of re-reading raw sources — the "strictly necessary tokens" mechanism.
All real logic lives in :mod:`harness.store`; this module only marshals arguments.

Run standalone:        uv run harness-mcp
Configured for Claude:  see .mcp.json (project-scoped stdio server).

The database path comes from ``$HARNESS_DB`` (default ``.harness/kb.sqlite``),
so every agent in the repo shares one substrate.
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from harness.store import Store

_DB_PATH = os.environ.get("HARNESS_DB", ".harness/kb.sqlite")
store = Store(_DB_PATH, check_same_thread=False)

mcp = FastMCP("harness")


def _resolve_entity(entity_type: str | None, entity_name: str | None) -> int | None:
    if entity_type and entity_name:
        return store.upsert_entity(entity_type, entity_name)
    return None


@mcp.tool()
def add_finding(
    content: str,
    task: str | None = None,
    source: str | None = None,
    confidence: float | None = None,
    entity_type: str | None = None,
    entity_name: str | None = None,
) -> dict[str, Any]:
    """Record an observation on the blackboard. The primary way a worker reports
    back. Optionally attach it to an entity (created on demand by type+name).
    """
    entity_id = _resolve_entity(entity_type, entity_name)
    fid = store.add_finding(
        content, task=task, source=source, confidence=confidence, entity_id=entity_id
    )
    return {"id": fid, "entity_id": entity_id}


@mcp.tool()
def query_findings(
    query: str | None = None,
    task: str | None = None,
    entity_type: str | None = None,
    entity_name: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Read findings instead of re-reading source material.

    With ``query`` this runs full-text search and returns a ``snippet`` per hit
    (the matched fragment — cheap to scan). Without ``query`` it filters by ``task``
    and/or entity and returns full ``content`` — use that to read a hit in full.
    Timestamps are omitted and null fields dropped to keep returns small.
    """
    entity_id = None
    if entity_type and entity_name:
        e = store.get_entity(type=entity_type, name=entity_name)
        if e is None:
            return []
        entity_id = e["id"]
    return store.query_findings(query, task=task, entity_id=entity_id, limit=limit)


@mcp.tool()
def upsert_entity(type: str, name: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create or update a durable node (paper, module, person, ...). Idempotent
    on ``(type, name)``; ``data`` is merged into any existing record.
    """
    eid = store.upsert_entity(type, name, data)
    return store.get_entity(id=eid)  # type: ignore[return-value]


@mcp.tool()
def get_entity(type: str, name: str) -> dict[str, Any] | None:
    """Fetch one entity by ``(type, name)``, or null if it does not exist."""
    return store.get_entity(type=type, name=name)


@mcp.tool()
def list_entities(type: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """List entities, most-recently-updated first, optionally filtered by type."""
    return store.list_entities(type=type, limit=limit)


@mcp.tool()
def link_entities(
    src_type: str,
    src_name: str,
    dst_type: str,
    dst_name: str,
    rel_type: str,
) -> dict[str, Any]:
    """Add a typed edge ``src -[rel_type]-> dst`` between two entities, creating
    either endpoint if needed. Turns the blackboard into a graph.
    """
    src = store.upsert_entity(src_type, src_name)
    dst = store.upsert_entity(dst_type, dst_name)
    store.link_entities(src, dst, rel_type)
    return {"src_id": src, "dst_id": dst, "rel_type": rel_type}


@mcp.tool()
def related(
    type: str, name: str, rel_type: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Entities reachable from ``(type, name)`` via outgoing edges."""
    e = store.get_entity(type=type, name=name)
    if e is None:
        return []
    return store.related(e["id"], rel_type=rel_type, limit=limit)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
