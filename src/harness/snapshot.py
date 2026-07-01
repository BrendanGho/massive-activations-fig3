"""Write a compact, resumable snapshot of harness state.

Invoked by the Stop hook so that when a session ends (or context is compacted) the
next session can recover *what was learned and what's left* by reading one small file
instead of replaying the transcript. Durable knowledge already lives in the SQLite
blackboard; this just surfaces a human/agent-readable pointer to it plus git state.

Writes ``.harness/checkpoints/latest.md`` (overwritten) and a timestamped copy, and
prints a one-line summary to stdout for the hook to echo.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from harness.store import Store


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return ""


def render(db_path: str, limit: int = 15) -> tuple[str, str]:
    """Return ``(markdown, one_line_summary)`` for the current state."""
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "?"
    dirty = _git("status", "--porcelain")
    changed = [ln[3:] for ln in dirty.splitlines()] if dirty else []

    entities: list = []
    findings: list = []
    if Path(db_path).exists():
        store = Store(db_path, check_same_thread=False)
        try:
            entities = store.list_entities(limit=1000)
            findings = store.query_findings(limit=limit)
        finally:
            store.close()

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Harness checkpoint — {ts}",
        "",
        f"- branch: `{branch}`",
        f"- uncommitted files: {len(changed)}",
        f"- entities: {len(entities)}  |  findings shown: {len(findings)}",
        "",
        "## Recent findings (newest first)",
    ]
    if findings:
        for f in findings:
            task = f" [{f['task']}]" if f.get("task") else ""
            src = f" ({f['source']})" if f.get("source") else ""
            seen = f" ×{f['seen']}" if f.get("seen", 1) > 1 else ""
            lines.append(f"- {f['content']}{task}{src}{seen}")
    else:
        lines.append("- (none recorded yet)")

    if changed:
        lines += ["", "## Uncommitted files"] + [f"- {c}" for c in changed[:50]]

    lines += [
        "",
        "## Resume",
        "Query the blackboard via the `harness` MCP tools "
        "(`query_findings`, `list_entities`) before re-reading sources.",
        "",
    ]
    md = "\n".join(lines)
    summary = (
        f"checkpoint: {len(entities)} entities, {len(findings)} recent findings, "
        f"{len(changed)} uncommitted files on {branch}"
    )
    return md, summary


def main() -> None:
    db_path = os.environ.get("HARNESS_DB", ".harness/kb.sqlite")
    out_dir = Path(".harness/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    md, summary = render(db_path)
    (out_dir / "latest.md").write_text(md)
    (out_dir / f"{time.strftime('%Y%m%dT%H%M%S')}.md").write_text(md)
    print(summary)


if __name__ == "__main__":
    main()
