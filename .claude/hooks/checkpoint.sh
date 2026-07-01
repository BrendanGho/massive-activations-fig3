#!/usr/bin/env bash
# Stop hook: snapshot harness state so a fresh session can resume without replaying
# the transcript. Non-destructive — writes .harness/checkpoints/, never commits.
# Kept fast and best-effort: never block or fail the session on snapshot problems.
set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

if command -v uv >/dev/null 2>&1; then
  uv run --quiet harness-snapshot 2>/dev/null || true
else
  python3 -m harness.snapshot 2>/dev/null || true
fi

exit 0
