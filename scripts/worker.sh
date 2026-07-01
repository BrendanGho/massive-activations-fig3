#!/usr/bin/env bash
# worker.sh "<detailed task>"
# Hands a bounded, well-specified task to a FREE model via free-claude-code (fcc), running headless
# in the current repo. Use for high-volume, mechanical, testable work to save premium-model budget.
# Always review the resulting `git diff` and tests before accepting.
#
# Requires: `fcc-server` running, with LM Studio / NIM configured in the fcc Admin UI.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: scripts/worker.sh \"<task: what to do, which files, how to test>\"" >&2
  exit 1
fi

TASK="$1"
PROXY_URL="${FCC_URL:-http://localhost:8082}"

# Safety: refuse to run on the default branch so a bad run is easy to discard.
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
  echo "Refusing to run the worker on '$BRANCH'. Create a feature branch first." >&2
  exit 1
fi

# Check the proxy is reachable.
if ! curl -sf "${PROXY_URL}/v1/models" >/dev/null 2>&1; then
  echo "fcc proxy not reachable at ${PROXY_URL}. Start it with: fcc-server" >&2
  exit 1
fi

echo ">> Delegating to fcc worker on branch '$BRANCH'..."
# Prefer fcc's launcher if present; otherwise point the real CLI at the proxy.
if command -v fcc-claude >/dev/null 2>&1; then
  fcc-claude -p "$TASK" --dangerously-skip-permissions
else
  ANTHROPIC_BASE_URL="$PROXY_URL" ANTHROPIC_AUTH_TOKEN="${FCC_TOKEN:-freecc}" \
    claude -p "$TASK" --dangerously-skip-permissions
fi
