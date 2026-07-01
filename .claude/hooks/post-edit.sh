#!/usr/bin/env bash
# post-edit.sh — runs after every file edit (PostToolUse: Write|Edit|MultiEdit).
# Best-effort format, then run the project's tests. Output is fed back to the agent so it can
# self-correct. Non-fatal by design: it never blocks an edit (always exits 0).
#
# Tuning: for large/slow test suites, change this to lint-only and run the full suite from a Stop
# hook instead, so tests run once per turn rather than after every edit.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

# --- best-effort format (only if the formatter is installed) ---
if [ -f pyproject.toml ] || ls ./*.py >/dev/null 2>&1; then
  command -v ruff  >/dev/null 2>&1 && ruff format . >/dev/null 2>&1
fi
if [ -f package.json ]; then
  command -v npx   >/dev/null 2>&1 && npx --no-install prettier -w . >/dev/null 2>&1
fi
[ -f go.mod ]     && command -v gofmt >/dev/null 2>&1 && gofmt -w . >/dev/null 2>&1
[ -f Cargo.toml ] && command -v cargo >/dev/null 2>&1 && cargo fmt   >/dev/null 2>&1

# --- run tests (auto-detected stack) ---
echo "--- post-edit test run ---"
if [ -f package.json ] && grep -q '"test"' package.json; then
  npm test --silent 2>&1 | tail -n 30
elif [ -f pyproject.toml ] || [ -f pytest.ini ] || [ -d tests ]; then
  if command -v pytest >/dev/null 2>&1; then pytest -q 2>&1 | tail -n 30; else echo "pytest not installed"; fi
elif [ -f go.mod ]; then
  go test ./... 2>&1 | tail -n 30
elif [ -f Cargo.toml ]; then
  cargo test --quiet 2>&1 | tail -n 30
else
  echo "No test framework detected — skipping."
fi
exit 0
