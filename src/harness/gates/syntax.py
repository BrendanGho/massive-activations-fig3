"""Syntax gate — checks that generated files parse correctly."""

from __future__ import annotations

import ast
import json
import subprocess
from typing import Any

from harness.gates.base import Gate, GateResult

# Extension -> checker function mapping
_CHECKERS: dict[str, Any] = {}


def _check_python(content: str, path: str) -> GateResult:
    try:
        ast.parse(content, filename=path)
        return GateResult(passed=True, gate_name="syntax", message=f"{path}: valid Python")
    except SyntaxError as e:
        return GateResult(
            passed=False,
            gate_name="syntax",
            message=f"{path}: Python syntax error at line {e.lineno}",
            details={"file": path, "line": e.lineno, "error": str(e)},
        )


def _check_json(content: str, path: str) -> GateResult:
    try:
        json.loads(content)
        return GateResult(passed=True, gate_name="syntax", message=f"{path}: valid JSON")
    except json.JSONDecodeError as e:
        return GateResult(
            passed=False,
            gate_name="syntax",
            message=f"{path}: JSON error at line {e.lineno}",
            details={"file": path, "line": e.lineno, "error": str(e)},
        )


def _check_yaml(content: str, path: str) -> GateResult:
    try:
        import yaml

        yaml.safe_load(content)
        return GateResult(passed=True, gate_name="syntax", message=f"{path}: valid YAML")
    except Exception as e:
        return GateResult(
            passed=False,
            gate_name="syntax",
            message=f"{path}: YAML error: {e}",
            details={"file": path, "error": str(e)},
        )


def _check_js_ts(content: str, path: str) -> GateResult:
    """Best-effort JS/TS syntax check via node --check or esbuild."""
    import tempfile, os

    ext = os.path.splitext(path)[1]
    with tempfile.NamedTemporaryFile(suffix=ext, mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        try:
            result = subprocess.run(
                ["node", "--check", f.name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return GateResult(passed=True, gate_name="syntax", message=f"{path}: valid JS")
            return GateResult(
                passed=False,
                gate_name="syntax",
                message=f"{path}: JS syntax error",
                details={"file": path, "error": result.stderr.strip()},
            )
        except FileNotFoundError:
            # node not available — skip gracefully
            return GateResult(passed=True, gate_name="syntax", message=f"{path}: skipped (no node)")
        finally:
            os.unlink(f.name)


_EXT_MAP = {
    ".py": _check_python,
    ".json": _check_json,
    ".yaml": _check_yaml,
    ".yml": _check_yaml,
    ".js": _check_js_ts,
    ".ts": _check_js_ts,
    ".tsx": _check_js_ts,
    ".jsx": _check_js_ts,
}


class SyntaxGate(Gate):
    """Checks every output file parses in its language. Fast, deterministic."""

    name = "syntax"

    async def check(self, stage_outputs: dict[str, str], context: dict[str, Any]) -> GateResult:
        import os

        failures: list[GateResult] = []
        for path, content in stage_outputs.items():
            ext = os.path.splitext(path)[1].lower()
            checker = _EXT_MAP.get(ext)
            if checker:
                result = checker(content, path)
                if not result.passed:
                    failures.append(result)

        if failures:
            return GateResult(
                passed=False,
                gate_name="syntax",
                message=f"{len(failures)} syntax error(s)",
                details={"failures": [f.details for f in failures]},
            )
        return GateResult(
            passed=True,
            gate_name="syntax",
            message=f"All {len(stage_outputs)} files parse OK",
        )
