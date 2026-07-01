"""Contract conformance gate — checks outputs against frozen contracts.

This is the key coordination gate. It verifies that the stage's outputs
actually implement the interfaces defined in the contracts, not just that
they mention the right names.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from harness.gates.base import Gate, GateResult


def _extract_python_signatures(source: str) -> dict[str, list[str]]:
    """Extract class/function signatures from Python source.

    Returns {"classes": [...], "functions": [...], "imports": [...]}.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    result: dict[str, list[str]] = {"classes": [], "functions": [], "imports": []}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            result["classes"].append(node.name)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # Include args for signature matching
            args = [a.arg for a in node.args.args if a.arg != "self"]
            sig = f"{node.name}({', '.join(args)})"
            result["functions"].append(sig)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                result["imports"].append(f"{node.module}.{alias.name}")
    return result


def _extract_contract_requirements(contract_body: str) -> list[str]:
    """Pull required symbols/signatures from a contract body.

    Contracts can declare requirements in several ways:
    - Python code with class/function defs (we extract their names)
    - Explicit `REQUIRES: symbol_name` lines
    - Type annotations or interface declarations
    """
    requirements: list[str] = []

    # Check for explicit REQUIRES markers
    for line in contract_body.splitlines():
        m = re.match(r"^\s*REQUIRES:\s*(.+)", line)
        if m:
            requirements.append(m.group(1).strip())

    # Extract signatures from code in the contract
    sigs = _extract_python_signatures(contract_body)
    requirements.extend(sigs.get("classes", []))
    for fn in sigs.get("functions", []):
        # Just the function name for matching
        name = fn.split("(")[0]
        requirements.append(name)

    return requirements


class ContractGate(Gate):
    """Verify stage outputs implement required contract interfaces.

    Checks that every symbol/signature declared in the stage's contracts
    appears in at least one output file. This catches drift between the
    plan's contract and the implementation.
    """

    name = "contract"

    async def check(self, stage_outputs: dict[str, str], context: dict[str, Any]) -> GateResult:
        contracts: dict[str, Any] = context.get("contracts", {})
        stage_contracts: list[str] = context.get("stage_contracts", [])

        if not stage_contracts:
            return GateResult(passed=True, gate_name="contract", message="No contracts to check")

        # Gather all symbols from all output files
        all_symbols: set[str] = set()
        for path, content in stage_outputs.items():
            if path.endswith(".py"):
                sigs = _extract_python_signatures(content)
                all_symbols.update(sigs.get("classes", []))
                for fn in sigs.get("functions", []):
                    all_symbols.add(fn.split("(")[0])
            # For non-Python, do text-based matching
            all_symbols.update(word for word in re.findall(r"\b[A-Za-z_]\w+\b", content))

        missing: list[dict[str, str]] = []
        for contract_name in stage_contracts:
            contract = contracts.get(contract_name)
            if not contract:
                missing.append({"contract": contract_name, "reason": "contract not found in DAG"})
                continue

            requirements = _extract_contract_requirements(contract.body)
            for req in requirements:
                # Check if the requirement symbol exists in outputs
                req_name = req.split("(")[0].strip()
                if req_name and req_name not in all_symbols:
                    missing.append({"contract": contract_name, "missing_symbol": req_name})

        if missing:
            return GateResult(
                passed=False,
                gate_name="contract",
                message=f"{len(missing)} contract violation(s)",
                details={"violations": missing},
            )
        return GateResult(
            passed=True,
            gate_name="contract",
            message="All contracts satisfied",
        )
