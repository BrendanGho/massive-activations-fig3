"""Tests for the gate system."""

import pytest

from harness.gates.syntax import SyntaxGate
from harness.gates.contract import ContractGate
from harness.gates.scope import ScopeGate


@pytest.fixture
def syntax_gate():
    return SyntaxGate()


@pytest.fixture
def contract_gate():
    return ContractGate()


@pytest.fixture
def scope_gate():
    return ScopeGate()


class TestSyntaxGate:
    @pytest.mark.asyncio
    async def test_valid_python(self, syntax_gate):
        result = await syntax_gate.check(
            {"main.py": "def hello():\n    return 'world'\n"},
            {},
        )
        assert result.passed

    @pytest.mark.asyncio
    async def test_invalid_python(self, syntax_gate):
        result = await syntax_gate.check(
            {"main.py": "def hello(\n    return"},
            {},
        )
        assert not result.passed
        assert "syntax error" in result.message.lower()

    @pytest.mark.asyncio
    async def test_valid_json(self, syntax_gate):
        result = await syntax_gate.check(
            {"config.json": '{"key": "value"}'},
            {},
        )
        assert result.passed

    @pytest.mark.asyncio
    async def test_invalid_json(self, syntax_gate):
        result = await syntax_gate.check(
            {"config.json": '{"key": value}'},
            {},
        )
        assert not result.passed

    @pytest.mark.asyncio
    async def test_unknown_extension_passes(self, syntax_gate):
        result = await syntax_gate.check(
            {"readme.md": "# Hello\nThis is fine"},
            {},
        )
        assert result.passed

    @pytest.mark.asyncio
    async def test_mixed_files(self, syntax_gate):
        result = await syntax_gate.check(
            {
                "good.py": "x = 1",
                "bad.py": "def (",
                "data.json": '{"ok": true}',
            },
            {},
        )
        assert not result.passed


class TestContractGate:
    @pytest.mark.asyncio
    async def test_no_contracts(self, contract_gate):
        result = await contract_gate.check({"main.py": "x = 1"}, {"stage_contracts": []})
        assert result.passed

    @pytest.mark.asyncio
    async def test_contract_satisfied(self, contract_gate):
        from harness.models.spec import Contract

        contract = Contract(
            name="api",
            body="class UserService:\n    def get_user(self, id): ...",
        )
        result = await contract_gate.check(
            {"service.py": "class UserService:\n    def get_user(self, id):\n        return {}"},
            {"contracts": {"api": contract}, "stage_contracts": ["api"]},
        )
        assert result.passed

    @pytest.mark.asyncio
    async def test_contract_violated(self, contract_gate):
        from harness.models.spec import Contract

        contract = Contract(
            name="api",
            body="class UserService:\n    def get_user(self, id): ...",
        )
        result = await contract_gate.check(
            {"service.py": "class OrderService:\n    def get_order(self, id):\n        return {}"},
            {"contracts": {"api": contract}, "stage_contracts": ["api"]},
        )
        assert not result.passed

    @pytest.mark.asyncio
    async def test_requires_marker(self, contract_gate):
        from harness.models.spec import Contract

        contract = Contract(
            name="schema",
            body="REQUIRES: validate_input\nREQUIRES: parse_response",
        )
        result = await contract_gate.check(
            {"utils.py": "def validate_input(data): pass\ndef parse_response(raw): pass"},
            {"contracts": {"schema": contract}, "stage_contracts": ["schema"]},
        )
        assert result.passed


class TestScopeGate:
    @pytest.mark.asyncio
    async def test_in_scope(self, scope_gate):
        result = await scope_gate.check(
            {"src/main.py": "x = 1", "src/utils.py": "y = 2"},
            {"owned_files": ["src/*.py"]},
        )
        assert result.passed

    @pytest.mark.asyncio
    async def test_out_of_scope(self, scope_gate):
        result = await scope_gate.check(
            {"src/main.py": "x = 1", "tests/test.py": "y = 2"},
            {"owned_files": ["src/*.py"]},
        )
        assert not result.passed
        assert "tests/test.py" in str(result.details)

    @pytest.mark.asyncio
    async def test_no_restriction(self, scope_gate):
        result = await scope_gate.check(
            {"anywhere.py": "x = 1"},
            {"owned_files": []},
        )
        assert result.passed
