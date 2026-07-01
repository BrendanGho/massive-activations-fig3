"""Abstract LLM provider interface.

The harness doesn't care whether you call the Anthropic API, shell out to the
Claude CLI, or use a local model. Implement this interface and plug it in.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelTier(Enum):
    """Route work to the right cost tier."""

    STRONG = "strong"  # Planning, review, escalation — needs judgment
    CHEAP = "cheap"  # Execute loop — volume work


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class GenerateRequest:
    """Everything the provider needs to make one LLM call."""

    messages: list[Message]
    tier: ModelTier = ModelTier.CHEAP
    system: str = ""
    max_tokens: int = 16384
    temperature: float = 0.0
    # Structured output schema (JSON schema dict) — provider may use tool_use or json_mode
    output_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerateResponse:
    content: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    # Parsed structured output if output_schema was provided
    parsed: dict[str, Any] | None = None
    raw: Any = None  # Provider-specific raw response


class Provider(abc.ABC):
    """Abstract LLM provider."""

    @abc.abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Send a prompt and get a completion."""
        ...

    @abc.abstractmethod
    def tier_model(self, tier: ModelTier) -> str:
        """Return the concrete model name for a given tier."""
        ...
