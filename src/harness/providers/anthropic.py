"""Anthropic API provider — calls Claude directly via the SDK."""

from __future__ import annotations

import json
import os
from typing import Any

from harness.providers.base import (
    GenerateRequest,
    GenerateResponse,
    ModelTier,
    Provider,
)


class AnthropicProvider(Provider):
    """Uses the Anthropic Python SDK. Requires ANTHROPIC_API_KEY."""

    def __init__(
        self,
        strong_model: str = "claude-sonnet-4-20250514",
        cheap_model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ):
        self._strong = strong_model
        self._cheap = cheap_model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def tier_model(self, tier: ModelTier) -> str:
        return self._strong if tier == ModelTier.STRONG else self._cheap

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("pip install anthropic  (required for AnthropicProvider)")

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        model = self.tier_model(request.tier)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.system:
            kwargs["system"] = request.system

        # If structured output requested, use tool_use to force JSON
        if request.output_schema:
            kwargs["tools"] = [
                {
                    "name": "structured_output",
                    "description": "Return structured data matching the schema.",
                    "input_schema": request.output_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}

        resp = await client.messages.create(**kwargs)

        content = ""
        parsed = None
        for block in resp.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use" and block.name == "structured_output":
                parsed = block.input
                content = json.dumps(parsed, indent=2)

        return GenerateResponse(
            content=content,
            model=model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            parsed=parsed,
            raw=resp,
        )
