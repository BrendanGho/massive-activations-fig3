"""Claude CLI provider — shells out to the `claude` command.

Useful when you want the model to have full tool access (file I/O, bash, etc.)
without reimplementing tool dispatch. The harness captures stdout as the response.
"""

from __future__ import annotations

import asyncio
import json
import os

from harness.providers.base import (
    GenerateRequest,
    GenerateResponse,
    ModelTier,
    Provider,
)


class ClaudeCliProvider(Provider):
    """Runs `claude -p <prompt>` as a subprocess."""

    def __init__(
        self,
        claude_bin: str = "claude",
        strong_model: str = "opus",
        cheap_model: str = "haiku",
        cwd: str | None = None,
        env_overrides: dict[str, str] | None = None,
    ):
        self._bin = claude_bin
        self._strong = strong_model
        self._cheap = cheap_model
        self._cwd = cwd or os.getcwd()
        self._env = env_overrides or {}

    def tier_model(self, tier: ModelTier) -> str:
        return self._strong if tier == ModelTier.STRONG else self._cheap

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        model = self.tier_model(request.tier)

        # Build the full prompt: system + messages concatenated
        parts: list[str] = []
        if request.system:
            parts.append(f"<system>\n{request.system}\n</system>\n")
        for msg in request.messages:
            parts.append(msg.content)
        prompt = "\n\n".join(parts)

        cmd = [
            self._bin,
            "-p",
            prompt,
            "--model",
            model,
            "--output-format",
            "json",
        ]

        env = {**os.environ, **self._env}
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI exited {proc.returncode}: {stderr.decode(errors='replace')}"
            )

        raw_out = stdout.decode(errors="replace")

        # Try to parse JSON output
        content = raw_out
        parsed = None
        try:
            data = json.loads(raw_out)
            if isinstance(data, dict):
                content = data.get("result", raw_out)
                parsed = data if request.output_schema else None
        except json.JSONDecodeError:
            pass

        return GenerateResponse(
            content=content,
            model=model,
            parsed=parsed,
            raw=raw_out,
        )
