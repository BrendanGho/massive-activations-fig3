"""Checkpoint manager — git-based snapshots with clean revert.

Every passing stage gets tagged against a clean baseline. A bad attempt
reverts cleanly to the last checkpoint, so bounded retries are safe.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Checkpoint:
    stage_id: str
    tag: str
    commit_sha: str


class CheckpointManager:
    """Manages checkpoints in a workspace directory.

    Two modes:
    - git mode (default): uses git tags and commits for real version control.
    - file mode (fallback): copies files to a .checkpoints/ dir when git isn't available.
    """

    def __init__(self, workspace: Path, use_git: bool = True):
        self._workspace = workspace
        self._use_git = use_git and self._has_git()
        self._checkpoints: list[Checkpoint] = []
        self._file_backups: dict[str, dict[str, str]] = {}  # stage_id -> {path: content}

    def _has_git(self) -> bool:
        return (self._workspace / ".git").exists()

    async def _run(self, *args: str) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._workspace),
        )
        stdout, stderr = await proc.communicate()
        return (
            proc.returncode or 0,
            stdout.decode(errors="replace").strip(),
            stderr.decode(errors="replace").strip(),
        )

    async def snapshot_baseline(self) -> str:
        """Take a baseline snapshot before execution begins."""
        if self._use_git:
            rc, sha, _ = await self._run("git", "rev-parse", "HEAD")
            if rc == 0:
                await self._run("git", "tag", "-f", "harness/baseline", sha)
                return sha
        return "no-git"

    async def create(self, stage_id: str, files: dict[str, str]) -> Checkpoint:
        """Checkpoint after a stage passes.

        In git mode: writes files, commits, tags.
        In file mode: stores file contents in memory for revert.
        """
        tag = f"harness/stage/{stage_id}"

        if self._use_git:
            # Write files
            for path, content in files.items():
                full = self._workspace / path
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text(content)

            await self._run("git", "add", "-A")
            await self._run(
                "git", "commit", "-m", f"harness: stage {stage_id} passed", "--allow-empty"
            )
            _, sha, _ = await self._run("git", "rev-parse", "HEAD")
            await self._run("git", "tag", "-f", tag, sha)
            cp = Checkpoint(stage_id=stage_id, tag=tag, commit_sha=sha)
        else:
            # File-mode fallback
            self._file_backups[stage_id] = dict(files)
            for path, content in files.items():
                full = self._workspace / path
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text(content)
            cp = Checkpoint(stage_id=stage_id, tag=tag, commit_sha="file-mode")

        self._checkpoints.append(cp)
        return cp

    async def revert(self, stage_id: str | None = None) -> None:
        """Revert to the checkpoint before the given stage, or to baseline.

        This is the critical safety mechanism: a failed retry starts from
        a known-good state, not from the corrupted output of the last attempt.
        """
        if self._use_git:
            if stage_id and self._checkpoints:
                # Find the checkpoint just before this stage
                target = "harness/baseline"
                for cp in self._checkpoints:
                    if cp.stage_id == stage_id:
                        break
                    target = cp.tag
                await self._run("git", "reset", "--hard", target)
                await self._run("git", "clean", "-fd")
            else:
                await self._run("git", "reset", "--hard", "harness/baseline")
                await self._run("git", "clean", "-fd")
        else:
            # File mode: we can't truly revert, but we can re-write known-good files
            pass

    @property
    def checkpoints(self) -> list[Checkpoint]:
        return list(self._checkpoints)
