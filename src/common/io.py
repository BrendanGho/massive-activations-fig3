"""IO helpers: reduced-cache shards, prompt loading, hashing, IoU, upsampling.

Pure ``numpy`` + stdlib. matplotlib/pandas/datasets are imported lazily only in
the optional code paths that need them, so importing this module stays cheap.

Cache layout (fused mode persists only these small artifacts, never full [N_I, D]):

    <activation_cache_dir>/
        shard_000000.npz     # arrays for a batch of prompts, keyed p{pid}_<field>
        shard_000000.json    # per-prompt metadata (prompt text, shapes, layer ids)
        ...
        completed.json       # {"prompt_ids": [...], "shard_of": {pid: "shard_000000"}}
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

COMPLETED_FILE = "completed.json"


# --- geometry / metrics -------------------------------------------------------


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary IoU. Both-empty counts as a perfect match (1.0)."""
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def upsample_nearest_2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbour upsample a 2D array to (out_h, out_w). dtype preserved."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {arr.shape}")
    in_h, in_w = arr.shape
    row_idx = (np.arange(out_h) * in_h // out_h).clip(0, in_h - 1)
    col_idx = (np.arange(out_w) * in_w // out_w).clip(0, in_w - 1)
    return arr[row_idx][:, col_idx]


# --- prompt loading -----------------------------------------------------------


def load_prompts(prompt_source: str) -> list[str]:
    """Load prompts from .txt / .json / .jsonl / .parquet, or an HF dataset id.

    Returns a list of prompt strings (order preserved).
    """
    if os.path.isfile(prompt_source):
        ext = os.path.splitext(prompt_source)[1].lower()
        if ext == ".txt":
            with open(prompt_source, "r") as fh:
                return [line.strip() for line in fh if line.strip()]
        if ext == ".json":
            with open(prompt_source, "r") as fh:
                data = json.load(fh)
            return _extract_prompt_list(data)
        if ext == ".jsonl":
            out: list[str] = []
            with open(prompt_source, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    out.append(_extract_prompt_scalar(obj))
            return out
        if ext == ".parquet":
            import pandas as pd  # lazy

            df = pd.read_parquet(prompt_source)
            return _extract_prompts_from_columns(df)
        raise ValueError(f"Unsupported prompt file extension: {ext} ({prompt_source})")

    # Not a local file: try an HF datasets id (e.g. a GenAI-Bench mirror).
    try:
        from datasets import load_dataset  # lazy
    except Exception as exc:  # pragma: no cover
        raise FileNotFoundError(
            f"prompt_source {prompt_source!r} is not a local file and `datasets` "
            f"is not available to treat it as an HF id ({exc})."
        )
    ds = load_dataset(prompt_source, split="train")
    for col in ("prompt", "text", "caption"):
        if col in ds.column_names:
            return [str(x) for x in ds[col]]
    raise ValueError(
        f"Could not find a prompt column in HF dataset {prompt_source!r}; "
        f"columns: {ds.column_names}"
    )


def _extract_prompt_scalar(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("prompt", "text", "caption"):
            if key in obj:
                return str(obj[key])
    raise ValueError(f"Cannot extract a prompt from record: {obj!r}")


def _extract_prompt_list(data: Any) -> list[str]:
    if isinstance(data, dict):
        for key in ("prompts", "prompt", "data"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("JSON prompt file must be a list (or {prompts: [...]}).")
    return [_extract_prompt_scalar(x) for x in data]


def _extract_prompts_from_columns(df: Any) -> list[str]:
    for col in ("prompt", "text", "caption"):
        if col in df.columns:
            return [str(x) for x in df[col].tolist()]
    raise ValueError(f"No prompt column found in parquet; columns: {list(df.columns)}")


def prompts_content_hash(prompts: list[str]) -> str:
    """Stable content hash of the prompt set (order-sensitive)."""
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# --- reduced-cache records ----------------------------------------------------

# Per-prompt array fields stored in each shard npz (key = f"p{pid}_{field}").
_ARRAY_FIELDS = (
    "rgb",  # uint8 (H_img, W_img, 3)
    "scores",  # float16 (L, D)
    "top_idx",  # int32 (L, k)
    "bottom_idx",  # int32 (L, k)
    "random_idx",  # int32 (L, R, k)
    "mask_top",  # uint8 (L, H_lat, W_lat)
    "mask_bottom",  # uint8 (L, H_lat, W_lat)
    "mask_random",  # uint8 (L, R, H_lat, W_lat)
)


@dataclass
class PromptRecord:
    prompt_id: int
    prompt: str
    h_lat: int
    w_lat: int
    d: int
    img_h: int
    img_w: int
    layers: list[int]
    n_random_trials: int
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def meta(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "h_lat": self.h_lat,
            "w_lat": self.w_lat,
            "d": self.d,
            "img_h": self.img_h,
            "img_w": self.img_w,
            "layers": list(self.layers),
            "n_random_trials": self.n_random_trials,
        }


def _shard_paths(cache_dir: str, index: int) -> tuple[str, str]:
    base = os.path.join(cache_dir, f"shard_{index:06d}")
    return base + ".npz", base + ".json"


def completed_prompt_ids(cache_dir: str) -> set[int]:
    path = os.path.join(cache_dir, COMPLETED_FILE)
    if not os.path.isfile(path):
        return set()
    with open(path, "r") as fh:
        data = json.load(fh)
    return set(int(x) for x in data.get("prompt_ids", []))


class ShardWriter:
    """Batches PromptRecords into shard files; resumable via completed.json.

    Flushes every ``batch_size`` prompts so writes land incrementally (important
    for a Colab session that can die at any time). Idempotent across restarts:
    already-completed prompt ids are tracked in completed.json.
    """

    def __init__(self, cache_dir: str, batch_size: int = 32):
        self.cache_dir = cache_dir
        self.batch_size = max(1, int(batch_size))
        os.makedirs(cache_dir, exist_ok=True)
        self._buffer: list[PromptRecord] = []
        self._completed_path = os.path.join(cache_dir, COMPLETED_FILE)
        self._completed: dict[str, Any] = self._load_completed()

    def _load_completed(self) -> dict[str, Any]:
        if os.path.isfile(self._completed_path):
            with open(self._completed_path, "r") as fh:
                return json.load(fh)
        return {"prompt_ids": [], "shard_of": {}}

    def _next_shard_index(self) -> int:
        existing = glob.glob(os.path.join(self.cache_dir, "shard_*.npz"))
        return len(existing)

    @property
    def completed_ids(self) -> set[int]:
        return set(int(x) for x in self._completed["prompt_ids"])

    def add(self, record: PromptRecord) -> None:
        if record.prompt_id in self.completed_ids:
            return
        self._buffer.append(record)
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        index = self._next_shard_index()
        npz_path, json_path = _shard_paths(self.cache_dir, index)

        arrays: dict[str, np.ndarray] = {}
        metas: list[dict[str, Any]] = []
        for rec in self._buffer:
            pid = rec.prompt_id
            for fname in _ARRAY_FIELDS:
                if fname in rec.arrays:
                    arrays[f"p{pid}_{fname}"] = rec.arrays[fname]
            metas.append(rec.meta())

        # Write shard payload first, then the completed manifest last, so a crash
        # mid-write never marks a prompt done without its data.
        np.savez_compressed(npz_path, **arrays)
        with open(json_path, "w") as fh:
            json.dump({"shard": os.path.basename(npz_path), "prompts": metas}, fh)

        shard_name = f"shard_{index:06d}"
        for rec in self._buffer:
            self._completed["prompt_ids"].append(rec.prompt_id)
            self._completed["shard_of"][str(rec.prompt_id)] = shard_name
        with open(self._completed_path, "w") as fh:
            json.dump(self._completed, fh)
        self._buffer.clear()

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def iter_cache(cache_dir: str) -> Iterator[PromptRecord]:
    """Yield every cached PromptRecord (arrays lazily reconstructed per shard)."""
    json_paths = sorted(glob.glob(os.path.join(cache_dir, "shard_*.json")))
    for jpath in json_paths:
        with open(jpath, "r") as fh:
            shard_meta = json.load(fh)
        npz_path = os.path.join(cache_dir, shard_meta["shard"])
        with np.load(npz_path) as npz:
            for meta in shard_meta["prompts"]:
                pid = meta["prompt_id"]
                arrays = {
                    fname: npz[f"p{pid}_{fname}"]
                    for fname in _ARRAY_FIELDS
                    if f"p{pid}_{fname}" in npz.files
                }
                yield PromptRecord(arrays=arrays, **_meta_kwargs(meta))


def _meta_kwargs(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_id": meta["prompt_id"],
        "prompt": meta["prompt"],
        "h_lat": meta["h_lat"],
        "w_lat": meta["w_lat"],
        "d": meta["d"],
        "img_h": meta["img_h"],
        "img_w": meta["img_w"],
        "layers": meta["layers"],
        "n_random_trials": meta["n_random_trials"],
    }


# --- misc json ----------------------------------------------------------------


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)


def load_json(path: str) -> Any:
    with open(path, "r") as fh:
        return json.load(fh)
