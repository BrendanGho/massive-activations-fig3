"""Stage 2 — per-layer channel ranking.

Core invariant (do not "simplify"): the channel score is ``abs(mean_over_tokens)``
— mean **then** abs, NOT ``mean(abs(...))``. The two give different rankings, and
the whole result depends on getting this right.

Ranking stats are computed per-sample / per-layer / per-stream. Never average raw
activations across samples before ranking.

Runs on CPU with plain numpy; the model is not needed here. Callable standalone
(``python -m src.stage2_channel_ranking``) for debugging a single cached prompt
produced by ``stage1 --no-fused``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from src.common.config import load_config, parse_set_overrides


@dataclass
class RankingResult:
    scores: np.ndarray  # (D,) float — abs(mean_over_tokens)
    top_idx: np.ndarray  # (k,) int32
    bottom_idx: np.ndarray  # (k,) int32
    random_idx: np.ndarray  # (R, k) int32


def derive_rng(seed: int, prompt_id: int, layer: int, trial: int) -> np.random.Generator:
    """Deterministic RNG from (seed, prompt_id, layer, trial) via SeedSequence."""
    ss = np.random.SeedSequence([int(seed), int(prompt_id), int(layer), int(trial)])
    return np.random.default_rng(ss)


def channel_scores(image_stream: np.ndarray) -> np.ndarray:
    """score = abs(mean over tokens). image_stream: (N_tokens, D) -> (D,).

    THE ORDERING MATTERS: mean first (over axis 0 = tokens), then abs.
    """
    acts = np.asarray(image_stream, dtype=np.float64)
    if acts.ndim != 2:
        raise ValueError(f"image_stream must be 2D (N_tokens, D), got {acts.shape}")
    mu = acts.mean(axis=0)  # (D,)
    return np.abs(mu)


def channel_scores_max(image_stream: np.ndarray, q: float = 0.999) -> np.ndarray:
    """Complementary token-localized score: quantile-over-tokens of ``abs(activation)``.

    ``q=1.0`` is the per-channel abs max. This is NOT a replacement for
    ``channel_scores`` (the fig3 mean-then-abs invariant); it exists to catch channels
    that are massive at only a few tokens, which the mean dilutes.
    image_stream: (N_tokens, D) -> (D,).
    """
    acts = np.asarray(image_stream, dtype=np.float64)
    if acts.ndim != 2:
        raise ValueError(f"image_stream must be 2D (N_tokens, D), got {acts.shape}")
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")
    return np.quantile(np.abs(acts), q, axis=0)


def rank_channels(
    image_stream: np.ndarray,
    top_k: int,
    random_k_trials: int,
    seed: int,
    prompt_id: int,
    layer: int,
) -> RankingResult:
    """Rank channels for one (prompt, layer) image stream.

    top_k_idx    = highest scores; bottom_k_idx = lowest scores (mean-then-abs).
    random_idx   = ``random_k_trials`` seeded draws without replacement, seed
                   derived from (seed, prompt_id, layer, trial).
    """
    score = channel_scores(image_stream)
    d = score.shape[0]
    k = int(top_k)
    if k > d:
        raise ValueError(f"top_k={k} exceeds channel dim D={d}")

    # Stable sort => deterministic tie-breaking by ascending channel index.
    order_desc = np.argsort(-score, kind="stable")
    order_asc = np.argsort(score, kind="stable")
    top_idx = order_desc[:k].astype(np.int32)
    bottom_idx = order_asc[:k].astype(np.int32)

    random_idx = np.empty((int(random_k_trials), k), dtype=np.int32)
    for t in range(int(random_k_trials)):
        rng = derive_rng(seed, prompt_id, layer, t)
        random_idx[t] = rng.choice(d, size=k, replace=False).astype(np.int32)

    return RankingResult(
        scores=score.astype(np.float32),
        top_idx=top_idx,
        bottom_idx=bottom_idx,
        random_idx=random_idx,
    )


# --- standalone debug entrypoint ----------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 2: channel ranking (debug on one cached prompt)."
    )
    p.add_argument("--config", required=True)
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a config key: --set key=value (repeatable).",
    )
    p.add_argument(
        "--prompt-id",
        type=int,
        required=True,
        help="Prompt id whose full-activation cache (from stage1 --no-fused) to rank.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    import os

    args = _build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, parse_set_overrides(args.overrides))

    full_path = os.path.join(cfg.activation_cache_dir, "full", f"prompt_{args.prompt_id}.npz")
    if not os.path.isfile(full_path):
        raise FileNotFoundError(
            f"No full-activation cache at {full_path}. Run stage1 with --no-fused first."
        )
    with np.load(full_path) as npz:
        layer_keys = sorted(
            (kk for kk in npz.files if kk.startswith("acts_")),
            key=lambda s: int(s.split("_")[1]),
        )
        out: dict[str, np.ndarray] = {}
        for kk in layer_keys:
            layer = int(kk.split("_")[1])
            res = rank_channels(
                npz[kk], cfg.top_k, cfg.random_k_trials, cfg.seed, args.prompt_id, layer
            )
            out[f"top_{layer}"] = res.top_idx
            out[f"bottom_{layer}"] = res.bottom_idx
            out[f"random_{layer}"] = res.random_idx
            out[f"scores_{layer}"] = res.scores
        ranking_path = os.path.join(
            cfg.activation_cache_dir, "full", f"prompt_{args.prompt_id}_ranking.npz"
        )
        np.savez_compressed(ranking_path, **out)
    print(f"[stage2] ranked {len(layer_keys)} layers -> {ranking_path}")


if __name__ == "__main__":
    main()
