"""Stage 3 — mask construction for every (layer, strategy).

Turns per-layer image-stream activations + channel rankings into binary
foreground masks (top-k / bottom-k / random-k) plus the Fig 3B per-token heatmap.
The numeric heavy-lifting lives in ``src.common.clustering``; this module drives
it across layers/strategies and packs results into a compact ``PromptRecord``.

Callable standalone (``python -m src.stage3_mask_construction``) for debugging one
cached prompt from ``stage1 --no-fused`` + ``stage2``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from src.common import clustering, io
from src.common.config import Config, load_config, parse_set_overrides
from src.stage2_channel_ranking import RankingResult, rank_channels

STRATEGIES = ("top", "bottom", "random")

# Strategy codes keep the per-mask K-means seed distinct across strategies/trials.
_STRAT_TOP = 1
_STRAT_BOTTOM = 2
_STRAT_RANDOM_BASE = 100


def mask_seed(seed: int, prompt_id: int, layer: int, strat_code: int) -> int:
    """Deterministic int seed for K-means random_state."""
    ss = np.random.SeedSequence([int(seed), int(prompt_id), int(layer), int(strat_code)])
    return int(ss.generate_state(1)[0])


def build_prompt_record(
    prompt_id: int,
    prompt: str,
    image_streams: dict[int, np.ndarray],
    rgb: np.ndarray,
    rankings: dict[int, RankingResult],
    cfg: Config,
    h_lat: int,
    w_lat: int,
    *,
    collect_qualitative: bool = False,
) -> tuple[io.PromptRecord, dict]:
    """Assemble masks for all strategies/layers into a reduced PromptRecord.

    Returns (record, qualitative) where qualitative maps
    ``layer -> {"top": (heatmap, mask), "bottom": (...)}`` when ``collect_qualitative``.
    """
    layers = sorted(image_streams.keys())
    r = int(cfg.random_k_trials)
    d = int(next(iter(image_streams.values())).shape[1])
    k = int(cfg.top_k)

    scores = np.zeros((len(layers), d), dtype=np.float16)
    top_idx = np.zeros((len(layers), k), dtype=np.int32)
    bottom_idx = np.zeros((len(layers), k), dtype=np.int32)
    random_idx = np.zeros((len(layers), r, k), dtype=np.int32)
    mask_top = np.zeros((len(layers), h_lat, w_lat), dtype=np.uint8)
    mask_bottom = np.zeros((len(layers), h_lat, w_lat), dtype=np.uint8)
    mask_random = np.zeros((len(layers), r, h_lat, w_lat), dtype=np.uint8)

    qualitative: dict = {}

    for li, layer in enumerate(layers):
        stream = np.asarray(image_streams[layer])
        rank = rankings[layer]
        scores[li] = rank.scores.astype(np.float16)
        top_idx[li] = rank.top_idx
        bottom_idx[li] = rank.bottom_idx
        random_idx[li] = rank.random_idx

        top_res = clustering.build_mask(
            stream,
            rank.top_idx,
            h_lat,
            w_lat,
            seed=mask_seed(cfg.seed, prompt_id, layer, _STRAT_TOP),
        )
        bottom_res = clustering.build_mask(
            stream,
            rank.bottom_idx,
            h_lat,
            w_lat,
            seed=mask_seed(cfg.seed, prompt_id, layer, _STRAT_BOTTOM),
        )
        mask_top[li] = top_res.mask.astype(np.uint8)
        mask_bottom[li] = bottom_res.mask.astype(np.uint8)

        for t in range(r):
            rand_res = clustering.build_mask(
                stream,
                rank.random_idx[t],
                h_lat,
                w_lat,
                seed=mask_seed(cfg.seed, prompt_id, layer, _STRAT_RANDOM_BASE + t),
            )
            mask_random[li, t] = rand_res.mask.astype(np.uint8)

        if collect_qualitative:
            qualitative[layer] = {
                "top": (top_res.heatmap, top_res.mask),
                "bottom": (bottom_res.heatmap, bottom_res.mask),
            }

    record = io.PromptRecord(
        prompt_id=prompt_id,
        prompt=prompt,
        h_lat=h_lat,
        w_lat=w_lat,
        d=d,
        img_h=int(rgb.shape[0]),
        img_w=int(rgb.shape[1]),
        layers=layers,
        n_random_trials=r,
        arrays={
            "rgb": np.asarray(rgb, dtype=np.uint8),
            "scores": scores,
            "top_idx": top_idx,
            "bottom_idx": bottom_idx,
            "random_idx": random_idx,
            "mask_top": mask_top,
            "mask_bottom": mask_bottom,
            "mask_random": mask_random,
        },
    )
    return record, qualitative


def save_qualitative_dump(
    output_dir: str,
    prompt_id: int,
    layer: int,
    strategy: str,
    heatmap: np.ndarray,
    mask: np.ndarray,
    rgb: np.ndarray | None = None,
) -> None:
    """Save Fig 3B heatmap + Fig 3C binary mask PNGs for visual sanity-checking.

    matplotlib is imported lazily so the module stays importable without it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qdir = os.path.join(output_dir, "qualitative", f"prompt_{prompt_id:05d}")
    os.makedirs(qdir, exist_ok=True)

    n_panels = 2 + (1 if rgb is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    axes = np.atleast_1d(axes)
    col = 0
    if rgb is not None:
        axes[col].imshow(rgb)
        axes[col].set_title("decoded")
        axes[col].axis("off")
        col += 1
    im = axes[col].imshow(heatmap, cmap="magma")
    axes[col].set_title(f"L{layer} {strategy} heatmap (Fig3B)")
    axes[col].axis("off")
    fig.colorbar(im, ax=axes[col], fraction=0.046)
    col += 1
    axes[col].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[col].set_title(f"L{layer} {strategy} mask (Fig3C)")
    axes[col].axis("off")

    fig.tight_layout()
    out = os.path.join(qdir, f"L{layer:03d}_{strategy}.png")
    fig.savefig(out, dpi=90)
    plt.close(fig)


# --- standalone debug entrypoint ----------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 3: mask construction (debug on one cached prompt)."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--prompt-id", type=int, required=True)
    p.add_argument("--qualitative", action="store_true", help="Also dump heatmap/mask PNGs.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, parse_set_overrides(args.overrides))

    full_dir = os.path.join(cfg.activation_cache_dir, "full")
    full_path = os.path.join(full_dir, f"prompt_{args.prompt_id}.npz")
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Missing {full_path}; run stage1 --no-fused first.")

    with np.load(full_path) as npz:
        image_streams = {
            int(kk.split("_")[1]): npz[kk] for kk in npz.files if kk.startswith("acts_")
        }
        rgb = npz["rgb"]
        prompt = str(npz["prompt"]) if "prompt" in npz.files else ""
        h_lat = (
            int(npz["h_lat"])
            if "h_lat" in npz.files
            else int(round(image_streams[next(iter(image_streams))].shape[0] ** 0.5))
        )
        w_lat = int(npz["w_lat"]) if "w_lat" in npz.files else h_lat

    rankings = {
        layer: rank_channels(
            stream, cfg.top_k, cfg.random_k_trials, cfg.seed, args.prompt_id, layer
        )
        for layer, stream in image_streams.items()
    }

    record, qualitative = build_prompt_record(
        args.prompt_id,
        prompt,
        image_streams,
        rgb,
        rankings,
        cfg,
        h_lat,
        w_lat,
        collect_qualitative=args.qualitative,
    )
    with io.ShardWriter(cfg.activation_cache_dir, cfg.cache_batch_size) as writer:
        writer.add(record)

    if args.qualitative:
        for layer, strat_map in qualitative.items():
            for strategy, (heatmap, mask) in strat_map.items():
                save_qualitative_dump(
                    cfg.output_dir, args.prompt_id, layer, strategy, heatmap, mask, rgb
                )
    print(f"[stage3] built masks for prompt {args.prompt_id} ({len(record.layers)} layers)")


if __name__ == "__main__":
    main()
