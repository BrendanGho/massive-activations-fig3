"""Quick qualitative look: isolated top-1 massive channel vs where high-norm tokens are.

The simplest possible version of the Part 3 question — no statistics, no CSV, no nulls.
For each prompt it renders one row:

    generated image | top-1 channel (the speckles) | high-norm (full) | high-norm (N_ex)

The two high-norm panels are the whole point of looking. "Full" is the token L2 norm; it
will look like a carbon copy of the speckle panel, because a token with a massive value in
one channel is high-norm *by that channel alone* (``‖x‖² = Σ_d x[d]²``). "N_ex" is the norm
with the isolated channel(s) removed — if the same spots still light up, the high-norm
tokens are more than just the massive activation; if they go dark, the two are the same
thing seen twice. Reuses ``highnorm`` (maps) and ``model_utils`` (generate/capture);
matplotlib/torch are imported lazily so the pure map-builder tests on CPU.

    python -m src.experiments.highnorm_qualitative --config configs/highnorm_tokens.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

from src.common import highnorm
from src.experiments.highnorm_tokens import load_highnorm_config

# --- pure map builder (no torch/matplotlib) -----------------------------------


def panel_maps(image_stream: np.ndarray, n_channels: int, h_lat: int, w_lat: int) -> dict[str, Any]:
    """Reshape the three per-token quantities to the latent grid, row-major.

    ``n_channels`` is how many of the top massive channels to isolate (1 = the user's
    "top 1 channel"). Returns the isolated channel ids plus the speckle map, the full
    norm map, and the deconfounded (channels-excluded) norm map.
    """
    x = np.asarray(image_stream)
    if x.shape[0] != h_lat * w_lat:
        raise ValueError(f"token count {x.shape[0]} != {h_lat}*{w_lat}={h_lat * w_lat}")
    chans = highnorm.top_channels(x, n_channels)
    return {
        "channels": chans,
        "speckle": highnorm.massive_score(x, chans).reshape(h_lat, w_lat),
        "n_full": highnorm.token_norms(x).reshape(h_lat, w_lat),
        "n_ex": highnorm.token_norms(x, exclude=chans).reshape(h_lat, w_lat),
    }


# --- figure (matplotlib lazy) -------------------------------------------------


def _save_figure(path: str, rows: list[dict[str, Any]], layer: int, n_channels: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chan_label = "top-1 channel" if n_channels == 1 else f"top-{n_channels} channels"
    titles = [
        "generated",
        f"isolated {chan_label}\n(the speckles)",
        "high-norm tokens\n(full norm — the confound)",
        f"high-norm tokens\n(norm minus {chan_label})",
    ]
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3.1 * n), squeeze=False)
    for r, row in enumerate(rows):
        maps = row["maps"]
        cells = [
            (row["rgb"], None),
            (maps["speckle"], "inferno"),
            (maps["n_full"], "viridis"),
            (maps["n_ex"], "viridis"),
        ]
        for c, (img, cmap) in enumerate(cells):
            ax = axes[r][c]
            ax.imshow(img, cmap=cmap, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
        axes[r][0].set_ylabel(row["prompt"][:32], fontsize=8)
    fig.suptitle(
        f"Massive-activation speckles vs high-norm tokens — layer {layer}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- runner -------------------------------------------------------------------


def run(cfg, n_channels: int, out_path: str, limit: int | None) -> str:
    from src.common import model_utils

    prompts = cfg.prompts if limit is None else cfg.prompts[:limit]

    pipe = model_utils.load_pipeline(cfg, offload=cfg.offload)
    blocks = model_utils.select_layers(
        model_utils.discover_blocks(pipe.transformer), [cfg.target_layer]
    )
    state = model_utils.CaptureState()
    handles = model_utils.register_capture_hooks(pipe.transformer, blocks, state)

    rows: list[dict[str, Any]] = []
    try:
        for pid, prompt in enumerate(prompts):
            rgb, info = model_utils.generate_with_capture(pipe, prompt, cfg, state)
            if cfg.target_layer not in state.image_streams:
                raise RuntimeError(f"no capture at layer {cfg.target_layer} for prompt {pid}")
            x = state.image_streams[cfg.target_layer]
            maps = panel_maps(x, n_channels, info["h_lat"], info["w_lat"])
            rows.append({"prompt": prompt, "rgb": rgb, "maps": maps})
            print(f"[qual] {pid + 1}/{len(prompts)}: {prompt[:50]} (channels {maps['channels']})")
    finally:
        for h in handles:
            h.remove()

    _save_figure(out_path, rows, cfg.target_layer, n_channels)
    print(f"[qual] wrote {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Qualitative: top-1 channel vs high-norm tokens.")
    p.add_argument("--config", required=True)
    p.add_argument("--channels", type=int, default=1, help="How many top channels to isolate.")
    p.add_argument("--limit", type=int, default=4, help="Only render the first N prompts.")
    p.add_argument(
        "--out", default=None, help="Output PNG (default: <output_dir>/qualitative_highnorm.png)."
    )
    args = p.parse_args(argv)

    cfg = load_highnorm_config(args.config)
    out_path = args.out or os.path.join(cfg.output_dir, "qualitative_highnorm.png")
    run(cfg, n_channels=args.channels, out_path=out_path, limit=args.limit)


if __name__ == "__main__":
    main()
