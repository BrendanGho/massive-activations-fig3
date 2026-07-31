"""Quick qualitative look: isolated top massive channel(s) vs where high-norm tokens are.

The simplest possible version of the Part 3 question — no statistics, no CSV, no nulls.
For each prompt it renders one row:

    generated | top-k channel (the speckles) | high-norm (full) | high-norm (minus top-k)

The two high-norm panels are the whole point of looking. "Full" is the token L2 norm; it
will look like a carbon copy of the speckle panel, because a token with a massive value in
one channel is high-norm *by that channel alone* (``‖x‖² = Σ_d x[d]²``). "minus top-k" is
the norm with the isolated channel(s) removed — if the same spots still light up, the
high-norm tokens are more than just the massive activation; if they go dark, the two are
the same thing seen twice.

``--subtract-ks 5,10,20`` adds one further "norm minus top-k" column per k, to watch the
high-norm token fade (or persist) as progressively more massive channels are peeled off.
Those deconfounded columns share one color scale per row, so the *magnitude* of the drop is
visible; independent auto-scaling would re-brighten each panel and hide it. Reuses
``highnorm`` (maps) and ``model_utils`` (generate/capture); matplotlib/torch are imported
lazily so the pure map-builder tests on CPU.

    python -m src.experiments.highnorm_qualitative --config configs/highnorm_tokens.yaml \
        --subtract-ks 5,10,20
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

from src.common import highnorm
from src.experiments.highnorm_tokens import load_highnorm_config


def parse_ks(spec: str | None) -> list[int]:
    """Parse a ``"5,10,20"`` flag into a sorted, deduped list of positive ints. ``""``/None -> []."""
    if not spec:
        return []
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k <= 0:
            raise ValueError(f"subtract-ks values must be positive, got {k}")
        out.add(k)
    return sorted(out)


def default_output_name(
    target_layer: int, n_channels: int, subtract_ks: list[int] | None = None
) -> str:
    """Filename encoding the swept hyperparameters, so runs at different layers / channel
    counts / subtract-sets land side by side in the same folder instead of overwriting.

    e.g. ``qualitative_L18_ch1.png`` or ``qualitative_L18_ch1_sub5-10-20.png``. The
    output_dir is already per-model (the Colab cell nests it under
    ``<drive>/<model>/highnorm_qualitative``), so the model need not be in the name.
    """
    name = f"qualitative_L{int(target_layer)}_ch{int(n_channels)}"
    if subtract_ks:
        name += "_sub" + "-".join(str(int(k)) for k in subtract_ks)
    return name + ".png"


# --- pure map builder (no torch/matplotlib) -----------------------------------


def panel_maps(
    image_stream: np.ndarray,
    n_channels: int,
    h_lat: int,
    w_lat: int,
    subtract_ks: list[int] | None = None,
) -> dict[str, Any]:
    """Reshape the per-token quantities to the latent grid, row-major.

    ``n_channels`` is how many of the top massive channels to isolate (1 = the user's
    "top 1 channel"). Returns the isolated channel ids, the speckle map, the full norm
    map, and the deconfounded (isolated-channels-excluded) norm map. For each k in
    ``subtract_ks`` it additionally returns ``subtract[k]`` = the norm with the top-k
    massive channels removed, to watch the high-norm token fade as more are peeled.
    """
    x = np.asarray(image_stream)
    if x.shape[0] != h_lat * w_lat:
        raise ValueError(f"token count {x.shape[0]} != {h_lat}*{w_lat}={h_lat * w_lat}")
    chans = highnorm.top_channels(x, n_channels)
    out: dict[str, Any] = {
        "channels": chans,
        "speckle": highnorm.massive_score(x, chans).reshape(h_lat, w_lat),
        "n_full": highnorm.token_norms(x).reshape(h_lat, w_lat),
        "n_ex": highnorm.token_norms(x, exclude=chans).reshape(h_lat, w_lat),
    }
    if subtract_ks:
        out["subtract"] = {
            int(k): highnorm.token_norms(x, exclude=highnorm.top_channels(x, k)).reshape(
                h_lat, w_lat
            )
            for k in subtract_ks
        }
    return out


# --- figure (matplotlib lazy) -------------------------------------------------


def _save_figure(
    path: str,
    rows: list[dict[str, Any]],
    layer: int,
    n_channels: int,
    subtract_ks: list[int] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subtract_ks = subtract_ks or []
    chan_label = "top-1 channel" if n_channels == 1 else f"top-{n_channels} channels"
    titles = [
        "generated",
        f"isolated {chan_label}\n(the speckles)",
        "high-norm tokens\n(full norm — the confound)",
        f"high-norm tokens\n(norm minus {chan_label})",
        *(f"high-norm tokens\n(norm minus top-{k})" for k in subtract_ks),
    ]
    ncols = len(titles)
    n = len(rows)
    fig, axes = plt.subplots(n, ncols, figsize=(3.0 * ncols, 3.1 * n), squeeze=False)
    for r, row in enumerate(rows):
        maps = row["maps"]
        deconf = [maps["n_ex"], *(maps["subtract"][k] for k in subtract_ks)]
        # Shared scale across the deconfounded columns so the high-norm token visibly
        # fades as more channels are peeled (per-panel autoscale would hide the drop).
        # n_full keeps its own scale — it is the confound reference, dominated by the
        # massive channel, and would otherwise crush every deconfounded panel to black.
        d_lo = float(min(m.min() for m in deconf))
        d_hi = float(max(m.max() for m in deconf))
        cells = [
            (row["rgb"], None, None, None),
            (maps["speckle"], "inferno", None, None),
            (maps["n_full"], "viridis", None, None),
            *((m, "viridis", d_lo, d_hi) for m in deconf),
        ]
        for c, (img, cmap, vlo, vhi) in enumerate(cells):
            ax = axes[r][c]
            ax.imshow(img, cmap=cmap, vmin=vlo, vmax=vhi, interpolation="nearest")
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


def run(
    cfg,
    n_channels: int,
    out_path: str,
    limit: int | None,
    subtract_ks: list[int] | None = None,
) -> str:
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
            maps = panel_maps(x, n_channels, info["h_lat"], info["w_lat"], subtract_ks)
            rows.append({"prompt": prompt, "rgb": rgb, "maps": maps})
            print(f"[qual] {pid + 1}/{len(prompts)}: {prompt[:50]} (channels {maps['channels']})")
    finally:
        for h in handles:
            h.remove()

    _save_figure(out_path, rows, cfg.target_layer, n_channels, subtract_ks)
    print(f"[qual] wrote {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Qualitative: top massive channel(s) vs high-norm tokens."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--channels", type=int, default=1, help="How many top channels to isolate.")
    p.add_argument(
        "--subtract-ks",
        default="",
        help="Comma-separated extra channel counts to subtract, e.g. '5,10,20'. Each adds a "
        "'norm minus top-k' column so you can watch the high-norm token fade. Empty = none.",
    )
    p.add_argument("--limit", type=int, default=4, help="Only render the first N prompts.")
    p.add_argument(
        "--out",
        default=None,
        help="Output PNG (default: <output_dir>/qualitative_L<layer>_ch<channels>[_sub..].png).",
    )
    args = p.parse_args(argv)

    subtract_ks = parse_ks(args.subtract_ks)
    cfg = load_highnorm_config(args.config)
    out_path = args.out or os.path.join(
        cfg.output_dir, default_output_name(cfg.target_layer, args.channels, subtract_ks)
    )
    run(cfg, n_channels=args.channels, out_path=out_path, limit=args.limit, subtract_ks=subtract_ks)


if __name__ == "__main__":
    main()
