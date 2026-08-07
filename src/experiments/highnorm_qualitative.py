"""Quick qualitative look: isolated top massive channel(s) vs where high-norm tokens are.

The simplest possible version of the Part 3 question — no statistics, no CSV, no nulls.
For each prompt it renders one row:

    generated | isolated channel(s) (speckles) | high-norm (full) | high-norm (minus those)

The two high-norm panels are the whole point of looking. "Full" is the token L2 norm; it
will look like a carbon copy of the speckle panel, because a token with a massive value in
one channel is high-norm *by that channel alone* (``‖x‖² = Σ_d x[d]²``). "minus those" is
the norm with the isolated channel(s) removed — if the same spots still light up, the
high-norm tokens are more than just the massive activation; if they go dark, the two are
the same thing seen twice.

Knobs:
* ``--channels N`` — isolate the top-N massive channels (default 1).
* ``--ablate-channels 154,1446`` — isolate/remove these *explicit* channels instead of the
  top-N (e.g. after reading the printed ranking below and picking specific ones).
* ``--subtract-ks 5,10,20`` — add one further "norm minus top-k" column per k, to watch the
  high-norm token fade (or persist) as progressively more massive channels are peeled off.
* ``--report-top N`` — print the top-N channels (by mean|abs|) per prompt, plus an aggregate,
  so you know which channels to ablate.

The deconfounded columns share one color scale per row, so the *magnitude* of the drop is
visible; independent auto-scaling would re-brighten each panel and hide it. Reuses
``highnorm`` (maps) and ``model_utils`` (generate/capture); matplotlib/torch are imported
lazily so the pure map-builder tests on CPU.

    python -m src.experiments.highnorm_qualitative --config configs/highnorm_tokens.yaml \
        --subtract-ks 5,10,20 --report-top 15
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Any

import numpy as np

from src.common import highnorm
from src.experiments.highnorm_tokens import load_highnorm_config


def parse_ks(spec: str | None) -> list[int]:
    """Parse ``"5,10,20"`` into a sorted, deduped list of positive ints. ``""``/None -> []."""
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


def parse_channels(spec: str | None) -> list[int]:
    """Parse ``"154,1446"`` into a sorted, deduped list of non-negative channel ids. -> []."""
    if not spec:
        return []
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        c = int(part)
        if c < 0:
            raise ValueError(f"ablate-channels values must be non-negative, got {c}")
        out.add(c)
    return sorted(out)


def default_output_name(
    target_layer: int,
    n_channels: int,
    subtract_ks: list[int] | None = None,
    explicit_channels: list[int] | None = None,
) -> str:
    """Filename encoding the swept hyperparameters, so runs at different layers / channel
    counts / ablation sets / subtract-sets land side by side instead of overwriting.

    e.g. ``qualitative_L18_ch1.png``, ``qualitative_L18_ablate154-1446.png``, or with a
    ``_sub5-10-20`` suffix. The output_dir is already per-model (the Colab cell nests it
    under ``<drive>/<model>/highnorm_qualitative``), so the model need not be in the name.
    """
    if explicit_channels:
        stem = f"qualitative_L{int(target_layer)}_ablate" + "-".join(
            str(int(c)) for c in explicit_channels
        )
    else:
        stem = f"qualitative_L{int(target_layer)}_ch{int(n_channels)}"
    if subtract_ks:
        stem += "_sub" + "-".join(str(int(k)) for k in subtract_ks)
    return stem + ".png"


# --- pure map builder (no torch/matplotlib) -----------------------------------


def top_channel_report(image_stream: np.ndarray, n: int) -> list[tuple[int, float]]:
    """Top-``n`` channels by ``mean(abs(activations))`` with their scores, best first.

    The same score stage 2 ranks by; this is what you read to pick channels for
    ``--ablate-channels``.
    """
    scores = highnorm.channel_scores(np.asarray(image_stream))
    order = np.argsort(-scores, kind="stable")[: max(0, int(n))]
    return [(int(c), float(scores[c])) for c in order]


def _primary_channels(
    image_stream: np.ndarray, n_channels: int, explicit_channels: list[int] | None
) -> np.ndarray:
    """The channel set for the speckle + primary deconfounded columns: explicit if given,
    else the top-``n_channels`` massive channels."""
    x = np.asarray(image_stream)
    if explicit_channels:
        chans = np.asarray(explicit_channels, dtype=np.int64)
        d = x.shape[1]
        bad = chans[(chans < 0) | (chans >= d)]
        if bad.size:
            raise ValueError(f"ablate-channels out of range for D={d}: {bad.tolist()}")
        return chans
    return highnorm.top_channels(x, n_channels)


def panel_maps(
    image_stream: np.ndarray,
    n_channels: int,
    h_lat: int,
    w_lat: int,
    subtract_ks: list[int] | None = None,
    explicit_channels: list[int] | None = None,
) -> dict[str, Any]:
    """Reshape the per-token quantities to the latent grid, row-major.

    The primary isolated set is the top-``n_channels`` massive channels, or
    ``explicit_channels`` when given. Returns those channel ids, the speckle map, the
    full norm map, and the deconfounded (isolated-channels-excluded) norm map. For each k
    in ``subtract_ks`` it additionally returns ``subtract[k]`` = the norm with the top-k
    massive channels removed, to watch the high-norm token fade as more are peeled.
    """
    x = np.asarray(image_stream)
    if x.shape[0] != h_lat * w_lat:
        raise ValueError(f"token count {x.shape[0]} != {h_lat}*{w_lat}={h_lat * w_lat}")
    chans = _primary_channels(x, n_channels, explicit_channels)
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


def _primary_label(n_channels: int, explicit_channels: list[int] | None) -> str:
    if explicit_channels:
        ids = ",".join(str(int(c)) for c in explicit_channels)
        return f"channel {ids}" if len(explicit_channels) == 1 else f"channels {ids}"
    return "top-1 channel" if n_channels == 1 else f"top-{n_channels} channels"


# --- figure (matplotlib lazy) -------------------------------------------------


def _save_figure(
    path: str,
    rows: list[dict[str, Any]],
    layer: int,
    n_channels: int,
    subtract_ks: list[int] | None = None,
    explicit_channels: list[int] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subtract_ks = subtract_ks or []
    label = _primary_label(n_channels, explicit_channels)
    titles = [
        "generated",
        f"isolated {label}\n(the speckles)",
        "high-norm tokens\n(full norm — the confound)",
        f"high-norm tokens\n(norm minus {label})",
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
    explicit_channels: list[int] | None = None,
    report_top: int = 10,
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
    top_counts: Counter[int] = Counter()
    try:
        for pid, prompt in enumerate(prompts):
            rgb, info = model_utils.generate_with_capture(pipe, prompt, cfg, state)
            if cfg.target_layer not in state.image_streams:
                raise RuntimeError(f"no capture at layer {cfg.target_layer} for prompt {pid}")
            x = state.image_streams[cfg.target_layer]
            maps = panel_maps(
                x, n_channels, info["h_lat"], info["w_lat"], subtract_ks, explicit_channels
            )
            rows.append({"prompt": prompt, "rgb": rgb, "maps": maps})

            report = top_channel_report(x, report_top)
            top_counts.update(c for c, _ in report)
            ranked = ", ".join(f"{c}({s:.1f})" for c, s in report)
            print(f"[qual] {pid + 1}/{len(prompts)}: {prompt[:40]}")
            print(f"        top-{report_top} channels @L{cfg.target_layer} (id(score)): {ranked}")
    finally:
        for h in handles:
            h.remove()

    if top_counts:
        # Channels that recur across prompts are the stable massive ones worth ablating.
        agg = ", ".join(f"{c}(x{n})" for c, n in top_counts.most_common(report_top))
        print(f"[qual] most frequent top channels across {len(rows)} prompt(s): {agg}")

    _save_figure(out_path, rows, cfg.target_layer, n_channels, subtract_ks, explicit_channels)
    print(f"[qual] wrote {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Qualitative: top / explicit massive channel(s) vs high-norm tokens."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--channels", type=int, default=1, help="How many top channels to isolate.")
    p.add_argument(
        "--ablate-channels",
        default="",
        help="Explicit channel ids to isolate/remove instead of the top-N, e.g. '154,1446'. "
        "Overrides --channels for the speckle + primary deconfounded columns.",
    )
    p.add_argument(
        "--subtract-ks",
        default="",
        help="Comma-separated extra channel counts to subtract, e.g. '5,10,20'. Each adds a "
        "'norm minus top-k' column so you can watch the high-norm token fade. Empty = none.",
    )
    p.add_argument(
        "--report-top",
        type=int,
        default=10,
        help="Print the top-N channels (by mean|abs|) per prompt + an aggregate. Default 10.",
    )
    p.add_argument("--limit", type=int, default=4, help="Only render the first N prompts.")
    p.add_argument(
        "--out",
        default=None,
        help="Output PNG (default: <output_dir>/qualitative_L<layer>_...[_sub..].png).",
    )
    args = p.parse_args(argv)

    subtract_ks = parse_ks(args.subtract_ks)
    explicit_channels = parse_channels(args.ablate_channels)
    cfg = load_highnorm_config(args.config)
    out_path = args.out or os.path.join(
        cfg.output_dir,
        default_output_name(cfg.target_layer, args.channels, subtract_ks, explicit_channels),
    )
    run(
        cfg,
        n_channels=args.channels,
        out_path=out_path,
        limit=args.limit,
        subtract_ks=subtract_ks,
        explicit_channels=explicit_channels,
        report_top=args.report_top,
    )


if __name__ == "__main__":
    main()
