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
* ``--layers all`` (or ``"0,5,10"``) — sweep layers: every requested layer is captured in one
  generation pass and written to its own ``qualitative_L<layer>.png`` in the channel folder.

All the norm columns (3 onward: full norm + every "minus ..." column) share ONE **absolute**
color scale per row (spanning the full-norm range) with a colorbar, so a color means the same
token norm in every column and they are directly comparable pixel for pixel. That is what lets
you see ablation work: a token whose norm is dominated by the ablated channel(s) drops toward
background and goes dark (disappears), while a token that stays high stays bright (persists).
Reuses ``highnorm`` (maps) and ``model_utils`` (generate/capture); matplotlib/torch are
imported lazily so the pure map-builder tests on CPU.

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


def variant_dir(n_channels: int, explicit_channels: list[int] | None = None) -> str:
    """Subfolder name for one *channel variant* (the thing held fixed while sweeping layers).

    ``ablate_154-1446`` for explicit channels, else ``top_ch1``. A full layer sweep for one
    channel set therefore lands together in one folder, and a different channel set gets its
    own folder — which is what you want when sweeping layers x a few channel sets.
    """
    if explicit_channels:
        return "ablate_" + "-".join(str(int(c)) for c in explicit_channels)
    return f"top_ch{int(n_channels)}"


def default_output_name(
    target_layer: int,
    n_channels: int,
    subtract_ks: list[int] | None = None,
    explicit_channels: list[int] | None = None,
) -> str:
    """Relative path ``<variant>/qualitative_L<layer>[_sub5-10-20].png``.

    The ``<variant>`` subfolder (see :func:`variant_dir`) groups a whole layer sweep for one
    channel set; the layer lives in the filename. Different layers -> same folder, different
    files; different channel sets -> different folders. The output_dir is already per-model
    (the Colab cell nests it under ``<drive>/<model>/highnorm_qualitative``).
    """
    fname = f"qualitative_L{int(target_layer)}"
    if subtract_ks:
        fname += "_sub" + "-".join(str(int(k)) for k in subtract_ks)
    return os.path.join(variant_dir(n_channels, explicit_channels), fname + ".png")


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


def norm_columns(maps: dict[str, Any], subtract_ks: list[int] | None = None) -> list[np.ndarray]:
    """The norm maps shown in columns 3+ (full norm, then each 'norm minus ...'), in order."""
    return [maps["n_full"], maps["n_ex"], *(maps["subtract"][k] for k in (subtract_ks or []))]


def shared_norm_scale(
    maps: dict[str, Any], subtract_ks: list[int] | None = None
) -> tuple[float, float]:
    """One ABSOLUTE (vmin, vmax) spanning ALL norm columns (full norm + every "norm minus
    ..."), so a color means the same token-norm value in every column and they are directly
    comparable pixel for pixel.

    This is what lets you actually SEE ablation work: a token whose norm is dominated by the
    ablated channel(s) drops toward background and therefore renders dark (it disappears),
    while a token that stays high renders bright (it persists). Removing channels can only
    lower a token's norm, so the full norm sets ``vmax`` and no column ever clips.

    Do NOT anchor to the post-ablation (deconfounded) range instead: that re-normalizes the
    ablated columns to their own tiny residual span, so a token that collapsed from (say) 700
    to 35 still paints at the top of the shrunken scale and looks like it "remains" — the
    opposite of the comparison this figure exists to make.
    """
    cols = norm_columns(maps, subtract_ks)
    lo = float(min(m.min() for m in cols))
    hi = float(max(m.max() for m in cols))
    if hi <= lo:  # degenerate (e.g. all-zero activations): keep a positive span
        hi = lo + 1.0
    return lo, hi


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
    # An extra thin trailing column holds each row's colorbar, so every image panel keeps
    # equal width and rows stay aligned (a colorbar stolen from the norm axes would shrink
    # only those, breaking the side-by-side comparison the figure exists for).
    fig, axes = plt.subplots(
        n,
        ncols + 1,
        figsize=(3.0 * ncols + 0.6, 3.1 * n),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0] * ncols + [0.08]},
    )
    for r, row in enumerate(rows):
        maps = row["maps"]
        norm_maps = norm_columns(maps, subtract_ks)
        # ONE absolute scale across all norm columns (full norm + every ablated norm), so
        # color = the same token-norm value everywhere. A token whose norm is dominated by
        # the ablated channel(s) drops toward background and renders dark (disappears); one
        # that stays high renders bright (persists). No clipping — full norm sets the top.
        v_lo, v_hi = shared_norm_scale(maps, subtract_ks)
        cells = [
            (row["rgb"], None, None, None),
            (maps["speckle"], "inferno", None, None),
            *((m, "viridis", v_lo, v_hi) for m in norm_maps),
        ]
        im_norm = None
        for c, (img, cmap, vlo, vhi) in enumerate(cells):
            ax = axes[r][c]
            im = ax.imshow(img, cmap=cmap, vmin=vlo, vmax=vhi, interpolation="nearest")
            if c == 2:  # first norm column carries the shared mappable for the colorbar
                im_norm = im
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
        axes[r][0].set_ylabel(row["prompt"][:32], fontsize=8)
        # Per-row colorbar in its dedicated column (each row has its own absolute scale).
        cbar_ax = axes[r][ncols]
        if im_norm is not None:
            fig.colorbar(im_norm, cax=cbar_ax, label="token L2 norm")
        else:
            cbar_ax.axis("off")
    fig.suptitle(
        f"Massive-activation speckles vs high-norm tokens — layer {layer}\n"
        f"(columns 3+ share one color scale — watch the high-norm tokens dim or persist)",
        fontsize=11,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- runner -------------------------------------------------------------------


def resolve_layers(spec: str | None, all_ids: list[int], target_layer: int) -> list[int]:
    """Which layers to render. ``""``/None -> [target_layer]; ``"all"`` -> every block;
    ``"0,5,10"`` -> those (validated against the model's block ids)."""
    available = set(all_ids)
    if not spec or not str(spec).strip():
        want = [int(target_layer)]  # default path is validated too, with the same message
    else:
        s = str(spec).strip().lower()
        if s == "all":
            return list(all_ids)
        want = sorted({int(p) for p in s.split(",") if p.strip()})
    missing = [ly for ly in want if ly not in available]
    if missing:
        raise ValueError(f"requested layers not in model (available 0..{max(all_ids)}): {missing}")
    return want


def run(
    cfg,
    n_channels: int,
    limit: int | None,
    subtract_ks: list[int] | None = None,
    explicit_channels: list[int] | None = None,
    report_top: int = 10,
    layers_spec: str | None = None,
    out_override: str | None = None,
) -> list[str]:
    """Render the qualitative figure for one or more layers.

    All requested layers are captured in a SINGLE generation pass (hooks on every wanted
    block), so a full layer sweep generates each image once rather than reloading the model
    or regenerating per layer. Only the small per-layer maps are kept; the big [N, D] streams
    live transiently in the capture buffer (~n_layers x N x D of CPU RAM at peak).
    """
    from src.common import model_utils

    prompts = cfg.prompts if limit is None else cfg.prompts[:limit]

    pipe = model_utils.load_pipeline(cfg, offload=cfg.offload)
    all_blocks = model_utils.discover_blocks(pipe.transformer)
    all_ids = [b.layer_id for b in all_blocks]
    want = resolve_layers(layers_spec, all_ids, cfg.target_layer)
    if out_override is not None and len(want) > 1:
        raise ValueError(
            "--out is only valid for a single layer; a multi-layer sweep writes one "
            "file per layer under output_dir. Drop --out or request a single layer."
        )

    blocks = model_utils.select_layers(all_blocks, want)
    state = model_utils.CaptureState()
    handles = model_utils.register_capture_hooks(pipe.transformer, blocks, state)

    rows_by_layer: dict[int, list[dict[str, Any]]] = {ly: [] for ly in want}
    counts_by_layer: dict[int, Counter[int]] = {ly: Counter() for ly in want}
    single = len(want) == 1
    try:
        for pid, prompt in enumerate(prompts):
            rgb, info = model_utils.generate_with_capture(pipe, prompt, cfg, state)
            if single:
                print(f"[qual] {pid + 1}/{len(prompts)}: {prompt[:40]}")
            for ly in want:
                if ly not in state.image_streams:
                    raise RuntimeError(f"no capture at layer {ly} for prompt {pid}")
                x = state.image_streams[ly]
                maps = panel_maps(
                    x, n_channels, info["h_lat"], info["w_lat"], subtract_ks, explicit_channels
                )
                rows_by_layer[ly].append({"prompt": prompt, "rgb": rgb, "maps": maps})
                report = top_channel_report(x, report_top)
                counts_by_layer[ly].update(c for c, _ in report)
                if single:  # per-prompt detail only when not sweeping (else 100s of lines)
                    ranked = ", ".join(f"{c}({s:.1f})" for c, s in report)
                    print(f"        top-{report_top} channels @L{ly} (id(score)): {ranked}")
    finally:
        for h in handles:
            h.remove()

    # Per-layer aggregate: channels that recur across prompts are the stable massive ones
    # worth ablating — this is the ranking you read to pick --ablate-channels per layer.
    for ly in want:
        counts = counts_by_layer[ly]
        if counts:
            agg = ", ".join(f"{c}(x{n})" for c, n in counts.most_common(report_top))
            print(f"[qual] L{ly}: top channels across {len(prompts)} prompt(s): {agg}")

    out_paths: list[str] = []
    for ly in want:
        out_path = out_override or os.path.join(
            cfg.output_dir,
            default_output_name(ly, n_channels, subtract_ks, explicit_channels),
        )
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)  # the per-variant subfolder
        _save_figure(out_path, rows_by_layer[ly], ly, n_channels, subtract_ks, explicit_channels)
        print(f"[qual] wrote {out_path}")
        out_paths.append(out_path)
    return out_paths


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
        "--layers",
        default="",
        help="Layers to render: 'all', a list '0,5,10', or empty for the config's "
        "target_layer. A sweep captures every requested layer in one generation pass and "
        "writes one file per layer under its channel-variant folder.",
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
        help="Output PNG for a SINGLE layer (default: the foldered per-layer path). "
        "Invalid when --layers requests more than one layer.",
    )
    args = p.parse_args(argv)

    subtract_ks = parse_ks(args.subtract_ks)
    explicit_channels = parse_channels(args.ablate_channels)
    cfg = load_highnorm_config(args.config)
    run(
        cfg,
        n_channels=args.channels,
        limit=args.limit,
        subtract_ks=subtract_ks,
        explicit_channels=explicit_channels,
        report_top=args.report_top,
        layers_spec=args.layers,
        out_override=args.out,
    )


if __name__ == "__main__":
    main()
