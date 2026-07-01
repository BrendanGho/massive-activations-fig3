"""Stage 4 — Figure 3D: layer-wise mIoU vs. BiRefNet pseudo-GT.

For every cached prompt: BiRefNet(decoded RGB) -> pseudo-GT foreground mask. For
every (layer, strategy) the stored latent-res mask is upsampled to the eval
resolution and scored with IoU against the GT. Averaging over all prompts gives
mIoU per (layer, strategy). random-k is averaged over its trials per prompt first.

Outputs:
    <output_dir>/figure3d_results.csv   (layer, strategy, mean_miou, std_miou, n)
    <output_dir>/figure3d_curve.png     (top-k / bottom-k / random-k curves)

    python -m src.stage4_evaluate_figure3d --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from src.common import io
from src.common.config import Config, load_config, parse_set_overrides

STRATEGIES = ("top", "bottom", "random")

# Approximate FLUX.2-klein targets from the paper (Fig 3D) for the sanity check.
_TARGET_TOP_PEAK = 0.5
_TARGET_PEAK_LAYER = 10
_TARGET_BOTTOM = 0.2


def evaluate(cfg: Config, limit: int | None = None) -> dict:
    """Compute per-(layer, strategy) IoU distributions over the cached prompts."""
    from src.common import model_utils

    model = model_utils.load_birefnet(cfg)

    # (layer_id, strategy) -> list of per-prompt IoU
    acc: dict[tuple[int, str], list[float]] = defaultdict(list)

    n_prompts = 0
    for record in io.iter_cache(cfg.activation_cache_dir):
        if limit is not None and n_prompts >= limit:
            break
        rgb = record.arrays["rgb"]
        out_hw = (record.img_h, record.img_w)
        gt = model_utils.birefnet_mask(model, rgb, cfg, out_hw=out_hw)

        layers = record.layers
        mask_top = record.arrays["mask_top"]
        mask_bottom = record.arrays["mask_bottom"]
        mask_random = record.arrays.get("mask_random")
        r = record.n_random_trials

        for li, layer in enumerate(layers):
            acc[(layer, "top")].append(_iou_upsampled(mask_top[li], gt, out_hw))
            acc[(layer, "bottom")].append(_iou_upsampled(mask_bottom[li], gt, out_hw))
            if mask_random is not None and r > 0:
                trial_ious = [_iou_upsampled(mask_random[li, t], gt, out_hw) for t in range(r)]
                acc[(layer, "random")].append(float(np.mean(trial_ious)))
        n_prompts += 1

    if n_prompts == 0:
        raise RuntimeError(
            f"No cached prompts found in {cfg.activation_cache_dir}. Run stage1 first."
        )

    results = {key: (float(np.mean(v)), float(np.std(v)), len(v)) for key, v in acc.items()}
    print(f"[stage4] evaluated {n_prompts} prompt(s)")
    return results


def _iou_upsampled(mask_lat: np.ndarray, gt: np.ndarray, out_hw: tuple[int, int]) -> float:
    pred = io.upsample_nearest_2d(mask_lat.astype(bool), out_hw[0], out_hw[1])
    return io.iou(pred, gt)


def write_results_csv(results: dict, output_dir: str) -> str:
    path = os.path.join(output_dir, "figure3d_results.csv")
    rows = sorted(results.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "strategy", "mean_miou", "std_miou", "n"])
        for (layer, strategy), (mean, std, n) in rows:
            w.writerow([layer, strategy, f"{mean:.6f}", f"{std:.6f}", n])
    return path


def plot_curve(results: dict, output_dir: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"top": "#d1495b", "bottom": "#66a182", "random": "#2e4057"}
    for strategy in STRATEGIES:
        pts = sorted(
            (layer, mean, std) for (layer, s), (mean, std, _n) in results.items() if s == strategy
        )
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.plot(xs, ys, marker="o", ms=3, label=f"{strategy}-k", color=colors.get(strategy))
        ax.fill_between(
            xs,
            [y - e for y, e in zip(ys, es)],
            [y + e for y, e in zip(ys, es)],
            alpha=0.15,
            color=colors.get(strategy),
        )
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("mIoU vs. BiRefNet pseudo-GT")
    ax.set_title("Figure 3D — layer-wise mIoU by channel-selection strategy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "figure3d_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def sanity_check(results: dict) -> list[str]:
    """Print (not assert) warnings if the curve shape deviates from the targets.

    Likely bug locations if this trips: normalization order (Stage 3 step 1) or the
    mean-then-abs ranking (Stage 2).
    """
    warnings: list[str] = []
    layers = sorted({layer for (layer, _s) in results})

    def mean_at(layer, strat):
        return results.get((layer, strat), (float("nan"),))[0]

    # 1. top-k should dominate at every layer.
    dominated = [
        layer
        for layer in layers
        if not (
            mean_at(layer, "top") >= mean_at(layer, "bottom")
            and mean_at(layer, "top") >= mean_at(layer, "random")
        )
    ]
    if dominated:
        warnings.append(
            f"top-k does NOT dominate at layers {dominated} — check mean-then-abs ranking."
        )

    # 2. bottom-k should be flat and low (~0.2).
    bmean = float("nan")
    bottoms = [mean_at(layer, "bottom") for layer in layers]
    bottoms = [b for b in bottoms if not np.isnan(b)]
    if bottoms:
        bmean = float(np.mean(bottoms))
        if not (0.1 <= bmean <= 0.35):
            warnings.append(f"bottom-k mean mIoU {bmean:.3f} far from target ~{_TARGET_BOTTOM}.")
        if float(np.std(bottoms)) > 0.15:
            warnings.append(f"bottom-k curve not flat (std {np.std(bottoms):.3f}).")

    # 3. random-k should sit between bottom and top on average.
    tops = [mean_at(layer, "top") for layer in layers]
    rands = [mean_at(layer, "random") for layer in layers]
    tmean = float(np.nanmean(tops)) if tops else float("nan")
    rmean = float(np.nanmean(rands)) if rands else float("nan")
    if not np.isnan(rmean) and not np.isnan(bmean) and not (bmean - 0.05 <= rmean <= tmean + 0.05):
        warnings.append(
            f"random-k mean {rmean:.3f} not between bottom {bmean:.3f} and top {tmean:.3f}."
        )

    # 4. top-k peak magnitude / location.
    if tops:
        peak_layer = layers[int(np.nanargmax(tops))]
        peak_val = float(np.nanmax(tops))
        if abs(peak_val - _TARGET_TOP_PEAK) > 0.15:
            warnings.append(f"top-k peak mIoU {peak_val:.3f} far from target ~{_TARGET_TOP_PEAK}.")
        if abs(peak_layer - _TARGET_PEAK_LAYER) > 5:
            warnings.append(
                f"top-k peaks at layer {peak_layer}, target ~layer {_TARGET_PEAK_LAYER}."
            )

    if warnings:
        print("[stage4][SANITY] deviations from Figure 3D targets:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("[stage4][SANITY] curve shape matches Figure 3D targets.")
    return warnings


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 4: evaluate Figure 3D mIoU curves.")
    p.add_argument("--config", required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument(
        "--limit", type=int, default=None, help="Only evaluate the first N cached prompts."
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, parse_set_overrides(args.overrides))
    results = evaluate(cfg, limit=args.limit)
    csv_path = write_results_csv(results, cfg.output_dir)
    png_path = plot_curve(results, cfg.output_dir)
    sanity_check(results)
    print(f"[stage4] wrote {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
