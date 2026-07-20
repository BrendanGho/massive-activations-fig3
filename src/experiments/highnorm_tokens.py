"""Do massive-activation outlier tokens coincide with ViT "high-norm" tokens? (FLUX)

Question and decision rule: ``SPEC_highnorm.md``. Numeric core: ``src/common/highnorm.py``.

Isolating the top 1-2 massive-activation channels at a FLUX double-stream block renders a
near-black image with sparse bright speckles. This asks whether those speckle tokens are the
same tokens Darcet et al. (arXiv:2309.16588) call high-norm / register tokens.

**The whole design turns on one confound.** ``‖x‖² = Σ_d x[d]²``, so a token with a massive
value in one channel is high-norm *by construction*; correlating the two directly is circular
and always returns overlap ~1. Every statistic here is computed against ``N_ex`` — the token
norm with the massive channels excised. The confounded full-norm version is also computed,
but only so the run reports the size of the artifact it is controlling for.

Reuses (unchanged): ``model_utils`` (load/capture/generate), ``stage2.channel_scores``
(abs-then-mean ranking), ``spatial`` (latent-grid reshape). Torch / matplotlib / PIL are
imported lazily, so this module imports and its pure functions test on CPU.

    python -m src.experiments.highnorm_tokens --config configs/highnorm_tokens.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np
import yaml

from src.common import highnorm

# --- config -------------------------------------------------------------------


@dataclass
class HighNormConfig:
    # Read by model_utils.load_pipeline / generate_with_capture.
    model_ckpt: str | None = None
    output_dir: str | None = None
    device: str = "cuda"
    dtype: str = "bf16"
    seed: int = 0
    num_denoising_steps: int = 4  # FLUX.1-schnell few-step schedule
    resolution: int = 1024
    guidance_scale: float | None = None
    offload: bool = False

    # Probe point.
    target_layer: int = 18  # last FLUX double-stream block
    base_k: int = 2  # channels that produce the speckles (the user's "top 1-2")
    rho_ks: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50])
    outlier_frac: float = 0.01  # top-1% of tokens by massive score

    # Nulls.
    n_null_trials: int = 20
    null_rank_lo: int = 50
    null_rank_hi: int = 500

    # E3 layer sweep. Every `norm_profile_stride`-th block, plus target and final.
    norm_profile_stride: int = 3

    prompts: list[str] = field(default_factory=list)
    n_spatial_panels: int = 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_REQUIRED = ("model_ckpt", "output_dir")


def load_highnorm_config(config_path: str, *, create_dirs: bool = True) -> HighNormConfig:
    """Load + validate the experiment YAML. Fails loud on missing required / unknown keys."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")

    valid = {f.name for f in fields(HighNormConfig)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {sorted(unknown)}")

    cfg = HighNormConfig(**raw)
    _validate(cfg, config_path)
    if create_dirs and cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)
    return cfg


def _validate(cfg: HighNormConfig, source: str) -> None:
    missing = [k for k in _REQUIRED if not getattr(cfg, k)]
    if missing:
        raise ValueError(f"Missing required config value(s) {missing} (source: {source}).")
    if not cfg.prompts:
        raise ValueError(f"`prompts` must be a non-empty list (source: {source}).")
    if cfg.base_k <= 0:
        raise ValueError(f"base_k must be positive, got {cfg.base_k}")
    if not cfg.rho_ks or any(int(k) <= 0 for k in cfg.rho_ks):
        raise ValueError(f"rho_ks must be a non-empty list of positive ints, got {cfg.rho_ks}")
    # Normalize so the E1 curve is drawn in ascending order (matplotlib would otherwise
    # zig-zag) and each k appears once.
    cfg.rho_ks = sorted({int(k) for k in cfg.rho_ks})
    if not 0.0 < cfg.outlier_frac <= 1.0:
        raise ValueError(f"outlier_frac must be in (0, 1], got {cfg.outlier_frac}")
    if cfg.dtype not in ("bf16", "fp16", "fp32"):
        raise ValueError(f"dtype must be one of bf16/fp16/fp32, got {cfg.dtype!r}")
    if cfg.null_rank_hi <= cfg.null_rank_lo:
        raise ValueError(
            f"null_rank_hi ({cfg.null_rank_hi}) must exceed null_rank_lo ({cfg.null_rank_lo})"
        )
    if cfg.norm_profile_stride <= 0:
        raise ValueError(f"norm_profile_stride must be positive, got {cfg.norm_profile_stride}")


# --- per-prompt analysis (pure numpy; no torch) --------------------------------


def _ratio(a: np.ndarray, b: np.ndarray) -> float:
    """median(a) / median(b). Empty input or 0/0 -> NaN (undefined); x>0 over 0 -> +inf.

    A zero-median denominator is not a failure: it means over half the typical tokens
    score exactly zero, so the outliers are *unboundedly* more selective/elevated — the
    strongest possible signal, which `summarize` reads as such rather than discarding.
    """
    if a.size == 0 or b.size == 0:
        return float("nan")
    num, den = float(np.median(a)), float(np.median(b))
    if den > 0:
        return num / den
    return float("nan") if num == 0 else float("inf")


def analyze_layer(
    image_stream: np.ndarray,
    cfg: HighNormConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """E1 + E2 for one (prompt, layer) image stream. Pure numpy => CPU-testable.

    Returns a flat dict of metrics. The verdict (see ``summarize``) is driven by
    ``selectivity`` and ``elevation``, NOT by ``rho_outlier_at_base_k`` or the overlap
    stats: rho cannot separate H1 from H3 and the IoU-vs-null test cannot detect H2, so
    those are reported for characterization only. ``iou_confounded`` (measured against the
    full norm rather than ``N_ex``) is expected to be ~1 and only quantifies the
    circularity being controlled for — it must never be read as a result.
    """
    x = np.asarray(image_stream)
    n_tokens = x.shape[0]
    base = highnorm.top_channels(x, cfg.base_k)

    m = highnorm.massive_score(x, base)
    n_ex = highnorm.token_norms(x, exclude=base)
    n_full = highnorm.token_norms(x)

    outliers = highnorm.top_fraction_indices(m, cfg.outlier_frac)
    hi_ex = highnorm.top_fraction_indices(n_ex, cfg.outlier_frac)
    hi_full = highnorm.top_fraction_indices(n_full, cfg.outlier_frac)

    deconf = highnorm.overlap_stats(outliers, hi_ex, n_tokens)
    conf = highnorm.overlap_stats(outliers, hi_full, n_tokens)

    labels = np.zeros(n_tokens, dtype=bool)
    labels[hi_ex] = True

    o_mask = np.zeros(n_tokens, dtype=bool)
    o_mask[outliers] = True
    # The two effect sizes the verdict actually rests on (set overlap alone cannot
    # separate the hypotheses — see the H2 note in `summarize`):
    #   selectivity — is the massive channel token-sparse, i.e. are there speckles at all?
    #   elevation   — do those tokens stay high-norm AFTER the massive channels are removed?
    selectivity = _ratio(m[o_mask], m[~o_mask])
    elevation = _ratio(n_ex[o_mask], n_ex[~o_mask])

    curve = highnorm.variance_explained_curve(
        x, ks=list(cfg.rho_ks), base_k=cfg.base_k, outlier_frac=cfg.outlier_frac
    )
    # rho at exactly base_k: `base` already IS the top-base_k channel set, so this is
    # exact — no need to interpolate from `curve` (whose ks need not contain base_k, and
    # np.interp would silently clamp / assume sorted xp).
    rho_base = highnorm.norm_fraction(x, base)
    rho_outlier_at_base = float(np.median(rho_base[o_mask]))
    rho_typical_at_base = float(np.median(rho_base[~o_mask])) if (~o_mask).any() else float("nan")

    # Null 1: scale-matched random channels, same k, same downstream pipeline.
    null_ious, null_aurocs = [], []
    for _ in range(int(cfg.n_null_trials)):
        try:
            ch = highnorm.scale_matched_channels(
                x, cfg.base_k, base, rng, cfg.null_rank_lo, cfg.null_rank_hi
            )
        except ValueError:
            break
        m_null = highnorm.massive_score(x, ch)
        n_ex_null = highnorm.token_norms(x, exclude=ch)
        o_null = highnorm.top_fraction_indices(m_null, cfg.outlier_frac)
        h_null = highnorm.top_fraction_indices(n_ex_null, cfg.outlier_frac)
        null_ious.append(highnorm.overlap_stats(o_null, h_null, n_tokens)["iou"])
        lab = np.zeros(n_tokens, dtype=bool)
        lab[h_null] = True
        null_aurocs.append(highnorm.auroc(m_null, lab))

    # Null 2: token permutation — fixes the spatial-sparsity baseline.
    perm_ious = []
    for _ in range(int(cfg.n_null_trials)):
        o_perm = highnorm.top_fraction_indices(rng.permutation(m), cfg.outlier_frac)
        perm_ious.append(highnorm.overlap_stats(o_perm, hi_ex, n_tokens)["iou"])

    return {
        "n_tokens": int(n_tokens),
        "base_channels": base.tolist(),
        # effect sizes driving the verdict
        "selectivity": selectivity,
        "elevation": elevation,
        # E1
        "rho_ks": curve["ks"],
        "rho_outlier": curve["rho_outlier"],
        "rho_typical": curve["rho_typical"],
        "rho_outlier_at_base_k": rho_outlier_at_base,
        "rho_typical_at_base_k": rho_typical_at_base,
        # E2 (deconfounded — the actual result)
        "iou_deconfounded": deconf["iou"],
        "n_intersection": deconf["n_intersection"],
        "expected_intersection": deconf["expected_intersection"],
        "p_value": deconf["p_value"],
        "auroc_deconfounded": highnorm.auroc(m, labels),
        "spearman_m_vs_nex": highnorm.spearman(m, n_ex),
        # confound size, for reporting only
        "iou_confounded": conf["iou"],
        "spearman_m_vs_nfull": highnorm.spearman(m, n_full),
        # nulls
        "null_iou_mean": float(np.mean(null_ious)) if null_ious else float("nan"),
        "null_iou_max": float(np.max(null_ious)) if null_ious else float("nan"),
        "null_auroc_mean": float(np.nanmean(null_aurocs)) if null_aurocs else float("nan"),
        "perm_iou_mean": float(np.mean(perm_ious)) if perm_ious else float("nan"),
    }


def outlier_mask(image_stream: np.ndarray, base_k: int, outlier_frac: float) -> np.ndarray:
    """Boolean (N,) mask of the speckle tokens — the top ``outlier_frac`` by massive score."""
    x = np.asarray(image_stream)
    m = highnorm.massive_score(x, highnorm.top_channels(x, base_k))
    mask = np.zeros(x.shape[0], dtype=bool)
    mask[highnorm.top_fraction_indices(m, outlier_frac)] = True
    return mask


#: Below this, the "massive" channel is not token-sparse and there are no speckles to
#: explain — the experiment's precondition fails. H1/H3 data both show rho ~1, so rho
#: cannot be used here; only selectivity separates them.
SELECTIVITY_MIN = 10.0
#: Above this, outlier tokens are still high-norm with the massive channels removed.
ELEVATION_H2 = 1.5


def summarize(per_prompt: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-prompt metrics and apply the SPEC's H1/H2/H3 decision rule.

    The verdict rests on two effect sizes, NOT on set overlap versus the nulls. Overlap
    cannot carry it: under genuine H2 the outlier tokens are elevated across the whole
    channel dimension, so a scale-matched random channel reproduces nearly the same
    overlap (~0.82 vs ~1.00 on planted data) and any "beats the null" test is dead on
    arrival. IoU and the nulls are still reported — they characterise the geometry — but
    the discrimination comes from:

    * ``selectivity`` = median m[outlier] / median m[typical] — are there speckles at all?
    * ``elevation``   = median N_ex[outlier] / median N_ex[typical] — do those tokens stay
      high-norm once the massive channels are excised? This is the question, stated numerically.
    """

    def med(key: str) -> float:
        # Keep +/-inf (a genuine unbounded selectivity/elevation signal); drop only NaN
        # (undefined). np.median propagates inf sensibly: median([inf, 1, 2]) == 2.
        vals = [p[key] for p in per_prompt if p.get(key) is not None]
        vals = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
        return float(np.median(vals)) if vals else float("nan")

    rho_out = med("rho_outlier_at_base_k")
    iou = med("iou_deconfounded")
    null_iou = med("null_iou_mean")
    perm_iou = med("perm_iou_mean")
    selectivity = med("selectivity")
    elevation = med("elevation")

    # Only NaN (undefined) is inconclusive; +inf is a valid unbounded signal that the
    # threshold comparisons below handle correctly (inf < 10 -> False, inf >= 1.5 -> True).
    if np.isnan(selectivity) or np.isnan(elevation):
        verdict, reading = "inconclusive", "Effect sizes were undefined; inspect per_prompt.csv."
    elif selectivity < SELECTIVITY_MIN:
        verdict, reading = (
            "H3",
            (
                f"No sparse outlier structure: the top channel is only {selectivity:.1f}x larger "
                "on its top tokens than elsewhere, i.e. it is uniformly large rather than "
                "token-sparse. There are no speckles to explain, so the premise does not hold "
                "at this layer."
            ),
        )
    elif elevation >= ELEVATION_H2:
        verdict, reading = (
            "H2",
            (
                f"Co-located but distinct: with the massive channels removed, outlier tokens are "
                f"still {elevation:.1f}x the typical token norm. These are genuine register-like "
                "tokens that are broadly elevated, of which the massive channels are one facet."
            ),
        )
    else:
        verdict, reading = (
            "H1",
            (
                f"One mechanism: with the massive channels removed, outlier tokens sit at "
                f"{elevation:.2f}x the typical token norm — i.e. unremarkable. The massive "
                "channels ARE how these tokens acquire high norm; massive activations and "
                "high-norm tokens are the same phenomenon seen two ways."
            ),
        )

    return {
        "n_prompts": len(per_prompt),
        "verdict": verdict,
        "reading": reading,
        "median_selectivity": selectivity,
        "median_elevation": elevation,
        "median_rho_outlier_at_base_k": rho_out,
        "median_rho_typical_at_base_k": med("rho_typical_at_base_k"),
        "median_iou_deconfounded": iou,
        "median_iou_confounded": med("iou_confounded"),
        "median_auroc_deconfounded": med("auroc_deconfounded"),
        "median_spearman_m_vs_nex": med("spearman_m_vs_nex"),
        "median_spearman_m_vs_nfull": med("spearman_m_vs_nfull"),
        "null_iou_scale_matched": null_iou,
        "null_iou_token_permutation": perm_iou,
        "median_p_value": med("p_value"),
    }


# --- figures (matplotlib imported lazily) -------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _fig_variance_explained(path: str, per_prompt: list[dict[str, Any]]) -> None:
    plt = _plt()
    ks = per_prompt[0]["rho_ks"]
    out = np.array([p["rho_outlier"] for p in per_prompt], dtype=float)
    typ = np.array([p["rho_typical"] for p in per_prompt], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for arr, label, color in (
        (out, "outlier tokens", "#c0392b"),
        (typ, "typical tokens", "#2c3e50"),
    ):
        med = np.nanmedian(arr, axis=0)
        ax.plot(ks, med, "o-", color=color, label=label)
        ax.fill_between(
            ks,
            np.nanpercentile(arr, 25, axis=0),
            np.nanpercentile(arr, 75, axis=0),
            color=color,
            alpha=0.18,
        )
    ax.axhline(0.9, ls=":", c="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("k (number of top massive channels)")
    ax.set_ylabel(r"$\rho$ = share of squared token norm")
    ax.set_title("E1: how much of a token's norm the massive channels own")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_overlap(path: str, per_prompt: list[dict[str, Any]]) -> None:
    plt = _plt()
    series = [
        ("deconfounded\n(vs $N_{ex}$)", [p["iou_deconfounded"] for p in per_prompt], "#c0392b"),
        ("scale-matched\nnull", [p["null_iou_mean"] for p in per_prompt], "#7f8c8d"),
        ("token-permutation\nnull", [p["perm_iou_mean"] for p in per_prompt], "#bdc3c7"),
        ("confounded\n(vs $N_{full}$)", [p["iou_confounded"] for p in per_prompt], "#95a5a6"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (label, vals, color) in enumerate(series):
        vals = np.array(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.scatter(
                np.full(vals.size, i) + rng_jitter(vals.size), vals, s=14, alpha=0.6, color=color
            )
            ax.hlines(np.median(vals), i - 0.25, i + 0.25, color="black", lw=2)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([s[0] for s in series], fontsize=8)
    ax.set_ylabel("IoU (massive-score top set vs high-norm top set)")
    ax.set_title(
        "E2: overlap, deconfounded vs nulls\n(confounded bar shows the artifact)", fontsize=10
    )
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rng_jitter(n: int, width: float = 0.08) -> np.ndarray:
    return np.random.default_rng(0).uniform(-width, width, size=n)


def _fig_norm_profile(path: str, profile: dict[int, dict[str, list[float]]]) -> None:
    """Darcet Fig. 4a analogue: norm distribution and bimodality across depth."""
    plt = _plt()
    layers = sorted(profile)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    pcts = [50, 90, 99, 99.9]
    for p, color in zip(pcts, ["#bdc3c7", "#7f8c8d", "#e67e22", "#c0392b"]):
        ax1.plot(
            layers,
            [np.median(profile[ly][f"p{p}"]) for ly in layers],
            "o-",
            color=color,
            ms=3,
            label=f"p{p}",
        )
    ax1.set_yscale("log")
    ax1.set_ylabel("token L2 norm")
    ax1.set_title("E3: norm distribution across depth")
    ax1.legend(fontsize=8, ncol=4)

    ax2.plot(layers, [np.median(profile[ly]["bc"]) for ly in layers], "o-", color="#2c3e50", ms=3)
    ax2.axhline(highnorm.BC_BIMODAL_CUTOFF, ls="--", c="#c0392b", lw=1, label="bimodal cutoff 5/9")
    ax2.axhline(highnorm.BC_GAUSSIAN, ls=":", c="gray", lw=1, label="Gaussian 1/3")
    ax2.set_ylabel("bimodality coefficient")
    ax2.set_xlabel("transformer block")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_norm_hist(path: str, norms: np.ndarray, split: dict[str, float], layer_name: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(norms[norms > 0], bins=120, log=True, color="#34495e")
    if np.isfinite(split["threshold"]):
        ax.axvline(
            split["threshold"],
            color="#c0392b",
            ls="--",
            label=f"2-means cutoff {split['threshold']:.1f}\n"
            f"({100 * split['frac_above']:.2f}% above)",
        )
    ax.set_xscale("log")
    ax.set_xlabel("token L2 norm")
    ax.set_ylabel("count")
    ax.set_title(
        f"Norm histogram, {layer_name}  (BC={split['bc']:.2f}; "
        f"bimodal if > {highnorm.BC_BIMODAL_CUTOFF:.2f})",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_spatial_panels(path: str, panels: list[dict[str, Any]]) -> None:
    """Per-prompt: massive-channel map, full norm, deconfounded norm — side by side."""
    plt = _plt()
    n = len(panels)
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.7 * n), squeeze=False)
    for r, panel in enumerate(panels):
        for c, (key, title, cmap) in enumerate(
            (
                ("rgb", "generated", None),
                ("massive", "massive-channel score $m$", "inferno"),
                ("n_full", r"full norm $N_{full}$", "viridis"),
                ("n_ex", r"deconfounded norm $N_{ex}$", "viridis"),
            )
        ):
            ax = axes[r][c]
            ax.imshow(panel[key], cmap=cmap, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
        axes[r][0].set_ylabel(panel["prompt"][:28], fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_position_stability(path: str, freq: np.ndarray, n_prompts: int) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(freq, cmap="magma", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="fraction of prompts where token is an outlier")
    ax.set_title(f"Outlier-token position stability\nacross {n_prompts} prompts", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- runner -------------------------------------------------------------------


def _profile_layers(all_ids: list[int], target: int, stride: int) -> list[int]:
    wanted = set(all_ids[::stride]) | {target, all_ids[-1], all_ids[0]}
    return sorted(w for w in wanted if w in set(all_ids))


def run(cfg: HighNormConfig) -> dict[str, Any]:
    from src.common import model_utils

    pipe = model_utils.load_pipeline(cfg, offload=cfg.offload)
    all_blocks = model_utils.discover_blocks(pipe.transformer)
    all_ids = [b.layer_id for b in all_blocks]
    if cfg.target_layer not in all_ids:
        raise ValueError(f"target_layer {cfg.target_layer} not in discovered blocks {all_ids}")
    layer_ids = _profile_layers(all_ids, cfg.target_layer, cfg.norm_profile_stride)
    final_layer = all_ids[-1]

    blocks = model_utils.select_layers(all_blocks, layer_ids)
    state = model_utils.CaptureState()
    handles = model_utils.register_capture_hooks(pipe.transformer, blocks, state)

    per_prompt: list[dict[str, Any]] = []
    profile: dict[int, dict[str, list[float]]] = {
        ly: {"p50": [], "p90": [], "p99": [], "p99.9": [], "bc": []} for ly in layer_ids
    }
    masks: list[np.ndarray] = []
    panels: list[dict[str, Any]] = []
    final_norms_pool: list[np.ndarray] = []
    cross_layer: list[dict[str, float]] = []
    neighbor_rows: list[dict[str, float]] = []
    h_lat = w_lat = None

    try:
        for pid, prompt in enumerate(cfg.prompts):
            rgb, info = model_utils.generate_with_capture(pipe, prompt, cfg, state)
            streams = dict(state.image_streams)
            if cfg.target_layer not in streams:
                raise RuntimeError(f"no capture at layer {cfg.target_layer} for prompt {pid}")
            h_lat, w_lat = info["h_lat"], info["w_lat"]
            rng = np.random.default_rng([int(cfg.seed), pid])

            x = streams[cfg.target_layer]
            row = analyze_layer(x, cfg, rng)
            row.update({"prompt_id": pid, "prompt": prompt, "layer": cfg.target_layer})
            per_prompt.append(row)

            # E3: norm distribution + bimodality at every profiled layer.
            for ly, s in streams.items():
                nrm = highnorm.token_norms(s)
                for p in (50, 90, 99, 99.9):
                    profile[ly][f"p{p}"].append(float(np.percentile(nrm, p)))
                profile[ly]["bc"].append(highnorm.bimodality_coefficient(np.log10(nrm[nrm > 0])))

            # Darcet's criterion is on the FINAL output: do layer-`target` speckle tokens
            # predict final-layer high-norm tokens (deconfounded at the final layer too)?
            mask = outlier_mask(x, cfg.base_k, cfg.outlier_frac)
            masks.append(mask)
            xf = streams[final_layer]
            base_f = highnorm.top_channels(xf, cfg.base_k)
            hi_final = highnorm.top_fraction_indices(
                highnorm.token_norms(xf, exclude=base_f), cfg.outlier_frac
            )
            cross_layer.append(highnorm.overlap_stats(np.flatnonzero(mask), hi_final, x.shape[0]))
            final_norms_pool.append(highnorm.token_norms(xf))

            # Darcet's positive control: are outlier tokens redundant with their neighbours?
            early = streams[min(streams)]
            sim = highnorm.neighbor_cosine_similarity(early, h_lat, w_lat)
            neighbor_rows.append(
                {
                    "neighbor_cos_outlier": float(np.median(sim[mask])),
                    "neighbor_cos_typical": float(np.median(sim[~mask])),
                }
            )

            if len(panels) < cfg.n_spatial_panels:
                base = highnorm.top_channels(x, cfg.base_k)
                panels.append(
                    {
                        "prompt": prompt,
                        "rgb": rgb,
                        "massive": highnorm.massive_score(x, base).reshape(h_lat, w_lat),
                        "n_full": highnorm.token_norms(x).reshape(h_lat, w_lat),
                        "n_ex": highnorm.token_norms(x, exclude=base).reshape(h_lat, w_lat),
                    }
                )
            print(
                f"[highnorm] prompt {pid + 1}/{len(cfg.prompts)}: "
                f"rho_out={row['rho_outlier_at_base_k']:.3f} "
                f"iou_deconf={row['iou_deconfounded']:.3f} "
                f"(null {row['null_iou_mean']:.3f})"
            )
    finally:
        for h in handles:
            h.remove()

    summary = summarize(per_prompt)
    summary["cross_layer_to_final"] = {
        "final_layer": final_layer,
        "median_iou": float(np.median([c["iou"] for c in cross_layer])),
        "median_expected": float(np.median([c["expected_intersection"] for c in cross_layer])),
    }
    summary["neighbor_redundancy"] = {
        "median_cos_outlier": float(np.median([r["neighbor_cos_outlier"] for r in neighbor_rows])),
        "median_cos_typical": float(np.median([r["neighbor_cos_typical"] for r in neighbor_rows])),
        "note": "layer-0 block output used as the early-representation proxy (no ViT patch embed)",
    }
    final_norms = np.concatenate(final_norms_pool)
    split = highnorm.bimodality_split(final_norms)
    summary["final_layer_norm_split"] = split
    summary["config"] = cfg.to_dict()
    summary["deviations"] = [
        "neighbour redundancy uses layer-0 output, not a ViT patch-embedding layer",
        "high-norm threshold derived per-run by 2-means on log-norms (Darcet's 150 is "
        "DINOv2-specific and stated to vary by model)",
        "last denoising step only (fig3 capture convention)",
    ]

    _write_outputs(
        cfg,
        per_prompt,
        summary,
        profile,
        panels,
        masks,
        final_norms,
        split,
        final_layer,
        h_lat,
        w_lat,
    )
    return summary


def _write_outputs(
    cfg, per_prompt, summary, profile, panels, masks, final_norms, split, final_layer, h_lat, w_lat
) -> None:
    out = cfg.output_dir
    scalar_keys = [k for k, v in per_prompt[0].items() if not isinstance(v, list)]
    with open(os.path.join(out, "per_prompt.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_prompt)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    _fig_variance_explained(os.path.join(out, "fig_variance_explained.png"), per_prompt)
    _fig_overlap(os.path.join(out, "fig_overlap.png"), per_prompt)
    _fig_norm_profile(os.path.join(out, "fig_norm_profile.png"), profile)
    _fig_norm_hist(
        os.path.join(out, "fig_norm_hist.png"), final_norms, split, f"block {final_layer} (final)"
    )
    if panels:
        _fig_spatial_panels(os.path.join(out, "fig_spatial_panels.png"), panels)
    if masks and h_lat:
        freq = np.mean(np.stack(masks), axis=0).reshape(h_lat, w_lat)
        _fig_position_stability(os.path.join(out, "fig_position_stability.png"), freq, len(masks))

    print(f"\n[highnorm] verdict: {summary['verdict']}\n{summary['reading']}")
    print(
        f"[highnorm] rho_outlier={summary['median_rho_outlier_at_base_k']:.3f} "
        f"iou_deconfounded={summary['median_iou_deconfounded']:.4f} "
        f"vs null {summary['null_iou_scale_matched']:.4f} "
        f"(confounded {summary['median_iou_confounded']:.3f})"
    )
    print(f"[highnorm] outputs -> {out}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Massive-activation vs high-norm-token overlap.")
    p.add_argument("--config", required=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    run(load_highnorm_config(args.config))


if __name__ == "__main__":
    main()
