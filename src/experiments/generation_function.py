"""Q7: paired generation interventions for the register/channel/sink circuit.

The numeric helpers are deliberately torch/diffusers-free at import time.  GPU-only model hooks
live behind lazy imports so the operators and analysis remain unit-testable on CPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CONDITIONS = (
    "baseline",
    "remove_vstar",
    "suppress_channel",
    "suppress_sink",
    "remove_top_registers",
    "norm_only",
)
EXPERIMENT_REVISION = "q7-generation-function-v3"
GENEVAL_METRICS = (
    "geneval_counting",
    "geneval_attribute",
    "geneval_spatial",
    "geneval_overall",
)
STRUCTURED_SCORE_KEYS = ("run_identity", "prompt_id", "seed", "condition", "phase", "zone")


@dataclass(frozen=True)
class Q7Config:
    model_ckpt: str = "black-forest-labs/FLUX.1-dev"
    model_preset: str | None = None
    model_family: str = "flux1"
    output_dir: str = "./q7_outputs"
    prompts: tuple[str, ...] = ()
    seeds: tuple[int, ...] = (0,)
    resolution: int = 1024
    num_steps: int = 28
    guidance_scale: float | None = 3.5
    dtype: str = "bf16"
    device: str = "cuda"
    offload: bool = False
    channel: int = 154
    register_threshold: float = 3.0
    max_registers: int = 8
    phases: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"early": (0, 8), "middle": (9, 18), "late": (19, 27)}
    )
    zones: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "writer": (18, 18),
            "early_register": (19, 22),
            "mid_register": (23, 34),
            "dissolution": (35, 39),
        }
    )
    conditions: tuple[str, ...] = CONDITIONS
    vstar_path: str | None = None

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> Q7Config:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("prompts", "seeds", "conditions"):
            if key in raw:
                raw[key] = tuple(raw[key])
        for key in ("phases", "zones"):
            if key in raw:
                raw[key] = {k: tuple(v) for k, v in raw[key].items()}
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.model_family not in {"flux1", "pixart_sigma"}:
            raise ValueError(f"unsupported Q7 model_family: {self.model_family!r}")
        bad = set(self.conditions) - set(CONDITIONS)
        if bad:
            raise ValueError(f"unknown conditions: {sorted(bad)}")
        if not self.prompts:
            raise ValueError("at least one prompt is required")
        max_layer = 56 if self.model_family == "flux1" else 27
        for group, ranges, upper in (
            ("phases", self.phases, self.num_steps - 1),
            ("zones", self.zones, max_layer),
        ):
            for name, (lo, hi) in ranges.items():
                if lo < 0 or hi < lo or hi > upper:
                    raise ValueError(f"invalid {group}.{name} range {(lo, hi)}")


def in_target(step: int, layer: int, phase: tuple[int, int], zone: tuple[int, int]) -> bool:
    return phase[0] <= step <= phase[1] and zone[0] <= layer <= zone[1]


def denoising_thirds(num_steps: int) -> dict[str, tuple[int, int]]:
    """Split a schedule into exhaustive early/middle/late integer ranges."""
    if num_steps < 3:
        raise ValueError("at least three denoising steps are required")
    edges = [round(i * num_steps / 3) for i in range(4)]
    names = ("early", "middle", "late")
    return {name: (edges[i], edges[i + 1] - 1) for i, name in enumerate(names)}


def validate_model_layout(blocks: list[Any], cfg: Q7Config) -> None:
    """Fail before generation unless the discovered layout matches the selected adapter."""
    layer_ids = [int(block.layer_id) for block in blocks]
    kinds = [block.kind for block in blocks]
    if cfg.model_family == "flux1":
        expected_ids = list(range(57))
        if (
            layer_ids != expected_ids
            or kinds[:19] != ["double"] * 19
            or kinds[19:] != ["single"] * 38
        ):
            raise RuntimeError(
                "Q7 FLUX.1 expects 57 blocks (19 dual-stream + 38 single-stream); "
                f"discovered {len(blocks)} blocks with kinds {kinds[:2]}...{kinds[-2:]}"
            )
        attention = getattr(blocks[0].module, "attn", None)
    else:
        expected_ids = list(range(28))
        if layer_ids != expected_ids:
            raise RuntimeError(
                "Q7 PixArt-Sigma expects 28 image-only transformer blocks; "
                f"discovered layer ids {layer_ids}"
            )
        attention = getattr(blocks[0].module, "attn1", None)
        if attention is None:
            raise RuntimeError("Q7 PixArt-Sigma blocks must expose image self-attention as attn1")
    targeted = {layer for lo, hi in cfg.zones.values() for layer in range(lo, hi + 1)}
    missing = targeted - set(layer_ids)
    if missing:
        raise RuntimeError(f"Q7 depth zones reference missing layers: {sorted(missing)}")
    width = getattr(getattr(attention, "to_q", None), "in_features", None)
    if width is not None and not 0 <= cfg.channel < int(width):
        raise RuntimeError(
            f"channel {cfg.channel} is outside the {cfg.model_family} residual width {width}"
        )


def validate_flux1_layout(blocks: list[Any], cfg: Q7Config) -> None:
    """Backward-compatible name for callers that validate the configured model adapter."""
    validate_model_layout(blocks, cfg)


def natural_register_mask(
    x: np.ndarray, threshold: float = 3.0, max_registers: int = 8
) -> np.ndarray:
    """Paper definition: > threshold * median norm, retaining at most the largest K."""
    x = np.asarray(x)
    norms = np.linalg.norm(x.astype(np.float64), axis=-1)
    eligible = np.flatnonzero(norms > float(threshold) * np.median(norms))
    if eligible.size > max_registers:
        order = np.lexsort((eligible, -norms[eligible]))[:max_registers]
        eligible = eligible[order]
    mask = np.zeros(x.shape[0], dtype=bool)
    mask[eligible] = True
    return mask


def fit_vstar(register_vectors: np.ndarray) -> np.ndarray:
    """Uncentered top right-singular direction, sign oriented to positive mean projection."""
    x = np.asarray(register_vectors, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("register_vectors must be a non-empty [M,D] array")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    u = x / np.maximum(n, np.finfo(np.float64).eps)
    _left, _s, vh = np.linalg.svd(u, full_matrices=False)
    v = vh[0]
    if np.mean(u @ v) < 0:
        v = -v
    return (v / np.linalg.norm(v)).astype(np.float32)


def apply_numpy_intervention(
    x: np.ndarray,
    condition: str,
    register_mask: np.ndarray,
    *,
    vstar: np.ndarray | None = None,
    channel: int = 154,
) -> np.ndarray:
    """Reference implementation used by tests and analysis; returns a fresh array."""
    y = np.array(x, copy=True)
    mask = np.asarray(register_mask, dtype=bool)
    if condition == "baseline" or condition == "suppress_sink":
        return y
    if condition == "suppress_channel":
        y[..., channel] = 0
    elif condition == "remove_top_registers":
        y[mask] = 0
    elif condition == "remove_vstar":
        if vstar is None:
            raise ValueError("remove_vstar requires vstar")
        v = np.asarray(vstar, dtype=y.dtype)
        v = v / np.linalg.norm(v)
        y[mask] -= (y[mask] @ v)[:, None] * v[None, :]
    elif condition == "norm_only":
        norms = np.linalg.norm(y, axis=-1)
        target = float(np.median(norms[~mask])) if np.any(~mask) else float(np.median(norms))
        denom = np.maximum(norms[mask], np.finfo(np.float32).eps)
        y[mask] *= (target / denom)[:, None]
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return y


def frequency_distances(
    clean: np.ndarray, edited: np.ndarray, sigma: float = 8.0
) -> dict[str, float]:
    """RMS change split into Gaussian low-pass layout and high-pass detail components."""
    from scipy.ndimage import gaussian_filter

    a = np.asarray(clean, dtype=np.float32) / 255.0
    b = np.asarray(edited, dtype=np.float32) / 255.0
    sig = (sigma, sigma, 0.0)
    alo, blo = gaussian_filter(a, sig), gaussian_filter(b, sig)
    low = float(np.sqrt(np.mean((alo - blo) ** 2)))
    high = float(np.sqrt(np.mean(((a - alo) - (b - blo)) ** 2)))
    return {
        "low_frequency_rms": low,
        "high_frequency_rms": high,
        "low_high_ratio": low / max(high, 1e-12),
    }


def paired_bootstrap(
    values: Iterable[float], seed: int = 0, trials: int = 2000
) -> dict[str, float]:
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(trials, x.size))].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n": int(x.size),
    }


def _finite_metric_values(rows: Iterable[dict[str, Any]], metric: str) -> list[float]:
    """Return finite numeric metric values, tolerating blank CSV cells."""
    values = []
    for row in rows:
        value = row.get(metric)
        if value in (None, ""):
            continue
        value = float(value)
        if np.isfinite(value):
            values.append(value)
    return values


def _load_paired_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _scenario_metric_values(rows: Iterable[dict[str, Any]], metric: str) -> list[float]:
    """Average repeated phase/zone cells within scenario before cross-scenario inference."""
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        values = _finite_metric_values([row], metric)
        if values:
            key = (str(row.get("prompt_id", row.get("prompt", ""))), str(row.get("seed", "")))
            grouped.setdefault(key, []).append(values[0])
    return [float(np.mean(values)) for values in grouped.values()]


def generate_figures(cfg: Q7Config) -> list[Path]:
    """Render causal, frequency, fidelity, v* and representative image figures."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from PIL import Image, ImageDraw

    root = Path(cfg.output_dir)
    metrics_path = root / "paired_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"paired metrics not found at {metrics_path}; run --evaluate first")
    rows = _load_paired_metrics(metrics_path)
    if not rows:
        raise RuntimeError(f"no paired rows found in {metrics_path}")

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    conditions = [c for c in cfg.conditions if c != "baseline"]
    phases, zones = list(cfg.phases), list(cfg.zones)
    pretty = {
        "remove_vstar": "Remove v*",
        "suppress_channel": f"Suppress channel {cfg.channel}",
        "suppress_sink": "Suppress sink",
        "remove_top_registers": "Remove registers",
        "norm_only": "Norm only",
    }
    outputs = []

    # Condition x (depth, time) causal maps with one shared perceptual-distance scale.
    matrices = {}
    for condition in conditions:
        matrix = np.full((len(zones), len(phases)), np.nan)
        for zi, zone in enumerate(zones):
            for pi, phase in enumerate(phases):
                cell = [
                    row
                    for row in rows
                    if row.get("condition") == condition
                    and row.get("phase") == phase
                    and row.get("zone") == zone
                ]
                values = _finite_metric_values(cell, "lpips")
                if values:
                    matrix[zi, pi] = float(np.mean(values))
        matrices[condition] = matrix
    finite_parts = [matrix[np.isfinite(matrix)] for matrix in matrices.values()]
    finite_lpips = np.concatenate(finite_parts) if finite_parts else np.array([])
    vmax = float(finite_lpips.max()) if finite_lpips.size else 1.0
    fig, axes = plt.subplots(
        1, len(conditions), figsize=(3.15 * len(conditions), 4.2), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    image = None
    for ax, condition in zip(axes, conditions):
        matrix = matrices[condition]
        image = ax.imshow(matrix, vmin=0, vmax=max(vmax, 1e-12), cmap="magma", aspect="auto")
        ax.set_title(pretty.get(condition, condition), fontsize=10)
        ax.set_xticks(range(len(phases)), phases, rotation=35, ha="right")
        ax.set_yticks(range(len(zones)), zones if ax is axes[0] else [])
        for zi in range(len(zones)):
            for pi in range(len(phases)):
                value = matrix[zi, pi]
                color = "white" if np.isfinite(value) and value > vmax * 0.45 else "black"
                ax.text(
                    pi,
                    zi,
                    "—" if not np.isfinite(value) else f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), label="LPIPS from clean same-seed image", shrink=0.72)
    fig.suptitle("Q7 causal effect across denoising time and depth")
    path = figures / "q7_causal_map.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    # Global/layout versus local/detail change over all available paired cells.
    condition_rows = {c: [r for r in rows if r["condition"] == c] for c in conditions}
    x = np.arange(len(conditions))
    low = [
        np.mean(_scenario_metric_values(condition_rows[c], "low_frequency_rms")) for c in conditions
    ]
    high = [
        np.mean(_scenario_metric_values(condition_rows[c], "high_frequency_rms"))
        for c in conditions
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    width = 0.38
    ax.bar(x - width / 2, low, width, label="Low frequency / layout", color="#4472C4")
    ax.bar(x + width / 2, high, width, label="High frequency / detail", color="#ED7D31")
    ax.set_xticks(x, [pretty.get(c, c) for c in conditions], rotation=20, ha="right")
    ax.set_ylabel("RMS difference from clean image")
    ax.set_title("Q7 structural versus local image change")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    path = figures / "q7_frequency_profile.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    # Paired prompt-fidelity effects. A singleton smoke run remains visibly n=1.
    fidelity = [
        metric
        for metric in (
            "clip_delta",
            "image_reward_delta",
            *(f"{name}_delta" for name in GENEVAL_METRICS),
        )
        if _finite_metric_values(rows, metric)
    ]
    if fidelity:
        ncols = min(3, len(fidelity))
        nrows = math.ceil(len(fidelity) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6.0 * ncols, 4.8 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        for ax, metric in zip(axes.flat, fidelity):
            for yi, condition in enumerate(conditions):
                values = _scenario_metric_values(condition_rows[condition], metric)
                stats = paired_bootstrap(values)
                if not values:
                    continue
                ax.errorbar(
                    stats["mean"],
                    yi,
                    xerr=[
                        [stats["mean"] - stats["ci_low"]],
                        [stats["ci_high"] - stats["mean"]],
                    ],
                    fmt="o",
                    color="#222222",
                    capsize=3,
                )
                ax.text(stats["mean"], yi + 0.2, f"n={stats['n']}", fontsize=7, ha="center")
            ax.axvline(0, color="#888888", linewidth=1, linestyle="--")
            ax.set_yticks(range(len(conditions)), [pretty.get(c, c) for c in conditions])
            ax.set_xlabel("Edited − clean score")
            metric_title = {
                "clip_delta": "CLIP",
                "image_reward_delta": "ImageReward",
                "geneval_counting_delta": "GenEval counting",
                "geneval_attribute_delta": "GenEval attributes",
                "geneval_spatial_delta": "GenEval spatial",
                "geneval_overall_delta": "GenEval overall",
            }
            ax.set_title(metric_title[metric])
            ax.tick_params(axis="x", labelsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes.flat[len(fidelity) :]:
            ax.set_visible(False)
        fig.suptitle("Q7 paired prompt-fidelity effects (95% bootstrap CI)")
        path = figures / "q7_prompt_fidelity.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)

    # Calibrated shared direction: expose whether v* is effectively one channel.
    calibrated = Path(cfg.vstar_path) if cfg.vstar_path else root / "vstar.npy"
    if calibrated.exists():
        vstar = np.load(calibrated)
        top = np.argsort(np.abs(vstar))[::-1][:12]
        colors = ["#C00000" if int(i) == cfg.channel else "#5B9BD5" for i in top]
        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        ax.bar(range(len(top)), vstar[top], color=colors)
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_xticks(range(len(top)), [str(int(i)) for i in top])
        ax.set_xlabel("Channel index (top 12 by |loading|)")
        ax.set_ylabel("v* loading")
        ax.set_title("Calibrated register direction v*")
        ax.spines[["top", "right"]].set_visible(False)
        path = figures / "q7_vstar_loadings.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)

    # Representative clean / edited / amplified-difference contact sheet.
    first = rows[0]
    representative = [
        row
        for row in rows
        if row.get("prompt_id") == first.get("prompt_id")
        and row.get("seed") == first.get("seed")
        and row.get("phase") == first.get("phase")
        and row.get("zone") == first.get("zone")
    ]
    representative.sort(key=lambda row: conditions.index(row["condition"]))
    image_inputs_exist = representative and all(
        Path(row["clean_path"]).exists() and Path(row["image_path"]).exists()
        for row in representative
    )
    if image_inputs_exist:
        tile, header = 256, 42
        sheet = Image.new("RGB", (3 * tile, len(representative) * (tile + header)), "white")
        draw = ImageDraw.Draw(sheet)
        for ri, row in enumerate(representative):
            clean = Image.open(row["clean_path"]).convert("RGB").resize((tile, tile))
            edited = Image.open(row["image_path"]).convert("RGB").resize((tile, tile))
            clean_array = np.asarray(clean, dtype=np.int16)
            edited_array = np.asarray(edited, dtype=np.int16)
            difference = Image.fromarray(
                np.clip(np.abs(clean_array - edited_array) * 4, 0, 255).astype(np.uint8)
            )
            y = ri * (tile + header)
            draw.text((6, y + 4), pretty.get(row["condition"], row["condition"]), fill="black")
            draw.text((6, y + 21), "clean", fill="#555555")
            draw.text((tile + 6, y + 21), "edited", fill="#555555")
            draw.text((2 * tile + 6, y + 21), "|difference| ×4", fill="#555555")
            sheet.paste(clean, (0, y + header))
            sheet.paste(edited, (tile, y + header))
            sheet.paste(difference, (2 * tile, y + header))
        path = figures / "q7_representative_contact_sheet.png"
        sheet.save(path)
        outputs.append(path)

    return outputs


def add_paired_prompt_deltas(row: dict[str, Any]) -> None:
    """Add edited-minus-clean effects for every available prompt-fidelity metric."""
    for metric in ("clip", "image_reward", *GENEVAL_METRICS):
        clean, edited = row.get(f"{metric}_clean"), row.get(f"{metric}_edited")
        row[f"{metric}_delta"] = (
            None if clean is None or edited is None else float(edited) - float(clean)
        )


def _structured_score_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(name, "")) for name in STRUCTURED_SCORE_KEYS)


def load_structured_scores(
    path: str | os.PathLike[str], expected_identity: str
) -> dict[tuple[str, ...], dict[str, float | None]]:
    """Load externally computed GenEval-style clean/edited scores for exact run cells."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        missing = set(STRUCTURED_SCORE_KEYS) - fields
        if missing:
            raise ValueError(f"structured score CSV missing key columns: {sorted(missing)}")
        rows = list(reader)
    scores = {}
    for row in rows:
        if row["run_identity"] != expected_identity:
            continue
        key = _structured_score_key(row)
        if key in scores:
            raise ValueError(f"duplicate structured score key: {key}")
        values = {}
        for metric in GENEVAL_METRICS:
            for suffix in ("clean", "edited"):
                name = f"{metric}_{suffix}"
                raw = row.get(name, "")
                value = None if raw == "" else float(raw)
                if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
                values[name] = value
        scores[key] = values
    if not scores:
        raise ValueError(f"structured score CSV has no rows for run identity {expected_identity}")
    return scores


def config_hash(cfg: Q7Config) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_identity(
    cfg: Q7Config, calibration_sha256: str, scheduler_config: dict[str, Any] | None = None
) -> str:
    scheduler_json = json.dumps(scheduler_config or {}, sort_keys=True, separators=(",", ":"))
    payload = f"{EXPERIMENT_REVISION}:{config_hash(cfg)}:{calibration_sha256}:{scheduler_json}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def expected_target_fires(
    n_steps: int, phase: tuple[int, int], zone: tuple[int, int], layer: int
) -> int:
    if not zone[0] <= layer <= zone[1]:
        return 0
    lo, hi = max(0, phase[0]), min(n_steps - 1, phase[1])
    return max(0, hi - lo + 1)


def audit_counts(
    calls: dict[int, int],
    fires: dict[int, int],
    n_steps: int,
    phase: tuple[int, int],
    zone: tuple[int, int],
) -> dict[str, Any]:
    errors = []
    for layer in sorted(calls):
        if calls[layer] != n_steps:
            errors.append(f"layer {layer}: {calls[layer]} calls != {n_steps}")
        expected = expected_target_fires(n_steps, phase, zone, layer)
        if fires.get(layer, 0) != expected:
            errors.append(f"layer {layer}: {fires.get(layer, 0)} fires != {expected}")
    return {"ok": not errors, "errors": errors, "calls": calls, "fires": fires}


def _torch_edit(x, condition: str, mask_np: np.ndarray, vstar, channel: int):
    """Edit the conditional row only (the sole row for FLUX; last CFG row for PixArt)."""
    import torch

    y = x.clone()
    mask = torch.as_tensor(mask_np, device=y.device, dtype=torch.bool)
    conditional = y[-1:]
    if condition == "suppress_channel":
        conditional[..., channel] = 0
    elif condition == "remove_top_registers":
        conditional[:, mask, :] = 0
    elif condition == "remove_vstar":
        v = torch.as_tensor(vstar, device=y.device, dtype=y.dtype)
        v = v / v.float().norm().to(y.dtype)
        z = conditional[:, mask, :]
        conditional[:, mask, :] = z - (z.float() @ v.float()).to(y.dtype).unsqueeze(-1) * v
    elif condition == "norm_only":
        z = conditional[:, mask, :]
        all_norm = conditional.float().norm(dim=-1)
        ordinary = all_norm[:, ~mask]
        target = (
            ordinary.median(dim=1).values if ordinary.shape[1] else all_norm.median(dim=1).values
        )
        scale = target[:, None] / z.float().norm(dim=-1).clamp_min(1e-12)
        conditional[:, mask, :] = z * scale.to(z.dtype).unsqueeze(-1)
    return y


def _infer_image_token_count(transformer, hidden_states) -> int:
    """Infer sequence length from packed FLUX tokens or PixArt's pre-patch latent grid."""
    if hidden_states.ndim == 3:
        return int(hidden_states.shape[1])
    if hidden_states.ndim == 4:
        patch_size = int(getattr(transformer.config, "patch_size", 0))
        if patch_size <= 0:
            raise RuntimeError("cannot infer PixArt token count without a positive patch_size")
        height, width = hidden_states.shape[-2:]
        if height % patch_size or width % patch_size:
            raise RuntimeError(
                f"latent grid {(height, width)} is not divisible by patch_size={patch_size}"
            )
        return int((height // patch_size) * (width // patch_size))
    raise RuntimeError(f"unsupported transformer hidden-state rank: {hidden_states.ndim}")


def _modify_image_output(output: Any, n_image: int, fn):
    """Modify a FLUX tuple image output or a single/concatenated tensor."""
    import torch

    if torch.is_tensor(output) and output.ndim == 3:
        if output.shape[1] == n_image:
            return fn(output), True
        if output.shape[1] > n_image:
            y = output.clone()
            y[:, -n_image:, :] = fn(output[:, -n_image:, :])
            return y, True
    if isinstance(output, (tuple, list)):
        vals = list(output)
        for i, value in enumerate(vals):
            if torch.is_tensor(value) and value.ndim == 3 and value.shape[1] == n_image:
                vals[i] = fn(value)
                return (tuple(vals) if isinstance(output, tuple) else vals), True
    return output, False


class NaturalTrace:
    """Clean-run register masks, vectors, and per-head attention sinks."""

    def __init__(self, threshold: float, max_registers: int, vector_layers: set[int] | None = None):
        self.threshold, self.max_registers = threshold, max_registers
        self.vector_layers = vector_layers
        self.masks: dict[tuple[int, int], np.ndarray] = {}
        self.sink_indices: dict[tuple[int, int], np.ndarray] = {}
        self.vectors: list[np.ndarray] = []
        self.n_image: int | None = None

    def observe(self, step: int, layer: int, x) -> None:
        arr = x[-1].detach().float().cpu().numpy()
        mask = natural_register_mask(arr, self.threshold, self.max_registers)
        self.masks[(step, layer)] = mask
        if mask.any() and (self.vector_layers is None or layer in self.vector_layers):
            self.vectors.extend(arr[mask])

    def observe_sinks(self, step: int, layer: int, indices: np.ndarray) -> None:
        self.sink_indices[(step, layer)] = np.asarray(indices, dtype=np.int64)


def _flux_qkv(attn, hidden_states, encoder_hidden_states, image_rotary_emb):
    """Construct the normalized, rotary-applied Q/K and V used by FLUX attention."""
    import torch
    from diffusers.models.embeddings import apply_rotary_emb

    if getattr(attn, "fused_projections", False):
        query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        if encoder_hidden_states is not None:
            encoder_query, encoder_key, encoder_value = attn.to_added_qkv(
                encoder_hidden_states
            ).chunk(3, dim=-1)
    else:
        query, key, value = (
            attn.to_q(hidden_states),
            attn.to_k(hidden_states),
            attn.to_v(hidden_states),
        )
        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)
    query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
    key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
    value = value.unflatten(-1, (attn.heads, -1))
    if encoder_hidden_states is not None:
        encoder_query = attn.norm_added_q(encoder_query.unflatten(-1, (attn.heads, -1)))
        encoder_key = attn.norm_added_k(encoder_key.unflatten(-1, (attn.heads, -1)))
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)
    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
        key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
    return query, key, value


def _incoming_image_attention_sinks(query, key, n_image: int, chunk_size: int = 128):
    """Exact per-head sink under paper's image-key-renormalized incoming-attention metric."""
    import torch

    query = query[:, -n_image:].float()
    key = key[:, -n_image:].float()
    incoming = torch.zeros(
        (query.shape[0], query.shape[2], n_image), device=query.device, dtype=torch.float32
    )
    scale = query.shape[-1] ** -0.5
    for lo in range(0, n_image, chunk_size):
        q = query[:, lo : lo + chunk_size]
        scores = torch.einsum("bqhd,bkhd->bhqk", q, key) * scale
        incoming += scores.softmax(dim=-1).sum(dim=2)
    # Last row is the conditional branch under the same convention as residual capture.
    return incoming[-1].argmax(dim=-1).detach().cpu().numpy()


class AttentionSinkTraceProcessor:
    """Delegating processor that records clean per-head image-to-image sink identities."""

    def __init__(self, original, trace: NaturalTrace, layer: int, counter: dict[int, int]):
        self.original, self.trace, self.layer, self.counter = original, trace, layer, counter

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        **kwargs,
    ):
        if self.trace.n_image is None:
            raise RuntimeError("image-token count unavailable before attention trace")
        step = self.counter[self.layer]
        self.counter[self.layer] += 1
        query, key, _value = _flux_qkv(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        sinks = _incoming_image_attention_sinks(query, key, self.trace.n_image)
        self.trace.observe_sinks(step, self.layer, sinks)
        return self.original(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )


def _pixart_qkv(attn, hidden_states, encoder_hidden_states=None, temb=None):
    """Construct the normalized Q/K/V used by PixArt's AttnProcessor2_0 self-attention."""
    if encoder_hidden_states is not None:
        raise RuntimeError("PixArt Q7 sink tracing must target attn1 self-attention, not attn2")
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
    input_ndim = hidden_states.ndim
    spatial_shape = None
    if input_ndim == 4:
        batch, channel, height, width = hidden_states.shape
        spatial_shape = (channel, height, width)
        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    batch_size, sequence_length, inner_dim = query.shape
    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, sequence_length, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, sequence_length, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, sequence_length, attn.heads, head_dim).transpose(1, 2)
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
    return query, key, value, residual, input_ndim, spatial_shape


def _pixart_incoming_attention_sinks(query, key, chunk_size: int = 128):
    """Per-head incoming-attention argmax for PixArt image self-attention."""
    # Reuse the FLUX helper after converting [B,H,Q,D] to [B,Q,H,D].
    return _incoming_image_attention_sinks(
        query.transpose(1, 2), key.transpose(1, 2), query.shape[2], chunk_size
    )


class PixArtAttentionSinkTraceProcessor:
    """Record PixArt attn1 sinks while delegating the unmodified clean forward."""

    def __init__(self, original, trace: NaturalTrace, layer: int, counter: dict[int, int]):
        self.original, self.trace, self.layer, self.counter = original, trace, layer, counter

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        if attention_mask is not None:
            raise RuntimeError("PixArt Q7 sink tracing does not support masked image tokens")
        step = self.counter[self.layer]
        self.counter[self.layer] += 1
        query, key, _value, *_metadata = _pixart_qkv(
            attn, hidden_states, encoder_hidden_states, temb
        )
        self.trace.observe_sinks(step, self.layer, _pixart_incoming_attention_sinks(query, key))
        return self.original(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            *args,
            **kwargs,
        )


def _suppressed_image_attention(query, key, value, n_image: int, sinks, chunk_size: int = 128):
    """Recompute image-query rows over all keys, masking one natural sink per head."""
    import torch

    image_query = query[:, -n_image:].float()
    key_float, value_float = key.float(), value.float()
    scale = image_query.shape[-1] ** -0.5
    sinks = torch.as_tensor(sinks, device=query.device, dtype=torch.long)
    key_offset = key.shape[1] - n_image
    heads = torch.arange(query.shape[2], device=query.device)
    chunks = []
    for lo in range(0, n_image, chunk_size):
        q = image_query[:, lo : lo + chunk_size]
        scores = torch.einsum("bqhd,bkhd->bhqk", q, key_float) * scale
        scores[:, heads, :, key_offset + sinks] = torch.finfo(scores.dtype).min
        probs = scores.softmax(dim=-1)
        chunk = torch.einsum("bhqk,bkhd->bqhd", probs, value_float)
        chunks.append(chunk)
    return torch.cat(chunks, dim=1).to(query.dtype)


def _dispatch_suppressed_image_attention(
    query,
    key,
    value,
    n_image: int,
    sinks,
    *,
    backend=None,
    parallel_config=None,
    chunk_size: int = 128,
):
    """Use Diffusers' native backend/dtype with an additive per-head sink mask."""
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    image_query = query[:, -n_image:]
    sinks = torch.as_tensor(sinks, device=query.device, dtype=torch.long)
    key_offset = key.shape[1] - n_image
    heads = torch.arange(query.shape[2], device=query.device)
    chunks = []
    for lo in range(0, n_image, chunk_size):
        q = image_query[:, lo : lo + chunk_size]
        mask = torch.zeros(
            (q.shape[0], q.shape[2], q.shape[1], key.shape[1]),
            device=q.device,
            dtype=q.dtype,
        )
        mask[:, heads, :, key_offset + sinks] = torch.finfo(q.dtype).min
        chunks.append(
            dispatch_attention_fn(
                q,
                key,
                value,
                attn_mask=mask,
                backend=backend,
                parallel_config=parallel_config,
            )
        )
    return torch.cat(chunks, dim=1).to(query.dtype)


class SinkSuppressProcessor:
    """Delegating processor that changes image-query routing and preserves text-query output."""

    def __init__(self, original, trace, layer, phase, zone, counts, fires):
        self.original, self.trace, self.layer = original, trace, layer
        self.phase, self.zone, self.counts, self.fires = phase, zone, counts, fires

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        **kwargs,
    ):
        output = self.original(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )
        step = self.counts[self.layer]
        self.counts[self.layer] += 1
        if not in_target(step, self.layer, self.phase, self.zone):
            return output
        if attention_mask is not None:
            raise RuntimeError("suppress_sink does not support a pre-existing attention mask")
        sinks = self.trace.sink_indices.get((step, self.layer))
        if sinks is None:
            raise RuntimeError(f"clean attention trace missing step {step}, layer {self.layer}")
        if self.trace.n_image is None:
            raise RuntimeError("clean trace has no image-token count")
        query, key, value = _flux_qkv(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        image = _dispatch_suppressed_image_attention(
            query,
            key,
            value,
            self.trace.n_image,
            sinks,
            backend=getattr(self.original, "_attention_backend", None),
            parallel_config=getattr(self.original, "_parallel_config", None),
        )
        image = image.flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            # Double-stream processor output is (projected image, projected text).
            image = attn.to_out[0](image.contiguous())
            image = attn.to_out[1](image)
            result = (image, output[1])
        else:
            # Single-stream processor output is unprojected [text,image]; replace only image rows.
            result = output.clone()
            result[:, -self.trace.n_image :, :] = image
        self.fires[self.layer] += 1
        return result


class PixArtSinkSuppressProcessor:
    """Suppress each clean-traced PixArt attn1 sink on the conditional CFG row only."""

    def __init__(self, original, trace, layer, phase, zone, counts, fires, chunk_size=128):
        self.original, self.trace, self.layer = original, trace, layer
        self.phase, self.zone, self.counts, self.fires = phase, zone, counts, fires
        self.chunk_size = chunk_size

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        import torch
        from torch.nn import functional

        output = self.original(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            *args,
            **kwargs,
        )
        step = self.counts[self.layer]
        self.counts[self.layer] += 1
        if not in_target(step, self.layer, self.phase, self.zone):
            return output
        if attention_mask is not None:
            raise RuntimeError("suppress_sink does not support masked PixArt image tokens")
        sinks = self.trace.sink_indices.get((step, self.layer))
        if sinks is None:
            raise RuntimeError(f"clean attention trace missing step {step}, layer {self.layer}")
        query, key, value, residual, input_ndim, spatial_shape = _pixart_qkv(
            attn, hidden_states, encoder_hidden_states, temb
        )
        # PixArt real CFG is [unconditional, conditional]. Preserve the unconditional row.
        query, key, value = query[-1:], key[-1:], value[-1:]
        sinks = torch.as_tensor(sinks, device=query.device, dtype=torch.long)
        heads = torch.arange(query.shape[1], device=query.device)
        chunks = []
        for lo in range(0, query.shape[2], self.chunk_size):
            q = query[:, :, lo : lo + self.chunk_size]
            mask = torch.zeros(
                (1, q.shape[1], q.shape[2], key.shape[2]), device=q.device, dtype=q.dtype
            )
            mask[:, heads, :, sinks] = torch.finfo(q.dtype).min
            chunks.append(
                functional.scaled_dot_product_attention(
                    q, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
                )
            )
        edited = torch.cat(chunks, dim=2).transpose(1, 2).reshape(1, query.shape[2], -1)
        edited = attn.to_out[0](edited)
        edited = attn.to_out[1](edited)
        if input_ndim == 4:
            channel, height, width = spatial_shape
            edited = edited.transpose(-1, -2).reshape(1, channel, height, width)
        if attn.residual_connection:
            edited = edited + residual[-1:]
        edited = edited / attn.rescale_output_factor
        if not torch.is_tensor(output) or output.shape[0] != hidden_states.shape[0]:
            raise RuntimeError("unexpected PixArt attn1 processor output contract")
        result = output.clone()
        result[-1:] = edited.to(result.dtype)
        self.fires[self.layer] += 1
        return result


def _block_self_attention(ref, model_family: str):
    attr = "attn" if model_family == "flux1" else "attn1"
    return getattr(ref.module, attr, None)


class ResidualInterventionHooks:
    """Post-block hooks using masks fixed by the paired clean run."""

    def __init__(self, blocks, trace, condition, phase, zone, vstar, channel, n_steps):
        self.blocks, self.trace, self.condition = blocks, trace, condition
        self.phase, self.zone, self.vstar, self.channel = phase, zone, vstar, channel
        self.n_steps, self.counts, self.fires, self.handles = n_steps, {}, {}, []

    def attach(self, transformer):
        def pre(_m, _a, kw):
            hs = kw.get("hidden_states")
            self.n_image = _infer_image_token_count(transformer, hs)

        self.handles.append(transformer.register_forward_pre_hook(pre, with_kwargs=True))
        for ref in self.blocks:
            self.counts[ref.layer_id] = 0
            self.fires[ref.layer_id] = 0

            def hook(_m, _a, out, layer=ref.layer_id):
                step = self.counts[layer]
                self.counts[layer] += 1
                if not in_target(step, layer, self.phase, self.zone):
                    return None
                mask = self.trace.masks.get((step, layer))
                if mask is None:
                    raise RuntimeError(f"clean trace missing step {step}, layer {layer}")
                new, found = _modify_image_output(
                    out,
                    self.n_image,
                    lambda x: _torch_edit(x, self.condition, mask, self.vstar, self.channel),
                )
                if not found:
                    raise RuntimeError(f"image output not found at layer {layer}")
                self.fires[layer] += 1
                return new

            self.handles.append(ref.module.register_forward_hook(hook))
        return self

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def audit(self):
        return audit_counts(self.counts, self.fires, self.n_steps, self.phase, self.zone)


class SinkAttentionHooks:
    """Replace self-attention processors to suppress clean-traced natural sink routing."""

    def __init__(self, blocks, trace, phase, zone, n_steps, model_family="flux1"):
        self.blocks, self.trace, self.phase, self.zone = blocks, trace, phase, zone
        self.model_family = model_family
        self.n_steps, self.counts, self.fires, self.originals = n_steps, {}, {}, []

    def attach(self):
        for ref in self.blocks:
            attn = _block_self_attention(ref, self.model_family)
            if attn is None or not hasattr(attn, "processor") or not hasattr(attn, "set_processor"):
                raise RuntimeError(f"block {ref.layer_id} has no replaceable attention processor")
            self.counts[ref.layer_id] = 0
            self.fires[ref.layer_id] = 0
            self.originals.append((attn, attn.processor))
            processor_class = (
                SinkSuppressProcessor
                if self.model_family == "flux1"
                else PixArtSinkSuppressProcessor
            )
            attn.set_processor(
                processor_class(
                    attn.processor,
                    self.trace,
                    ref.layer_id,
                    self.phase,
                    self.zone,
                    self.counts,
                    self.fires,
                )
            )
        return self

    def detach(self):
        for attn, processor in self.originals:
            attn.set_processor(processor)
        self.originals.clear()

    def audit(self):
        return audit_counts(self.counts, self.fires, self.n_steps, self.phase, self.zone)


def _generate(pipe, cfg: Q7Config, prompt: str, seed: int):
    import torch

    gen_device = "cpu" if cfg.device == "cuda" else cfg.device
    generator = torch.Generator(gen_device).manual_seed(seed)
    kwargs = {
        "prompt": prompt,
        "height": cfg.resolution,
        "width": cfg.resolution,
        "num_inference_steps": cfg.num_steps,
        "generator": generator,
        "output_type": "pil",
    }
    if cfg.guidance_scale is not None:
        kwargs["guidance_scale"] = cfg.guidance_scale
    with torch.inference_mode():
        return pipe(**kwargs).images[0]


def _trace_clean(pipe, blocks, cfg, prompt, seed, capture_sinks: bool = True):
    register_layers = {
        layer for lo, hi in cfg.zones.values() for layer in range(int(lo), int(hi) + 1)
    }
    trace = NaturalTrace(cfg.register_threshold, cfg.max_registers, register_layers)
    counters = {b.layer_id: 0 for b in blocks}
    state = {"n_image": None}
    handles = []
    original_processors = []
    sink_layers = register_layers
    attention_counters = {b.layer_id: 0 for b in blocks if b.layer_id in sink_layers}

    def pre(_m, _a, kw):
        state["n_image"] = _infer_image_token_count(pipe.transformer, kw["hidden_states"])
        trace.n_image = state["n_image"]

    handles.append(pipe.transformer.register_forward_pre_hook(pre, with_kwargs=True))
    for ref in blocks:
        if capture_sinks and ref.layer_id in sink_layers:
            attn = _block_self_attention(ref, cfg.model_family)
            if attn is None or not hasattr(attn, "processor") or not hasattr(attn, "set_processor"):
                raise RuntimeError(
                    f"block {ref.layer_id} does not expose a replaceable attention processor"
                )
            original_processors.append((attn, attn.processor))
            processor_class = (
                AttentionSinkTraceProcessor
                if cfg.model_family == "flux1"
                else PixArtAttentionSinkTraceProcessor
            )
            attn.set_processor(
                processor_class(attn.processor, trace, ref.layer_id, attention_counters)
            )

        def hook(_m, _a, out, layer=ref.layer_id):
            step = counters[layer]
            counters[layer] += 1

            def observe(x):
                trace.observe(step, layer, x)
                return x

            _unchanged, found = _modify_image_output(out, state["n_image"], observe)
            if not found:
                raise RuntimeError(f"image output not found at layer {layer}")

        handles.append(ref.module.register_forward_hook(hook))
    try:
        image = _generate(pipe, cfg, prompt, seed)
    finally:
        for h in handles:
            h.remove()
        for attn, processor in original_processors:
            attn.set_processor(processor)
    if capture_sinks:
        bad = {
            layer: count for layer, count in attention_counters.items() if count != cfg.num_steps
        }
        if bad:
            raise RuntimeError(f"clean attention trace call-count mismatch: {bad}")
    return image, trace


def _save_image(image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def calibrate_vstar(cfg: Q7Config, smoke: bool = False) -> Path:
    """Pool clean register-zone vectors across prompts, seeds, steps, and layers."""
    from src.common.model_utils import discover_blocks, load_pipeline

    pipe_cfg = type(
        "Cfg",
        (),
        {"model_ckpt": cfg.model_ckpt, "dtype": cfg.dtype, "device": cfg.device},
    )()
    pipe = load_pipeline(pipe_cfg, offload=cfg.offload)
    blocks = discover_blocks(pipe.transformer)
    validate_model_layout(blocks, cfg)
    jobs = [(p, s) for p in cfg.prompts for s in cfg.seeds]
    if smoke:
        jobs = jobs[:1]
    vectors = []
    for prompt, seed in jobs:
        _image, trace = _trace_clean(pipe, blocks, cfg, prompt, seed, capture_sinks=False)
        vectors.extend(trace.vectors)
    if not vectors:
        raise RuntimeError("no natural register vectors found during vstar calibration")
    path = Path(cfg.vstar_path) if cfg.vstar_path else Path(cfg.output_dir) / "vstar.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, fit_vstar(np.asarray(vectors)))
    metadata = {
        "path": str(path),
        "model_preset": cfg.model_preset,
        "model_family": cfg.model_family,
        "model_ckpt": cfg.model_ckpt,
        "n_vectors": len(vectors),
        "n_scenarios": len(jobs),
        "calibration_layers": sorted(
            {layer for lo, hi in cfg.zones.values() for layer in range(lo, hi + 1)}
        ),
        "config_hash": config_hash(cfg),
        "experiment_revision": EXPERIMENT_REVISION,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def run(cfg: Q7Config, smoke: bool = False) -> None:
    """Run the paired grid. `suppress_sink` requires the optional attention-mask adapter below."""
    from src.common.model_utils import discover_blocks, load_pipeline

    pipe_cfg = type(
        "Cfg", (), {"model_ckpt": cfg.model_ckpt, "dtype": cfg.dtype, "device": cfg.device}
    )()
    pipe = load_pipeline(pipe_cfg, offload=cfg.offload)
    blocks = discover_blocks(pipe.transformer)
    validate_model_layout(blocks, cfg)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    calibrated = Path(cfg.vstar_path) if cfg.vstar_path else out / "vstar.npy"
    if not calibrated.exists():
        raise FileNotFoundError(
            f"calibrated vstar not found at {calibrated}; run --calibrate-vstar first"
        )
    scheduler_config = json.loads(json.dumps(dict(pipe.scheduler.config), default=str))
    calibration_sha = file_sha256(calibrated)
    identity = run_identity(cfg, calibration_sha, scheduler_config)
    generation_params = {
        "experiment_revision": EXPERIMENT_REVISION,
        "model_preset": cfg.model_preset,
        "model_family": cfg.model_family,
        "model_ckpt": cfg.model_ckpt,
        "resolution": cfg.resolution,
        "num_steps": cfg.num_steps,
        "guidance_scale": cfg.guidance_scale,
        "dtype": cfg.dtype,
        "device": cfg.device,
        "offload": cfg.offload,
        "scheduler_class": type(pipe.scheduler).__name__,
        "scheduler_config": scheduler_config,
        "calibration_path": str(calibrated),
        "calibration_sha256": calibration_sha,
    }
    manifest = out / "runs.jsonl"
    completed = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            old = json.loads(line)
            if old.get("run_identity") != identity or not Path(old.get("image_path", "")).exists():
                continue
            completed.add(
                (
                    old["run_identity"],
                    old["prompt_id"],
                    old["prompt"],
                    old["seed"],
                    old["condition"],
                    old["phase"],
                    old["zone"],
                )
            )
    jobs = [(p, s) for p in cfg.prompts for s in cfg.seeds]
    if smoke:
        jobs = jobs[:1]
    for prompt_id, (prompt, seed) in enumerate(jobs):
        clean, trace = _trace_clean(pipe, blocks, cfg, prompt, seed)
        clean_path = out / "images" / f"{identity}_p{prompt_id:03d}_s{seed}_baseline.png"
        _save_image(clean, clean_path)
        vstar = np.load(calibrated)
        phase_items = list(cfg.phases.items())[:1] if smoke else list(cfg.phases.items())
        zone_items = list(cfg.zones.items())[:1] if smoke else list(cfg.zones.items())
        for phase_name, phase in phase_items:
            for zone_name, zone in zone_items:
                conditions = [c for c in cfg.conditions if c != "baseline"]
                for condition in conditions:
                    job_key = (
                        identity,
                        prompt_id,
                        prompt,
                        seed,
                        condition,
                        phase_name,
                        zone_name,
                    )
                    if job_key in completed:
                        continue
                    if condition == "suppress_sink":
                        hooks = SinkAttentionHooks(
                            blocks, trace, phase, zone, cfg.num_steps, cfg.model_family
                        ).attach()
                    else:
                        hooks = ResidualInterventionHooks(
                            blocks,
                            trace,
                            condition,
                            phase,
                            zone,
                            vstar,
                            cfg.channel,
                            cfg.num_steps,
                        ).attach(pipe.transformer)
                    try:
                        image = _generate(pipe, cfg, prompt, seed)
                    finally:
                        hooks.detach()
                    audit = hooks.audit()
                    if not audit["ok"]:
                        raise RuntimeError(
                            f"intervention audit failed for {condition}/{phase_name}/{zone_name}: "
                            + "; ".join(audit["errors"])
                        )
                    stem = (
                        f"{identity}_p{prompt_id:03d}_s{seed}_{condition}_{phase_name}_{zone_name}"
                    )
                    image_path = out / "images" / f"{stem}.png"
                    _save_image(image, image_path)
                    row = {
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "seed": seed,
                        "condition": condition,
                        "phase": phase_name,
                        "zone": zone_name,
                        "clean_path": str(clean_path),
                        "image_path": str(image_path),
                        "config_hash": config_hash(cfg),
                        "run_identity": identity,
                        "generation_params": generation_params,
                        "intervention_audit": audit,
                    }
                    with manifest.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, sort_keys=True) + "\n")


def evaluate(cfg: Q7Config, structured_scores_path: str | None = None) -> None:
    """Compute paired perceptual, prompt-fidelity, and frequency-band metrics.

    LPIPS, OpenCLIP, and ImageReward are optional: absent packages produce explicit blank
    columns rather than silently substituting a different metric.
    """
    from PIL import Image

    root = Path(cfg.output_dir)
    calibrated = Path(cfg.vstar_path) if cfg.vstar_path else root / "vstar.npy"
    if not calibrated.exists():
        raise FileNotFoundError(f"calibrated vstar not found at {calibrated}")
    calibration_sha = file_sha256(calibrated)
    manifest_rows = [
        json.loads(line) for line in (root / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    identities = {
        row["run_identity"]
        for row in manifest_rows
        if row.get("config_hash") == config_hash(cfg)
        and row.get("generation_params", {}).get("experiment_revision") == EXPERIMENT_REVISION
        and row.get("generation_params", {}).get("calibration_sha256") == calibration_sha
    }
    if len(identities) != 1:
        raise RuntimeError(
            f"expected exactly one compatible run identity, found {sorted(identities)}"
        )
    identity = identities.pop()
    structured_scores = (
        load_structured_scores(structured_scores_path, identity)
        if structured_scores_path is not None
        else None
    )
    rows = []
    lpips_model = clip_model = clip_preprocess = clip_tokenizer = reward_model = None
    try:
        import lpips

        lpips_model = lpips.LPIPS(net="alex")
    except ImportError:
        pass
    try:
        import open_clip

        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        clip_model.eval()
    except ImportError:
        pass
    try:
        import ImageReward as RM

        reward_model = RM.load("ImageReward-v1.0")
    except ImportError:
        pass
    for row in manifest_rows:
        if row.get("run_identity") != identity:
            continue
        clean_pil = Image.open(row["clean_path"]).convert("RGB")
        edited_pil = Image.open(row["image_path"]).convert("RGB")
        clean, edited = np.asarray(clean_pil), np.asarray(edited_pil)
        row.update(frequency_distances(clean, edited))
        row.update(
            lpips=None,
            clip_clean=None,
            clip_edited=None,
            clip_delta=None,
            image_reward_clean=None,
            image_reward_edited=None,
            image_reward_delta=None,
        )
        for metric in GENEVAL_METRICS:
            row.update(
                {
                    f"{metric}_clean": None,
                    f"{metric}_edited": None,
                    f"{metric}_delta": None,
                }
            )
        if structured_scores is not None:
            key = _structured_score_key(row)
            if key not in structured_scores:
                raise ValueError(f"structured score CSV is missing run cell {key}")
            row.update(structured_scores[key])
        if lpips_model is not None:
            import torch

            def lp_tensor(a):
                return torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float() / 127.5 - 1

            with torch.inference_mode():
                row["lpips"] = float(lpips_model(lp_tensor(clean), lp_tensor(edited)).item())
        if clip_model is not None:
            import torch

            with torch.inference_mode():
                ims = torch.stack([clip_preprocess(clean_pil), clip_preprocess(edited_pil)])
                imf = clip_model.encode_image(ims)
                imf = imf / imf.norm(dim=-1, keepdim=True)
                txt = clip_model.encode_text(clip_tokenizer([row["prompt"]]))
                txt = txt / txt.norm(dim=-1, keepdim=True)
                row["clip_clean"], row["clip_edited"] = [float(v) for v in (imf @ txt.T)[:, 0]]
        if reward_model is not None:
            row["image_reward_clean"] = float(reward_model.score(row["prompt"], row["clean_path"]))
            row["image_reward_edited"] = float(reward_model.score(row["prompt"], row["image_path"]))
        add_paired_prompt_deltas(row)
        rows.append(row)
    fields = sorted({k for r in rows for k in r if k != "hook_counts"})
    with (root / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})

    metric_names = [
        "lpips",
        "low_frequency_rms",
        "high_frequency_rms",
        "low_high_ratio",
        "clip_delta",
        "image_reward_delta",
        *(f"{metric}_delta" for metric in GENEVAL_METRICS),
    ]
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        for metric in metric_names:
            value = row.get(metric)
            if value is not None and np.isfinite(float(value)):
                key = (row["condition"], row["phase"], row["zone"], metric)
                grouped.setdefault(key, []).append(float(value))
    summary_rows = []
    for (condition, phase, zone, metric), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "condition": condition,
                "phase": phase,
                "zone": zone,
                "metric": metric,
                **paired_bootstrap(values),
            }
        )
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        summary_fields = ["condition", "phase", "zone", "metric", "mean", "ci_low", "ci_high", "n"]
        w = csv.DictWriter(fh, summary_fields)
        w.writeheader()
        w.writerows(summary_rows)
    for figure in generate_figures(cfg):
        print(figure)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument(
        "--structured-scores",
        help="optional GenEval-style clean/edited CSV to merge during --evaluate",
    )
    p.add_argument("--plot", action="store_true", help="regenerate figures from paired_metrics.csv")
    p.add_argument("--calibrate-vstar", action="store_true")
    args = p.parse_args(argv)
    cfg = Q7Config.from_json(args.config)
    if args.calibrate_vstar:
        print(calibrate_vstar(cfg, smoke=args.smoke))
    elif args.evaluate:
        evaluate(cfg, structured_scores_path=args.structured_scores)
    elif args.plot:
        for figure in generate_figures(cfg):
            print(figure)
    else:
        run(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
