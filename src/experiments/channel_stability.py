"""Per-generation channel-stability experiment (FLUX.1-dev).

Question: *for each individual image generation (prompt + seed), which channels are the
largest, and do their identities change with what the model generates / from one
generation to the next?*

Method (deliberately narrower than the fig3 pipeline): fix ONE transformer block and ONE
timestep, run a small cross-product of ``prompts × seeds`` **scenarios**, rank channels
**independently per scenario** by the massive-activation score ``abs(mean_over_tokens)``,
and analyze how stable the top-channel *identities* are across scenarios. No BiRefNet / mIoU.

Reuses (unchanged): ``model_utils`` (load/capture/generate), ``stage2.rank_channels`` /
``channel_scores`` (ranking), ``clustering.minmax_normalize_channels`` (via
``spatial.aggregated_topk_heatmap``). Torch / matplotlib / PIL are imported lazily so this
module imports (and its pure functions test) on CPU with no model stack.

    python -m src.experiments.channel_stability --config configs/channel_stability.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

import numpy as np
import yaml

from src.common import io, spatial
from src.stage2_channel_ranking import rank_channels

# top-k prefixes we report identity-overlap for (all derivable from the stored top-20).
REPORT_KS = (1, 5, 10, 20)


# --- config -------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    # Attributes read by model_utils.load_pipeline / generate_with_capture.
    model_ckpt: str | None = None
    output_dir: str | None = None
    device: str = "cuda"
    dtype: str = "bf16"
    seed: int = 0  # base seed; per-scenario seed comes from `seeds` below
    num_denoising_steps: int = 50
    resolution: int = 1024
    guidance_scale: float | None = 3.5

    # Experiment knobs.
    fixed_layer: int = 11
    top_n: int = 20
    agg_k: int = 10
    n_channel_maps: int = 5
    n_control_maps: int = 0
    prompts: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [0])
    representative_scenarios: list[int] | None = None  # None -> seed-jitter + content mix
    offload: bool = False
    # Multi-timestep capture: denoising-step indices to snapshot in addition to the
    # last step (primary analysis stays last-step). None -> last step only.
    capture_steps: list[int] | None = None
    # Complementary token-localized ranking metric: "p999" (abs 99.9th pct over tokens),
    # "max" (abs max), or None to disable. Primary metric stays abs(mean_over_tokens).
    secondary_metric: str | None = "p999"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_REQUIRED = ("model_ckpt", "output_dir")


def load_scenario_config(config_path: str, *, create_dirs: bool = True) -> ScenarioConfig:
    """Load + validate the experiment YAML. Fails loud on missing required keys / unknowns.

    Kept separate from ``src/common/config.py`` so the fig3 Config (which requires
    ``birefnet_weights`` / ``prompt_source``, unused here) and its tests are untouched.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")

    valid = {f.name for f in fields(ScenarioConfig)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {sorted(unknown)}")

    cfg = ScenarioConfig(**raw)
    _validate(cfg, config_path)
    if create_dirs and cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)
    return cfg


def _validate(cfg: ScenarioConfig, source: str) -> None:
    missing = [k for k in _REQUIRED if not getattr(cfg, k)]
    if missing:
        raise ValueError(f"Missing required config value(s) {missing} (source: {source}).")
    if not cfg.prompts:
        raise ValueError(f"`prompts` must be a non-empty list (source: {source}).")
    if not cfg.seeds:
        raise ValueError(f"`seeds` must be a non-empty list (source: {source}).")
    if cfg.top_n <= 0:
        raise ValueError(f"top_n must be positive, got {cfg.top_n}")
    if cfg.agg_k <= 0 or cfg.agg_k > cfg.top_n:
        raise ValueError(f"agg_k must be in 1..top_n ({cfg.top_n}), got {cfg.agg_k}")
    if cfg.dtype not in ("bf16", "fp16", "fp32"):
        raise ValueError(f"dtype must be one of bf16/fp16/fp32, got {cfg.dtype!r}")
    if cfg.capture_steps is not None:
        if len(cfg.capture_steps) > 4:
            raise ValueError(
                f"capture_steps holds full activation snapshots in memory; at most 4 "
                f"allowed, got {len(cfg.capture_steps)}"
            )
        bad = [s for s in cfg.capture_steps if not 0 <= int(s) < cfg.num_denoising_steps]
        if bad:
            raise ValueError(
                f"capture_steps must be in 0..{cfg.num_denoising_steps - 1}, got {bad}"
            )
    if cfg.secondary_metric not in (None, "max", "p999"):
        raise ValueError(
            f"secondary_metric must be one of null/max/p999, got {cfg.secondary_metric!r}"
        )


# --- scenarios (pure) ---------------------------------------------------------


@dataclass
class Scenario:
    scenario_id: int
    prompt_id: int
    prompt: str
    seed_id: int
    seed: int


def build_scenarios(prompts: list[str], seeds: list[int]) -> list[Scenario]:
    """Cross-product of prompts × seeds, row-major (prompt outer, seed inner).

    scenario_id = prompt_id * len(seeds) + seed_id, so ``scenario_id // len(seeds)``
    recovers the prompt group (used to separate same-prompt/different-seed pairs from
    different-prompt pairs in the stability analysis).
    """
    out: list[Scenario] = []
    for pid, prompt in enumerate(prompts):
        for sid, seed in enumerate(seeds):
            out.append(
                Scenario(
                    scenario_id=pid * len(seeds) + sid,
                    prompt_id=pid,
                    prompt=prompt,
                    seed_id=sid,
                    seed=int(seed),
                )
            )
    return out


# --- table + stability assembly (pure) ----------------------------------------


def build_scenario_channel_matrix(
    scenarios: list[Scenario],
    top_idx_by_scenario: dict[int, np.ndarray],
    scores_by_scenario: dict[int, np.ndarray],
) -> tuple[list[int], dict[int, dict[int, int]], dict[int, dict[int, float]]]:
    """Scenario × channel table.

    Returns ``(channels, rank_of, score_of)`` where ``channels`` is the sorted union of
    every channel selected in any scenario's top-n, ``rank_of[sid][ch]`` is the 1-based
    rank of channel ``ch`` in scenario ``sid`` (absent if not selected), and
    ``score_of[sid][ch]`` its score.
    """
    channel_union: set[int] = set()
    rank_of: dict[int, dict[int, int]] = {}
    score_of: dict[int, dict[int, float]] = {}
    for sc in scenarios:
        top_idx = np.asarray(top_idx_by_scenario[sc.scenario_id]).ravel()
        scores = np.asarray(scores_by_scenario[sc.scenario_id]).ravel()
        rank_of[sc.scenario_id] = {}
        score_of[sc.scenario_id] = {}
        for rank0, ch in enumerate(top_idx):
            ch = int(ch)
            channel_union.add(ch)
            rank_of[sc.scenario_id][ch] = rank0 + 1
            score_of[sc.scenario_id][ch] = float(scores[ch])
    return sorted(channel_union), rank_of, score_of


def compute_stability_summary(
    scenarios: list[Scenario],
    top_idx_by_scenario: dict[int, np.ndarray],
    d: int,
    n_seeds: int,
) -> dict[str, Any]:
    """Numeric answer to "do the top-channel identities change across scenarios?".

    For each k in REPORT_KS: distinct channels used across all scenarios, per-channel
    selection frequency, and mean pairwise Jaccard split into same-prompt/different-seed
    pairs vs different-prompt pairs.
    """
    sids = [sc.scenario_id for sc in scenarios]
    prompt_of = {sc.scenario_id: sc.prompt_id for sc in scenarios}
    summary: dict[str, Any] = {"n_scenarios": len(scenarios), "n_seeds": n_seeds, "per_k": {}}

    for k in REPORT_KS:
        sets = {sid: np.asarray(top_idx_by_scenario[sid]).ravel()[:k] for sid in sids}
        freq = spatial.selection_frequency([sets[sid] for sid in sids], d)
        used = sorted(int(c) for c in np.nonzero(freq)[0])

        same_prompt: list[float] = []
        diff_prompt: list[float] = []
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                a, b = sids[i], sids[j]
                jac = spatial.topk_jaccard(sets[a], sets[b])
                (same_prompt if prompt_of[a] == prompt_of[b] else diff_prompt).append(jac)

        summary["per_k"][str(k)] = {
            "n_distinct_channels_used": len(used),
            "channels_used": used,
            "selection_frequency": {str(c): int(freq[c]) for c in used},
            "mean_jaccard_same_prompt_diff_seed": (
                float(np.mean(same_prompt)) if same_prompt else None
            ),
            "mean_jaccard_diff_prompt": float(np.mean(diff_prompt)) if diff_prompt else None,
            # Raw pair values (12 same-prompt + 54 diff-prompt pairs for 4 prompts x 3
            # seeds); the overlap figure plots these directly.
            "pairs_same_prompt": [float(v) for v in same_prompt],
            "pairs_diff_prompt": [float(v) for v in diff_prompt],
        }
    return summary


def rank_channels_secondary(
    stream: np.ndarray, metric: str, top_n: int
) -> tuple[np.ndarray, np.ndarray]:
    """(top_idx, scores) under the token-localized secondary metric ("max" or "p999").

    Same deterministic tie-breaking as ``rank_channels`` (stable sort, ascending id).
    """
    from src.stage2_channel_ranking import channel_scores_max

    q = 1.0 if metric == "max" else 0.999
    score = channel_scores_max(stream, q=q)
    top_idx = np.argsort(-score, kind="stable")[:top_n].astype(np.int32)
    return top_idx, score.astype(np.float32)


def compute_step_consistency(
    scenarios: list[Scenario],
    top_idx_by_scenario: dict[int, np.ndarray],
    step_top_by_scenario: dict[int, dict[int, np.ndarray]],
) -> dict[str, Any]:
    """Mean Jaccard(top-k at captured step vs top-k at the last step), per step and k.

    Answers: is the last-step probe representative, or do the massive-channel
    identities drift over denoising?
    """
    out: dict[str, Any] = {}
    steps = sorted({s for per in step_top_by_scenario.values() for s in per})
    for step in steps:
        per_k: dict[str, float | None] = {}
        for k in REPORT_KS:
            vals = [
                spatial.topk_jaccard(
                    step_top_by_scenario[sc.scenario_id][step][:k],
                    np.asarray(top_idx_by_scenario[sc.scenario_id]).ravel()[:k],
                )
                for sc in scenarios
                if step in step_top_by_scenario.get(sc.scenario_id, {})
            ]
            per_k[str(k)] = float(np.mean(vals)) if vals else None
        out[str(step)] = per_k
    return out


def compute_metric_agreement(
    scenarios: list[Scenario],
    top_idx_by_scenario: dict[int, np.ndarray],
    secondary_top_by_scenario: dict[int, np.ndarray],
) -> dict[str, float]:
    """Mean per-scenario Jaccard(primary top-k, secondary top-k) for each k."""
    out: dict[str, float] = {}
    for k in REPORT_KS:
        vals = [
            spatial.topk_jaccard(
                np.asarray(top_idx_by_scenario[sc.scenario_id]).ravel()[:k],
                np.asarray(secondary_top_by_scenario[sc.scenario_id]).ravel()[:k],
            )
            for sc in scenarios
        ]
        out[str(k)] = float(np.mean(vals))
    return out


# --- CSV writers --------------------------------------------------------------


def write_topk_csv(
    path: str,
    scenarios: list[Scenario],
    top_idx_by_scenario: dict[int, np.ndarray],
    scores_by_scenario: dict[int, np.ndarray],
) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario_id", "prompt_id", "seed", "prompt", "rank", "channel_id", "score"])
        for sc in scenarios:
            top_idx = np.asarray(top_idx_by_scenario[sc.scenario_id]).ravel()
            scores = np.asarray(scores_by_scenario[sc.scenario_id]).ravel()
            for rank0, ch in enumerate(top_idx):
                ch = int(ch)
                w.writerow(
                    [
                        sc.scenario_id,
                        sc.prompt_id,
                        sc.seed,
                        sc.prompt,
                        rank0 + 1,
                        ch,
                        f"{scores[ch]:.6g}",
                    ]
                )


def write_matrix_csv(
    path: str,
    scenarios: list[Scenario],
    channels: list[int],
    cell_by_scenario: dict[int, dict[int, Any]],
    value_fmt=str,
) -> None:
    """Wide scenario × channel table; blank where a channel is not in that scenario's top-n."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario_id", "prompt_id", "seed"] + [f"ch{c}" for c in channels])
        for sc in scenarios:
            cells = cell_by_scenario[sc.scenario_id]
            row = [sc.scenario_id, sc.prompt_id, sc.seed]
            row += [value_fmt(cells[c]) if c in cells else "" for c in channels]
            w.writerow(row)


# --- rendering (lazy matplotlib / PIL) ----------------------------------------


def _save_image(path: str, rgb: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def _save_heatmap(path: str, arr: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(arr, cmap="magma")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _save_scenario_channel_heatmap(
    path: str,
    scenarios: list[Scenario],
    channels: list[int],
    rank_of: dict[int, dict[int, int]],
    top_n: int,
    n_seeds: int,
) -> None:
    """Visual companion to ``scenario_channel_matrix.csv``: rows=scenarios, cols=channels,
    color = 1-based RANK within the scenario (rank 1 darkest), blank (grey) where a
    channel isn't in that scenario's top-n. Rank, not raw score, so color is comparable
    across scenarios (scores live in ``scenario_channel_scores.csv``). Columns are
    ordered by descending selection frequency (mean-rank tie-break); a vertical line
    separates channels selected in every scenario from the rest.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy.ma as ma

    def freq_of(c: int) -> int:
        return sum(1 for sc in scenarios if c in rank_of[sc.scenario_id])

    def mean_rank_of(c: int) -> float:
        ranks = [rank_of[sc.scenario_id][c] for sc in scenarios if c in rank_of[sc.scenario_id]]
        return float(np.mean(ranks)) if ranks else float("inf")

    cols = sorted(channels, key=lambda c: (-freq_of(c), mean_rank_of(c), c))
    n_always = sum(1 for c in cols if freq_of(c) == len(scenarios))

    mat = np.full((len(scenarios), len(cols)), np.nan)
    for r, sc in enumerate(scenarios):
        row = rank_of[sc.scenario_id]
        for c, ch in enumerate(cols):
            if ch in row:
                mat[r, c] = row[ch]
    masked = ma.masked_invalid(mat)

    fig_w = max(6.0, 0.3 * len(cols) + 2.0)
    fig_h = max(4.0, 0.4 * len(scenarios) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap("magma").copy()  # low (rank 1) = dark = most salient
    cmap.set_bad(color="0.92")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=1, vmax=top_n)

    for r in range(1, len(scenarios)):
        if scenarios[r].prompt_id != scenarios[r - 1].prompt_id:
            ax.axhline(r - 0.5, color="white", linewidth=1.5)
    if 0 < n_always < len(cols):
        ax.axvline(n_always - 0.5, color="white", linewidth=1.5, linestyle="--")

    if len(cols) <= 40:
        for r in range(len(scenarios)):
            for c in range(len(cols)):
                if not np.isnan(mat[r, c]):
                    rank = int(mat[r, c])
                    ax.text(
                        c,
                        r,
                        str(rank),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if rank <= top_n // 2 else "black",
                    )

    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([f"p{sc.prompt_id}/s{sc.seed}" for sc in scenarios], fontsize=7)
    if len(cols) <= 60:
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([str(c) for c in cols], fontsize=6, rotation=90)
    else:
        ax.set_xlabel("channel id (sorted by selection frequency, most-selected first)")
    ax.set_title(
        f"scenario x channel rank (1 = largest; grey = not in top-{top_n}; "
        "left of dashed line = selected in every scenario)",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="rank in scenario")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _save_stability_overlap(
    path: str,
    mat: np.ndarray,
    k: int,
    scenarios: list[Scenario],
    n_seeds: int,
    summary: dict[str, Any],
) -> None:
    """Two-panel headline figure.

    Left: block-structured pairwise top-k Jaccard matrix — thick dividers every
    ``n_seeds`` rows/cols so same-prompt (diagonal) blocks vs cross-prompt blocks are
    visible at a glance. Right: same-prompt/diff-seed vs diff-prompt mean Jaccard over
    k, with every individual pair shown as a jittered dot; the vertical gap between the
    two series is the experiment's finding (content dependence vs seed jitter).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5), width_ratios=[1.1, 1])

    im = ax0.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
    labels = [f"p{sc.prompt_id}/s{sc.seed}" for sc in scenarios]
    ax0.set_xticks(range(len(labels)))
    ax0.set_xticklabels(labels, fontsize=6, rotation=90)
    ax0.set_yticks(range(len(labels)))
    ax0.set_yticklabels(labels, fontsize=6)
    for pos in range(n_seeds, len(scenarios), n_seeds):
        ax0.axhline(pos - 0.5, color="white", linewidth=2.0)
        ax0.axvline(pos - 0.5, color="white", linewidth=2.0)
    ax0.set_title(
        f"pairwise top-{k} Jaccard\n(diagonal blocks = same prompt, different seed)",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax0, fraction=0.046)

    ks = [int(k_) for k_ in summary["per_k"]]
    rng = np.random.default_rng(0)
    for key, color, label in (
        ("pairs_same_prompt", "#1b7837", "same prompt, diff seed"),
        ("pairs_diff_prompt", "#762a83", "different prompt"),
    ):
        means = []
        for i, k_ in enumerate(ks):
            pairs = summary["per_k"][str(k_)].get(key, [])
            if pairs:
                x = i + rng.uniform(-0.12, 0.12, size=len(pairs))
                ax1.scatter(x, pairs, s=12, color=color, alpha=0.35, linewidths=0)
            means.append(float(np.mean(pairs)) if pairs else np.nan)
        ax1.plot(range(len(ks)), means, "o-", color=color, label=label, markersize=6)
    ax1.set_xticks(range(len(ks)))
    ax1.set_xticklabels([f"top-{k_}" for k_ in ks])
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("channel-identity Jaccard")
    ax1.set_title("seed jitter vs content dependence", fontsize=9)
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(axis="y", color="0.9", linewidth=0.7)
    ax1.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _save_step_consistency(path: str, per_step_jaccard: dict[str, Any], n_steps: int) -> None:
    """Mean Jaccard(top-k at captured step vs last step) — one line per captured step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    steps = sorted(per_step_jaccard, key=int)
    cmap = plt.get_cmap("viridis")
    for i, step in enumerate(steps):
        per_k = per_step_jaccard[step]
        ks = sorted(per_k, key=int)
        vals = [per_k[k_] for k_ in ks]
        color = cmap(0.15 + 0.7 * (int(step) / max(1, n_steps - 1)))
        ax.plot(
            range(len(ks)),
            vals,
            "o-",
            color=color,
            markersize=5,
            label=f"step {step} vs last",
        )
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"top-{k_}" for k_ in ks])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("mean Jaccard vs last step")
    ax.set_title("top-channel identity across denoising steps", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _save_contact_sheet(
    path: str,
    rows: list[dict[str, Any]],
    agg_k: int,
    h_lat: int,
    w_lat: int,
    n_top_panels: int = 3,
) -> None:
    """One qualitative summary figure: one row per representative scenario —
    [generated image | top-1..n channel maps | aggregated top-k | low-rank control].

    ``rows``: dicts with keys sc (Scenario), rgb (H,W,3 uint8), stream (N,D),
    top_idx, bottom_idx, scores.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 1 + n_top_panels + 1 + 1  # image + top maps + aggregate + control
    fig, axes = plt.subplots(
        len(rows), n_cols, figsize=(2.3 * n_cols, 2.5 * len(rows)), squeeze=False
    )
    for r, row in enumerate(rows):
        sc, stream, scores = row["sc"], row["stream"], row["scores"]
        panels = axes[r]

        panels[0].imshow(row["rgb"])
        panels[0].set_title(
            f'"{sc.prompt[:38]}…"' if len(sc.prompt) > 38 else f'"{sc.prompt}"', fontsize=6
        )
        panels[0].set_ylabel(f"p{sc.prompt_id}/s{sc.seed}", fontsize=9)
        panels[0].set_xticks([])
        panels[0].set_yticks([])

        for i in range(n_top_panels):
            ax = panels[1 + i]
            ch = int(row["top_idx"][i])
            smap = spatial.channel_spatial_map(stream, ch, h_lat, w_lat, normalize=True)
            ax.imshow(smap, cmap="magma")
            ax.set_title(f"rank {i + 1} · ch {ch}\nscore {scores[ch]:.3g}", fontsize=7)
            ax.axis("off")

        agg = spatial.aggregated_topk_heatmap(stream, row["top_idx"][:agg_k], h_lat, w_lat)
        panels[1 + n_top_panels].imshow(agg, cmap="magma")
        panels[1 + n_top_panels].set_title(f"aggregated top-{agg_k}", fontsize=7)
        panels[1 + n_top_panels].axis("off")

        ax = panels[-1]
        ch = int(row["bottom_idx"][0])
        smap = spatial.channel_spatial_map(stream, ch, h_lat, w_lat, normalize=True)
        ax.imshow(smap, cmap="magma")
        ax.set_title(f"control (low rank)\nch {ch} · score {scores[ch]:.3g}", fontsize=7)
        ax.axis("off")

    fig.suptitle(
        "generated image · top-channel spatial maps · aggregate · low-rank control",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def dump_scenario_qualitative(
    cfg: ScenarioConfig,
    sc: Scenario,
    stream: np.ndarray,
    rgb: np.ndarray,
    top_idx: np.ndarray,
    bottom_idx: np.ndarray,
    scores: np.ndarray,
    h_lat: int,
    w_lat: int,
) -> None:
    """Image + per-channel spatial maps + aggregated top-k heatmap (+ optional controls)."""
    sdir = os.path.join(cfg.output_dir, "scenarios", f"p{sc.prompt_id}_s{sc.seed}")
    os.makedirs(sdir, exist_ok=True)

    _save_image(os.path.join(sdir, "image.png"), rgb)

    for r in range(min(cfg.n_channel_maps, len(top_idx))):
        ch = int(top_idx[r])
        smap = spatial.channel_spatial_map(stream, ch, h_lat, w_lat, normalize=True)
        _save_heatmap(
            os.path.join(sdir, f"channel_rank{r + 1:02d}_ch{ch}.png"),
            smap,
            f"rank {r + 1} · ch {ch} · score {scores[ch]:.3g}",
        )

    agg = spatial.aggregated_topk_heatmap(stream, top_idx[: cfg.agg_k], h_lat, w_lat)
    _save_heatmap(
        os.path.join(sdir, f"aggregated_top{cfg.agg_k}.png"),
        agg,
        f"aggregated top-{cfg.agg_k} heatmap",
    )

    for r in range(min(cfg.n_control_maps, len(bottom_idx))):
        ch = int(bottom_idx[r])
        smap = spatial.channel_spatial_map(stream, ch, h_lat, w_lat, normalize=True)
        _save_heatmap(
            os.path.join(sdir, f"control_rank{r + 1:02d}_ch{ch}.png"),
            smap,
            f"control (low-rank) · ch {ch} · score {scores[ch]:.3g}",
        )


# --- metadata -----------------------------------------------------------------


def write_run_metadata(
    cfg: ScenarioConfig, scenarios: list[Scenario], model_info: dict | None
) -> str:
    meta = {
        "experiment": "channel_stability",
        "resolved_config": cfg.to_dict(),
        "versions": io.package_versions(),
        "n_scenarios": len(scenarios),
        "assumptions": {
            "score": "abs(mean_over_tokens) — mean then abs (fig3 massive-activation score)",
            "secondary_score": (
                f"{cfg.secondary_metric}: quantile-over-tokens of abs(activation) "
                "(token-localized complement)"
                if cfg.secondary_metric
                else None
            ),
            "probe": {
                "fixed_layer": cfg.fixed_layer,
                "timestep": (
                    "primary analysis = last denoising step; additionally snapshotting "
                    f"steps {cfg.capture_steps} for step-consistency check"
                    if cfg.capture_steps
                    else "last denoising step only (hook overwrite -> last forward wins)"
                ),
                "stream": "image-stream tokens only; N_I derived at runtime",
            },
        },
        "model_info": model_info or {},
    }
    path = os.path.join(cfg.output_dir, "run_metadata.json")
    io.save_json(path, meta)
    return path


# --- driver -------------------------------------------------------------------


def run(cfg: ScenarioConfig) -> dict[str, Any]:
    """Generate every scenario, rank channels, write tables + stability summary + dumps."""
    from src.common import model_utils

    # Nest under a layer-specific subfolder so different `fixed_layer` runs (e.g. sweeping
    # layers for the same model/prompts/seeds) don't overwrite each other's outputs.
    cfg = replace(cfg, output_dir=os.path.join(cfg.output_dir, f"layer_{cfg.fixed_layer}"))
    os.makedirs(cfg.output_dir, exist_ok=True)

    scenarios = build_scenarios(cfg.prompts, cfg.seeds)
    # Default representative scenarios cover BOTH axes of the design: prompt 0 at every
    # seed (seed-to-seed jitter) plus prompts 1 and 2 at their first seed (content
    # dependence).
    n_seeds = len(cfg.seeds)
    if cfg.representative_scenarios is not None:
        rep = set(cfg.representative_scenarios)
    else:
        rep = set(range(n_seeds))  # prompt 0, all seeds
        rep |= {p * n_seeds for p in (1, 2) if p < len(cfg.prompts)}
    write_run_metadata(cfg, scenarios, model_info=None)

    pipe = model_utils.load_pipeline(cfg, offload=cfg.offload)
    transformer = pipe.transformer
    all_blocks = model_utils.discover_blocks(transformer)
    blocks = model_utils.select_layers(all_blocks, [cfg.fixed_layer])
    state = model_utils.CaptureState(
        capture_steps=set(int(s) for s in cfg.capture_steps) if cfg.capture_steps else None
    )
    handles = model_utils.register_capture_hooks(transformer, blocks, state)

    top_idx_by_scenario: dict[int, np.ndarray] = {}
    scores_by_scenario: dict[int, np.ndarray] = {}
    secondary_top_by_scenario: dict[int, np.ndarray] = {}
    secondary_scores_by_scenario: dict[int, np.ndarray] = {}
    step_top_by_scenario: dict[int, dict[int, np.ndarray]] = {}
    contact_rows: list[dict[str, Any]] = []
    h_lat = w_lat = None
    d = None
    model_info_written = False

    try:
        for sc in scenarios:
            sccfg = replace(cfg, seed=sc.seed)
            rgb, info = model_utils.generate_with_capture(pipe, sc.prompt, sccfg, state)
            stream = state.image_streams.get(cfg.fixed_layer)
            if stream is None:
                raise RuntimeError(
                    f"No activations captured at layer {cfg.fixed_layer} for scenario "
                    f"{sc.scenario_id}; check that fixed_layer is a valid block index."
                )
            h_lat, w_lat = info["h_lat"], info["w_lat"]
            d = int(stream.shape[1])

            rank = rank_channels(
                stream,
                top_k=cfg.top_n,
                random_k_trials=0,
                seed=sc.seed,
                prompt_id=sc.prompt_id,
                layer=cfg.fixed_layer,
            )
            top_idx_by_scenario[sc.scenario_id] = rank.top_idx
            scores_by_scenario[sc.scenario_id] = rank.scores

            if cfg.secondary_metric:
                sec_idx, sec_scores = rank_channels_secondary(
                    stream, cfg.secondary_metric, cfg.top_n
                )
                secondary_top_by_scenario[sc.scenario_id] = sec_idx
                secondary_scores_by_scenario[sc.scenario_id] = sec_scores

            if cfg.capture_steps:
                from src.stage2_channel_ranking import channel_scores

                per_step: dict[int, np.ndarray] = {}
                for step in cfg.capture_steps:
                    st = state.step_streams.get((int(step), cfg.fixed_layer))
                    if st is None:
                        continue
                    sc_scores = channel_scores(st)
                    per_step[int(step)] = np.argsort(-sc_scores, kind="stable")[: cfg.top_n].astype(
                        np.int32
                    )
                step_top_by_scenario[sc.scenario_id] = per_step

            if not model_info_written:
                write_run_metadata(
                    cfg,
                    scenarios,
                    model_info={
                        "n_layers": len(all_blocks),
                        "fixed_layer": cfg.fixed_layer,
                        "d": d,
                        "n_image_tokens": info["n_image"],
                        "h_lat": h_lat,
                        "w_lat": w_lat,
                    },
                )
                model_info_written = True

            if sc.scenario_id in rep:
                dump_scenario_qualitative(
                    cfg, sc, stream, rgb, rank.top_idx, rank.bottom_idx, rank.scores, h_lat, w_lat
                )
                contact_rows.append(
                    {
                        "sc": sc,
                        "rgb": rgb,
                        "stream": stream,
                        "top_idx": rank.top_idx,
                        "bottom_idx": rank.bottom_idx,
                        "scores": rank.scores,
                    }
                )
    finally:
        for h in handles:
            h.remove()

    return _write_outputs(
        cfg,
        scenarios,
        top_idx_by_scenario,
        scores_by_scenario,
        int(d),
        secondary_top_by_scenario=secondary_top_by_scenario or None,
        secondary_scores_by_scenario=secondary_scores_by_scenario or None,
        step_top_by_scenario=step_top_by_scenario or None,
        contact_rows=contact_rows or None,
        latent_grid=(h_lat, w_lat) if h_lat is not None else None,
    )


def _write_outputs(
    cfg: ScenarioConfig,
    scenarios: list[Scenario],
    top_idx_by_scenario: dict[int, np.ndarray],
    scores_by_scenario: dict[int, np.ndarray],
    d: int,
    secondary_top_by_scenario: dict[int, np.ndarray] | None = None,
    secondary_scores_by_scenario: dict[int, np.ndarray] | None = None,
    step_top_by_scenario: dict[int, dict[int, np.ndarray]] | None = None,
    contact_rows: list[dict[str, Any]] | None = None,
    latent_grid: tuple[int, int] | None = None,
) -> dict[str, Any]:
    out = cfg.output_dir
    write_topk_csv(
        os.path.join(out, "channel_stability_topk.csv"),
        scenarios,
        top_idx_by_scenario,
        scores_by_scenario,
    )

    channels, rank_of, score_of = build_scenario_channel_matrix(
        scenarios, top_idx_by_scenario, scores_by_scenario
    )
    write_matrix_csv(
        os.path.join(out, "scenario_channel_matrix.csv"), scenarios, channels, rank_of, str
    )
    write_matrix_csv(
        os.path.join(out, "scenario_channel_scores.csv"),
        scenarios,
        channels,
        score_of,
        lambda v: f"{v:.6g}",
    )

    summary = compute_stability_summary(scenarios, top_idx_by_scenario, d, len(cfg.seeds))

    if secondary_top_by_scenario and secondary_scores_by_scenario:
        write_topk_csv(
            os.path.join(out, f"channel_stability_topk_{cfg.secondary_metric}.csv"),
            scenarios,
            secondary_top_by_scenario,
            secondary_scores_by_scenario,
        )
        summary["secondary_metric"] = {
            "name": cfg.secondary_metric,
            "mean_jaccard_primary_vs_secondary_per_k": compute_metric_agreement(
                scenarios, top_idx_by_scenario, secondary_top_by_scenario
            ),
            "stability": compute_stability_summary(
                scenarios, secondary_top_by_scenario, d, len(cfg.seeds)
            ),
        }

    if step_top_by_scenario:
        summary["per_step_jaccard"] = compute_step_consistency(
            scenarios, top_idx_by_scenario, step_top_by_scenario
        )

    io.save_json(os.path.join(out, "stability_summary.json"), summary)

    # Figures alongside the CSVs above. Failures never abort the run, but they are
    # reported loudly (full traceback) and surfaced in summary["figure_errors"].
    figure_errors: dict[str, str] = {}

    def _try_fig(name: str, fn) -> None:
        import traceback

        try:
            fn()
        except Exception:
            tb = traceback.format_exc()
            print(f"[channel_stability] FAILED to render {name}:\n{tb}")
            figure_errors[name] = tb

    _try_fig(
        "scenario_channel_heatmap",
        lambda: _save_scenario_channel_heatmap(
            os.path.join(out, "scenario_channel_heatmap.png"),
            scenarios,
            channels,
            rank_of,
            cfg.top_n,
            len(cfg.seeds),
        ),
    )

    def _overlap() -> None:
        sids = [sc.scenario_id for sc in scenarios]
        sets = [np.asarray(top_idx_by_scenario[s]).ravel()[: cfg.agg_k] for s in sids]
        _save_stability_overlap(
            os.path.join(out, "stability_overlap.png"),
            spatial.pairwise_jaccard_matrix(sets),
            cfg.agg_k,
            scenarios,
            len(cfg.seeds),
            summary,
        )

    _try_fig("stability_overlap", _overlap)

    if step_top_by_scenario and summary.get("per_step_jaccard"):
        _try_fig(
            "step_consistency",
            lambda: _save_step_consistency(
                os.path.join(out, "step_consistency.png"),
                summary["per_step_jaccard"],
                cfg.num_denoising_steps,
            ),
        )

    if contact_rows and latent_grid is not None:
        _try_fig(
            "qualitative_summary",
            lambda: _save_contact_sheet(
                os.path.join(out, "qualitative_summary.png"),
                contact_rows,
                cfg.agg_k,
                latent_grid[0],
                latent_grid[1],
            ),
        )

    summary["figure_errors"] = figure_errors
    if figure_errors:
        io.save_json(os.path.join(out, "stability_summary.json"), summary)

    print(
        f"[channel_stability] {len(scenarios)} scenarios, {len(channels)} distinct top-{cfg.top_n} "
        f"channels; outputs in {out}"
    )
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-generation channel-stability experiment (FLUX.1-dev)."
    )
    p.add_argument("--config", required=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_scenario_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
