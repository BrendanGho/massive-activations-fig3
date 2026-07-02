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
    representative_scenarios: list[int] | None = None  # None -> first 3
    offload: bool = False

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
        }
    return summary


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


def _save_jaccard_heatmap(path: str, mat: np.ndarray, k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"pairwise top-{k} channel-identity Jaccard", fontsize=10)
    ax.set_xlabel("scenario_id")
    ax.set_ylabel("scenario_id")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
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
    sdir = os.path.join(cfg.output_dir, "scenarios", f"scenario_{sc.scenario_id:03d}")
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


def _versions() -> dict:
    out = {}
    for name in ("numpy", "sklearn", "torch", "diffusers", "transformers"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = "not installed"
    return out


def write_run_metadata(
    cfg: ScenarioConfig, scenarios: list[Scenario], model_info: dict | None
) -> str:
    meta = {
        "experiment": "channel_stability",
        "resolved_config": cfg.to_dict(),
        "versions": _versions(),
        "n_scenarios": len(scenarios),
        "assumptions": {
            "score": "abs(mean_over_tokens) — mean then abs (fig3 massive-activation score)",
            "probe": {
                "fixed_layer": cfg.fixed_layer,
                "timestep": "last denoising step only (hook overwrite -> last forward wins)",
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

    scenarios = build_scenarios(cfg.prompts, cfg.seeds)
    # Default representative scenarios: the first (up to) 3 DISTINCT prompts at their first
    # seed, so the qualitative dumps show different content (not one prompt at three seeds).
    n_seeds = len(cfg.seeds)
    rep = (
        set(cfg.representative_scenarios)
        if cfg.representative_scenarios is not None
        else {p * n_seeds for p in range(min(3, len(cfg.prompts)))}
    )
    write_run_metadata(cfg, scenarios, model_info=None)

    pipe = model_utils.load_pipeline(cfg, offload=cfg.offload)
    transformer = pipe.transformer
    all_blocks = model_utils.discover_blocks(transformer)
    blocks = model_utils.select_layers(all_blocks, [cfg.fixed_layer])
    state = model_utils.CaptureState()
    handles = model_utils.register_capture_hooks(transformer, blocks, state)

    top_idx_by_scenario: dict[int, np.ndarray] = {}
    scores_by_scenario: dict[int, np.ndarray] = {}
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
                seed=cfg.seed,
                prompt_id=sc.scenario_id,
                layer=cfg.fixed_layer,
            )
            top_idx_by_scenario[sc.scenario_id] = rank.top_idx
            scores_by_scenario[sc.scenario_id] = rank.scores

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
    finally:
        for h in handles:
            h.remove()

    return _write_outputs(cfg, scenarios, top_idx_by_scenario, scores_by_scenario, int(d))


def _write_outputs(
    cfg: ScenarioConfig,
    scenarios: list[Scenario],
    top_idx_by_scenario: dict[int, np.ndarray],
    scores_by_scenario: dict[int, np.ndarray],
    d: int,
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
    io.save_json(os.path.join(out, "stability_summary.json"), summary)

    # Optional Jaccard heatmap at top-10 (lazy matplotlib; skip if it can't be drawn).
    try:
        sids = [sc.scenario_id for sc in scenarios]
        sets = [np.asarray(top_idx_by_scenario[s]).ravel()[:10] for s in sids]
        _save_jaccard_heatmap(
            os.path.join(out, "stability_overlap.png"), spatial.pairwise_jaccard_matrix(sets), 10
        )
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"[channel_stability] skipped overlap heatmap: {exc}")

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
