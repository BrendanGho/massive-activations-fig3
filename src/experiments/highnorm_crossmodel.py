"""Part 3, cross-model: one row per model, each with its own ablated massive channel.

The single-model qualitative figure (``highnorm_qualitative``) puts one prompt per row.
This one puts one **model** per row — same prompt, same seed, four columns:

    generated | isolated channel C | high-norm tokens | high-norm tokens, C ablated

with ``C`` chosen **per model**, because massive-channel ids are per model and per layer
(e.g. FLUX 154, PixArt-Sigma 293). Every row therefore carries its own column titles, not
just the top one — the label has to name that row's channel or it is wrong for two rows out
of three. The row label on the left is the model name and nothing else.

The figure carries **no suptitle, no colorbar and no layer label** — it is built to drop into
a paper, where the prompt, the color mapping and the layer belong in the caption. The layer
lives in the *filename* instead: one figure per layer, ``crossmodel_L<layer>.png``.

The norm columns share ONE absolute scale **per row** (never across rows): different models
have different widths D and different activation magnitudes, so a cross-row color scale would
say nothing. Within a row the scale is absolute, which is what makes the ablation legible —
see ``highnorm_qualitative.shared_norm_scale``.

Capture and figure are separate stages joined by a cache, because the three models do not fit
in memory together and one of them (FLUX.1-dev) is gated: each model is loaded alone, hooked
on every requested layer in a SINGLE generation pass (a full-depth sweep costs one generation,
not one per layer), written to ``<output_dir>/cache/<key>/L<layer>.npz``, then freed. A row
with a populated cache is reused as-is, so re-runs are free and the figures can be reassembled
(different channels, different rows) without touching a GPU.

    python -m src.experiments.highnorm_crossmodel --config configs/highnorm_crossmodel.yaml
    python -m src.experiments.highnorm_crossmodel --config ... --only pixart-sigma --refresh
    python -m src.experiments.highnorm_crossmodel --config ... --layers 0,9,18
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import yaml

from src.experiments.highnorm_qualitative import (
    _primary_label,
    panel_maps,
    parse_channels,
    resolve_layers,
    shared_norm_scale,
)

# --- config -------------------------------------------------------------------


@dataclass
class RowSpec:
    """One model = one row of the figure."""

    key: str  # cache folder + --only selector, e.g. "flux-schnell"
    model_ckpt: str
    label: str = ""  # row label on the figure (defaults to `key`)
    target_layer: int = 18  # the block this row probes
    layers: str = ""  # layers to capture: "all", "0,5,10", or "" for target_layer only
    ablate_channels: list[int] = field(default_factory=list)  # explicit ids, e.g. [154]
    n_channels: int = 1  # used only when ablate_channels is empty (top-N per layer)
    num_denoising_steps: int = 4
    guidance_scale: float | None = None
    offload: bool | None = None  # None => inherit the top-level default
    seed: int | None = None  # None => inherit the top-level default

    def __post_init__(self) -> None:
        self.ablate_channels = [int(c) for c in (self.ablate_channels or [])]
        if not self.label:
            self.label = self.key


@dataclass
class CrossModelConfig:
    output_dir: str | None = None
    prompt: str = ""
    models: list[RowSpec] = field(default_factory=list)
    resolution: int = 1024
    dtype: str = "bf16"
    device: str = "cuda"
    seed: int = 0
    offload: bool = False

    def row_seed(self, spec: RowSpec) -> int:
        return int(self.seed if spec.seed is None else spec.seed)

    def row_offload(self, spec: RowSpec) -> bool:
        return bool(self.offload if spec.offload is None else spec.offload)


def load_crossmodel_config(config_path: str, *, create_dirs: bool = True) -> CrossModelConfig:
    """Load + validate the cross-model YAML. Fails loud on missing/unknown keys."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")

    valid = {f.name for f in fields(CrossModelConfig)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {sorted(unknown)}")

    raw_models = raw.pop("models", None) or []
    if not isinstance(raw_models, list):
        raise ValueError("`models` must be a list of mappings, one per figure row.")
    row_valid = {f.name for f in fields(RowSpec)}
    models = []
    for i, m in enumerate(raw_models):
        if not isinstance(m, dict):
            raise ValueError(f"models[{i}] must be a mapping, got {type(m).__name__}")
        bad = set(m) - row_valid
        if bad:
            raise ValueError(f"Unknown keys in models[{i}] of {config_path}: {sorted(bad)}")
        for req in ("key", "model_ckpt"):
            if not m.get(req):
                raise ValueError(f"models[{i}] is missing required `{req}` (source: {config_path})")
        models.append(RowSpec(**m))

    cfg = CrossModelConfig(models=models, **raw)
    _validate(cfg, config_path)
    if create_dirs and cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)
    return cfg


def _validate(cfg: CrossModelConfig, source: str) -> None:
    if not cfg.output_dir:
        raise ValueError(f"Missing required config value `output_dir` (source: {source}).")
    if not cfg.prompt or not str(cfg.prompt).strip():
        raise ValueError(f"`prompt` must be a non-empty string (source: {source}).")
    if not cfg.models:
        raise ValueError(f"`models` must list at least one model row (source: {source}).")
    keys = [m.key for m in cfg.models]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:  # keys are cache folders; duplicates would overwrite each other
        raise ValueError(f"Duplicate model keys {dupes} (source: {source}).")
    if cfg.dtype not in ("bf16", "fp16", "fp32"):
        raise ValueError(f"dtype must be one of bf16/fp16/fp32, got {cfg.dtype!r}")
    for m in cfg.models:
        if m.n_channels <= 0:
            raise ValueError(f"models[{m.key}].n_channels must be positive, got {m.n_channels}")
        if any(c < 0 for c in m.ablate_channels):
            raise ValueError(f"models[{m.key}].ablate_channels must be non-negative")


def select_rows(cfg: CrossModelConfig, only: str | None) -> list[RowSpec]:
    """The rows named by ``--only k1,k2`` (config order preserved); all rows when empty."""
    if not only or not str(only).strip():
        return list(cfg.models)
    want = [p.strip() for p in str(only).split(",") if p.strip()]
    known = {m.key for m in cfg.models}
    missing = [k for k in want if k not in known]
    if missing:
        raise ValueError(f"--only names unknown model key(s) {missing}; config has {sorted(known)}")
    return [m for m in cfg.models if m.key in set(want)]


# --- cache --------------------------------------------------------------------


def cache_path(output_dir: str, key: str, layer: int) -> str:
    """Where one row's captured maps live: ``<out>/cache/<key>/L<layer>.npz``."""
    return os.path.join(output_dir, "cache", key, f"L{int(layer)}.npz")


def save_row(path: str, row: dict[str, Any]) -> None:
    """Persist one captured row (rgb + the three maps + which channels were isolated)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(
        path,
        rgb=row["rgb"],
        speckle=row["speckle"],
        n_full=row["n_full"],
        n_ex=row["n_ex"],
        channels=np.asarray(row["channels"], dtype=np.int64),
        layer=np.int64(row["layer"]),
        key=np.array(str(row["key"])),
        label=np.array(str(row["label"])),
        prompt=np.array(str(row.get("prompt", ""))),
    )


def load_row(path: str) -> dict[str, Any]:
    """Read back a row written by :func:`save_row`."""
    with np.load(path, allow_pickle=False) as z:
        return {
            "rgb": z["rgb"],
            "speckle": z["speckle"],
            "n_full": z["n_full"],
            "n_ex": z["n_ex"],
            "channels": z["channels"].astype(np.int64),
            "layer": int(z["layer"]),
            "key": str(z["key"]),
            "label": str(z["label"]),
            "prompt": str(z["prompt"]),
        }


# --- figure plumbing (pure) ---------------------------------------------------


def row_titles(channels: np.ndarray | list[int]) -> list[str]:
    """The four column titles for ONE row, named for that row's own channel(s).

    Per-row rather than per-figure: with a different massive channel ablated in each model,
    a single header row would mislabel every row but the first.
    """
    ids = [int(c) for c in np.asarray(channels).ravel().tolist()]
    label = _primary_label(len(ids), ids)  # "channel 154" / "channels 154,1446"
    return [
        "generated",
        f"isolated {label}",
        "high-norm tokens",
        f"high-norm tokens\n{label} ablated",
    ]


def sweep_figure_layers(layers_by_key: dict[str, list[int]]) -> list[int]:
    """Layers a cross-model figure can be drawn at: present in EVERY row, ascending.

    Models differ in depth (FLUX 0..56, PixArt-Sigma 0..27), so a sweep can only render the
    common prefix; the caller reports what that drops rather than dropping it silently.
    """
    if not layers_by_key:
        return []
    common: set[int] | None = None
    for layers in layers_by_key.values():
        s = set(int(ly) for ly in layers)
        common = s if common is None else (common & s)
    return sorted(common or set())


def figure_name(layer: int) -> str:
    """``crossmodel_L<layer>.png`` — the layer is in the filename, one file per layer."""
    return f"crossmodel_L{int(layer)}.png"


def _save_figure(path: str, rows: list[dict[str, Any]]) -> None:
    """Draw the figure: no suptitle, no colorbar, no layer in the row label.

    All three are deliberately absent — the layer is in the *filename* and the rest belongs
    in the paper caption, so the panels get the space instead.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 4
    n = len(rows)
    # Compact: panels sized to leave only what the two-line per-row titles need.
    fig, axes = plt.subplots(
        n,
        ncols,
        figsize=(2.3 * ncols, 2.45 * n),
        squeeze=False,
        gridspec_kw={"wspace": 0.04, "hspace": 0.24},
    )
    for r, row in enumerate(rows):
        titles = row_titles(row["channels"])
        # One ABSOLUTE scale per row across both norm columns: a color means the same token
        # norm in each, so a token whose norm was owned by the ablated channel goes dark.
        # Per row, never per figure — models differ in D and in activation magnitude.
        v_lo, v_hi = shared_norm_scale(row, subtract_ks=None)
        cells = [
            (row["rgb"], None, None, None),
            (row["speckle"], "inferno", None, None),
            (row["n_full"], "viridis", v_lo, v_hi),
            (row["n_ex"], "viridis", v_lo, v_hi),
        ]
        for c, (img, cmap, vlo, vhi) in enumerate(cells):
            ax = axes[r][c]
            ax.imshow(img, cmap=cmap, vmin=vlo, vmax=vhi, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(titles[c], fontsize=9, pad=4)  # every row, not just the first
        axes[r][0].set_ylabel(row["label"], fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# --- capture (GPU) ------------------------------------------------------------


def _model_cfg(cfg: CrossModelConfig, spec: RowSpec):
    """A HighNormConfig view of one row, for model_utils.load_pipeline/generate_with_capture."""
    from src.experiments.highnorm_tokens import HighNormConfig

    return HighNormConfig(
        model_ckpt=spec.model_ckpt,
        output_dir=cfg.output_dir,
        device=cfg.device,
        dtype=cfg.dtype,
        seed=cfg.row_seed(spec),
        num_denoising_steps=spec.num_denoising_steps,
        resolution=cfg.resolution,
        guidance_scale=spec.guidance_scale,
        offload=cfg.row_offload(spec),
        target_layer=spec.target_layer,
    )


def _free(pipe) -> None:
    """Drop a pipeline and its GPU memory before the next model loads."""
    import gc

    del pipe
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def capture_model(
    cfg: CrossModelConfig, spec: RowSpec, layers_spec: str | None = None
) -> dict[int, dict[str, Any]]:
    """Generate once for this model and return ``{layer: row}`` for every requested layer.

    Every wanted layer is hooked in the SAME forward pass, so a full-depth sweep costs one
    generation, not one per layer. The pipeline is freed before returning.
    """
    from src.common import model_utils

    mcfg = _model_cfg(cfg, spec)
    pipe = model_utils.load_pipeline(mcfg, offload=mcfg.offload)
    try:
        all_blocks = model_utils.discover_blocks(pipe.transformer)
        all_ids = [b.layer_id for b in all_blocks]
        want = resolve_layers(
            layers_spec if layers_spec is not None else spec.layers, all_ids, spec.target_layer
        )
        blocks = model_utils.select_layers(all_blocks, want)
        state = model_utils.CaptureState()
        handles = model_utils.register_capture_hooks(pipe.transformer, blocks, state)
        try:
            rgb, info = model_utils.generate_with_capture(pipe, cfg.prompt, mcfg, state)
        finally:
            for h in handles:
                h.remove()
    finally:
        _free(pipe)

    out: dict[int, dict[str, Any]] = {}
    for ly in want:
        if ly not in state.image_streams:
            raise RuntimeError(f"no capture at layer {ly} for model {spec.key}")
        maps = panel_maps(
            state.image_streams[ly],
            spec.n_channels,
            info["h_lat"],
            info["w_lat"],
            explicit_channels=spec.ablate_channels or None,
        )
        out[ly] = {
            "key": spec.key,
            "label": spec.label,
            "layer": int(ly),
            "prompt": cfg.prompt,
            "rgb": rgb,
            "channels": np.asarray(maps["channels"], dtype=np.int64),
            "speckle": maps["speckle"],
            "n_full": maps["n_full"],
            "n_ex": maps["n_ex"],
        }
    return out


# --- runner -------------------------------------------------------------------


def cached_layers(output_dir: str, key: str) -> list[int]:
    """Layers already captured for one row, ascending, read off the cache folder."""
    d = os.path.join(output_dir, "cache", key)
    if not os.path.isdir(d):
        return []
    return sorted(int(f[1:-4]) for f in os.listdir(d) if f.startswith("L") and f.endswith(".npz"))


def run(
    cfg: CrossModelConfig,
    only: str | None = None,
    layers_override: str | None = None,
    refresh: bool = False,
) -> list[str]:
    """Capture any rows not already cached, then write ONE FIGURE PER LAYER. Returns paths.

    Each row is captured over its own ``layers`` spec (``all`` by default), and a figure is
    written for every layer present in *every* row — ``crossmodel_L<layer>.png``, so the
    layer is in the filename rather than on the figure. Layers a row cannot supply (FLUX
    has 57 blocks, PixArt-Sigma 28) are reported, not silently dropped.

    A row whose cache folder is already populated is reused as-is and not regenerated; pass
    ``refresh`` to re-capture it. Rows missing from the cache are reported and omitted from
    the figure rather than aborting it — a gated or OOM model can be run later and the
    figures reassembled from cache for free.
    """
    specs = select_rows(cfg, only)
    assert cfg.output_dir  # validated

    for spec in specs:
        layers_spec = layers_override if layers_override is not None else spec.layers
        have = cached_layers(cfg.output_dir, spec.key)
        if have and not refresh:
            print(f"[xm] {spec.key}: {len(have)} layer(s) already cached — skipping generation")
            continue
        print(
            f"[xm] {spec.key}: generating ({spec.model_ckpt}, "
            f"layers={layers_spec or spec.target_layer})"
        )
        rows = capture_model(cfg, spec, layers_spec)
        for ly, row in rows.items():
            save_row(cache_path(cfg.output_dir, spec.key, ly), row)
        if len(rows) <= 4:  # a full sweep would print hundreds of ids
            chans = {ly: row["channels"].tolist() for ly, row in rows.items()}
            print(f"[xm] {spec.key}: isolated channels per layer {chans}")
        print(f"[xm] {spec.key}: cached {len(rows)} layer(s) in cache/{spec.key}")

    # Assemble from cache in CONFIG order (the row order the figure is specified in),
    # including rows captured by an earlier run.
    layers_by_key = {}
    for spec in cfg.models:
        found = cached_layers(cfg.output_dir, spec.key)
        if found:
            layers_by_key[spec.key] = found
    missing_rows = [m.key for m in cfg.models if m.key not in layers_by_key]
    if missing_rows:
        print(f"[xm] no cached rows for {missing_rows}; their row(s) will be omitted")

    fig_layers = sweep_figure_layers(layers_by_key)
    for key, found in layers_by_key.items():
        dropped = [ly for ly in found if ly not in set(fig_layers)]
        if dropped:  # never drop coverage silently
            print(
                f"[xm] {key}: {len(dropped)} captured layer(s) not shared by every row, "
                f"no figure for {dropped[0]}..{dropped[-1]}"
            )
    if not fig_layers:
        print(f"[xm] no layer is present in every row; cached layers: {layers_by_key}")

    out_paths: list[str] = []
    for ly in fig_layers:
        rows = [
            load_row(cache_path(cfg.output_dir, spec.key, ly))
            for spec in cfg.models
            if os.path.isfile(cache_path(cfg.output_dir, spec.key, ly))
        ]
        if not rows:
            continue
        out_path = os.path.join(cfg.output_dir, figure_name(ly))
        _save_figure(out_path, rows)
        out_paths.append(out_path)
    print(f"[xm] wrote {len(out_paths)} figure(s) to {cfg.output_dir}")
    return out_paths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Cross-model Part 3 figure: one row per model, its own ablated channel."
    )
    p.add_argument("--config", required=True)
    p.add_argument(
        "--only",
        default="",
        help="Capture only these model keys (comma-separated). Other rows still come from "
        "cache if present — use this to run a gated/large model on its own.",
    )
    p.add_argument(
        "--layers",
        default=None,
        help="Override every row's `layers` spec: 'all' (the config default), a list "
        "'0,5,10', or '' for each row's target_layer only. One figure is written per layer "
        "present in every row, named crossmodel_L<layer>.png.",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Regenerate even when a cached row exists (e.g. after changing the prompt/seed).",
    )
    p.add_argument(
        "--channels",
        default="",
        help="Override the ablated channels for the models named by --only, e.g. '154'. "
        "Applies to every selected row; leave empty to use each model's config value.",
    )
    args = p.parse_args(argv)

    cfg = load_crossmodel_config(args.config)
    override = parse_channels(args.channels)
    if override:
        for spec in select_rows(cfg, args.only):
            spec.ablate_channels = override
    run(cfg, only=args.only, layers_override=args.layers, refresh=args.refresh)


if __name__ == "__main__":
    main()
