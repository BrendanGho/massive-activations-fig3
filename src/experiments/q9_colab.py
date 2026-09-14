"""No-code Colab configuration shared by the main and standalone Q9 notebooks."""

from .text_image_coupling import EDGE_METHODS, PRESETS, preset_config


def _csv(value, convert=str):
    return [convert(part.strip()) for part in value.split(",") if part.strip()]


def build_config(
    model="flux1-dev",
    mode="screen",
    resolution="preset",
    *,
    workload="pilot",
    vram_gib=40,
    bf16=True,
    advanced=None,
):
    """Pilot is explicit; full retains the original grid. Form overrides apply last."""
    if workload not in {"pilot", "full"}:
        raise ValueError("workload must be pilot or full")
    cfg = preset_config(model, mode)
    if workload == "pilot" and mode != "smoke":
        # Independent calibration/evaluation, but intentionally too small for a final claim.
        cfg.calibration_prompts = [cfg.calibration_prompts[i] for i in (0, 4, 8)]
        cfg.calibration_seeds = [0]
        indices = (0, 8, 16) if mode == "confirm" else (0, 2, 3)
        cfg.prompts = [cfg.prompts[i] for i in indices]
        cfg.seeds = [7]
        cfg.sites, cfg.steps = [17], [0]
        cfg.attention_layers = [17, 18, 19, 39]
        cfg.methods = ["remove_direction", "norm_matched", "zero", "ordinary_zero"]
        cfg.rescues = ["none"]
        cfg.include_empty = False
    cfg.dtype = "bf16" if bf16 else "fp16"
    cfg.offload = vram_gib < 38
    if resolution != "preset":
        cfg.resolution = int(resolution)
    fields = dict(advanced or {})
    for name in ("sites", "steps", "seeds", "attention_layers", "calibration_seeds"):
        value = fields.pop(name, "")
        if value.strip():
            setattr(cfg, name, _csv(value, int))
    for name in ("methods", "candidate_classes"):
        value = fields.pop(name, "")
        if value.strip():
            setattr(cfg, name, _csv(value))
    for name in ("prompts", "calibration_prompts"):
        value = fields.pop(name, "")
        if value.strip():
            setattr(cfg, name, [p.strip() for p in value.split("||") if p.strip()])
    for field, count_field in (
        ("prompts", "prompt_count"),
        ("calibration_prompts", "calibration_prompt_count"),
    ):
        count = fields.pop(count_field, 0)
        if not isinstance(count, int) or not 0 <= count <= len(getattr(cfg, field)):
            raise ValueError(
                f"{count_field}: use 0 for preset/all or 1..{len(getattr(cfg, field))}"
            )
        if count:
            setattr(cfg, field, getattr(cfg, field)[:count])
    rescues = fields.pop("rescues", "preset")
    if rescues != "preset":
        cfg.rescues = _csv(rescues)
    memory = fields.pop("memory", "auto")
    if memory not in {"auto", "offload", "gpu"}:
        raise ValueError("memory must be auto, offload or gpu")
    if memory != "auto":
        cfg.offload = memory == "offload"
    for name in ("rescue_layer", "readout_layer"):
        value = fields.pop(name, "")
        if str(value).strip():
            setattr(cfg, name, int(value))
    for name in (
        "candidate_source",
        "optimize_probes",
        "skip_unavailable",
        "evaluate_lpips",
        "evaluate_clip",
    ):
        if name in fields:
            setattr(cfg, name, fields.pop(name))
    include_empty = fields.pop("include_empty", "preset")
    if include_empty != "preset":
        if include_empty not in (True, False, "yes", "no"):
            raise ValueError("include_empty must be preset, yes or no")
        cfg.include_empty = include_empty in (True, "yes")
    cfg.structured_scores = fields.pop("structured_scores", "").strip() or None
    if fields:
        raise ValueError(f"Unknown Q9 form settings: {sorted(fields)}")
    if any(seed < 0 for seed in cfg.seeds + cfg.calibration_seeds):
        raise ValueError("Seeds must be nonnegative integers")
    cfg.validate()
    return cfg


def show_results(cfg):
    """Display after the run without requiring another notebook cell."""
    from IPython.display import Image, display

    from .q9_report import locate

    result = locate(cfg)
    print((result / "report_status.json").read_text())
    for figure in sorted((result / "figures").glob("*.png")):
        print(figure.name)
        display(Image(filename=str(figure)))
    print("Results:", result)
    print("Check audits.csv and direction_stability.csv before interpreting a negative result.")
    return result


def run_budget(cfg):
    """Planned upper bounds before resume/no-candidate skips, excluding donor replays."""
    discovery = cfg.mode == "discovery"
    jobs_per_pair = (
        sum(
            1
            for site in cfg.sites
            for _step in cfg.steps
            for method in cfg.methods
            for rescue in cfg.rescues
            if rescue == "none" or method == "remove_direction"
            if not (site == -1 and (method in EDGE_METHODS or method.startswith("image_")))
        )
        if not discovery
        else 0
    )
    pairs = sum(bool(p) for p in cfg.prompts) * len(cfg.seeds) if not discovery else 0
    baselines = (
        ((len(cfg.prompts) + int(cfg.include_empty and "" not in cfg.prompts)) * len(cfg.seeds))
        if not discovery
        else 0
    )
    jobs = jobs_per_pair * pairs
    return {
        "model": cfg.model_preset,
        "mode": cfg.mode,
        "denoising_steps": PRESETS[cfg.model_preset]["num_steps"],
        "resolution": cfg.resolution,
        "calibration_trajectories_if_uncached": len(cfg.calibration_prompts)
        * len(cfg.calibration_seeds),
        "clean_evaluation_trajectories": baselines,
        "evaluation_pairs": pairs,
        "jobs_per_pair": jobs_per_pair,
        "edited_full_trajectories": jobs if cfg.mode == "confirm" else 0,
        "edited_single_forwards": jobs if cfg.mode in {"screen", "smoke"} else 0,
        "replay_check_forwards": baselines * len(cfg.steps),
        "additional_donor_replays_possible": not discovery and "donor_swap" in cfg.methods,
    }
