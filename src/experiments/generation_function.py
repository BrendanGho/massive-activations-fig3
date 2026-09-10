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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CONDITIONS = (
    "baseline",
    "remove_vstar",
    "suppress_channel_154",
    "suppress_sink",
    "remove_top_registers",
    "norm_only",
)


@dataclass(frozen=True)
class Q7Config:
    model_ckpt: str = "black-forest-labs/FLUX.1-dev"
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
    def from_json(cls, path: str | os.PathLike[str]) -> "Q7Config":
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
        bad = set(self.conditions) - set(CONDITIONS)
        if bad:
            raise ValueError(f"unknown conditions: {sorted(bad)}")
        if not self.prompts:
            raise ValueError("at least one prompt is required")
        for group, ranges, upper in (
            ("phases", self.phases, self.num_steps - 1),
            ("zones", self.zones, 56),
        ):
            for name, (lo, hi) in ranges.items():
                if lo < 0 or hi < lo or hi > upper:
                    raise ValueError(f"invalid {group}.{name} range {(lo, hi)}")


def in_target(step: int, layer: int, phase: tuple[int, int], zone: tuple[int, int]) -> bool:
    return phase[0] <= step <= phase[1] and zone[0] <= layer <= zone[1]


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
    if condition == "suppress_channel_154":
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


def config_hash(cfg: Q7Config) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _torch_edit(x, condition: str, mask_np: np.ndarray, vstar, channel: int):
    """Device-local version of the residual interventions."""
    import torch

    y = x.clone()
    mask = torch.as_tensor(mask_np, device=y.device, dtype=torch.bool)
    if condition == "suppress_channel_154":
        y[..., channel] = 0
    elif condition == "remove_top_registers":
        y[:, mask, :] = 0
    elif condition == "remove_vstar":
        v = torch.as_tensor(vstar, device=y.device, dtype=y.dtype)
        v = v / v.float().norm().to(y.dtype)
        z = y[:, mask, :]
        y[:, mask, :] = z - (z.float() @ v.float()).to(y.dtype).unsqueeze(-1) * v
    elif condition == "norm_only":
        z = y[:, mask, :]
        all_norm = y.float().norm(dim=-1)
        ordinary = all_norm[:, ~mask]
        target = (
            ordinary.median(dim=1).values if ordinary.shape[1] else all_norm.median(dim=1).values
        )
        scale = target[:, None] / z.float().norm(dim=-1).clamp_min(1e-12)
        y[:, mask, :] = z * scale.to(z.dtype).unsqueeze(-1)
    return y


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
    """Clean-run masks and register vectors keyed by (step, layer)."""

    def __init__(self, threshold: float, max_registers: int):
        self.threshold, self.max_registers = threshold, max_registers
        self.masks: dict[tuple[int, int], np.ndarray] = {}
        self.vectors: list[np.ndarray] = []

    def observe(self, step: int, layer: int, x) -> None:
        arr = x[-1].detach().float().cpu().numpy()
        mask = natural_register_mask(arr, self.threshold, self.max_registers)
        self.masks[(step, layer)] = mask
        if mask.any():
            self.vectors.extend(arr[mask])


class ResidualInterventionHooks:
    """Post-block hooks using masks fixed by the paired clean run."""

    def __init__(self, blocks, trace, condition, phase, zone, vstar, channel, n_steps):
        self.blocks, self.trace, self.condition = blocks, trace, condition
        self.phase, self.zone, self.vstar, self.channel = phase, zone, vstar, channel
        self.n_steps, self.counts, self.handles = n_steps, {}, []

    def attach(self, transformer):
        def pre(_m, _a, kw):
            hs = kw.get("hidden_states")
            self.n_image = int(hs.shape[1])

        self.handles.append(transformer.register_forward_pre_hook(pre, with_kwargs=True))
        for ref in self.blocks:
            self.counts[ref.layer_id] = 0

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
                return new

            self.handles.append(ref.module.register_forward_hook(hook))
        return self

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class SinkAttentionHooks:
    """Mask incoming attention to clean-run natural register/sink keys.

    FLUX attention concatenates text keys before image keys.  The hook supplies an additive
    key mask to each targeted block's ``Attention.forward``; the block input and residual skip
    are never edited.  This relies on the public ``attention_mask`` argument and validates it
    before attachment so API drift fails during the smoke test.
    """

    def __init__(self, blocks, trace, phase, zone, n_steps):
        self.blocks, self.trace, self.phase, self.zone = blocks, trace, phase, zone
        self.n_steps, self.counts, self.handles = n_steps, {}, []

    def attach(self):
        import inspect
        import torch

        for ref in self.blocks:
            attn = getattr(ref.module, "attn", None)
            if attn is None:
                raise RuntimeError(f"block {ref.layer_id} has no .attn module")
            if "attention_mask" not in inspect.signature(attn.forward).parameters:
                raise RuntimeError(
                    f"{type(attn).__name__}.forward lacks attention_mask; pin a compatible "
                    "diffusers version before running sink suppression"
                )
            self.counts[ref.layer_id] = 0

            def pre(_module, args, kwargs, layer=ref.layer_id):
                step = self.counts[layer]
                self.counts[layer] += 1
                if not in_target(step, layer, self.phase, self.zone):
                    return None
                mask_np = self.trace.masks.get((step, layer))
                if mask_np is None:
                    raise RuntimeError(f"clean trace missing step {step}, layer {layer}")
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                encoder = kwargs.get("encoder_hidden_states")
                if hidden is None or hidden.ndim != 3:
                    raise RuntimeError(f"unexpected attention input at layer {layer}")
                # Joint FLUX attention orders encoder/text then hidden/image. Single-stream
                # attention already receives the concatenated sequence, with image tokens last.
                n_img = int(mask_np.size)
                total = int(hidden.shape[1] + (encoder.shape[1] if encoder is not None else 0))
                offset = total - n_img
                additive = torch.zeros(
                    (hidden.shape[0], 1, 1, total), device=hidden.device, dtype=hidden.dtype
                )
                sink_idx = torch.as_tensor(np.flatnonzero(mask_np), device=hidden.device)
                additive[..., offset + sink_idx] = torch.finfo(hidden.dtype).min
                prior = kwargs.get("attention_mask")
                if prior is not None:
                    additive = additive + prior.to(device=hidden.device, dtype=hidden.dtype)
                kwargs["attention_mask"] = additive
                return args, kwargs

            self.handles.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
        return self

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _generate(pipe, cfg: Q7Config, prompt: str, seed: int):
    import torch

    gen_device = "cpu" if cfg.device == "cuda" else cfg.device
    generator = torch.Generator(gen_device).manual_seed(seed)
    kwargs = dict(
        prompt=prompt,
        height=cfg.resolution,
        width=cfg.resolution,
        num_inference_steps=cfg.num_steps,
        generator=generator,
        output_type="pil",
    )
    if cfg.guidance_scale is not None:
        kwargs["guidance_scale"] = cfg.guidance_scale
    with torch.inference_mode():
        return pipe(**kwargs).images[0]


def _trace_clean(pipe, blocks, cfg, prompt, seed):
    trace = NaturalTrace(cfg.register_threshold, cfg.max_registers)
    counters = {b.layer_id: 0 for b in blocks}
    state = {"n_image": None}
    handles = []

    def pre(_m, _a, kw):
        state["n_image"] = int(kw["hidden_states"].shape[1])

    handles.append(pipe.transformer.register_forward_pre_hook(pre, with_kwargs=True))
    for ref in blocks:

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
    return image, trace


def _save_image(image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def calibrate_vstar(cfg: Q7Config, smoke: bool = False) -> Path:
    """Pool clean natural-register vectors across prompt/seed scenarios and save v*."""
    from src.common.model_utils import discover_blocks, load_pipeline

    pipe_cfg = type(
        "Cfg",
        (),
        {"model_ckpt": cfg.model_ckpt, "dtype": cfg.dtype, "device": cfg.device},
    )()
    pipe = load_pipeline(pipe_cfg, offload=cfg.offload)
    blocks = discover_blocks(pipe.transformer)
    jobs = [(p, s) for p in cfg.prompts for s in cfg.seeds]
    if smoke:
        jobs = jobs[:1]
    vectors = []
    for prompt, seed in jobs:
        _image, trace = _trace_clean(pipe, blocks, cfg, prompt, seed)
        vectors.extend(trace.vectors)
    if not vectors:
        raise RuntimeError("no natural register vectors found during vstar calibration")
    path = Path(cfg.vstar_path) if cfg.vstar_path else Path(cfg.output_dir) / "vstar.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, fit_vstar(np.asarray(vectors)))
    metadata = {
        "path": str(path),
        "n_vectors": len(vectors),
        "n_scenarios": len(jobs),
        "config_hash": config_hash(cfg),
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
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "runs.jsonl"
    completed = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            old = json.loads(line)
            completed.add(
                (old["prompt_id"], old["seed"], old["condition"], old["phase"], old["zone"])
            )
    jobs = [(p, s) for p in cfg.prompts for s in cfg.seeds]
    if smoke:
        jobs = jobs[:1]
    for prompt_id, (prompt, seed) in enumerate(jobs):
        clean, trace = _trace_clean(pipe, blocks, cfg, prompt, seed)
        clean_path = out / "images" / f"p{prompt_id:03d}_s{seed}_baseline.png"
        _save_image(clean, clean_path)
        calibrated = Path(cfg.vstar_path) if cfg.vstar_path else out / "vstar.npy"
        vstar = np.load(calibrated) if calibrated.exists() else fit_vstar(np.asarray(trace.vectors))
        phase_items = list(cfg.phases.items())[:1] if smoke else list(cfg.phases.items())
        zone_items = list(cfg.zones.items())[:1] if smoke else list(cfg.zones.items())
        for phase_name, phase in phase_items:
            for zone_name, zone in zone_items:
                conditions = [c for c in cfg.conditions if c != "baseline"]
                for condition in conditions:
                    job_key = (prompt_id, seed, condition, phase_name, zone_name)
                    if job_key in completed:
                        continue
                    if condition == "suppress_sink":
                        hooks = SinkAttentionHooks(
                            blocks, trace, phase, zone, cfg.num_steps
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
                    stem = f"p{prompt_id:03d}_s{seed}_{condition}_{phase_name}_{zone_name}"
                    image_path = out / "images" / f"{stem}.png"
                    _save_image(image, image_path)
                    row = dict(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        seed=seed,
                        condition=condition,
                        phase=phase_name,
                        zone=zone_name,
                        clean_path=str(clean_path),
                        image_path=str(image_path),
                        config_hash=config_hash(cfg),
                        hook_counts=hooks.counts,
                    )
                    with manifest.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, sort_keys=True) + "\n")


def evaluate(output_dir: str) -> None:
    """Compute paired perceptual, prompt-fidelity, and frequency-band metrics.

    LPIPS, OpenCLIP, and ImageReward are optional: absent packages produce explicit blank
    columns rather than silently substituting a different metric.
    """
    from PIL import Image

    root = Path(output_dir)
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
    for line in (root / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        clean_pil = Image.open(row["clean_path"]).convert("RGB")
        edited_pil = Image.open(row["image_path"]).convert("RGB")
        clean, edited = np.asarray(clean_pil), np.asarray(edited_pil)
        row.update(frequency_distances(clean, edited))
        row.update(
            lpips=None,
            clip_clean=None,
            clip_edited=None,
            image_reward_clean=None,
            image_reward_edited=None,
        )
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
        "clip_edited",
        "image_reward_edited",
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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--calibrate-vstar", action="store_true")
    args = p.parse_args(argv)
    cfg = Q7Config.from_json(args.config)
    if args.calibrate_vstar:
        print(calibrate_vstar(cfg, smoke=args.smoke))
    elif args.evaluate:
        evaluate(cfg.output_dir)
    else:
        run(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
