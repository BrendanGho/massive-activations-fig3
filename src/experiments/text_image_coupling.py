"""Q9 configuration, numeric analysis and CLI. Heavy model imports are lazy."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REVISION = "q9-v1"
PRESETS = {
    "flux1-dev": {
        "model_ckpt": "black-forest-labs/FLUX.1-dev",
        "num_steps": 28,
        "guidance_scale": 3.5,
    },
    "flux-schnell": {
        "model_ckpt": "black-forest-labs/FLUX.1-schnell",
        "num_steps": 4,
        "guidance_scale": 0.0,
    },
}
CALIBRATION_PROMPTS = (
    "a red apple",
    "a green apple",
    "a blue car",
    "a red car",
    "two cats beside a chair",
    "three dogs beside a table",
    "a yellow cube above a green sphere",
    "a green cube below a yellow sphere",
    "a bicycle leaning against a brick wall in the afternoon sunlight",
    "a boat floating on a still lake with mountains in the distance",
    "a striped ceramic vase holding white flowers on a wooden desk",
    "a black horse standing beside a white fence under a cloudy sky",
)
SCREEN_PROMPTS = (
    "a red bird",
    "a blue bird",
    "two oranges to the left of a purple bowl",
    "two oranges to the right of a purple bowl",
)
CONFIRM_PROMPTS = tuple(
    f"{count} {color} {obj} {relation} a gray box"
    for count in ("two", "three")
    for color in ("yellow", "green")
    for obj in ("cups", "balls", "bottles")
    for relation in ("beside", "above")
)
STATE_METHODS = (
    "remove_direction",
    "norm_matched",
    "suppress_channel",
    "zero",
    "ordinary_zero",
    "random_direction",
    "donor_swap",
)
EDGE_METHODS = (
    "image_reads_text_score",
    "image_reads_text_value",
    "text_reads_register_score",
    "text_reads_register_value",
    "text_reads_content_score",
    "text_reads_content_value",
)
METHODS = (
    STATE_METHODS + EDGE_METHODS + ("image_remove_direction", "image_zero", "image_ordinary_zero")
)
RESCUES = ("none", "projection", "state", "ordinary_projection", "sham")


@dataclass
class Q9Config:
    model_preset: str = "flux1-dev"
    mode: str = "screen"
    output_dir: str = "/content/q9_work/flux1-dev"
    calibration_dir: str | None = None
    resolution: int = 1024
    dtype: str = "bf16"
    device: str = "cuda"
    offload: bool = False
    calibration_prompts: list[str] = field(default_factory=lambda: list(CALIBRATION_PROMPTS))
    calibration_seeds: list[int] = field(default_factory=lambda: [0, 42])
    prompts: list[str] = field(default_factory=lambda: list(SCREEN_PROMPTS))
    seeds: list[int] = field(default_factory=lambda: [7, 19])
    steps: list[int] = field(default_factory=lambda: [0, 14, 27])
    sites: list[int] = field(default_factory=lambda: [-1, 17, 24, 35])
    # -1 is the output of FLUX's context_embedder, before DiT block 0.
    attention_layers: list[int] = field(default_factory=lambda: [0, 17, 18, 19, 24, 35, 39])
    methods: list[str] = field(default_factory=lambda: list(METHODS))
    candidate_classes: list[str] = field(default_factory=lambda: ["eos", "pad"])
    candidate_source: str = "union"
    text_norm_threshold: float = 3.0
    text_max_candidates: int = 0  # zero means uncapped, never force K candidates
    sink_enrichment: float = 3.0
    sink_min_mass: float = 0.01
    image_norm_threshold: float = 3.0
    image_max_registers: int = 8
    image_channel: int = 154
    reservoir_size: int = 128
    query_chunk: int = 64
    include_empty: bool = True
    rescues: list[str] = field(default_factory=lambda: ["none"])
    rescue_layer: int = 19
    readout_layer: int = 39
    save_images: bool = True
    evaluate_lpips: bool = True
    evaluate_clip: bool = True
    structured_scores: str | None = None
    equivalence_bound: float | None = None

    def validate(self):
        if self.model_preset not in PRESETS:
            raise ValueError(
                "Q9 bidirectional adapter supports FLUX.1-dev/Schnell only; "
                "PixArt has no live DiT text read-back pathway."
            )
        if self.mode not in {"smoke", "discovery", "screen", "confirm"}:
            raise ValueError("unknown Q9 mode")
        if not self.steps or not self.sites or not self.prompts or not self.calibration_prompts:
            raise ValueError("steps/sites/prompts/calibration_prompts cannot be empty")
        if any(t < 0 or t >= PRESETS[self.model_preset]["num_steps"] for t in self.steps):
            raise ValueError("step out of preset range")
        if any(l < -1 or l > 55 for l in self.sites):
            raise ValueError("sites must be -1 (projected T5) or 0..55 with downstream readouts")
        if any(l < 0 or l > 56 for l in self.attention_layers):
            raise ValueError("attention layer out of range")
        if set(self.methods) - set(METHODS) or not self.methods:
            raise ValueError("unknown/empty methods")
        if set(self.rescues) - set(RESCUES) or "none" not in self.rescues:
            raise ValueError("rescues must include none, with valid rescue names")
        if len(self.rescues) > 1 and (
            self.mode != "confirm" or self.rescue_layer <= max(self.sites) or self.rescue_layer > 55
        ):
            raise ValueError("rescue requires confirm mode and a downstream layer <=55")
        if len(self.rescues) > 1 and any(m not in STATE_METHODS for m in self.methods):
            raise ValueError("rescue confirmation must select text-state methods only")
        if set(self.prompts) & set(self.calibration_prompts):
            raise ValueError("evaluation prompts must be held out from calibration")
        if self.candidate_source not in {"norm", "sink", "union", "intersection"}:
            raise ValueError("invalid candidate_source")
        if not self.candidate_classes or set(self.candidate_classes) - {
            "content",
            "eos",
            "pad",
            "special",
        }:
            raise ValueError("invalid candidate classes")
        if min(self.text_norm_threshold, self.image_norm_threshold, self.sink_enrichment) <= 1:
            raise ValueError("candidate enrichment thresholds must exceed one")
        if self.query_chunk < 1 or self.reservoir_size < 2 or self.text_max_candidates < 0:
            raise ValueError("invalid chunk/reservoir/cap")
        if not 0 <= self.sink_min_mass <= 1 or self.image_max_registers < 1:
            raise ValueError("invalid sink mass or image cap")
        if self.resolution < 64 or self.resolution % 16:
            raise ValueError("resolution must be a positive multiple of 16 >=64")
        if not max(self.sites) < self.readout_layer <= 56:
            raise ValueError("readout_layer must follow every intervention site and be <=56")
        if self.equivalence_bound is not None and self.equivalence_bound <= 0:
            raise ValueError("equivalence_bound must be positive and selected before confirmation")
        if (
            self.mode == "confirm"
            and not self.save_images
            and (self.evaluate_clip or self.evaluate_lpips)
        ):
            raise ValueError("Image evaluation requires save_images=True")
        for values in (
            self.steps,
            self.sites,
            self.prompts,
            self.seeds,
            self.calibration_prompts,
            self.calibration_seeds,
            self.methods,
        ):
            if not values or len(values) != len(set(values)):
                raise ValueError("empty/duplicate configuration entries")

    def calibration_identity(self):
        keys = (
            "model_preset",
            "resolution",
            "dtype",
            "calibration_prompts",
            "calibration_seeds",
            "steps",
            "sites",
            "attention_layers",
            "candidate_classes",
            "candidate_source",
            "text_norm_threshold",
            "text_max_candidates",
            "sink_enrichment",
            "sink_min_mass",
            "image_norm_threshold",
            "image_max_registers",
            "image_channel",
            "reservoir_size",
        )
        return fingerprint(
            {k: asdict(self)[k] for k in keys}
            | {"revision": REVISION, "preset": PRESETS[self.model_preset]}
        )


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def token_classes(ids, eos, pad, special=()):
    return np.array(
        [
            "eos" if i == eos else "pad" if i == pad else "special" if i in special else "content"
            for i in ids
        ]
    )


def norm_candidates(x, threshold, cap=0):
    norms = np.linalg.norm(np.asarray(x, dtype=np.float64), axis=-1)
    mask = norms > threshold * np.median(norms)
    count = int(mask.sum())
    if cap and count > cap:
        indices = np.flatnonzero(mask)
        indices = indices[np.argsort(-norms[indices], kind="stable")[:cap]]
        mask[:] = False
        mask[indices] = True
    return mask, count


def select_text(x, classes, cfg, sinks=None):
    norm, _ = norm_candidates(x, cfg.text_norm_threshold)
    sink = np.zeros(len(x), bool) if sinks is None else np.asarray(sinks, bool)
    eligible = np.isin(classes, cfg.candidate_classes)
    selected = {"norm": norm, "sink": sink, "union": norm | sink, "intersection": norm & sink}[
        cfg.candidate_source
    ] & eligible
    count = int(selected.sum())
    if cfg.text_max_candidates and count > cfg.text_max_candidates:
        ix = np.flatnonzero(selected)
        ix = ix[
            np.argsort(-np.linalg.norm(x[ix], axis=-1), kind="stable")[: cfg.text_max_candidates]
        ]
        selected[:] = False
        selected[ix] = True
    return selected, count


def matched_positions(x, selected, classes):
    """Greedy same-class, nearest-norm matching without replacement; fail if impossible."""
    norms = np.linalg.norm(x, axis=-1)
    result = np.zeros(len(x), bool)
    for i in np.flatnonzero(selected):
        pool = np.flatnonzero(~selected & ~result & (classes == classes[i]))
        if not len(pool):
            return None
        result[pool[np.argmin(np.abs(norms[pool] - norms[i]))]] = True
    return result


class Reservoir:
    """Bounded uniform sample of unit vectors; no activation archive."""

    def __init__(self, capacity, seed=0):
        self.capacity, self.seen = capacity, 0
        self.rng = np.random.default_rng(seed)
        self.rows = []

    def add(self, rows):
        for row in np.asarray(rows, dtype=np.float32):
            norm = np.linalg.norm(row)
            if norm == 0:
                continue
            row = row / norm
            self.seen += 1
            if len(self.rows) < self.capacity:
                self.rows.append(row.copy())
            else:
                j = int(self.rng.integers(self.seen))
                if j < self.capacity:
                    self.rows[j] = row.copy()

    def fit(self):
        if not self.rows:
            return None
        x = np.stack(self.rows).astype(np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(x @ x.T)
        v = eigenvectors[:, -1] @ x
        v /= max(np.linalg.norm(v), 1e-12)
        if np.mean(x @ v) < 0:
            v = -v
        return {
            "vector": v.astype(np.float32),
            "energy": float(eigenvalues[-1] / np.trace(x @ x.T)),
            "seen": self.seen,
            "sampled": len(x),
            "channel": int(np.argmax(np.abs(v))),
        }


def state_metrics(x, mask, vector=None, channel=None):
    x = np.asarray(x, np.float32)
    norms = np.linalg.norm(x, axis=-1)
    y = x[mask]
    result = {
        "count": int(mask.sum()),
        "median_norm": float(np.median(norms)),
        "max_norm_ratio": float(norms.max() / max(np.median(norms), 1e-12)),
    }
    if not len(y):
        return result
    yn = np.linalg.norm(y, axis=-1)
    result["selected_norm"] = float(yn.mean())
    if channel is not None:
        result["channel_abs"] = float(np.abs(y[:, channel]).mean())
        result["channel_energy"] = float(np.mean(y[:, channel] ** 2 / np.maximum(yn**2, 1e-20)))
    if vector is not None:
        projection = y @ vector
        result["projection_abs"] = float(np.abs(projection).mean())
        result["alignment_squared"] = float(np.mean(projection**2 / np.maximum(yn**2, 1e-20)))
    return result


def cluster_summary(rows):
    """Prompt-cluster bootstrap; seeds/positions never become independent replicates."""
    groups = {}
    for row in rows:
        key = tuple(
            row[k]
            for k in (
                "condition",
                "site",
                "target_step",
                "rescue",
                "stage",
                "step",
                "layer",
                "stream",
                "population",
                "metric",
            )
        )
        entry = groups.setdefault(key, {}).setdefault(row["prompt_id"], [0.0, 0])
        entry[0] += float(row["delta"])
        entry[1] += 1
    output = []
    for key, prompts in groups.items():
        values = np.array([v[0] / v[1] for v in prompts.values()])
        stats = {
            "mean_delta": float(values.mean()),
            "n_prompts": len(values),
            "n_pairs": sum(v[1] for v in prompts.values()),
            "ci_low": None,
            "ci_high": None,
        }
        if len(values) > 1:
            rng = np.random.default_rng(0)
            draws = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
            stats.update(
                ci_low=float(np.quantile(draws, 0.025)), ci_high=float(np.quantile(draws, 0.975))
            )
        output.append(
            dict(
                zip(
                    (
                        "condition",
                        "site",
                        "target_step",
                        "rescue",
                        "stage",
                        "step",
                        "layer",
                        "stream",
                        "population",
                        "metric",
                    ),
                    key,
                )
            )
            | stats
        )
    return output


def preset_config(model="flux1-dev", mode="screen", output_dir=None):
    cfg = Q9Config(model_preset=model, mode=mode)
    n = PRESETS[model]["num_steps"]
    cfg.steps = sorted({0, n // 2, n - 1})
    cfg.output_dir = output_dir or f"/content/q9_work/{model}"
    if mode == "smoke":
        cfg.calibration_prompts = list(CALIBRATION_PROMPTS[:2])
        cfg.calibration_seeds = [0]
        cfg.prompts, cfg.seeds, cfg.steps, cfg.sites = list(SCREEN_PROMPTS[:2]), [7], [0], [-1, 17]
        cfg.attention_layers = [0, 17, 18, 19]
    if mode == "confirm":
        cfg.prompts, cfg.seeds = list(CONFIRM_PROMPTS), [7, 19, 31]
        cfg.steps = [0]
        # Small predefined contrast set; users freeze sites after separate discovery.
        cfg.methods = ["remove_direction", "norm_matched", "zero", "ordinary_zero"]
        cfg.sites = [17]
        cfg.rescues = ["none", "projection", "state", "ordinary_projection", "sham"]
    cfg.validate()
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export-compact")
    args = parser.parse_args()
    cfg = Q9Config(**json.loads(Path(args.config).read_text()))
    cfg.validate()
    if args.export_compact:
        from .q9_report import export_compact

        export_compact(cfg, Path(args.export_compact))
    elif args.plot:
        from .q9_report import report

        report(cfg)
    else:
        from .q9_runtime import run

        run(cfg)


if __name__ == "__main__":
    main()
