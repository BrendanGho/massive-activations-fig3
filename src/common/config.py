"""Config loading with a strict precedence: CLI flag > FIG3_* env var > YAML.

Nothing here imports torch/diffusers, so it is cheap and fully unit-testable.
Required paths/ids fail loudly (this pipeline runs unattended).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import Any

import yaml

ENV_PREFIX = "FIG3_"

# Keys that MUST be provided (blank/None is an error). model_ckpt / birefnet_weights
# may be HF ids rather than local paths, so we check "is set", not "exists on disk";
# stage code resolves local existence when it actually needs a file.
REQUIRED_KEYS = (
    "model_ckpt",
    "prompt_source",
    "birefnet_weights",
    "output_dir",
    "activation_cache_dir",
)


@dataclass
class Config:
    # Required (defaults are None so a missing value is caught by validation).
    model_ckpt: str | None = None
    prompt_source: str | None = None
    birefnet_weights: str | None = None
    output_dir: str | None = None
    activation_cache_dir: str | None = None

    # Fixed experiment knobs.
    num_denoising_steps: int = 4
    resolution: int = 1024
    top_k: int = 12

    # Tunables.
    layers: Any = "all"  # "all" or list[int]
    random_k_trials: int = 5
    seed: int = 0
    device: str = "cuda"
    dtype: str = "bf16"
    batch_size: int = 1
    num_example_prompts: int = 8

    # Storage / IO.
    cache_batch_size: int = 32
    guidance_scale: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# --- coercion helpers ---------------------------------------------------------

_INT_FIELDS = {
    "num_denoising_steps",
    "resolution",
    "top_k",
    "random_k_trials",
    "seed",
    "batch_size",
    "num_example_prompts",
    "cache_batch_size",
}
_STR_FIELDS = {
    "model_ckpt",
    "prompt_source",
    "birefnet_weights",
    "output_dir",
    "activation_cache_dir",
    "device",
    "dtype",
}


def _parse_layers(value: Any) -> Any:
    """Accept 'all', a real list, or a string like '[0, 5, 10]' / '0,5,10'."""
    if value is None:
        return "all"
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, int):
        return [value]
    s = str(value).strip()
    if s.lower() == "all":
        return "all"
    s = s.strip("[]() ")
    if not s:
        return "all"
    return [int(part) for part in s.replace(",", " ").split()]


def _coerce(key: str, value: Any) -> Any:
    """Coerce a raw (possibly-string, from env/CLI) value to the field's type."""
    if value is None:
        return None
    if key == "layers":
        return _parse_layers(value)
    if key == "guidance_scale":
        if isinstance(value, str) and value.strip() == "":
            return None
        return float(value)
    if key in _INT_FIELDS:
        return int(value)
    if key in _STR_FIELDS:
        s = str(value)
        return s if s != "" else None
    return value


def parse_set_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Turn ['top_k=12', 'seed=1'] (from repeated --set) into a dict of raw strings."""
    out: dict[str, Any] = {}
    valid = {f.name for f in fields(Config)}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in valid:
            raise ValueError(f"Unknown config key in --set {item!r}. Valid keys: {sorted(valid)}")
        out[key] = value
    return out


def _env_overrides() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(Config):
        env_key = ENV_PREFIX + f.name.upper()
        if env_key in os.environ:
            out[f.name] = os.environ[env_key]
    return out


def load_config(
    config_path: str,
    cli_overrides: dict[str, Any] | None = None,
    *,
    create_dirs: bool = True,
) -> Config:
    """Load YAML, overlay env (FIG3_*), overlay CLI, coerce, validate.

    Precedence, lowest to highest: YAML file, env vars, CLI overrides.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")

    valid = {f.name for f in fields(Config)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {sorted(unknown)}")

    merged: dict[str, Any] = dict(raw)
    merged.update(_env_overrides())
    merged.update(cli_overrides or {})

    coerced = {key: _coerce(key, val) for key, val in merged.items()}

    cfg = Config(**coerced)
    _validate(cfg, config_path)

    if create_dirs:
        for d in (cfg.output_dir, cfg.activation_cache_dir):
            os.makedirs(d, exist_ok=True)  # type: ignore[arg-type]
    return cfg


def _validate(cfg: Config, source: str) -> None:
    missing = [k for k in REQUIRED_KEYS if not getattr(cfg, k)]
    if missing:
        raise ValueError(
            "Missing required config value(s) "
            f"{missing} (source: {source}). Set them in the YAML, via "
            f"{', '.join(ENV_PREFIX + k.upper() for k in missing)}, or with --set key=value."
        )
    if cfg.top_k <= 0:
        raise ValueError(f"top_k must be positive, got {cfg.top_k}")
    if cfg.num_denoising_steps <= 0:
        raise ValueError(f"num_denoising_steps must be positive, got {cfg.num_denoising_steps}")
    if cfg.random_k_trials < 0:
        raise ValueError(f"random_k_trials must be >= 0, got {cfg.random_k_trials}")
    if cfg.dtype not in ("bf16", "fp16", "fp32"):
        raise ValueError(f"dtype must be one of bf16/fp16/fp32, got {cfg.dtype!r}")
    if cfg.layers != "all" and not isinstance(cfg.layers, list):
        raise ValueError(f"layers must be 'all' or a list, got {cfg.layers!r}")


def config_json(cfg: Config) -> str:
    """Stable JSON dump of the resolved config (for run_metadata / logging)."""
    return json.dumps(cfg.to_dict(), indent=2, sort_keys=True, default=str)
