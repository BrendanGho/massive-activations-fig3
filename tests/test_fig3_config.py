"""Config precedence (CLI > env > YAML), validation, coercion, layers parsing."""

from __future__ import annotations

import pytest

from src.common.config import load_config, parse_set_overrides


def _write_cfg(tmp_path, **overrides):
    out = tmp_path / "outputs"
    cache = tmp_path / "cache"
    body = {
        "model_ckpt": "black-forest-labs/FLUX.2-klein",
        "prompt_source": str(tmp_path / "prompts.txt"),
        "birefnet_weights": "ZhengPeng7/BiRefNet",
        "output_dir": str(out),
        "activation_cache_dir": str(cache),
        "top_k": 12,
        "seed": 0,
    }
    body.update(overrides)
    lines = []
    for k, v in body.items():
        lines.append(f"{k}: {v}")
    path = tmp_path / "cfg.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_defaults_load(tmp_path):
    cfg = load_config(str(_write_cfg(tmp_path)))
    assert cfg.top_k == 12
    assert cfg.num_denoising_steps == 4
    assert cfg.resolution == 1024
    assert cfg.random_k_trials == 5
    assert cfg.layers == "all"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path, top_k=12)
    monkeypatch.setenv("FIG3_TOP_K", "20")
    monkeypatch.setenv("FIG3_SEED", "7")
    cfg = load_config(str(cfg_path))
    assert cfg.top_k == 20
    assert cfg.seed == 7


def test_cli_beats_env_and_yaml(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path, top_k=12)
    monkeypatch.setenv("FIG3_TOP_K", "20")
    cfg = load_config(str(cfg_path), parse_set_overrides(["top_k=33"]))
    assert cfg.top_k == 33  # CLI wins over env (20) and yaml (12)


def test_missing_required_fails_loudly(tmp_path):
    # Blank required value -> loud error.
    cfg_path = _write_cfg(tmp_path)
    text = cfg_path.read_text().replace(
        "birefnet_weights: ZhengPeng7/BiRefNet", "birefnet_weights:"
    )
    cfg_path.write_text(text)
    with pytest.raises(ValueError, match="birefnet_weights"):
        load_config(str(cfg_path))


def test_unknown_key_rejected(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    cfg_path.write_text(cfg_path.read_text() + "bogus_key: 1\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(str(cfg_path))


@pytest.mark.parametrize(
    "raw,expected",
    [("all", "all"), ("[0, 5, 10]", [0, 5, 10]), ("0,5,10", [0, 5, 10])],
)
def test_layers_parsing(tmp_path, raw, expected):
    cfg = load_config(str(_write_cfg(tmp_path)), parse_set_overrides([f"layers={raw}"]))
    assert cfg.layers == expected


def test_guidance_scale_optional(tmp_path):
    cfg = load_config(str(_write_cfg(tmp_path)))
    assert cfg.guidance_scale is None
    cfg2 = load_config(str(_write_cfg(tmp_path)), parse_set_overrides(["guidance_scale=3.5"]))
    assert cfg2.guidance_scale == pytest.approx(3.5)


def test_bad_dtype_rejected(tmp_path):
    with pytest.raises(ValueError, match="dtype"):
        load_config(str(_write_cfg(tmp_path)), parse_set_overrides(["dtype=int8"]))


def test_set_rejects_unknown_key():
    with pytest.raises(ValueError, match="Unknown config key"):
        parse_set_overrides(["not_a_key=1"])


def test_dirs_created(tmp_path):
    cfg = load_config(str(_write_cfg(tmp_path)))
    import os

    assert os.path.isdir(cfg.output_dir)
    assert os.path.isdir(cfg.activation_cache_dir)
