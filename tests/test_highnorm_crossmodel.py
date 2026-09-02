"""CPU tests for the cross-model Part 3 figure (config, cache, per-row labels, layout)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import yaml

from src.experiments import highnorm_crossmodel as xm
from src.experiments.highnorm_qualitative import resolve_layers

_ROWS = [
    {
        "key": "flux-schnell",
        "label": "FLUX.1-schnell",
        "model_ckpt": "black-forest-labs/FLUX.1-schnell",
        "target_layer": 18,
        "ablate_channels": [154],
    },
    {
        "key": "pixart-sigma",
        "label": "PixArt-Sigma",
        "model_ckpt": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "target_layer": 27,
        "ablate_channels": [293],
    },
]


def _write_cfg(tmp_path, **over):
    raw = {
        "output_dir": str(tmp_path / "out"),
        "prompt": "a red bicycle leaning against a brick wall",
        "models": _ROWS,
    }
    raw.update(over)
    p = tmp_path / "cfg.yaml"
    with open(p, "w") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
    return str(p)


def _fake_row(key="flux-schnell", label="FLUX.1-schnell", layer=18, channels=(154,), h=8, w=8):
    rng = np.random.default_rng(abs(hash(key)) % 2**31)
    return {
        "key": key,
        "label": label,
        "layer": layer,
        "prompt": "a red bicycle leaning against a brick wall",
        "rgb": rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8),
        "channels": np.asarray(channels, dtype=np.int64),
        "speckle": rng.random((h, w)),
        "n_full": rng.random((h, w)) * 100,
        "n_ex": rng.random((h, w)) * 10,
    }


# --- config -------------------------------------------------------------------


def test_load_config_parses_rows_in_order(tmp_path):
    cfg = xm.load_crossmodel_config(_write_cfg(tmp_path))

    assert [m.key for m in cfg.models] == ["flux-schnell", "pixart-sigma"]
    assert cfg.models[0].ablate_channels == [154]
    assert cfg.models[1].ablate_channels == [293], "each row keeps its OWN channel"
    assert cfg.models[1].target_layer == 27


def test_label_defaults_to_key_and_row_overrides_inherit(tmp_path):
    rows = [{"key": "m1", "model_ckpt": "x/y", "seed": 7}, {"key": "m2", "model_ckpt": "x/z"}]
    cfg = xm.load_crossmodel_config(_write_cfg(tmp_path, models=rows, seed=3, offload=True))

    assert cfg.models[0].label == "m1"
    assert cfg.row_seed(cfg.models[0]) == 7, "per-row seed wins"
    assert cfg.row_seed(cfg.models[1]) == 3, "else inherit the global seed"
    assert cfg.row_offload(cfg.models[1]) is True


@pytest.mark.parametrize(
    "over",
    [
        {"prompt": ""},
        {"models": []},
        {"output_dir": ""},
        {"dtype": "int8"},
        {"models": [dict(_ROWS[0]), dict(_ROWS[0])]},  # duplicate keys share a cache folder
    ],
)
def test_config_rejects_bad_values(tmp_path, over):
    with pytest.raises(ValueError):
        xm.load_crossmodel_config(_write_cfg(tmp_path, **over))


def test_config_rejects_unknown_keys(tmp_path):
    with pytest.raises(ValueError, match="Unknown config keys"):
        xm.load_crossmodel_config(_write_cfg(tmp_path, wobble=1))
    bad = [{"key": "m", "model_ckpt": "x/y", "wobble": 1}]
    with pytest.raises(ValueError, match="Unknown keys in models"):
        xm.load_crossmodel_config(_write_cfg(tmp_path, models=bad))


def test_config_requires_model_ckpt(tmp_path):
    with pytest.raises(ValueError, match="model_ckpt"):
        xm.load_crossmodel_config(_write_cfg(tmp_path, models=[{"key": "m"}]))


def test_select_rows(tmp_path):
    cfg = xm.load_crossmodel_config(_write_cfg(tmp_path))

    assert [m.key for m in xm.select_rows(cfg, None)] == ["flux-schnell", "pixart-sigma"]
    assert [m.key for m in xm.select_rows(cfg, "pixart-sigma")] == ["pixart-sigma"]
    with pytest.raises(ValueError, match="unknown model key"):
        xm.select_rows(cfg, "nope")


def test_shipped_config_is_valid_and_has_the_three_rows(tmp_path):
    """The shipped YAML is a template (output_dir blank, filled in by Colab/CLI)."""
    with open("configs/highnorm_crossmodel.yaml") as fh:
        raw = yaml.safe_load(fh)
    raw["output_dir"] = str(tmp_path / "out")
    p = tmp_path / "shipped.yaml"
    with open(p, "w") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
    cfg = xm.load_crossmodel_config(str(p), create_dirs=False)

    assert [m.key for m in cfg.models] == ["flux-schnell", "flux1-dev", "pixart-sigma"]
    assert [m.ablate_channels for m in cfg.models] == [[154], [154], [293]]
    assert [m.target_layer for m in cfg.models] == [18, 18, 18]


# --- per-row column titles ----------------------------------------------------


def test_row_titles_name_that_rows_channel():
    assert xm.row_titles([154]) == [
        "generated",
        "isolated channel 154",
        "high-norm tokens",
        "high-norm tokens\nchannel 154 ablated",
    ]
    # a different row, a different channel — the titles must follow it
    assert xm.row_titles(np.asarray([293])) == [
        "generated",
        "isolated channel 293",
        "high-norm tokens",
        "high-norm tokens\nchannel 293 ablated",
    ]


def test_row_titles_pluralize_for_several_channels():
    titles = xm.row_titles([154, 1446])
    assert titles[1] == "isolated channels 154,1446"
    assert titles[3].endswith("channels 154,1446 ablated")


# --- cache --------------------------------------------------------------------


def test_cache_round_trip(tmp_path):
    row = _fake_row()
    p = xm.cache_path(str(tmp_path), row["key"], row["layer"])
    xm.save_row(p, row)
    back = xm.load_row(p)

    assert p.endswith(os.path.join("cache", "flux-schnell", "L18.npz"))
    assert back["label"] == "FLUX.1-schnell"
    assert back["layer"] == 18
    assert back["channels"].tolist() == [154]
    for key in ("rgb", "speckle", "n_full", "n_ex"):
        assert np.array_equal(back[key], row[key])


def test_cache_path_separates_models_and_layers(tmp_path):
    a = xm.cache_path(str(tmp_path), "flux-schnell", 18)
    b = xm.cache_path(str(tmp_path), "pixart-sigma", 18)
    c = xm.cache_path(str(tmp_path), "flux-schnell", 27)
    assert len({a, b, c}) == 3


# --- layer sweep --------------------------------------------------------------


def test_sweep_figure_layers_is_the_intersection():
    layers = {"flux-schnell": list(range(57)), "pixart-sigma": list(range(28))}
    common = xm.sweep_figure_layers(layers)

    assert common == list(range(28)), "FLUX-only depth cannot be drawn cross-model"


def test_sweep_figure_layers_empty_inputs():
    assert xm.sweep_figure_layers({}) == []
    assert xm.sweep_figure_layers({"a": [1, 2], "b": [3]}) == []


def test_figure_name():
    assert xm.figure_name(None) == "crossmodel.png"
    assert xm.figure_name(12) == "crossmodel_L12.png"


# --- figure -------------------------------------------------------------------


def test_save_figure_gives_every_row_its_own_titles(tmp_path):
    import matplotlib

    matplotlib.use("Agg")

    rows = [
        _fake_row("flux-schnell", "FLUX.1-schnell", 18, (154,)),
        _fake_row("flux1-dev", "FLUX.1-dev", 18, (154,)),
        _fake_row("pixart-sigma", "PixArt-Sigma", 27, (293,)),
    ]
    out = tmp_path / "crossmodel.png"
    xm._save_figure(str(out), rows, "a red bicycle leaning against a brick wall")

    assert out.is_file() and out.stat().st_size > 0


def test_row_scale_is_per_row_not_shared_across_models():
    """Different models have different D and magnitudes; the scale must not be pooled."""
    a = _fake_row("a", "A", 18, (154,))
    b = _fake_row("b", "B", 27, (293,))
    b["n_full"] = b["n_full"] * 1000.0

    assert xm.shared_norm_scale(a, None)[1] != pytest.approx(xm.shared_norm_scale(b, None)[1])


# --- runner (capture stubbed; no GPU) -----------------------------------------


def _stub_capture(monkeypatch, calls, layers=(18,)):
    """Replace the GPU capture with a fake that records which models it was asked for."""

    def fake(cfg, spec, layers_spec=None):
        calls.append(spec.key)
        want = resolve_layers(layers_spec or spec.layers, list(range(28)), spec.target_layer)
        return {ly: _fake_row(spec.key, spec.label, ly, spec.ablate_channels or [7]) for ly in want}

    monkeypatch.setattr(xm, "capture_model", fake)
    return calls


def test_run_captures_then_assembles_one_figure(tmp_path, monkeypatch):
    cfg = xm.load_crossmodel_config(_write_cfg(tmp_path))
    calls = _stub_capture(monkeypatch, [])

    paths = xm.run(cfg)

    assert calls == ["flux-schnell", "pixart-sigma"]
    assert [os.path.basename(p) for p in paths] == ["crossmodel.png"]
    assert os.path.isfile(xm.cache_path(cfg.output_dir, "flux-schnell", 18))
    assert os.path.isfile(xm.cache_path(cfg.output_dir, "pixart-sigma", 27)), "own target layer"


def test_run_reuses_the_cache_and_refresh_overrides_it(tmp_path, monkeypatch):
    cfg = xm.load_crossmodel_config(_write_cfg(tmp_path))
    calls = _stub_capture(monkeypatch, [])
    xm.run(cfg)
    calls.clear()

    xm.run(cfg)
    assert calls == [], "a cached row must not regenerate"

    xm.run(cfg, refresh=True)
    assert calls == ["flux-schnell", "pixart-sigma"]


def test_run_only_captures_one_model_but_draws_the_cached_rest(tmp_path, monkeypatch):
    """The gated/OOM workflow: fill in one row later, reassemble from cache."""
    cfg = xm.load_crossmodel_config(_write_cfg(tmp_path))
    calls = _stub_capture(monkeypatch, [])

    xm.run(cfg, only="flux-schnell")
    assert calls == ["flux-schnell"]
    first = xm.load_row(xm.cache_path(cfg.output_dir, "flux-schnell", 18))
    assert first["channels"].tolist() == [154]

    calls.clear()
    paths = xm.run(cfg, only="pixart-sigma")
    assert calls == ["pixart-sigma"]
    assert len(paths) == 1, "figure still drawn, now with both rows"


def test_run_sweep_writes_one_figure_per_common_layer(tmp_path, monkeypatch):
    rows = [dict(r, layers="0,5,27") for r in _ROWS[:1]] + [dict(_ROWS[1], layers="0,5")]
    cfg = xm.load_crossmodel_config(_write_cfg(tmp_path, models=rows))
    _stub_capture(monkeypatch, [])

    paths = xm.run(cfg, sweep_layers=True)

    names = sorted(os.path.basename(p) for p in paths)
    assert names == ["crossmodel_L0.png", "crossmodel_L5.png"], "L27 is not in every row"
