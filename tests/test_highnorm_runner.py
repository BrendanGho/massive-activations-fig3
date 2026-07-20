"""CPU tests for the high-norm experiment runner (SPEC_highnorm.md AC1-AC9 + verdict).

The important tests here are the planted-data ones: they check that `analyze_layer` +
`summarize` actually return H1 on data where one channel owns the outlier norm and H2 on
data where outlier tokens are broadly elevated. If the decision rule can't tell those
apart on synthetic data it cannot be trusted on FLUX.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import yaml

from src.experiments import highnorm_tokens as hn


def _cfg(**kw) -> hn.HighNormConfig:
    base = dict(
        model_ckpt="dummy",
        output_dir="/tmp/unused",
        prompts=["a"],
        base_k=1,
        outlier_frac=0.05,
        rho_ks=[1, 2, 5, 10],
        n_null_trials=5,
    )
    base.update(kw)
    return hn.HighNormConfig(**base)


# --- planted data -------------------------------------------------------------


def _planted_h1(n=400, d=128, n_out=20, seed=0):
    """Outlier tokens are high-norm ONLY because of one massive channel."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    x[:n_out, 0] += 800.0
    return x


def _planted_h2(n=400, d=128, n_out=20, seed=1):
    """Outlier tokens are broadly elevated across all channels, plus a massive channel."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    x[:n_out, :] *= 30.0
    x[:n_out, 0] += 150.0
    return x


def _planted_h3(n=400, d=128, seed=2):
    """Massive channel is massive everywhere (not token-sparse); no outlier structure."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    x[:, 0] *= 60.0
    return x


# --- decision rule ------------------------------------------------------------


def test_verdict_h1_on_single_channel_data():
    row = hn.analyze_layer(_planted_h1(), _cfg(), np.random.default_rng(0))
    assert row["rho_outlier_at_base_k"] > 0.95
    assert row["rho_typical_at_base_k"] < 0.1
    assert row["selectivity"] > hn.SELECTIVITY_MIN, "speckles must be token-sparse"
    assert row["elevation"] == pytest.approx(1.0, abs=0.2), "no residual norm after deconfounding"
    summary = hn.summarize([row])
    assert summary["verdict"] == "H1", summary["reading"]


def test_verdict_h2_on_broadly_elevated_data():
    rows = [
        hn.analyze_layer(_planted_h2(seed=s), _cfg(), np.random.default_rng(s)) for s in range(5)
    ]
    summary = hn.summarize(rows)
    assert summary["median_rho_outlier_at_base_k"] < 0.9
    assert summary["median_elevation"] > hn.ELEVATION_H2
    assert summary["verdict"] == "H2", summary["reading"]


def test_verdict_h3_when_channel_is_not_token_sparse():
    rows = [
        hn.analyze_layer(_planted_h3(seed=s), _cfg(), np.random.default_rng(s)) for s in range(5)
    ]
    summary = hn.summarize(rows)
    assert summary["median_selectivity"] < hn.SELECTIVITY_MIN
    assert summary["verdict"] == "H3", summary["reading"]


def test_rho_alone_cannot_separate_h1_from_h3():
    """Guard on the reason the verdict is not driven by rho.

    H1 (one sparse massive channel) and H3 (a uniformly large channel) both show
    rho ~= 1. Only `selectivity` tells them apart, so if someone later simplifies the
    rule back to a rho threshold, this fails.
    """
    r1 = hn.analyze_layer(_planted_h1(), _cfg(), np.random.default_rng(0))
    r3 = hn.analyze_layer(_planted_h3(), _cfg(), np.random.default_rng(0))
    assert r1["rho_outlier_at_base_k"] > 0.9 and r3["rho_outlier_at_base_k"] > 0.9
    assert r1["selectivity"] > 10 * r3["selectivity"]


def test_nulls_cannot_discriminate_h2_which_is_why_they_do_not_drive_the_verdict():
    """Under H2 a scale-matched random channel reproduces nearly the same overlap.

    Documents why `summarize` uses effect sizes rather than an "IoU beats the null" test:
    that test is dead on arrival for the one hypothesis it would need to detect.
    """
    row = hn.analyze_layer(_planted_h2(), _cfg(), np.random.default_rng(0))
    assert row["iou_deconfounded"] > 0.8
    assert row["null_iou_mean"] > 0.5, "null tracks the signal when tokens are broadly elevated"


# --- the confound the whole spec exists to control ----------------------------


def test_confounded_overlap_is_near_perfect_on_h1_but_deconfounded_is_not():
    """The circular measurement must look like a slam dunk; the honest one must not.

    This is the guard against silently reverting to the full norm: if both numbers ever
    agree on H1 data, the deconfounding has stopped working.
    """
    row = hn.analyze_layer(_planted_h1(), _cfg(), np.random.default_rng(0))
    assert row["iou_confounded"] > 0.9, "full-norm overlap should be circular/near-perfect"
    assert row["iou_deconfounded"] < 0.5, "deconfounded overlap must not inherit the artifact"
    assert row["spearman_m_vs_nfull"] > row["spearman_m_vs_nex"]


def test_nulls_are_reported_and_finite():
    row = hn.analyze_layer(_planted_h2(), _cfg(), np.random.default_rng(0))
    for key in ("null_iou_mean", "null_iou_max", "perm_iou_mean", "null_auroc_mean"):
        assert np.isfinite(row[key]), f"{key} must be finite"
    assert row["perm_iou_mean"] < 0.2, "permuted tokens must not overlap the high-norm set"


def test_analyze_layer_reports_expected_intersection_and_p_value():
    row = hn.analyze_layer(_planted_h2(), _cfg(), np.random.default_rng(0))
    assert 0.0 <= row["p_value"] <= 1.0
    assert row["expected_intersection"] == pytest.approx(20 * 20 / 400)
    assert row["n_tokens"] == 400


# --- outlier mask -------------------------------------------------------------


def test_outlier_mask_finds_planted_tokens():
    mask = hn.outlier_mask(_planted_h1(), base_k=1, outlier_frac=0.05)
    assert mask.sum() == 20
    assert mask[:20].all()


# --- summarize edge cases -----------------------------------------------------


def test_summarize_handles_nan_metrics():
    row = hn.analyze_layer(_planted_h1(), _cfg(), np.random.default_rng(0))
    row["auroc_deconfounded"] = float("nan")
    summary = hn.summarize([row])
    assert summary["n_prompts"] == 1
    assert np.isnan(summary["median_auroc_deconfounded"])
    assert summary["verdict"] in {"H1", "H2", "H3", "inconclusive"}


# --- config -------------------------------------------------------------------


def _write_cfg(tmp_path, **kw):
    raw = dict(model_ckpt="m", output_dir=str(tmp_path / "out"), prompts=["a"])
    raw.update(kw)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return str(path)


def test_config_roundtrip_and_dir_creation(tmp_path):
    cfg = hn.load_highnorm_config(_write_cfg(tmp_path, target_layer=18, base_k=2))
    assert cfg.target_layer == 18 and cfg.base_k == 2
    assert os.path.isdir(cfg.output_dir)


def test_config_rejects_unknown_keys(tmp_path):
    with pytest.raises(ValueError, match="Unknown config keys"):
        hn.load_highnorm_config(_write_cfg(tmp_path, layer=18))


def test_config_rejects_missing_required(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"output_dir": str(tmp_path), "prompts": ["a"]}))
    with pytest.raises(ValueError, match="Missing required"):
        hn.load_highnorm_config(str(path))


@pytest.mark.parametrize(
    "kw",
    [
        {"prompts": []},
        {"base_k": 0},
        {"outlier_frac": 0.0},
        {"outlier_frac": 1.5},
        {"dtype": "int8"},
        {"null_rank_lo": 500, "null_rank_hi": 50},
        {"norm_profile_stride": 0},
    ],
)
def test_config_validation_rejects_bad_values(tmp_path, kw):
    with pytest.raises(ValueError):
        hn.load_highnorm_config(_write_cfg(tmp_path, **kw))


def test_shipped_config_is_valid(tmp_path):
    with open("configs/highnorm_tokens.yaml") as fh:
        raw = yaml.safe_load(fh)
    raw["output_dir"] = str(tmp_path / "out")
    path = tmp_path / "shipped.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = hn.load_highnorm_config(str(path))
    assert cfg.target_layer == 18
    assert cfg.base_k == 2


# --- layer selection ----------------------------------------------------------


def test_profile_layers_always_include_target_first_and_final():
    all_ids = list(range(57))
    sel = hn._profile_layers(all_ids, target=18, stride=3)
    assert 18 in sel and 0 in sel and 56 in sel
    assert sel == sorted(set(sel))
    assert all(s in all_ids for s in sel)


def test_profile_layers_stride_one_keeps_everything():
    all_ids = list(range(20))
    assert hn._profile_layers(all_ids, target=5, stride=1) == all_ids
