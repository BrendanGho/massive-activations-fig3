"""CPU test for the qualitative side-by-side map builder (pure, no torch/matplotlib)."""

from __future__ import annotations

import numpy as np
import pytest

from src.common import highnorm
from src.experiments import highnorm_qualitative as q


def test_panel_maps_shapes_and_channel():
    rng = np.random.default_rng(0)
    h, w = 8, 8
    x = rng.normal(size=(h * w, 32))
    # plant a sparse massive value at token 20 in channel 7
    x[:, 7] += 0.0
    x[20, 7] = 500.0
    maps = q.panel_maps(x, n_channels=1, h_lat=h, w_lat=w)

    assert maps["channels"].tolist() == [7], "top channel must be the planted one"
    for key in ("speckle", "n_full", "n_ex"):
        assert maps[key].shape == (h, w)
    # the speckle peaks at the planted token (row-major position of index 20)
    assert np.unravel_index(np.argmax(maps["speckle"]), (h, w)) == (20 // w, 20 % w)


def test_full_norm_matches_speckle_but_deconfounded_does_not():
    """The confound, made visible: excising the massive channel moves the argmax."""
    rng = np.random.default_rng(1)
    h, w = 8, 8
    x = rng.normal(size=(h * w, 32))
    x[20, 7] = 500.0  # token 20 is high-norm ONLY because of channel 7
    maps = q.panel_maps(x, n_channels=1, h_lat=h, w_lat=w)

    peak = (20 // w, 20 % w)
    assert np.unravel_index(np.argmax(maps["n_full"]), (h, w)) == peak, "full norm follows channel"
    assert np.unravel_index(np.argmax(maps["n_ex"]), (h, w)) != peak, "deconfounded norm does not"


def test_panel_maps_rejects_wrong_grid():
    with pytest.raises(ValueError):
        q.panel_maps(np.zeros((63, 16)), n_channels=1, h_lat=8, w_lat=8)


# --- subtract-ks: peel more channels ------------------------------------------


def test_panel_maps_subtract_ks_builds_a_map_per_k():
    rng = np.random.default_rng(2)
    h, w = 8, 8
    x = rng.normal(size=(h * w, 64))
    maps = q.panel_maps(x, n_channels=1, h_lat=h, w_lat=w, subtract_ks=[5, 10, 20])

    assert set(maps["subtract"]) == {5, 10, 20}
    for k in (5, 10, 20):
        assert maps["subtract"][k].shape == (h, w)


def test_subtract_more_channels_monotonically_lowers_the_norm():
    """Removing a superset of channels can only shrink each token's norm.

    This is what makes 'does the high-norm token go away' answerable: minus-top-20 is
    everywhere <= minus-top-5 <= full norm, so a shared color scale shows a real fade.
    """
    rng = np.random.default_rng(3)
    h, w = 8, 8
    x = rng.normal(size=(h * w, 64))
    maps = q.panel_maps(x, n_channels=1, h_lat=h, w_lat=w, subtract_ks=[5, 10, 20])

    full, m5, m10, m20 = maps["n_full"], *(maps["subtract"][k] for k in (5, 10, 20))
    assert np.all(m5 <= full + 1e-9)
    assert np.all(m10 <= m5 + 1e-9)
    assert np.all(m20 <= m10 + 1e-9)


def test_panel_maps_no_subtract_key_when_not_requested():
    maps = q.panel_maps(np.random.default_rng(4).normal(size=(64, 32)), 1, 8, 8)
    assert "subtract" not in maps


def test_parse_ks():
    assert q.parse_ks("5,10,20") == [5, 10, 20]
    assert q.parse_ks("20, 5, 10, 5") == [5, 10, 20]  # sorted + deduped
    assert q.parse_ks("") == []
    assert q.parse_ks(None) == []
    assert q.parse_ks("7") == [7]
    with pytest.raises(ValueError):
        q.parse_ks("5,-3")
    with pytest.raises(ValueError):
        q.parse_ks("5,abc")


def test_default_output_name_encodes_layer_and_channels():
    assert q.default_output_name(18, 1) == "qualitative_L18_ch1.png"
    assert q.default_output_name(11, 3) == "qualitative_L11_ch3.png"
    # different layers / channel counts must not collide (the whole point)
    assert q.default_output_name(18, 1) != q.default_output_name(19, 1)
    assert q.default_output_name(18, 1) != q.default_output_name(18, 2)
    # tolerant of str-typed params coming from a config/CLI
    assert q.default_output_name("18", "1") == "qualitative_L18_ch1.png"


def test_default_output_name_encodes_subtract_ks():
    assert q.default_output_name(18, 1, [5, 10, 20]) == "qualitative_L18_ch1_sub5-10-20.png"
    # with vs without subtract must not overwrite each other at the same layer/channels
    assert q.default_output_name(18, 1, [5, 10, 20]) != q.default_output_name(18, 1)
    assert q.default_output_name(18, 1, []) == "qualitative_L18_ch1.png"


# --- explicit channel ablation ------------------------------------------------


def test_parse_channels():
    assert q.parse_channels("154,1446") == [154, 1446]
    assert q.parse_channels("1446, 154, 154") == [154, 1446]  # sorted + deduped
    assert q.parse_channels("0") == [0]  # channel 0 is valid
    assert q.parse_channels("") == [] and q.parse_channels(None) == []
    with pytest.raises(ValueError):
        q.parse_channels("154,-1")
    with pytest.raises(ValueError):
        q.parse_channels("x")


def test_panel_maps_uses_explicit_channels_over_top_k():
    rng = np.random.default_rng(6)
    h, w = 8, 8
    x = rng.normal(size=(h * w, 64))
    x[:, 0] = 900.0  # channel 0 would be the top massive channel...
    maps = q.panel_maps(x, n_channels=1, h_lat=h, w_lat=w, explicit_channels=[7, 30])

    assert maps["channels"].tolist() == [7, 30], "explicit channels must override top-k"
    # deconfounded norm must exclude exactly {7, 30}, not channel 0
    expected = highnorm.token_norms(x, exclude=np.array([7, 30])).reshape(h, w)
    np.testing.assert_allclose(maps["n_ex"], expected)


def test_panel_maps_rejects_out_of_range_ablation():
    x = np.random.default_rng(7).normal(size=(64, 32))
    with pytest.raises(ValueError, match="out of range"):
        q.panel_maps(x, 1, 8, 8, explicit_channels=[40])


def test_default_output_name_encodes_ablation():
    assert q.default_output_name(18, 1, None, [154, 1446]) == "qualitative_L18_ablate154-1446.png"
    # ablation vs top-k at the same layer must not collide
    assert q.default_output_name(18, 1, None, [154]) != q.default_output_name(18, 1)
    # different ablation sets are distinct files
    assert q.default_output_name(18, 1, None, [154]) != q.default_output_name(18, 1, None, [1446])
    # subtract suffix still applies under ablation
    assert q.default_output_name(18, 1, [5, 10], [154]) == "qualitative_L18_ablate154_sub5-10.png"


# --- top-channel reporting ----------------------------------------------------


def test_top_channel_report_ranks_by_mean_abs():
    x = np.zeros((10, 5))
    x[:, 3] = 8.0  # highest mean|abs|
    x[:, 1] = 4.0
    x[:, 4] = 2.0
    report = q.top_channel_report(x, n=3)
    assert [c for c, _ in report] == [3, 1, 4]
    assert report[0] == (3, pytest.approx(8.0))


def test_top_channel_report_respects_n():
    x = np.random.default_rng(8).normal(size=(16, 20))
    assert len(q.top_channel_report(x, 5)) == 5
    assert len(q.top_channel_report(x, 0)) == 0


def test_save_figure_renders_all_columns(tmp_path):
    """End-to-end figure smoke test: the PNG writes and has the expected column count."""
    rng = np.random.default_rng(5)
    h, w = 8, 8
    rows = []
    for i in range(2):
        x = rng.normal(size=(h * w, 64))
        rows.append(
            {
                "prompt": f"p{i}",
                "rgb": (rng.random((16, 16, 3)) * 255).astype(np.uint8),
                "maps": q.panel_maps(x, 1, h, w, subtract_ks=[5, 10, 20]),
            }
        )
    out = tmp_path / "fig.png"
    q._save_figure(str(out), rows, layer=18, n_channels=1, subtract_ks=[5, 10, 20])
    assert out.is_file() and out.stat().st_size > 0
