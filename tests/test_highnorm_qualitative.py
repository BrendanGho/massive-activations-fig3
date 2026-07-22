"""CPU test for the qualitative side-by-side map builder (pure, no torch/matplotlib)."""

from __future__ import annotations

import numpy as np
import pytest

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
