"""Stage 3 clustering: normalization order, foreground = higher-mean cluster."""

from __future__ import annotations

import numpy as np

from src.common.clustering import build_mask, kmeans_foreground, minmax_normalize_channels


def test_minmax_per_channel_and_constant_channel():
    feat = np.array([[0.0, 5.0], [10.0, 5.0], [5.0, 5.0]])
    norm = minmax_normalize_channels(feat)
    # ch0 spans 0..10 -> 0,1,0.5 ; ch1 constant -> all zeros (no div-by-zero).
    assert np.allclose(norm[:, 0], [0.0, 1.0, 0.5])
    assert np.allclose(norm[:, 1], [0.0, 0.0, 0.0])


def test_build_mask_separable_foreground():
    # 4x4 latent grid, 2 channels. Tokens 0..7 (top two rows) are the subject.
    h = w = 4
    n = h * w
    stream = np.zeros((n, 2), dtype=np.float64)
    stream[:8] = 1.0  # high activation = foreground
    stream[8:] = 0.0
    # add tiny noise so kmeans has non-degenerate but clearly separable clusters
    stream += np.random.default_rng(0).normal(scale=1e-3, size=stream.shape)

    res = build_mask(stream, channel_idx=np.array([0, 1]), h_lat=h, w_lat=w, seed=0)
    assert res.mask.shape == (h, w)
    assert res.heatmap.shape == (h, w)
    # Foreground is the high-activation top half.
    assert res.mask[:2].all()
    assert not res.mask[2:].any()
    assert res.n_channels == 2


def test_foreground_is_higher_mean_cluster():
    normalized = np.array([[0.0, 0.0]] * 5 + [[1.0, 1.0]] * 5)
    labels, fg, s = kmeans_foreground(normalized, seed=0)
    # s for the all-ones tokens (sum 2) must exceed the all-zeros tokens (sum 0),
    # and the fg cluster must be the one containing the high-s tokens.
    assert s[5:].mean() > s[:5].mean()
    assert (labels[5:] == fg).all()
    assert (labels[:5] != fg).all()


def test_build_mask_token_count_mismatch_raises():
    stream = np.zeros((15, 2))
    try:
        build_mask(stream, np.array([0, 1]), h_lat=4, w_lat=4, seed=0)
    except ValueError as e:
        assert "token count" in str(e)
    else:
        raise AssertionError("expected ValueError on token/grid mismatch")
