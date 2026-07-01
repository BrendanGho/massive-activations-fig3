"""Stage 2 ranking: the mean-then-abs invariant and seeded random draws."""

from __future__ import annotations

import numpy as np

from src.stage2_channel_ranking import channel_scores, derive_rng, rank_channels


def test_score_is_abs_of_mean_not_mean_of_abs():
    # ch0: +/-10 -> mean 0, |mean| 0  (mean-of-abs would be 10 -> WRONG ranking)
    # ch1: constant 1 -> |mean| 1
    # ch2: constant 0.5 -> |mean| 0.5
    stream = np.array(
        [[10.0, 1.0, 0.5], [-10.0, 1.0, 0.5]],
        dtype=np.float64,
    )
    score = channel_scores(stream)
    assert np.allclose(score, [0.0, 1.0, 0.5])

    res = rank_channels(stream, top_k=1, random_k_trials=0, seed=0, prompt_id=0, layer=0)
    assert res.top_idx.tolist() == [1]  # highest |mean| is ch1, NOT ch0
    assert res.bottom_idx.tolist() == [0]  # lowest |mean| is ch0


def test_top_bottom_ordering():
    rng = np.random.default_rng(0)
    stream = rng.normal(size=(50, 16))
    res = rank_channels(stream, top_k=4, random_k_trials=0, seed=0, prompt_id=1, layer=2)
    score = channel_scores(stream)
    # top scores >= every non-top score; bottom scores <= every non-bottom score.
    top_min = score[res.top_idx].min()
    bottom_max = score[res.bottom_idx].max()
    others_top = np.setdiff1d(np.arange(16), res.top_idx)
    others_bottom = np.setdiff1d(np.arange(16), res.bottom_idx)
    assert top_min >= score[others_top].max() - 1e-9
    assert bottom_max <= score[others_bottom].min() + 1e-9
    assert set(res.top_idx).isdisjoint(res.bottom_idx)


def test_random_draw_reproducible_and_no_replacement():
    rng = np.random.default_rng(1)
    stream = rng.normal(size=(30, 64))
    a = rank_channels(stream, top_k=12, random_k_trials=5, seed=0, prompt_id=3, layer=4)
    b = rank_channels(stream, top_k=12, random_k_trials=5, seed=0, prompt_id=3, layer=4)
    assert np.array_equal(a.random_idx, b.random_idx)  # deterministic
    assert a.random_idx.shape == (5, 12)
    for t in range(5):
        assert len(set(a.random_idx[t].tolist())) == 12  # no replacement
        assert a.random_idx[t].min() >= 0 and a.random_idx[t].max() < 64


def test_random_seed_varies_with_context():
    stream = np.random.default_rng(2).normal(size=(30, 64))
    base = rank_channels(stream, 12, 1, seed=0, prompt_id=0, layer=0)
    diff_layer = rank_channels(stream, 12, 1, seed=0, prompt_id=0, layer=1)
    # Different (prompt,layer,trial) context => different derived seed => different draw.
    assert not np.array_equal(base.random_idx, diff_layer.random_idx)


def test_derive_rng_deterministic():
    r1 = derive_rng(0, 1, 2, 3).integers(0, 1000, size=5)
    r2 = derive_rng(0, 1, 2, 3).integers(0, 1000, size=5)
    assert np.array_equal(r1, r2)
