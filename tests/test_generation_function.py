import numpy as np

from src.experiments.generation_function import (
    apply_numpy_intervention,
    fit_vstar,
    frequency_distances,
    in_target,
    natural_register_mask,
    paired_bootstrap,
)


def test_targeting_is_inclusive():
    assert in_target(2, 10, (2, 4), (10, 12))
    assert in_target(4, 12, (2, 4), (10, 12))
    assert not in_target(5, 12, (2, 4), (10, 12))


def test_register_mask_threshold_and_cap():
    x = np.ones((12, 4), np.float32)
    x[:4] *= np.array([20, 15, 10, 5], dtype=np.float32)[:, None]
    mask = natural_register_mask(x, threshold=3, max_registers=2)
    assert np.flatnonzero(mask).tolist() == [0, 1]


def test_fit_vstar_and_remove_projection():
    x = np.array([[10.0, 1.0], [20.0, -1.0], [1.0, 0.0]])
    v = fit_vstar(x[:2])
    assert abs(v[0]) > 0.99
    y = apply_numpy_intervention(x, "remove_vstar", np.array([1, 1, 0], bool), vstar=v)
    assert np.allclose(y[:2] @ v, 0, atol=1e-5)
    assert np.array_equal(y[2], x[2])


def test_channel_and_full_register_removal_are_selective():
    x = np.arange(20, dtype=np.float32).reshape(5, 4)
    mask = np.array([0, 1, 0, 0, 0], bool)
    ch = apply_numpy_intervention(x, "suppress_channel_154", mask, channel=2)
    assert np.all(ch[:, 2] == 0) and np.array_equal(ch[:, [0, 1, 3]], x[:, [0, 1, 3]])
    reg = apply_numpy_intervention(x, "remove_top_registers", mask)
    assert np.all(reg[1] == 0) and np.array_equal(reg[~mask], x[~mask])


def test_norm_only_preserves_direction_and_matches_ordinary_median():
    x = np.array([[10.0, 0.0], [0.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    y = apply_numpy_intervention(x, "norm_only", np.array([1, 0, 0], bool))
    assert np.allclose(y[0] / np.linalg.norm(y[0]), x[0] / np.linalg.norm(x[0]))
    assert np.isclose(np.linalg.norm(y[0]), 3.0)


def test_frequency_split_and_bootstrap():
    clean = np.zeros((32, 32, 3), np.uint8)
    edited = clean.copy()
    edited[:, 16:] = 255
    metrics = frequency_distances(clean, edited, sigma=3)
    assert metrics["low_frequency_rms"] > metrics["high_frequency_rms"]
    stats = paired_bootstrap([1, 2, 3], seed=7, trials=100)
    assert stats["n"] == 3 and stats["ci_low"] <= 2 <= stats["ci_high"]


def test_sink_condition_does_not_edit_residual_reference_operator():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = apply_numpy_intervention(x, "suppress_sink", np.array([0, 1, 0], bool))
    assert np.array_equal(x, y) and y is not x
