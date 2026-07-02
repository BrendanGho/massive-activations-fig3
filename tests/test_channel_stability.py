"""CPU tests for the channel-stability experiment: spatial maps, stability metrics,
scenario/table assembly, config validation, and torch-free importability.

No model stack required (torch/diffusers/matplotlib are lazy in the driver).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from src.common import spatial
from src.experiments import channel_stability as cs


# --- spatial maps -------------------------------------------------------------


def test_channel_spatial_map_reshape_and_normalize():
    # 4 tokens = 2x2 grid, D=2. Channel 0 = [0,1,2,3] -> row-major [[0,1],[2,3]].
    stream = np.array([[0.0, 9], [1, 9], [2, 9], [3, 9]])
    raw = spatial.channel_spatial_map(stream, 0, 2, 2, normalize=False)
    assert np.array_equal(raw, np.array([[0, 1], [2, 3]], dtype=np.float32))

    norm = spatial.channel_spatial_map(stream, 0, 2, 2, normalize=True)
    assert np.allclose(norm, np.array([[0.0, 1 / 3], [2 / 3, 1.0]]))

    # Constant channel -> all zeros (no div-by-zero).
    const = spatial.channel_spatial_map(stream, 1, 2, 2, normalize=True)
    assert np.array_equal(const, np.zeros((2, 2), dtype=np.float32))


def test_channel_spatial_map_grid_mismatch_raises():
    stream = np.zeros((5, 2))
    with pytest.raises(ValueError):
        spatial.channel_spatial_map(stream, 0, 2, 2)  # 5 != 4


def test_aggregated_topk_heatmap_matches_manual():
    # col0 ascending, col1 descending & symmetric -> normalized sums to 1 per token.
    stream = np.array([[0.0, 3], [1, 2], [2, 1], [3, 0]])
    agg = spatial.aggregated_topk_heatmap(stream, np.array([0, 1]), 2, 2)
    assert agg.shape == (2, 2)
    assert np.allclose(agg, np.ones((2, 2)))


# --- stability metrics --------------------------------------------------------


def test_topk_jaccard():
    assert spatial.topk_jaccard([0, 1, 2], [1, 2, 3]) == pytest.approx(0.5)
    assert spatial.topk_jaccard([], []) == 1.0
    assert spatial.topk_jaccard([0, 1], [2, 3]) == 0.0


def test_pairwise_jaccard_matrix_symmetry_and_diagonal():
    mat = spatial.pairwise_jaccard_matrix([np.array([0, 1]), np.array([1, 2]), np.array([0, 1])])
    assert mat.shape == (3, 3)
    assert np.allclose(np.diag(mat), 1.0)
    assert np.allclose(mat, mat.T)
    assert mat[0, 2] == 1.0  # identical sets
    assert mat[0, 1] == pytest.approx(1 / 3)  # {0,1} vs {1,2}


def test_selection_frequency():
    freq = spatial.selection_frequency([np.array([0, 1]), np.array([1, 2]), np.array([0])], d=4)
    assert np.array_equal(freq, np.array([2, 2, 1, 0]))


# --- scenarios + tables -------------------------------------------------------


def test_build_scenarios_cross_product():
    scs = cs.build_scenarios(["a", "b"], [10, 20])
    assert [(s.scenario_id, s.prompt_id, s.prompt, s.seed) for s in scs] == [
        (0, 0, "a", 10),
        (1, 0, "a", 20),
        (2, 1, "b", 10),
        (3, 1, "b", 20),
    ]
    # scenario_id // len(seeds) recovers the prompt group.
    assert all(s.scenario_id // 2 == s.prompt_id for s in scs)


def test_build_scenario_channel_matrix():
    scs = cs.build_scenarios(["a", "b"], [0])  # 2 scenarios
    top = {0: np.array([2, 0]), 1: np.array([0, 3])}
    scores = {0: np.array([9.0, 8, 7, 6]), 1: np.array([9.0, 8, 7, 6])}
    channels, rank_of, score_of = cs.build_scenario_channel_matrix(scs, top, scores)
    assert channels == [0, 2, 3]  # sorted union
    assert rank_of[0] == {2: 1, 0: 2}
    assert rank_of[1] == {0: 1, 3: 2}
    assert score_of[0][2] == pytest.approx(7.0)


def test_compute_stability_summary_groups_pairs():
    scs = cs.build_scenarios(["p0", "p1"], [0, 1])  # ids 0,1 -> p0 ; 2,3 -> p1
    top = {
        0: np.array([0, 1, 2, 3, 4]),
        1: np.array([0, 9, 8, 7, 6]),  # shares top-1 ch0 with scenario 0
        2: np.array([7, 10, 11, 12, 13]),
        3: np.array([7, 20, 21, 22, 23]),  # shares top-1 ch7 with scenario 2
    }
    summ = cs.compute_stability_summary(scs, top, d=30, n_seeds=2)
    k1 = summ["per_k"]["1"]
    assert k1["mean_jaccard_same_prompt_diff_seed"] == pytest.approx(1.0)
    assert k1["mean_jaccard_diff_prompt"] == pytest.approx(0.0)
    assert k1["n_distinct_channels_used"] == 2  # ch0 and ch7
    assert summ["n_scenarios"] == 4


def test_topk_ordering_via_rank_channels():
    from src.stage2_channel_ranking import rank_channels

    # mean over tokens = [5,1,3,2,4]; abs unchanged -> desc order [0,4,2,3,1].
    means = np.array([5.0, 1, 3, 2, 4])
    stream = np.tile(means, (6, 1))
    res = rank_channels(stream, top_k=5, random_k_trials=0, seed=0, prompt_id=0, layer=11)
    assert list(res.top_idx) == [0, 4, 2, 3, 1]
    assert list(res.top_idx[:1]) == [0]  # top-1
    assert list(res.top_idx[:3]) == [0, 4, 2]  # top-3 prefix


# --- CSV writers --------------------------------------------------------------


def test_write_topk_and_matrix_csv(tmp_path):
    scs = cs.build_scenarios(["a", "b"], [0])
    top = {0: np.array([2, 0]), 1: np.array([0, 3])}
    scores = {0: np.array([9.0, 8, 7, 6]), 1: np.array([9.0, 8, 7, 6])}

    topk_path = str(tmp_path / "topk.csv")
    cs.write_topk_csv(topk_path, scs, top, scores)
    lines = open(topk_path).read().splitlines()
    assert lines[0] == "scenario_id,prompt_id,seed,prompt,rank,channel_id,score"
    assert len(lines) == 1 + 2 * 2  # header + 2 scenarios * top_n(2)

    channels, rank_of, _ = cs.build_scenario_channel_matrix(scs, top, scores)
    mat_path = str(tmp_path / "matrix.csv")
    cs.write_matrix_csv(mat_path, scs, channels, rank_of, str)
    mlines = open(mat_path).read().splitlines()
    assert mlines[0] == "scenario_id,prompt_id,seed,ch0,ch2,ch3"
    # scenario 0 selected ch2(rank1), ch0(rank2), not ch3 -> blank.
    assert mlines[1] == "0,0,0,2,1,"


# --- config -------------------------------------------------------------------


def _write_cfg(tmp_path, **overrides):
    import yaml

    base = {
        "model_ckpt": "black-forest-labs/FLUX.1-dev",
        "output_dir": str(tmp_path / "out"),
        "prompts": ["a cat", "a dog"],
        "seeds": [0, 1],
        "fixed_layer": 11,
    }
    base.update(overrides)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(base))
    return str(path)


def test_load_scenario_config_ok(tmp_path):
    cfg = cs.load_scenario_config(_write_cfg(tmp_path))
    assert cfg.model_ckpt == "black-forest-labs/FLUX.1-dev"
    assert cfg.fixed_layer == 11
    assert cfg.prompts == ["a cat", "a dog"]
    assert len(cs.build_scenarios(cfg.prompts, cfg.seeds)) == 4


def test_load_scenario_config_missing_required_raises(tmp_path):
    import yaml

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"prompts": ["x"], "seeds": [0]}))  # no model_ckpt/output_dir
    with pytest.raises(ValueError):
        cs.load_scenario_config(str(path), create_dirs=False)


def test_load_scenario_config_unknown_key_raises(tmp_path):
    with pytest.raises(ValueError):
        cs.load_scenario_config(_write_cfg(tmp_path, bogus_key=1))


def test_load_scenario_config_empty_prompts_raises(tmp_path):
    with pytest.raises(ValueError):
        cs.load_scenario_config(_write_cfg(tmp_path, prompts=[]))


def test_load_scenario_config_bad_agg_k_raises(tmp_path):
    with pytest.raises(ValueError):
        cs.load_scenario_config(_write_cfg(tmp_path, top_n=20, agg_k=99))


# --- torch-free import --------------------------------------------------------


def test_experiment_modules_import_without_torch():
    for name in ("src.common.spatial", "src.experiments.channel_stability"):
        mod = importlib.import_module(name)
        assert mod is not None
    assert hasattr(cs, "main")
    assert hasattr(cs, "run")
