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


def test_compute_stability_summary_pair_lists():
    scs = cs.build_scenarios(["p0", "p1", "p2", "p3"], [0, 1, 2])  # 4 prompts x 3 seeds
    top = {sc.scenario_id: np.arange(sc.prompt_id * 5, sc.prompt_id * 5 + 5) for sc in scs}
    summ = cs.compute_stability_summary(scs, top, d=100, n_seeds=3)
    k5 = summ["per_k"]["5"]
    # 4 prompts x C(3,2)=3 same-prompt pairs; C(12,2)=66 total -> 54 diff-prompt pairs.
    assert len(k5["pairs_same_prompt"]) == 12
    assert len(k5["pairs_diff_prompt"]) == 54
    assert k5["mean_jaccard_same_prompt_diff_seed"] == pytest.approx(
        np.mean(k5["pairs_same_prompt"])
    )
    assert k5["mean_jaccard_diff_prompt"] == pytest.approx(np.mean(k5["pairs_diff_prompt"]))


# --- secondary (token-localized) metric ----------------------------------------


def test_channel_scores_max_catches_localized_channel():
    from src.stage2_channel_ranking import channel_scores, channel_scores_max

    rng = np.random.default_rng(0)
    stream = rng.normal(0, 0.1, size=(1000, 8))
    stream[:, 3] += 0.5  # uniformly shifted channel -> wins under mean-abs
    stream[7, 5] = 100.0  # single-token massive channel -> invisible to the mean

    mean_top = int(np.argmax(channel_scores(stream)))
    max_top = int(np.argmax(channel_scores_max(stream, q=1.0)))
    assert mean_top == 3
    assert max_top == 5

    # q=1.0 equals the per-channel abs max exactly.
    assert np.allclose(channel_scores_max(stream, q=1.0), np.abs(stream).max(axis=0))

    with pytest.raises(ValueError):
        channel_scores_max(stream, q=0.0)


def test_rank_channels_secondary_ordering():
    stream = np.zeros((10, 4))
    stream[0, 2] = 50.0
    stream[1, 0] = 10.0
    top_idx, scores = cs.rank_channels_secondary(stream, "max", top_n=4)
    assert list(top_idx[:2]) == [2, 0]
    assert scores.shape == (4,)


def test_compute_metric_agreement():
    scs = cs.build_scenarios(["a"], [0])
    primary = {0: np.arange(20)}
    secondary = {0: np.arange(20)}
    agree = cs.compute_metric_agreement(scs, primary, secondary)
    assert all(v == 1.0 for v in agree.values())
    secondary = {0: np.arange(100, 120)}
    agree = cs.compute_metric_agreement(scs, primary, secondary)
    assert all(v == 0.0 for v in agree.values())


# --- multi-step capture ---------------------------------------------------------


def test_compute_step_consistency():
    scs = cs.build_scenarios(["a", "b"], [0])  # 2 scenarios
    last = {0: np.arange(20), 1: np.arange(20)}
    steps = {
        0: {0: np.arange(20), 24: np.arange(100, 120)},  # step 0 identical, step 24 disjoint
        1: {0: np.arange(20), 24: np.arange(100, 120)},
    }
    out = cs.compute_step_consistency(scs, last, steps)
    assert out["0"]["20"] == pytest.approx(1.0)
    assert out["24"]["20"] == pytest.approx(0.0)


def test_capture_state_step_streams():
    torch = pytest.importorskip("torch")
    from src.common import model_utils

    state = model_utils.CaptureState(capture_steps={0, 2})
    n_image, d = 4, 3

    class Block(torch.nn.Module):
        def forward(self, x):
            return x

    block = Block()
    refs = [model_utils.BlockRef(layer_id=7, module=block, kind="double")]

    class Transformer(torch.nn.Module):
        def forward(self, hidden_states):
            return block(hidden_states)

    tr = Transformer()
    handles = model_utils.register_capture_hooks(tr, refs, state)
    try:
        for step in range(3):
            tr(torch.full((1, n_image, d), float(step)))
    finally:
        for h in handles:
            h.remove()

    # Last-step buffer always holds the final forward; step snapshots only 0 and 2.
    assert np.allclose(state.image_streams[7], 2.0)
    assert set(state.step_streams) == {(0, 7), (2, 7)}
    assert np.allclose(state.step_streams[(0, 7)], 0.0)

    state.reset()
    assert state.capture_steps == {0, 2}  # request survives reset
    assert state.step_streams == {}


# --- figures (Agg backend, synthetic data) --------------------------------------


def _synthetic_run_data(n_prompts=2, n_seeds=2, d=16, top_n=4):
    scs = cs.build_scenarios([f"prompt {i}" for i in range(n_prompts)], list(range(n_seeds)))
    rng = np.random.default_rng(1)
    top = {sc.scenario_id: rng.permutation(d)[:top_n] for sc in scs}
    scores = {sc.scenario_id: rng.uniform(1, 10, size=d) for sc in scs}
    return scs, top, scores


def test_scenario_channel_heatmap_renders(tmp_path):
    pytest.importorskip("matplotlib")
    scs, top, scores = _synthetic_run_data()
    channels, rank_of, _ = cs.build_scenario_channel_matrix(scs, top, scores)
    path = str(tmp_path / "heat.png")
    cs._save_scenario_channel_heatmap(path, scs, channels, rank_of, top_n=4, n_seeds=2)
    assert (tmp_path / "heat.png").stat().st_size > 0


def test_stability_overlap_renders(tmp_path):
    pytest.importorskip("matplotlib")
    from src.common import spatial as sp

    scs, top, _ = _synthetic_run_data()
    summ = cs.compute_stability_summary(scs, top, d=16, n_seeds=2)
    mat = sp.pairwise_jaccard_matrix([top[sc.scenario_id] for sc in scs])
    path = str(tmp_path / "overlap.png")
    cs._save_stability_overlap(path, mat, 4, scs, 2, summ)
    assert (tmp_path / "overlap.png").stat().st_size > 0


def test_step_consistency_figure_renders(tmp_path):
    pytest.importorskip("matplotlib")
    per_step = {"0": {"1": 0.5, "5": 0.6}, "24": {"1": 0.9, "5": 0.8}}
    path = str(tmp_path / "steps.png")
    cs._save_step_consistency(path, per_step, n_steps=50)
    assert (tmp_path / "steps.png").stat().st_size > 0


def test_contact_sheet_renders(tmp_path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(2)
    scs = cs.build_scenarios(["a tiny prompt"], [0])
    rows = [
        {
            "sc": scs[0],
            "rgb": rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8),
            "stream": rng.normal(size=(16, 8)),
            "top_idx": np.arange(8),
            "bottom_idx": np.arange(8)[::-1].copy(),
            "scores": rng.uniform(1, 10, size=8),
        }
    ]
    path = str(tmp_path / "sheet.png")
    cs._save_contact_sheet(path, rows, agg_k=3, h_lat=4, w_lat=4)
    assert (tmp_path / "sheet.png").stat().st_size > 0


def test_aggregate_sweep_ks_always_spans_5_to_20_and_dedupes():
    # Default agg_k=10 with a full top-20 ranking -> [5, 10, 20].
    assert cs._aggregate_sweep_ks(10, 20) == [5, 10, 20]
    # agg_k coincides with a sweep endpoint -> no duplicate.
    assert cs._aggregate_sweep_ks(5, 20) == [5, 20]
    assert cs._aggregate_sweep_ks(20, 20) == [5, 20]
    # Fewer ranked channels than 20 clamps the top of the sweep.
    assert cs._aggregate_sweep_ks(3, 8) == [3, 5]
    # A sub-5 agg_k is still honored when it fits.
    assert cs._aggregate_sweep_ks(3, 4) == [3]
    # Fewer than 5 ranked channels and agg_k out of range falls back to a single panel.
    assert cs._aggregate_sweep_ks(10, 4) == [4]
    assert cs._aggregate_sweep_ks(10, 0) == []


def test_write_outputs_surfaces_figure_errors(tmp_path, monkeypatch):
    scs, top, scores = _synthetic_run_data()
    cfg = cs.ScenarioConfig(
        model_ckpt="x",
        output_dir=str(tmp_path),
        prompts=["prompt 0", "prompt 1"],
        seeds=[0, 1],
        top_n=4,
        agg_k=4,
        secondary_metric=None,
    )

    def boom(*args, **kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(cs, "_save_scenario_channel_heatmap", boom)
    monkeypatch.setattr(cs, "_save_stability_overlap", boom)
    summary = cs._write_outputs(cfg, scs, top, scores, d=16)
    assert set(summary["figure_errors"]) == {"scenario_channel_heatmap", "stability_overlap"}
    assert "render failed" in summary["figure_errors"]["stability_overlap"]


def test_dump_scenario_qualitative_dir_naming(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    rng = np.random.default_rng(3)
    cfg = cs.ScenarioConfig(
        model_ckpt="x",
        output_dir=str(tmp_path),
        prompts=["a"],
        seeds=[7],
        top_n=4,
        agg_k=2,
        n_channel_maps=1,
        n_control_maps=1,
    )
    sc = cs.build_scenarios(cfg.prompts, cfg.seeds)[0]
    stream = rng.normal(size=(16, 8))
    rgb = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
    cs.dump_scenario_qualitative(
        cfg,
        sc,
        stream,
        rgb,
        np.arange(4),
        np.arange(4)[::-1].copy(),
        rng.uniform(1, 10, size=8),
        4,
        4,
    )
    sdir = tmp_path / "scenarios" / "p0_s7"
    assert (sdir / "image.png").is_file()
    assert (sdir / "aggregated_top2.png").is_file()
    assert any(f.name.startswith("control_rank") for f in sdir.iterdir())


# --- new config keys ------------------------------------------------------------


def test_config_capture_steps_and_secondary_metric(tmp_path):
    cfg = cs.load_scenario_config(
        _write_cfg(tmp_path, capture_steps=[0, 24, 49], secondary_metric="max")
    )
    assert cfg.capture_steps == [0, 24, 49]
    assert cfg.secondary_metric == "max"

    with pytest.raises(ValueError):
        cs.load_scenario_config(_write_cfg(tmp_path, capture_steps=[0, 99]))  # out of range
    with pytest.raises(ValueError):
        cs.load_scenario_config(_write_cfg(tmp_path, capture_steps=[0, 1, 2, 3, 4]))  # too many
    with pytest.raises(ValueError):
        cs.load_scenario_config(_write_cfg(tmp_path, secondary_metric="median"))


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
