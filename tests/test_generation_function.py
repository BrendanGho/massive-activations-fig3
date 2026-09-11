import csv
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.experiments.generation_function import (
    EXPERIMENT_REVISION,
    NaturalTrace,
    PixArtSinkSuppressProcessor,
    Q7Config,
    SinkSuppressProcessor,
    _conditioning_for_prompt,
    _dispatch_suppressed_image_attention,
    _infer_image_token_count,
    _suppressed_image_attention,
    _torch_edit,
    add_paired_prompt_deltas,
    apply_numpy_intervention,
    audit_counts,
    calibration_is_compatible,
    config_hash,
    denoising_thirds,
    fit_vstar,
    frequency_distances,
    generate_figures,
    in_target,
    load_structured_scores,
    natural_register_mask,
    paired_bootstrap,
    run_identity,
    run_is_complete,
    selected_grid_cells,
    validate_flux1_layout,
    validate_model_layout,
)


def test_targeting_is_inclusive():
    assert in_target(2, 10, (2, 4), (10, 12))
    assert in_target(4, 12, (2, 4), (10, 12))
    assert not in_target(5, 12, (2, 4), (10, 12))


def test_denoising_thirds_are_exhaustive_for_q7_presets():
    assert denoising_thirds(4) == {"early": (0, 0), "middle": (1, 2), "late": (3, 3)}
    assert denoising_thirds(28) == {
        "early": (0, 8),
        "middle": (9, 18),
        "late": (19, 27),
    }
    assert denoising_thirds(20) == {
        "early": (0, 6),
        "middle": (7, 12),
        "late": (13, 19),
    }


def test_selected_grid_cells_supports_screen_full_and_smoke():
    cfg = Q7Config(
        prompts=("p",),
        grid_cells=(("early", "mid_register"), ("middle", "writer")),
    )
    assert selected_grid_cells(cfg) == [
        ("early", "mid_register"),
        ("middle", "writer"),
    ]
    assert selected_grid_cells(cfg, smoke=True) == [("early", "writer")]
    full = Q7Config(prompts=("p",))
    assert len(selected_grid_cells(full)) == 12


def test_prompt_conditioning_is_cached_for_flux():
    torch = pytest.importorskip("torch")

    class FakePipe:
        _execution_device = "cpu"

        def __init__(self):
            self.calls = 0

        def encode_prompt(self, **_kwargs):
            self.calls += 1
            return torch.ones(1, 2, 3), torch.ones(1, 3), torch.zeros(2, 3)

    pipe = FakePipe()
    cfg = Q7Config(prompts=("p",))
    cache = {}
    first = _conditioning_for_prompt(pipe, cfg, "p", cache)
    second = _conditioning_for_prompt(pipe, cfg, "p", cache)
    assert first is second
    assert pipe.calls == 1
    assert set(first) == {"prompt_embeds", "pooled_prompt_embeds"}


def test_prompt_conditioning_uses_pixart_cfg_contract():
    torch = pytest.importorskip("torch")

    class FakePipe:
        _execution_device = "cpu"

        def encode_prompt(self, **kwargs):
            self.kwargs = kwargs
            return tuple(torch.full((1,), value) for value in range(4))

    pipe = FakePipe()
    cfg = Q7Config(prompts=("p",), model_family="pixart_sigma", guidance_scale=4.5)
    result = _conditioning_for_prompt(pipe, cfg, "p", {})
    assert pipe.kwargs["do_classifier_free_guidance"] is True
    assert pipe.kwargs["clean_caption"] is True
    assert list(result) == [
        "prompt_embeds",
        "prompt_attention_mask",
        "negative_prompt_embeds",
        "negative_prompt_attention_mask",
    ]


def test_completed_run_preflight_requires_matching_calibration_and_files(tmp_path):
    cfg = Q7Config(
        output_dir=str(tmp_path),
        prompts=("p",),
        conditions=("baseline", "remove_vstar"),
        grid_cells=(("early", "writer"),),
    )
    vstar_path = tmp_path / "vstar.npy"
    np.save(vstar_path, np.ones(4, np.float32))
    digest = hashlib.sha256(vstar_path.read_bytes()).hexdigest()
    (tmp_path / "vstar.json").write_text(
        json.dumps(
            {
                "config_hash": config_hash(cfg),
                "experiment_revision": EXPERIMENT_REVISION,
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    assert calibration_is_compatible(cfg)
    clean, edited = tmp_path / "clean.png", tmp_path / "edited.png"
    clean.touch()
    edited.touch()
    row = {
        "prompt_id": 0,
        "prompt": "p",
        "seed": 0,
        "condition": "remove_vstar",
        "phase": "early",
        "zone": "writer",
        "clean_path": str(clean),
        "image_path": str(edited),
        "config_hash": config_hash(cfg),
        "generation_params": {
            "experiment_revision": EXPERIMENT_REVISION,
            "calibration_sha256": digest,
        },
    }
    (tmp_path / "runs.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert run_is_complete(cfg)
    edited.unlink()
    assert not run_is_complete(cfg)


def test_q7_rejects_non_flux1_block_layout():
    blocks = [SimpleNamespace(layer_id=i, kind="block", module=None) for i in range(28)]
    with pytest.raises(RuntimeError, match="57 blocks"):
        validate_flux1_layout(blocks, Q7Config(prompts=("p",)))


def test_q7_accepts_pixart_layout_and_checks_attn1_width():
    attention = SimpleNamespace(to_q=SimpleNamespace(in_features=1152))
    blocks = [
        SimpleNamespace(layer_id=i, kind="double", module=SimpleNamespace(attn1=attention))
        for i in range(28)
    ]
    cfg = Q7Config(
        prompts=("p",),
        model_family="pixart_sigma",
        channel=293,
        num_steps=20,
        phases=denoising_thirds(20),
        zones={"writer": (13, 13), "register": (14, 20)},
    )
    cfg.validate()
    validate_model_layout(blocks, cfg)


def test_pixart_image_token_count_uses_latent_patch_grid():
    torch = pytest.importorskip("torch")
    transformer = SimpleNamespace(config=SimpleNamespace(patch_size=2))
    assert _infer_image_token_count(transformer, torch.zeros(2, 4, 128, 128)) == 4096


def test_torch_intervention_preserves_unconditional_cfg_row():
    torch = pytest.importorskip("torch")
    x = torch.ones(2, 3, 4)
    edited = _torch_edit(x, "suppress_channel", np.zeros(3, bool), None, 2)
    assert torch.equal(edited[0], x[0])
    assert torch.all(edited[1, :, 2] == 0)
    assert torch.equal(edited[1, :, [0, 1, 3]], x[1, :, [0, 1, 3]])


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
    ch = apply_numpy_intervention(x, "suppress_channel", mask, channel=2)
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


def test_generate_figures_from_smoke_metrics(tmp_path):
    pytest.importorskip("matplotlib")
    image_module = pytest.importorskip("PIL.Image")
    conditions = [
        "remove_vstar",
        "suppress_channel",
        "suppress_sink",
        "remove_top_registers",
        "norm_only",
    ]
    clean_path = tmp_path / "clean.png"
    edited_path = tmp_path / "edited.png"
    image_module.fromarray(np.zeros((16, 16, 3), np.uint8)).save(clean_path)
    image_module.fromarray(np.full((16, 16, 3), 32, np.uint8)).save(edited_path)
    fields = [
        "condition",
        "phase",
        "zone",
        "prompt_id",
        "seed",
        "clean_path",
        "image_path",
        "lpips",
        "low_frequency_rms",
        "high_frequency_rms",
        "clip_delta",
        "image_reward_delta",
    ]
    with (tmp_path / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fields)
        writer.writeheader()
        for index, condition in enumerate(conditions, start=1):
            writer.writerow(
                {
                    "condition": condition,
                    "phase": "early",
                    "zone": "writer",
                    "prompt_id": 0,
                    "seed": 0,
                    "clean_path": clean_path,
                    "image_path": edited_path,
                    "lpips": index / 100,
                    "low_frequency_rms": index / 200,
                    "high_frequency_rms": index / 300,
                    "clip_delta": index / 1000,
                    "image_reward_delta": "",
                }
            )
    vstar = np.zeros(8, np.float32)
    vstar[2] = 1
    np.save(tmp_path / "vstar.npy", vstar)
    cfg = Q7Config(
        output_dir=str(tmp_path),
        prompts=("test",),
        channel=2,
        phases={"early": (0, 0)},
        zones={"writer": (0, 0)},
        num_steps=1,
    )
    outputs = generate_figures(cfg)
    assert {path.name for path in outputs} == {
        "q7_causal_map.png",
        "q7_frequency_profile.png",
        "q7_prompt_fidelity.png",
        "q7_vstar_loadings.png",
        "q7_representative_contact_sheet.png",
    }
    assert all(path.stat().st_size > 0 for path in outputs)


def test_sink_condition_does_not_edit_residual_reference_operator():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = apply_numpy_intervention(x, "suppress_sink", np.array([0, 1, 0], bool))
    assert np.array_equal(x, y) and y is not x


def test_attention_sinks_are_traced_separately_from_registers():
    trace = NaturalTrace(3.0, 8)
    trace.masks[(1, 2)] = np.array([True, False, False])
    trace.observe_sinks(1, 2, np.array([2, 1]))
    assert trace.sink_indices[(1, 2)].tolist() == [2, 1]
    assert trace.masks[(1, 2)].tolist() == [True, False, False]


def test_vstar_trace_only_pools_configured_register_layers():
    torch = pytest.importorskip("torch")
    trace = NaturalTrace(3.0, 8, vector_layers={2})
    hidden = torch.ones(1, 4, 2)
    hidden[:, 0] = 10
    trace.observe(0, 1, hidden)
    assert trace.vectors == []
    trace.observe(0, 2, hidden)
    assert len(trace.vectors) == 1


def test_audit_checks_calls_and_actual_target_fires():
    calls = {0: 5, 1: 5, 2: 5}
    fires = {0: 0, 1: 3, 2: 0}
    assert audit_counts(calls, fires, 5, (1, 3), (1, 1))["ok"]
    bad = audit_counts(calls, {**fires, 1: 5}, 5, (1, 3), (1, 1))
    assert not bad["ok"] and "fires" in bad["errors"][0]


def test_run_identity_changes_with_calibration_and_scheduler():
    cfg = Q7Config(prompts=("p",))
    base = run_identity(cfg, "aaa", {"name": "one"})
    assert base != run_identity(cfg, "bbb", {"name": "one"})
    assert base != run_identity(cfg, "aaa", {"name": "two"})


def test_prompt_fidelity_effects_are_edited_minus_clean():
    row = {
        "clip_clean": 0.5,
        "clip_edited": 0.4,
        "image_reward_clean": 1.0,
        "image_reward_edited": 1.25,
    }
    add_paired_prompt_deltas(row)
    assert np.isclose(row["clip_delta"], -0.1)
    assert np.isclose(row["image_reward_delta"], 0.25)


def test_structured_scores_match_exact_run_cells_and_validate_range(tmp_path):
    path = tmp_path / "structured.csv"
    fields = [
        "run_identity",
        "prompt_id",
        "seed",
        "condition",
        "phase",
        "zone",
        "geneval_counting_clean",
        "geneval_counting_edited",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fields)
        writer.writeheader()
        writer.writerow(
            {
                "run_identity": "current",
                "prompt_id": 0,
                "seed": 42,
                "condition": "remove_vstar",
                "phase": "early",
                "zone": "writer",
                "geneval_counting_clean": 1,
                "geneval_counting_edited": 0,
            }
        )
    scores = load_structured_scores(path, "current")
    key = ("current", "0", "42", "remove_vstar", "early", "writer")
    assert scores[key]["geneval_counting_clean"] == 1
    assert scores[key]["geneval_counting_edited"] == 0

    text = path.read_text(encoding="utf-8").replace(",0\n", ",2\n")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        load_structured_scores(path, "current")


def test_sink_suppression_is_per_head():
    torch = pytest.importorskip("torch")
    query = torch.ones(1, 2, 2, 1)
    key = torch.ones(1, 3, 2, 1)
    value = torch.tensor([[[[100.0], [100.0]], [[1.0], [10.0]], [[2.0], [20.0]]]])
    result = _suppressed_image_attention(query, key, value, n_image=2, sinks=[0, 1])
    # Head 0 suppresses image key 0; head 1 suppresses image key 1. Text remains available.
    assert result.shape == (1, 2, 2, 1)
    assert not torch.allclose(result[:, :, 0], result[:, :, 1])


def test_native_sink_suppression_matches_reference_attention():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    query = torch.ones(1, 2, 2, 1)
    key = torch.ones(1, 3, 2, 1)
    value = torch.tensor([[[[100.0], [100.0]], [[1.0], [10.0]], [[2.0], [20.0]]]])
    reference = _suppressed_image_attention(query, key, value, n_image=2, sinks=[0, 1])
    native = _dispatch_suppressed_image_attention(query, key, value, n_image=2, sinks=[0, 1])
    assert torch.allclose(native, reference, atol=1e-5)


def test_sink_processor_preserves_double_stream_text_output_identity():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")

    class FakeAttention:
        heads = 2
        fused_projections = False

        def __init__(self):
            self.to_q = self.to_k = self.to_v = lambda x: x
            self.add_q_proj = self.add_k_proj = self.add_v_proj = lambda x: x
            self.norm_q = self.norm_k = lambda x: x
            self.norm_added_q = self.norm_added_k = lambda x: x
            self.dropout_calls = 0

            def dropout(x):
                self.dropout_calls += 1
                return x

            self.to_out = [lambda x: x, dropout]

    text_sentinel = torch.randn(1, 1, 2)

    def original(_attn, hidden, **_kwargs):
        return torch.zeros_like(hidden), text_sentinel

    trace = NaturalTrace(3.0, 8)
    trace.n_image = 2
    trace.observe_sinks(0, 1, np.array([0, 1]))
    counts, fires = {1: 0}, {1: 0}
    processor = SinkSuppressProcessor(original, trace, 1, (0, 0), (1, 1), counts, fires)
    attention = FakeAttention()
    output = processor(
        attention,
        torch.randn(1, 2, 2),
        encoder_hidden_states=torch.randn(1, 1, 2),
    )
    assert output[1] is text_sentinel
    assert attention.dropout_calls == 1
    assert counts[1] == 1 and fires[1] == 1


def test_pixart_sink_processor_preserves_unconditional_cfg_row():
    torch = pytest.importorskip("torch")
    attention_processor = pytest.importorskip("diffusers.models.attention_processor")

    class FakePixArtAttention:
        heads = 2
        spatial_norm = None
        group_norm = None
        norm_q = None
        norm_k = None
        residual_connection = False
        rescale_output_factor = 1.0

        def __init__(self):
            self.to_q = torch.nn.Linear(4, 4, bias=False)
            self.to_k = torch.nn.Linear(4, 4, bias=False)
            self.to_v = torch.nn.Linear(4, 4, bias=False)
            self.to_out = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4, bias=False), torch.nn.Dropout(0.0)]
            )
            with torch.no_grad():
                for layer in (self.to_q, self.to_k, self.to_v, self.to_out[0]):
                    layer.weight.copy_(torch.eye(4))

    original = attention_processor.AttnProcessor2_0()
    attention = FakePixArtAttention()
    hidden = torch.tensor(
        [
            [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]],
            [[2.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 2.0, 0]],
        ]
    )
    clean = original(attention, hidden)
    trace = NaturalTrace(3.0, 8)
    trace.n_image = 3
    trace.observe_sinks(0, 4, np.array([0, 1]))
    counts, fires = {4: 0}, {4: 0}
    processor = PixArtSinkSuppressProcessor(
        original, trace, 4, (0, 0), (4, 4), counts, fires, chunk_size=2
    )
    edited = processor(attention, hidden)
    assert torch.equal(edited[0], clean[0])
    assert not torch.allclose(edited[1], clean[1])
    assert counts[4] == 1 and fires[4] == 1
