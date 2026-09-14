import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from src.experiments import q9_runtime as rt
from src.experiments.text_image_coupling import (
    Q9Config,
    Reservoir,
    cluster_summary,
    matched_positions,
    norm_candidates,
    preset_config,
)


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_presets_and_split_validation():
    for model in ("flux1-dev", "flux-schnell"):
        for mode in ("smoke", "discovery", "screen", "confirm"):
            preset_config(model, mode).validate()
    cfg = preset_config()
    cfg.prompts = cfg.calibration_prompts[:1]
    with pytest.raises(ValueError, match="held out"):
        cfg.validate()
    cfg = preset_config()
    cfg.model_preset = "pixart-sigma"
    with pytest.raises(ValueError, match="read-back"):
        cfg.validate()


def test_registers_can_be_empty_and_caps_are_audited():
    x = np.ones((12, 3))
    assert norm_candidates(x, 3, 8)[0].sum() == 0
    x[:3] *= 10
    mask, count = norm_candidates(x, 3, 2)
    assert count == 3 and mask.sum() == 2


def test_same_class_matching_fails_instead_of_silently_switching_class():
    x = np.arange(12).reshape(4, 3)
    assert (
        matched_positions(x, np.array([1, 0, 0, 0], bool), np.array(["eos", "pad", "pad", "pad"]))
        is None
    )


def test_reservoir_direction_and_calibration_identity():
    r = Reservoir(8)
    r.add(np.tile([0.0, 4.0, 0.0], (100, 1)))
    fit = r.fit()
    assert fit["seen"] == 100 and fit["sampled"] == 8
    np.testing.assert_allclose(fit["vector"], [0, 1, 0], atol=1e-7)
    cfg = preset_config()
    before = cfg.calibration_identity()
    cfg.sink_enrichment = 4
    assert before != cfg.calibration_identity()


@pytest.mark.parametrize("method", ["remove_direction", "norm_matched", "random_direction"])
def test_direction_controls_match_norm_and_preserve_untargeted_states(method):
    x = torch.tensor([[[3.0, 4.0, 0.0], [2.0, 1.0, 8.0]]])
    v = np.array([1.0, 0.0, 0.0])
    y, audit = rt.edit_state(x, [True, False], method, v)
    torch.testing.assert_close(x[:, 1], y[:, 1], rtol=0, atol=0)
    assert y[:, 0].norm().item() == pytest.approx(4)
    assert audit["status"] == "edited"
    if method == "norm_matched":
        torch.testing.assert_close(y[:, 0] / 4, x[:, 0] / 5)
    else:
        assert (y - x).norm().item() == pytest.approx(3, abs=1e-5)


def test_projection_rescue_preserves_orthogonal_components():
    x = torch.tensor([[[9.0, 3.0, 4.0], [1.0, 2.0, 3.0]]])
    y = rt.rescue_state(x, [True, False], [[2.0, 99.0, 99.0]], [1.0, 0.0, 0.0], "projection")
    torch.testing.assert_close(y, torch.tensor([[[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]]]))


def test_attention_reductions_match_dense_and_do_not_renormalize_away_cross_stream_mass():
    torch.manual_seed(1)
    q, k = torch.randn(1, 7, 2, 4), torch.randn(1, 7, 2, 4)
    rows, incoming = rt.attention_reductions(
        q, k, 3, np.array([1, 0, 0, 0], bool), np.array(["content", "eos", "pad"]), 2
    )
    p = (torch.einsum("bqhd,bkhd->bhqk", q, k) / 2).softmax(-1)[0]
    np.testing.assert_allclose(incoming, p[:, 3:, :3].mean(1), atol=1e-7)
    for r in rows:
        assert r["text_mass"] + r["image_mass"] == pytest.approx(1.0, abs=1e-6)


def test_edge_score_and_value_have_distinct_meanings():
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    torch.manual_seed(3)
    q, k, v = [torch.randn(1, 6, 2, 4) for _ in range(3)]
    qi, ki = np.array([2, 4]), np.array([0])
    original = SimpleNamespace()
    masked = rt.edge_attention(q, k, v, qi, ki, "score", original, 1)
    contribution = rt.edge_attention(q, k, v, qi, ki, "value", original, 1)
    dense = (torch.einsum("bqhd,bkhd->bhqk", q[:, qi], k) / 2).softmax(-1)
    expected = torch.einsum("bhqk,bkhd->bqhd", dense[..., :1], v[:, :1]).flatten(2)
    torch.testing.assert_close(contribution, expected)
    score = torch.einsum("bqhd,bkhd->bhqk", q[:, qi], k) / 2
    score[..., 0] = -float("inf")
    expected_masked = torch.einsum("bhqk,bkhd->bqhd", score.softmax(-1), v).flatten(2)
    torch.testing.assert_close(masked, expected_masked, rtol=1e-5, atol=1e-6)
    clean = dispatch_attention_fn(q, k, v).flatten(2)
    patched = rt.patch_attention_output(None, clean, masked, qi, 3, False)
    untouched = [0, 1, 3, 5]
    torch.testing.assert_close(patched[:, untouched], clean[:, untouched], rtol=0, atol=0)


def tiny_flux():
    from diffusers import FluxTransformer2DModel

    from src.common.model_utils import discover_blocks

    torch.manual_seed(123)
    model = FluxTransformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        pooled_projection_dim=8,
        guidance_embeds=True,
        axes_dims_rope=(2, 2, 4),
    ).eval()
    pipe = SimpleNamespace(transformer=model, _execution_device="cpu")
    kw = {
        "hidden_states": torch.randn(1, 5, 4),
        "encoder_hidden_states": torch.randn(1, 4, 12),
        "pooled_projections": torch.randn(1, 8),
        "timestep": torch.tensor([0.5]),
        "guidance": torch.tensor([3.5]),
        "txt_ids": torch.zeros(4, 3),
        "img_ids": torch.zeros(5, 3),
        "return_dict": False,
    }
    cfg = Q9Config(
        sites=[0, 2],
        steps=[0],
        attention_layers=[0, 1, 2, 3],
        rescue_layer=1,
        image_channel=1,
        text_norm_threshold=1.01,
        candidate_classes=["content", "pad", "eos"],
        reservoir_size=8,
    )
    classes = np.array(["content", "eos", "pad", "pad"])
    return pipe, discover_blocks(model), cfg, classes, kw


@pytest.mark.parametrize("site", [0, 2])
@pytest.mark.parametrize(
    "method",
    [
        "remove_direction",
        "norm_matched",
        "zero",
        "image_zero",
        "image_reads_text_score",
        "text_reads_register_score",
        "image_reads_text_value",
        "text_reads_register_value",
    ],
)
def test_real_tiny_flux_interventions_and_audits(site, method):
    pipe, blocks, cfg, classes, kw = tiny_flux()
    clean_prediction = rt.replay(pipe, kw)
    baseline = rt.Q9Hooks(pipe, blocks, cfg, classes)
    with rt.installed(baseline):
        traced_prediction = rt.replay(pipe, kw)
    baseline.validate(1)
    torch.testing.assert_close(traced_prediction, clean_prediction, rtol=0, atol=0)
    clean = baseline.trace
    for frames in (clean.frames, clean.inputs):
        frames[(0, site)]["text_mask"] = np.array([0, 0, 1, 0], bool)
        frames[(0, site)]["image_mask"] = np.array([1, 0, 0, 0, 0], bool)
    vector = np.eye(16, dtype=np.float32)[0]
    bank = {f"{stream}_{site}": {"vector": vector, "channel": 0} for stream in ("text", "image")}
    hooks = rt.Q9Hooks(pipe, blocks, cfg, classes, bank, clean, method, site, 0, forced_step=0)
    with rt.installed(hooks):
        prediction = rt.replay(pipe, kw)
    hooks.validate(1)
    assert hooks.trace.audit[0]["status"] == "edited"
    assert not torch.equal(prediction, clean_prediction)
    assert all(type(b.module.attn.processor).__name__ != "Q9Attention" for b in blocks)


def test_real_tiny_flux_rescue_is_downstream_and_audited():
    pipe, blocks, cfg, classes, kw = tiny_flux()
    cfg.image_norm_threshold = 1.01
    baseline = rt.Q9Hooks(pipe, blocks, cfg, classes)
    with rt.installed(baseline):
        rt.replay(pipe, kw)
    clean = baseline.trace
    clean.frames[(0, 0)]["text_mask"] = np.array([0, 0, 1, 0], bool)
    hooks = rt.Q9Hooks(
        pipe,
        blocks,
        cfg,
        classes,
        reference=clean,
        condition="zero",
        site=0,
        target_step=0,
        rescue="state",
        forced_step=0,
    )
    with rt.installed(hooks):
        rt.replay(pipe, kw)
    hooks.validate(1)
    assert hooks.trace.audit[1]["kind"] == "rescue"
    assert hooks.trace.audit[1]["layer"] == 1
    assert hooks.trace.audit[1]["status"] == "restored"


def test_clusters_use_prompts_not_tokens_or_seeds():
    row = {
        "condition": "zero",
        "site": 0,
        "target_step": 0,
        "rescue": "none",
        "stage": "dit",
        "step": 0,
        "layer": 1,
        "stream": "image",
        "population": "fixed",
        "metric": "channel_energy",
    }
    rows = [row | {"prompt_id": 0, "delta": 1.0}] * 10 + [row | {"prompt_id": 1, "delta": 3.0}]
    result = cluster_summary(rows)[0]
    assert result["mean_delta"] == 2 and result["n_prompts"] == 2
    assert cluster_summary(rows[:1])[0]["ci_low"] is None


def test_real_t5_native_attention_is_observed_without_altering_embeddings(monkeypatch):
    from transformers import T5Config, T5EncoderModel

    torch.manual_seed(1)
    encoder = T5EncoderModel(
        T5Config(
            vocab_size=12, d_model=16, d_kv=4, d_ff=32, num_layers=2, num_heads=2, dropout_rate=0.0
        )
    ).eval()
    ids = torch.tensor([[2, 3, 1] + [0] * 509])

    class Tokenizer:
        eos_token_id, pad_token_id, all_special_ids = 1, 0, [0, 1]

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_ids=ids, attention_mask=(ids != 0).long())

    pipe = SimpleNamespace(tokenizer_2=Tokenizer(), text_encoder_2=encoder)
    with torch.no_grad():
        expected = encoder(input_ids=ids).last_hidden_state

    def encode(*args):
        with torch.no_grad():
            return {
                "prompt_embeds": encoder(input_ids=ids).last_hidden_state,
                "pooled_prompt_embeds": torch.zeros(1, 8),
            }

    monkeypatch.setattr(rt.q7, "_conditioning_for_prompt", encode)
    rows = []
    (out, classes, meta) = rt.conditioning(pipe, Q9Config(), "test", None, {}, rows)
    torch.testing.assert_close(out["prompt_embeds"], expected, atol=0, rtol=0)
    assert meta["encoder_mask"] is None and len(classes) == 512
    assert any("incoming_mass" in r for r in rows)


def test_notebook_is_valid_and_python_cells_compile():
    from pathlib import Path

    nb = json.loads(Path("Q9_Colab.ipynb").read_text())
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "Q9_Colab.ipynb", "exec")


def test_export_excludes_raw_and_enforces_budget(tmp_path):
    from src.experiments.q9_report import export_compact

    cfg = Q9Config(output_dir=str(tmp_path / "work"))
    root = tmp_path / "work" / "runs" / "identity"
    root.mkdir(parents=True)
    rt.write_json(tmp_path / "work" / "latest_screen.json", {"run_root": str(root)})
    rt.write_json(root / "config.json", asdict(cfg))
    rt.write_json(root / "raw_activations.json", {"huge": "not exported"})
    with pytest.raises(RuntimeError, match="exceeding"):
        export_compact(cfg, tmp_path / "export", max_bytes=1)
    export_compact(cfg, tmp_path / "export")
    assert (tmp_path / "export" / "identity" / "config.json").exists()
    assert not (tmp_path / "export" / "identity" / "raw_activations.json").exists()


def test_full_runner_reports_resumes_and_keeps_calibration_separate(tmp_path, monkeypatch):
    from PIL import Image

    pipe, blocks, cfg, classes, template = tiny_flux()
    cfg.mode, cfg.model_preset = "confirm", "flux-schnell"
    cfg.calibration_prompts, cfg.calibration_seeds = ["calibration"], [0]
    cfg.prompts, cfg.seeds, cfg.steps, cfg.sites = ["held-out"], [7], [0], [0]
    cfg.methods, cfg.rescues = ["zero", "remove_direction"], ["none", "state", "sham"]
    cfg.rescue_layer, cfg.output_dir, cfg.include_empty = 1, str(tmp_path / "run"), True
    cfg.evaluate_clip = cfg.evaluate_lpips = False
    cfg.resolution = 64

    class Pipe:
        transformer = pipe.transformer
        _execution_device = "cpu"
        scheduler = SimpleNamespace(
            config={"name": "test"}, timesteps=torch.arange(4), sigmas=torch.arange(5)
        )

        def __call__(self, **kwargs):
            state = torch.randn((1, 5, 4), generator=kwargs["generator"])
            for step in range(4):
                params = dict(
                    template,
                    hidden_states=state,
                    encoder_hidden_states=kwargs["prompt_embeds"],
                    pooled_projections=kwargs["pooled_prompt_embeds"],
                    timestep=torch.tensor([1.0 - step / 4]),
                )
                state = state - self.transformer(**params)[0] / 4
            image = (
                state[0]
                if kwargs["output_type"] == "latent"
                else Image.new("RGB", (64, 64), (int(state.mean() * 10) % 255, 5, 6))
            )
            return SimpleNamespace(images=[image])

    def fake_conditioning(pipe, cfg, prompt, q7cfg, cache, rows):
        if prompt not in cache:
            cache[prompt] = (
                {
                    "prompt_embeds": template["encoder_hidden_states"].clone(),
                    "pooled_prompt_embeds": template["pooled_projections"].clone(),
                },
                classes,
                {"classes": classes.tolist(), "encoder_mask": None},
            )
        return cache[prompt]

    monkeypatch.setattr(rt.q7, "_load_q7_model", lambda cfg: (Pipe(), blocks))
    monkeypatch.setattr(rt, "conditioning", fake_conditioning)
    rt.run(cfg)
    from src.experiments.q9_report import locate

    root = locate(cfg)
    assert (root / "summary.csv").exists()
    assert (root / "paired_metrics.csv.gz").exists()
    assert (root / "figures" / "q9_example_pairs.png").exists()
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest) == 4
    stamps = {p: p.stat().st_mtime_ns for p in root.glob("p*_s*/*.json")}
    rt.run(cfg)
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in stamps.items())


def test_projected_t5_edit_precedes_first_block_and_keeps_pooled_conditioning():
    pipe, blocks, cfg, classes, kw = tiny_flux()
    cfg.sites = [-1]
    base = rt.Q9Hooks(pipe, blocks, cfg, classes)
    with rt.installed(base):
        clean = rt.replay(pipe, kw)
    base.trace.frames[(0, -1)]["text_mask"] = np.array([0, 0, 1, 0], bool)
    pooled = kw["pooled_projections"].clone()
    hooks = rt.Q9Hooks(
        pipe,
        blocks,
        cfg,
        classes,
        reference=base.trace,
        condition="zero",
        site=-1,
        target_step=0,
        forced_step=0,
    )
    with rt.installed(hooks):
        edited = rt.replay(pipe, kw)
    hooks.validate(1)
    assert hooks.trace.audit[0]["layer"] == -1
    assert not torch.equal(clean, edited)
    torch.testing.assert_close(kw["pooled_projections"], pooled, rtol=0, atol=0)


def test_dual_stream_value_removal_preserves_other_queries_and_does_not_subtract_bias():
    projection = torch.nn.Linear(4, 4)
    torch.nn.init.eye_(projection.weight)
    torch.nn.init.constant_(projection.bias, 99)
    attn = SimpleNamespace(to_out=[projection, torch.nn.Identity()], to_add_out=projection)
    image, text = torch.randn(1, 3, 4), torch.randn(1, 2, 4)
    original = image.clone(), text.clone()
    raw = torch.ones(1, 1, 4)
    result = rt.patch_attention_output(attn, original, raw, np.array([1]), 2, True, subtract=True)
    torch.testing.assert_close(result[0], image, rtol=0, atol=0)
    torch.testing.assert_close(result[1][:, 0], text[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(result[1][:, 1], text[:, 1] - 1)


def test_resume_requires_retained_images_in_confirmation(tmp_path):
    cfg = preset_config(mode="confirm")
    rt.write_json(tmp_path / "job.json", {})
    assert not rt.job_complete(tmp_path, "job", cfg)
    cfg.save_images = False
    assert rt.job_complete(tmp_path, "job", cfg)


@pytest.mark.parametrize("method", ["zero", "image_reads_text_score", "text_reads_register_value"])
def test_optimized_probes_preserve_predictions_and_every_readout(method):
    import time

    pipe, blocks, cfg, classes, kw = tiny_flux()
    cfg.image_norm_threshold = 1.01
    cfg.optimize_probes = False
    bank = {
        f"{stream}_{layer}": {"vector": np.eye(16, dtype=np.float32)[0], "channel": 0}
        for stream in ("text", "image")
        for layer in (-1, 0, 1, 2, 3)
    }
    baseline = rt.Q9Hooks(pipe, blocks, cfg, classes, bank)
    with rt.installed(baseline):
        rt.replay(pipe, kw)
    observations, predictions, timings, workloads = [], [], [], []
    for optimized in (False, True):
        cfg.optimize_probes = optimized
        hooks = rt.Q9Hooks(
            pipe, blocks, cfg, classes, bank, baseline.trace, method, 2, 0, forced_step=0
        )
        start = time.perf_counter()
        with rt.installed(hooks):
            predictions.append(rt.replay(pipe, kw))
        timings.append(time.perf_counter() - start)
        hooks.validate(1)
        observations.append(hooks.trace.rows)
        workloads.append(hooks.work)
    torch.testing.assert_close(predictions[0], predictions[1], atol=0, rtol=0)
    assert observations[0] == observations[1]
    assert workloads[1]["attention_computed"] < workloads[0]["attention_computed"]
    assert workloads[1]["observations_computed"] < workloads[0]["observations_computed"]
    print(
        f"Q9 CPU fixture {method}: legacy={timings[0]:.4f}s fast={timings[1]:.4f}s; work={workloads}"
    )


def test_donor_capture_avoids_all_diagnostics_and_preserves_states():
    pipe, blocks, cfg, classes, kw = tiny_flux()
    cfg.sites = [-1, 0, 2]
    baseline = rt.Q9Hooks(pipe, blocks, cfg, classes, forced_step=0)
    with rt.installed(baseline):
        expected = rt.replay(pipe, kw)
    donor = rt.Q9Hooks(pipe, blocks, cfg, classes, forced_step=0, capture_only=True)
    with rt.installed(donor):
        actual = rt.replay(pipe, kw)
    donor.validate(1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not donor.trace.rows and not donor.trace.snapshots and not donor.trace.inputs
    assert donor.work["attention_computed"] == donor.work["observations_computed"] == 0
    for layer in cfg.sites:
        np.testing.assert_array_equal(
            donor.trace.frames[(0, layer)]["text_full"],
            baseline.trace.frames[(0, layer)]["text_full"],
        )


def test_optimized_multistep_trajectory_preserves_downstream_measurements():
    pipe, blocks, cfg, classes, kw = tiny_flux()
    cfg.steps = [0, 1, 2]
    cfg.image_norm_threshold = 1.01

    def trajectory(hooks):
        latent = kw["hidden_states"].clone()
        with rt.installed(hooks):
            for _ in cfg.steps:
                prediction = rt.replay(pipe, kw | {"hidden_states": latent})
                latent = latent - 0.1 * prediction
        hooks.validate(3)
        return latent

    cfg.optimize_probes = False
    baseline = rt.Q9Hooks(pipe, blocks, cfg, classes)
    trajectory(baseline)
    legacy = rt.Q9Hooks(
        pipe,
        blocks,
        cfg,
        classes,
        reference=baseline.trace,
        condition="zero",
        site=2,
        target_step=1,
    )
    expected = trajectory(legacy)
    cfg.optimize_probes = True
    fast = rt.Q9Hooks(
        pipe,
        blocks,
        cfg,
        classes,
        reference=baseline.trace,
        condition="zero",
        site=2,
        target_step=1,
    )
    actual = trajectory(fast)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert fast.trace.rows == legacy.trace.rows
    assert fast.work["attention_reused"] == 7  # four before the step, three before the site
    assert fast.work["attention_computed"] == 5  # one downstream, four at the next step


def test_prefix_reuse_stops_at_intervention_and_does_not_cross_later_steps():
    pipe, blocks, cfg, classes, _kw = tiny_flux()
    h = rt.Q9Hooks(pipe, blocks, cfg, classes, reference=rt.Trace(), site=2, target_step=1)
    h.step = 0
    assert h.clean_prefix(56)
    h.step = 1
    assert h.clean_prefix(1)
    assert not h.clean_prefix(2)
    assert h.clean_prefix(2, before_attention=True)
    assert not h.clean_prefix(3, before_attention=True)
    h.step = 2
    assert not h.clean_prefix(-1)


def test_preflight_marks_missing_controls_and_resume_needs_no_fake_image(tmp_path):
    cfg = preset_config(mode="confirm")
    trace = rt.Trace()
    trace.frames[(0, 17)] = {"text_mask": np.array([True, False]), "text_control": None}
    status = rt.unavailable_job(
        cfg, trace, {}, np.array(["eos", "pad"]), None, "ordinary_zero", 17, 0, "none"
    )
    assert status == "no_matched_control"
    rt.write_json(tmp_path / "job.json", {"execution": "skipped_unavailable"})
    assert rt.job_complete(tmp_path, "job", cfg)
