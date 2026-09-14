"""No pretrained weights or Colab account needed to verify the form workflow."""

# Execute only repository-owned notebook cells with model execution mocked out.
# ruff: noqa: S102

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from src.experiments.q9_colab import build_config, run_budget
from src.experiments.text_image_coupling import preset_config


@pytest.mark.parametrize("model", ["flux1-dev", "flux-schnell"])
@pytest.mark.parametrize("mode", ["smoke", "discovery", "screen", "confirm"])
def test_default_forms_preserve_presets(model, mode):
    assert asdict(build_config(model, mode, workload="full")) == asdict(preset_config(model, mode))


def test_form_overrides_need_no_python_and_stay_local():
    cfg = build_config(
        "flux-schnell",
        "screen",
        "512",
        vram_gib=16,
        bf16=False,
        advanced={
            "sites": "-1, 17",
            "steps": "0,3",
            "seeds": "8,9",
            "prompts": "a blue cup || a red cup",
            "methods": "zero",
            "prompt_count": 1,
        },
    )
    assert cfg.prompts == ["a blue cup"] and cfg.steps == [0, 3]
    assert cfg.offload and cfg.dtype == "fp16" and cfg.resolution == 512
    assert cfg.output_dir.startswith("/content/q9_work/")


@pytest.mark.parametrize(
    "advanced",
    [
        {"sites": "39"},
        {"steps": "999"},
        {"prompt_count": 999},
        {"methods": "typo"},
        {"seeds": "-1"},
        {"output_dir": "/content/drive"},
        {"prompts": "same", "calibration_prompts": "same"},
    ],
)
def test_invalid_form_settings_fail_before_model_load(advanced):
    with pytest.raises(ValueError):
        build_config(advanced=advanced)


def test_budget_distinguishes_discovery_screen_and_confirm():
    discovery = run_budget(build_config(mode="discovery", workload="full"))
    assert discovery["calibration_trajectories_if_uncached"] == 24
    assert discovery["evaluation_pairs"] == discovery["edited_single_forwards"] == 0
    confirm = run_budget(build_config(mode="confirm", workload="full"))
    assert confirm["edited_full_trajectories"] == 576
    assert confirm["clean_evaluation_trajectories"] == 75
    screen = run_budget(build_config(mode="screen"))
    assert screen["edited_single_forwards"] > 0 and screen["edited_full_trajectories"] == 0


@pytest.mark.parametrize("model", ["flux1-dev", "flux-schnell"])
def test_pilot_limits_work_and_preserves_pairing_and_calibration_split(model):
    cfg = build_config(model)
    budget = run_budget(cfg)
    assert len(cfg.prompts) == len(cfg.calibration_prompts) == 3
    assert len(cfg.seeds) == len(cfg.calibration_seeds) == 1
    assert not set(cfg.prompts) & set(cfg.calibration_prompts)
    assert cfg.sites == [17] and cfg.steps == [0]
    assert cfg.methods == ["remove_direction", "norm_matched", "zero", "ordinary_zero"]
    assert cfg.rescues == ["none"] and not cfg.include_empty
    assert budget["edited_single_forwards"] == 12
    assert budget["clean_evaluation_trajectories"] == 3
    assert budget["calibration_trajectories_if_uncached"] == 3
    confirm = build_config(model, "confirm")
    assert not set(cfg.prompts) & set(confirm.prompts)
    assert cfg.calibration_identity() == confirm.calibration_identity()
    assert run_budget(confirm)["edited_full_trajectories"] == 12
    assert run_budget(build_config(model, "discovery"))["calibration_trajectories_if_uncached"] == 3


def test_optional_empty_baseline_can_be_restored_without_changing_preset_default():
    assert not build_config(advanced={"include_empty": "preset"}).include_empty
    assert build_config(advanced={"include_empty": "yes"}).include_empty
    assert build_config(workload="full", advanced={"include_empty": "preset"}).include_empty


def test_q9_cells_are_identical_and_self_contained_in_both_notebooks():
    main, standalone = [
        json.loads(Path(name).read_text(encoding="utf-8"))
        for name in ("Figure3_Colab.ipynb", "Q9_Colab.ipynb")
    ]
    shared = {c["id"]: c for c in standalone["cells"] if c["cell_type"] == "code"}
    for cell in main["cells"]:
        if cell.get("id") in shared:
            assert cell["source"] == shared.pop(cell["id"])["source"]
            compile("".join(cell["source"]), cell["id"], "exec")
    assert not shared
    text = "".join("".join(c["source"]) for c in standalone["cells"])
    assert "edit cfg." not in text
    assert "build_config(" in text and "Q9_CONFIRM_FULL_RUN" in text
    ids = [c.get("id") for c in main["cells"]]
    assert ids.index("q9_setup") < ids.index("XcA1FvnJhZlY")


@pytest.mark.parametrize("advanced", [False, True])
def test_notebook_forms_run_without_earlier_sections_or_user_code(tmp_path, advanced, monkeypatch):
    import sys
    from types import SimpleNamespace

    nb = json.loads(Path("Q9_Colab.ipynb").read_text(encoding="utf-8"))
    cells = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    commands = []
    monkeypatch.setattr("src.experiments.q9_colab.show_results", lambda cfg: "shown")
    namespace = {
        "Q9_REPO_DIR": "/content/massive-activations-fig3",
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_properties=lambda _: SimpleNamespace(total_memory=40 * 2**30),
                is_bf16_supported=lambda: True,
            )
        ),
        "Path": lambda name: tmp_path / Path(name).name,
        "sys": sys,
        "subprocess": SimpleNamespace(run=lambda args, **kwargs: commands.append(args)),
    }
    if advanced:
        exec(cells["q9_advanced"], namespace)
    exec(cells["q9_run"], namespace)
    assert len(commands) == 1 and namespace["q9_finished"]
    assert "src.experiments.text_image_coupling" in commands[0]
    saved = json.loads((tmp_path / Path(namespace["Q9_CONFIG_PATH"]).name).read_text())
    assert saved == asdict(build_config())
    assert namespace["q9_result"] == "shown"
    full_confirm = (
        cells["q9_run"]
        .replace("Q9_MODE = 'screen'", "Q9_MODE = 'confirm'")
        .replace("Q9_WORKLOAD = 'pilot'", "Q9_WORKLOAD = 'full'")
    )
    exec(full_confirm, namespace)
    assert len(commands) == 1 and not namespace["q9_finished"]
    exec(
        full_confirm.replace("Q9_CONFIRM_FULL_RUN = False", "Q9_CONFIRM_FULL_RUN = True"),
        namespace,
    )
    assert len(commands) == 2 and namespace["q9_finished"]

    exec(
        cells["q9_run"].replace("Q9_RUN_EXPERIMENT = True", "Q9_RUN_EXPERIMENT = False"), namespace
    )
    assert len(commands) == 2 and not namespace["q9_finished"]


def test_notebook_failed_run_stays_unfinished():
    from types import SimpleNamespace

    nb = json.loads(Path("Q9_Colab.ipynb").read_text(encoding="utf-8"))
    cells = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    namespace = {
        "Q9_REPO_DIR": "/content/massive-activations-fig3",
        "q9_advanced": {"steps": "999"},
        "q9_finished": True,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_properties=lambda _: SimpleNamespace(total_memory=40 * 2**30),
                is_bf16_supported=lambda: True,
            )
        ),
    }
    with pytest.raises(ValueError, match="step out"):
        exec(cells["q9_run"], namespace)
    assert not namespace["q9_finished"]
