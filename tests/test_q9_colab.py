"""No pretrained weights or Colab account needed to verify the form workflow."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from src.experiments.q9_colab import build_config, run_budget
from src.experiments.text_image_coupling import preset_config


@pytest.mark.parametrize("model", ["flux1-dev", "flux-schnell"])
@pytest.mark.parametrize("mode", ["smoke", "discovery", "screen", "confirm"])
def test_default_forms_preserve_presets(model, mode):
    assert asdict(build_config(model, mode)) == asdict(preset_config(model, mode))


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
    discovery = run_budget(build_config(mode="discovery"))
    assert discovery["calibration_trajectories_if_uncached"] == 24
    assert discovery["evaluation_pairs"] == discovery["edited_single_forwards"] == 0
    confirm = run_budget(build_config(mode="confirm"))
    assert confirm["edited_full_trajectories"] == 576
    assert confirm["clean_evaluation_trajectories"] == 75
    screen = run_budget(build_config(mode="screen"))
    assert screen["edited_single_forwards"] > 0 and screen["edited_full_trajectories"] == 0


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
def test_notebook_forms_run_without_earlier_sections_or_user_code(tmp_path, advanced):
    import sys
    from types import SimpleNamespace

    nb = json.loads(Path("Q9_Colab.ipynb").read_text(encoding="utf-8"))
    cells = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    commands = []
    namespace = {
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
    exec(cells["q9_config"], namespace)
    if advanced:
        exec(cells["q9_advanced"], namespace)
    exec(cells["q9_run"], namespace)
    assert len(commands) == 1 and namespace["q9_finished"]
    assert "src.experiments.text_image_coupling" in commands[0]
    saved = json.loads((tmp_path / Path(namespace["Q9_CONFIG_PATH"]).name).read_text())
    assert saved == asdict(build_config())

    namespace["Q9_MODE"] = "confirm"
    exec(cells["q9_run"], namespace)
    assert len(commands) == 1 and not namespace["q9_finished"]
    exec(
        cells["q9_run"].replace("Q9_CONFIRM_FULL_RUN = False", "Q9_CONFIRM_FULL_RUN = True"),
        namespace,
    )
    assert len(commands) == 2 and namespace["q9_finished"]

    exec(
        cells["q9_run"].replace("Q9_RUN_EXPERIMENT = True", "Q9_RUN_EXPERIMENT = False"), namespace
    )
    assert len(commands) == 2 and not namespace["q9_finished"]


def test_notebook_selection_resets_old_overrides_and_failed_run_stays_unfinished(tmp_path):
    from types import SimpleNamespace

    nb = json.loads(Path("Q9_Colab.ipynb").read_text(encoding="utf-8"))
    cells = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    namespace = {
        "q9_advanced": {"steps": "999"},
        "q9_finished": True,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_properties=lambda _: SimpleNamespace(total_memory=40 * 2**30),
                is_bf16_supported=lambda: True,
            )
        ),
    }
    exec(cells["q9_config"], namespace)
    assert namespace["q9_advanced"] == {} and not namespace["q9_finished"]
    namespace["q9_advanced"] = {"steps": "999"}
    with pytest.raises(ValueError, match="step out"):
        exec(cells["q9_run"], namespace)
    assert not namespace["q9_finished"]
