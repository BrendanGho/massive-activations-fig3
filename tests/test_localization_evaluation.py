"""Research outcomes are valid even when the inherited reference ordering is reversed."""

import csv
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import stage4_evaluate_figure3d as evaluation
from src.experiments.__main__ import main as study_main


def unexpected_results():
    return {(42, "top"): (0.1, 0.02, 4), (42, "bottom"): (0.9, 0.02, 4)}


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {(0, "top"): (float("nan"), 0.0, 1)},
        {(0, "top"): (1.1, 0.0, 1)},
        {(0, "top"): (-0.1, 0.0, 1)},
        {(0, "top"): (0.5, float("inf"), 1)},
        {(0, "top"): (0.5, -0.1, 1)},
        {(0, "top"): (0.5, 0.6, 1)},
        {(0, "top"): (0.5, 0.0, 0)},
        {(0, "top"): (0.5, 0.0, 1.5)},
        {(0, "top"): (0.5, 0.0, True)},
    ],
)
def test_invalid_summaries_rejected(bad):
    with pytest.raises(ValueError):
        evaluation.validate_results(bad)


@pytest.mark.parametrize(
    "launcher,prefix", [(evaluation.main, "figure3d"), (study_main, "localization")]
)
@pytest.mark.parametrize("reference", [False, True])
def test_evaluation_names_and_opt_in_reference(monkeypatch, tmp_path, launcher, prefix, reference):
    monkeypatch.setattr(
        evaluation, "load_config", lambda *a: SimpleNamespace(output_dir=str(tmp_path))
    )
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **kw: unexpected_results())
    plot = Mock(return_value="plot.png")
    compare = Mock()
    monkeypatch.setattr(evaluation, "plot_curve", plot)
    monkeypatch.setattr(evaluation, "sanity_check", compare)
    args = ["--config", "unused.yaml"]
    if launcher is study_main:
        args.insert(0, "localization")
    if reference:
        args.append("--reference-check")
    launcher(args)
    with (tmp_path / f"{prefix}_results.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert {r["strategy"]: float(r["mean_miou"]) for r in rows} == {"top": 0.1, "bottom": 0.9}
    plot.assert_called_once_with(unexpected_results(), str(tmp_path), prefix)
    assert compare.call_count == int(reference)


def test_explicit_artifact_style_overrides_launcher_default(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluation, "load_config", lambda *a: SimpleNamespace(output_dir=str(tmp_path))
    )
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **kw: unexpected_results())
    monkeypatch.setattr(evaluation, "plot_curve", Mock(return_value="plot.png"))
    study_main(["localization", "--config", "unused.yaml", "--artifact-prefix=figure3d"])
    assert (tmp_path / "figure3d_results.csv").exists()
    assert not (tmp_path / "localization_results.csv").exists()


def test_invalid_summary_fails_before_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluation, "load_config", lambda *a: SimpleNamespace(output_dir=str(tmp_path))
    )
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **kw: {})
    with pytest.raises(ValueError, match="No localization results"):
        evaluation.main(["--config", "unused.yaml"])
    assert not list(tmp_path.iterdir())


def test_reference_comparison_handles_missing_top_and_random():
    assert evaluation.sanity_check({(0, "bottom"): (0.8, 0.02, 4)})
