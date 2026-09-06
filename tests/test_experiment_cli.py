"""The study launcher preserves driver arguments and keeps global help lightweight."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.experiments import __main__ as cli


@pytest.mark.parametrize("study", cli.STUDIES)
def test_dispatch_preserves_driver_arguments(monkeypatch, study):
    driver = Mock()
    importer = Mock(return_value=SimpleNamespace(main=driver))
    monkeypatch.setattr(cli.importlib, "import_module", importer)
    args = ["--config", "my study.yaml", "--help"]
    cli.main([study, *args])
    importer.assert_called_once_with(cli.STUDIES[study][0])
    expected = ["--artifact-prefix", "localization", *args] if study == "localization" else args
    driver.assert_called_once_with(expected)


@pytest.mark.parametrize("args,code", [([], 2), (["unknown"], 2), (["--help"], 0)])
def test_usage_without_importing_driver(monkeypatch, capsys, args, code):
    importer = Mock()
    monkeypatch.setattr(cli.importlib, "import_module", importer)
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == code
    importer.assert_not_called()
    captured = capsys.readouterr()
    assert "usage:" in captured.out + captured.err


def test_help_in_fresh_process_does_not_load_model_libraries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import runpy
import sys
sys.argv = ['src.experiments', '--help']
try:
    runpy.run_module('src.experiments', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
assert not {'torch', 'diffusers', 'transformers', 'matplotlib'} & sys.modules.keys()
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert all(study in result.stdout for study in cli.STUDIES)
