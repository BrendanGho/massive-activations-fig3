"""Launch a study without importing its model dependencies until it is selected."""

from __future__ import annotations

import argparse
import importlib
import sys

STUDIES = {
    "stability": ("src.experiments.channel_stability", "Channel identity across generations"),
    "norms": ("src.experiments.highnorm_tokens", "Token norm decomposition and nulls"),
    "norm-panels": ("src.experiments.highnorm_qualitative", "Channel exclusion panels"),
    "cross-model": ("src.experiments.highnorm_crossmodel", "Norm structure across models"),
    "text": ("src.experiments.text_stream_qualitative", "Text-token norms and channel overlap"),
    "localization": ("src.stage4_evaluate_figure3d", "Inherited foreground localization baseline"),
}


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Diffusion Activation Studies",
        epilog="Use <study> --help for that driver's configuration and options.",
    )
    parser.add_argument(
        "study", choices=STUDIES, help="; ".join(f"{k}: {v[1]}" for k, v in STUDIES.items())
    )
    # Parse only the command so --help and every subsequent flag reach the driver.
    selected = parser.parse_args(args[:1]).study
    driver_args = args[1:]
    if selected == "localization":
        # A later explicit user argument takes precedence in argparse.
        driver_args = ["--artifact-prefix", "localization", *driver_args]
    importlib.import_module(STUDIES[selected][0]).main(driver_args)


if __name__ == "__main__":
    main()
