#!/usr/bin/env bash
# End-to-end Figure 3 pipeline. Resumable: stage1 skips already-cached prompts,
# so a killed job (e.g. a Colab session ending) resumes without recomputation.
#
# Usage: scripts/run_pipeline.sh [CONFIG]
#   CONFIG defaults to configs/default.yaml
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
cd "$(dirname "$0")/.."

echo "[run_pipeline] Stage 1: generate + cache (fused, resumable)"
python -m src.stage1_generate_and_cache --config "$CONFIG" --fused --skip-if-cached

echo "[run_pipeline] Stage 4: evaluate Figure 3D"
python -m src.stage4_evaluate_figure3d --config "$CONFIG"

echo "[run_pipeline] done."
