# SPEC — Reproduce Figure 3 (Massive Activations, FLUX.2-klein)

Reproduce **Figure 3 / Section 3.2** of arXiv:2605.13974 for FLUX.2-klein: four
config-driven, cacheable, resumable stages producing the qualitative heatmaps/masks
(Fig 3A–C) and the quantitative layer-wise mIoU curve (Fig 3D).

## Acceptance criteria (testable)

Automated (CPU, `uv run pytest` — currently green):

- **AC1 — config precedence & fail-loud.** CLI > `FIG3_*` env > YAML; unknown keys and
  any empty required key raise. *(test_fig3_config.py)*
- **AC2 — mean-then-abs ranking.** `score = abs(mean_over_tokens)`; on data where
  `mean(abs)` and `abs(mean)` disagree, top-/bottom-k follow `abs(mean)`. *(test_fig3_ranking.py)*
- **AC3 — top/bottom selection.** top-k = highest scores, bottom-k = lowest, disjoint,
  correct against a brute-force check. *(test_fig3_ranking.py)*
- **AC4 — seeded random-k.** `random_k_trials` draws without replacement, reproducible,
  seed derived from `(seed, prompt_id, layer, trial)` and varying with context.
  *(test_fig3_ranking.py)*
- **AC5 — Stage 3 order.** per-channel min-max first (constant channel → 0), then
  KMeans(2) on k-dim vectors; foreground = higher-mean-`s` cluster; mask/heatmap reshape
  to `H_lat×W_lat`. *(test_fig3_clustering.py)*
- **AC6 — IoU & upsample.** binary IoU (both-empty → 1.0) and nearest-neighbour upsample.
  *(test_fig3_io.py)*
- **AC7 — resumable cache.** reduced `PromptRecord` shard round-trips exactly; completed
  prompts are skipped on restart; batching creates multiple shards. *(test_fig3_io.py)*
- **AC8 — lazy imports.** all four stage modules import with no torch/diffusers/matplotlib
  installed. *(test_fig3_io.py)*

Manual / Colab-only (require GPU + FLUX.2-klein + BiRefNet weights):

- **AC9 — capture correctness.** Stage 1 hooks capture only image-stream tokens at the
  last denoising step; `N_I` derived at runtime; `run_metadata.json` records geometry.
- **AC10 — Fig 3D shape.** top-k dominates every layer; bottom-k flat ≈ 0.2; random-k
  between; top-k peak ≈ 0.5 near layer 10. `figure3d_results.csv` + `figure3d_curve.png`
  produced; `sanity_check` prints warnings (does not assert) on deviation.
- **AC11 — qualitative dump.** heatmap+mask PNGs saved for `num_example_prompts` prompts;
  top-k subject-coherent, bottom-k diffuse.

## Logged ambiguities (paper-unspecified → written to run_metadata.json, not silently picked)

- `random_k_trials` default = 5 (our assumption).
- KMeans init/`n_init` = scikit-learn library defaults; version logged; `random_state`
  derived per (seed, prompt_id, layer, strategy).
- GenAI-Bench split/version = logged as source path + content SHA-256 of the prompt set.

## Fixed (do not sweep)

`num_denoising_steps=4`, `resolution=1024`, `top_k=12`.
