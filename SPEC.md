# SPEC — Diffusion Activation Studies

## Scope

Make channel identity, token norm decomposition, cross-model comparison, and text/image
stream analysis the primary project interface. Keep the inherited Figure 3 localization
baseline attributed and usable. This refactor adds no experimental evidence or priority claim.
The detailed norm protocol is in `SPEC_highnorm.md`; the baseline guide is in
`docs/localization_baseline.md`.

## Research interpretation

- Compare channel identities independently per prompt/seed/layer; report same-prompt and
  different-prompt comparisons separately.
- Report norm concentration and residual elevation after channel exclusion, alongside nulls.
- Treat qualitative cross-model panels as exploratory and norm exclusion as post-hoc analysis.
- Treat an unexpected localization curve as a possible result, not a failed reproduction.

## Refactor acceptance criteria

- `python -m src.experiments --help` lists all study commands without loading model libraries.
- Each study command forwards its arguments to the existing driver, including driver help;
  missing or unknown commands fail with usage information.
- `localization` writes neutrally named CSV/plot artifacts. The historical stage-4 command
  retains its filenames, and both commands allow an explicit artifact naming override.
- Default evaluation rejects empty, non-finite, out-of-range, or invalid-count summaries;
  it does not require top-k superiority or a particular peak layer or magnitude.
- Historical curve comparisons execute only when `--reference-check` is requested.
- README and Colab introduce the research questions, preserve baseline attribution, and
  describe possible norm-decomposition outcomes without assuming a finding.
- Existing CPU tests and the launcher/evaluation regression tests pass. GPU execution is
  separate validation and is not implied by these tests.

## Inherited baseline contract

The following records the original numerical and cache protocol. Historical curve targets
are optional comparisons; they are not acceptance criteria for the research studies.

## Acceptance criteria (testable)

Automated (CPU, `uv run pytest` — currently green):

- **AC1 — config precedence & fail-loud.** CLI > `FIG3_*` env > YAML; unknown keys and
  any empty required key raise. *(test_fig3_config.py)*
- **AC2 — abs-then-mean ranking.** `score = mean(abs(activations))` over tokens; on data
  where `mean(abs)` and `abs(mean)` disagree, top-/bottom-k follow `mean(abs)`.
  *(test_fig3_ranking.py)*
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
- **AC10 — evaluation artifacts.** Write the results CSV and curve PNG. The historical
  command uses `figure3d_*` filenames; the study launcher uses `localization_*`.
  Historical comparison targets (optional): top-k dominates every layer, bottom-k
  flat ≈ 0.2, random-k between, top-k peak ≈ 0.5 near layer 10. `--reference-check`
  reports differences without treating them as implementation errors.
- **AC11 — qualitative dump.** heatmap+mask PNGs saved for `num_example_prompts` prompts;
  inspect subject coherence across all strategies without requiring a particular outcome.

## Logged ambiguities (paper-unspecified → written to run_metadata.json, not silently picked)

- `random_k_trials` default = 5 (our assumption).
- KMeans init/`n_init` = scikit-learn library defaults; version logged; `random_state`
  derived per (seed, prompt_id, layer, strategy).
- GenAI-Bench split/version = logged as source path + content SHA-256 of the prompt set.

## Historical baseline settings

`num_denoising_steps=4`, `resolution=1024`, `top_k=12`.
