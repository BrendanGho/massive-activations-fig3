# Diffusion Activation Studies

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BrendanGho/massive-activations-fig3/blob/main/Figure3_Colab.ipynb)

Q9 text/image causal coupling: [![Open Q9 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BrendanGho/massive-activations-fig3/blob/main/Q9_Colab.ipynb)

Experiments on how activation magnitude is distributed across channels and tokens in
diffusion transformers: whether channel identities persist across generations, how much
of a token's norm comes from a few channels, and how these patterns vary across models
and text/image streams.


| Study | Question | Evidence produced |
|---|---|---|
| Channel stability | Do top-channel identities depend on prompt, seed, or denoising step? | Pairwise Jaccard, same-prompt vs different-prompt comparisons, rank agreement |
| Norm decomposition | Do outlier tokens remain high-norm after excluding selected channels? | Selectivity, residual elevation, norm fractions, null comparisons |
| Cross-model panels | Does norm concentration recur across architectures and layers? | Cached per-model, per-layer panels on a shared prompt and seed |
| Text/image streams | Which text positions have high norms, and are dominant channels shared with images? | Token-position profiles and within-model channel overlap |
| Localization baseline | How do selected channels align with foreground pseudo-labels? | Per-layer top/bottom/random-k mIoU |
| Q7 generation function | What causal role do registers, channel 154, and sink routing play during generation? | Same-seed interventions, perceptual/frequency effects, prompt-fidelity deltas, causal maps |
| Q9 text/image coupling | Do text states seed, maintain, or read back from image registers? | T5/DiT discovery, paired forward probes, directed routing lesions, reverse lesions, rescue |

## Install and run

```bash
uv sync                    # development and CPU tests
uv sync --extra fig3       # historical extra name for GPU experiment dependencies
uv run python -m src.experiments --help
```

Set `output_dir` in the chosen study YAML and review its model, prompts, seeds, layers,
and device settings. Localization additionally requires the paths and model IDs in
`configs/default.yaml`; see the baseline guide. Run commands from the repository root.

```bash
uv run python -m src.experiments stability --config configs/channel_stability.yaml
uv run python -m src.experiments norms --config configs/highnorm_tokens.yaml
uv run python -m src.experiments norm-panels --config configs/highnorm_tokens.yaml --subtract-ks 5,10,20
uv run python -m src.experiments cross-model --config configs/highnorm_crossmodel.yaml
uv run python -m src.experiments text --config configs/highnorm_tokens.yaml --layers all
```

Each command accepts its existing driver's arguments; use `<study> --help` for details.
The original module commands still work. Localization evaluates an existing reduced cache:

```bash
uv run python -m src.stage1_generate_and_cache --config configs/default.yaml --fused
uv run python -m src.experiments localization --config configs/default.yaml
```

This produces `localization_results.csv` and `localization_curve.png`. Numerical validation
checks finite, bounded summary statistics and sample counts. Comparison with historical
Figure 3 expectations is available explicitly through `--reference-check`.

## Layout and provenance

- `src/experiments/`: study drivers and the shared command launcher.
- `src/common/`: capture hooks, ranking support, spatial utilities, and norm statistics.
- `src/stage1_*` through `src/stage4_*`: inherited cache and localization pipeline.
- `configs/`: explicit settings for each experiment family.
- `SPEC.md`: current project scope and acceptance criteria.
- `SPEC_highnorm.md`: detailed norm-decomposition protocol.
- `docs/localization_baseline.md`: inherited protocol, attribution, and compatibility details.
- `src/harness/`: supporting agent and knowledge-store tooling.

The localization implementation follows the Figure 3 / Section 3.2 protocol attributed in
this repository to *Few Channels Draw The Whole Picture: Revealing Massive Activations in
Diffusion Transformers* (arXiv:2605.13974). The high-norm study builds on the register-token
question attributed to Darcet et al. in `SPEC_highnorm.md` (arXiv:2309.16588).
Shared capture conventions, channel ranking, and baseline methods retain that provenance.

Norm exclusion operates on captured activations. It measures component contributions;
it does not intervene in generation or demonstrate a functional register role. The
cross-model panels are exploratory comparisons, and the configured prompt grids are small.

## Channel identity across generations

`src/experiments/channel_stability.py` + `configs/channel_stability.yaml` (driven by the
Part 2 cells of `Figure3_Colab.ipynb`). Question: *for each individual generation
(prompt + seed), which channels are largest, and do their identities change with what
the model generates?*

**Design.** One fixed transformer block (`fixed_layer: 11`), image-stream tokens only,
primary analysis at the last denoising step. Scenarios are the cross-product of
4 prompts × 3 seeds = 12 generations; channels are ranked **independently per scenario**.
The prompt/seed grid is deliberate — it separates the two comparisons the experiment
exists to make:

- **same prompt, different seed** → generation-to-generation identity jitter
- **different prompt** → content dependence of the top channels

`stability_summary.json` reports mean top-k Jaccard for each split (k ∈ {1, 5, 10, 20}),
and `stability_overlap.png` shows the block-structured pairwise matrix (diagonal blocks =
same prompt) beside a same-prompt vs diff-prompt pair plot.

**Scores.** Primary = `mean(abs(activations))` over tokens (abs then mean).
Secondary (`secondary_metric: p999`) = 99.9th percentile of
`abs(activation)` over tokens — a token-localized complement that catches channels
massive at only a few tokens, which the mean dilutes. The summary reports per-k
agreement between the two rankings. Part 2 is **channel-space only**; there is no
high-norm *token* analysis here.

**Timesteps.** `capture_steps: [0, 24, 49]` additionally snapshots early/mid/last steps;
`step_consistency.png` reports how much top-channel identity drifts across denoising
(a check that the last-step probe is representative). Costs memory only, no extra GPU time.

**Outputs** (under `output_dir/layer_{fixed_layer}/`):

- `channel_stability_topk.csv` (+ `..._p999.csv`) — ordered top-20 per scenario;
  top-1/5/10 are prefixes.
- `scenario_channel_matrix.csv` / `scenario_channel_scores.csv` — wide scenario × channel
  tables (rank / score).
- `scenario_channel_heatmap.png` — scenario × channel colored by **rank** (comparable
  across scenarios), always-selected channels left of the dashed divider.
- `stability_overlap.png`, `step_consistency.png` — see above.
- `qualitative_summary.png` — contact sheet: one row per representative scenario
  (prompt 0 at all seeds for seed jitter + prompts 1–2 for content): generated image,
  top-1..3 channel spatial maps, a **top-5 / `agg_k` / top-20 aggregate sweep** (deduped;
  the spatial counterpart to the top-k Jaccard sweep in `stability_overlap.png`, showing
  the subject mask tighten or dilute as channels are added), low-rank control map.
- `scenarios/p{pid}_s{seed}/` — the per-scenario loose PNGs behind the contact sheet.
- `stability_summary.json` — all numbers above plus `figure_errors` (any figure that
  failed to render, with traceback).

## Token norm decomposition

`src/experiments/highnorm_tokens.py` + `configs/highnorm_tokens.yaml`, numeric core in
`src/common/highnorm.py`, spec in [`SPEC_highnorm.md`](SPEC_highnorm.md). Question:
isolating the top 1–2 massive channels at layer 18 renders a near-black image with sparse
bright speckles — are those speckle tokens the **high-norm / register tokens** of
Darcet et al. ([arXiv:2309.16588](https://arxiv.org/abs/2309.16588))? Where Part 2 is
channel-space only, this is a **token-space** question.

**Start here — the qualitative look.** `src/experiments/highnorm_qualitative.py` is the
simplest version: no statistics, one row per prompt — `generated` | `isolated top-1 channel`
(the speckles) | `high-norm tokens` (the full L2 norm) | `high-norm tokens, top-1 channel
ablated` (that channel deleted from the norm). The full-norm panel can resemble the
speckles because it includes the selected channel; whether the last panel still lights up
at those spots is the question. `--subtract-ks 5,10,20` adds one further `top-k channels
ablated` column per k, to watch the high-norm token fade (or persist) as more massive channels
are peeled off. The figure is titled `Layer <n>` and carries no per-row prompt label — the
prompts are in the config, and stripping them keeps the panels the same width. All the
norm columns (3 onward: full norm + every ablated column) share **one absolute**
color scale per row (spanning the full-norm range, no clipping) with a colorbar, so a color
means the same token norm in every column and they are directly comparable pixel for pixel.
That is what lets you see whether subtracting the massive channels makes a high-norm token
disappear: a token whose norm is dominated by the ablated channel drops toward background and
renders dark (disappears), while a genuinely elevated token stays bright (persists).
`--report-top 15` prints the top channels (by mean|abs|) per prompt plus a cross-prompt
aggregate, and `--ablate-channels 154,1446` isolates/removes those *explicit* channels
instead of the top-N — read the printed ranking, then ablate the ones you care about.
`--layers all` (or `"0,5,10"`) sweeps layers: every requested block is captured in a
**single** generation pass (the image is generated once, not reloaded/regenerated per
layer), and each layer is written to its own file. Outputs are foldered by channel variant:
`<output_dir>/{ablate_<ids>|top_ch<n>}/qualitative_L<layer>[_sub5-10-20].png`. So a full
layer sweep for one channel set lands in one folder, and each different ablated channel gets
its own folder — built for sweeping all layers × a few channel sets. (Capturing every layer
holds ~`n_layers × N × D` of CPU RAM at peak; pass a layer subset if memory-constrained.)
`python -m src.experiments norm-panels --config configs/highnorm_tokens.yaml --subtract-ks 5,10,20 --report-top 15`

**Cross-model version — one row per model, its own ablated channel.**
`src/experiments/highnorm_crossmodel.py` + `configs/highnorm_crossmodel.yaml` swaps the row
axis from prompts to **models**: same prompt, same seed, one row per model, four columns —
`generated` | `isolated channel C` | `high-norm tokens` | `high-norm tokens, C ablated` — with
`C` set **per model**, because massive-channel ids are per model *and* per layer (FLUX 154,
PixArt-Sigma 293). Every row therefore carries its **own** column titles naming its own
channel; a single header row would mislabel every row but the first. The row label on the
left names the model and the layer it was probed at. The absolute norm scale is **per row, never pooled across rows** — different models have different widths
`D` and different activation magnitudes, so a shared cross-model scale would say nothing.

The figure carries **no suptitle, no colorbar and no layer label** — it's built to drop into a
paper, where the prompt, the color mapping and the layer belong in the caption, so the panels
get that space instead. The layer lives in the **filename**: the run sweeps every layer
(`layers: all`) and writes one figure per layer present in *every* row —
`crossmodel_L0.png`, `crossmodel_L1.png`, … FLUX has 57 blocks and PixArt-Sigma 28, so a full
sweep draws 0–27 and reports the FLUX-only layers it can't pair rather than dropping them
silently. (Hooking every FLUX block holds ~3 GB of CPU RAM at peak — `n_layers × N_I × D`
float32 — so pass `--layers 0,9,18` if memory-constrained.)

Capture and figure are separate stages joined by a cache, because the three models don't fit
in memory together and one of them (FLUX.1-dev) is gated: each model is loaded **alone**,
hooked on every requested layer in a single generation pass (a full-depth sweep costs one
generation, not one per layer), written to `<output_dir>/cache/<key>/L<layer>.npz`, then
freed. A row whose cache is populated is reused as-is, so re-runs cost nothing and the
figures can be reassembled — different channels, different rows — without a GPU. A row
missing from the cache is reported and omitted rather than aborting the figure, so a gated or
OOM model can be filled in later. Flags: `--only pixart-sigma` captures just that row (others
still come from cache), `--refresh` regenerates over the cache, `--channels 154` overrides the
ablated ids for the selected rows, and `--layers` overrides every row's sweep spec. Leave a
row's `ablate_channels: []` to isolate that layer's top-`n_channels` channel instead and have
the titles report which one it picked.
`python -m src.experiments cross-model --config configs/highnorm_crossmodel.yaml`

Model scope: FLUX.1 (schnell/dev) and FLUX.2-klein, plus **PixArt-Sigma** (a DiT that feeds
the transformer a 4D conv latent and uses real classifier-free guidance — the capture hooks
handle both; see `model_utils.register_capture_hooks`). Pick the model in the Colab
`HN_ACTIVE_MODEL` dropdown.

**The confound the quantitative design exists to control.** `‖x‖² = Σ_d x[d]²`, so a token
with a massive value in one channel is high-norm *by construction*. Correlating the
massive-channel score against the full token norm is circular and always returns overlap ≈ 1.
Every statistic is therefore computed against `N_ex` — the norm with the massive channels
**excised**. The confounded number is still reported, but only to show the size of the artifact.

**Verdict rests on two effect sizes**, not on set overlap:

- `selectivity` = median `m`[outlier] / median `m`[typical] — is the channel token-sparse,
  i.e. are there speckles at all?
- `elevation` = median `N_ex`[outlier] / median `N_ex`[typical] — do those tokens stay
  high-norm once the massive channels are removed? *This is the question, numerically.*

**H1** (selectivity ≫ 1, elevation ≈ 1) describes concentrated norm in the selected
channels. **H2** (both ≫ 1) describes residual elevation across other channels.
**H3** (selectivity ≈ 1) describes a uniformly large channel. These are descriptive
signatures; they do not establish a register function or a causal mechanism.

Two seemingly natural measurements are unusable and the tests pin this down: `ρ` (share of
squared norm owned by the massive channels) **cannot separate H1 from H3** — both give ≈ 1 —
and an *"IoU beats the scale-matched null"* test **cannot detect H2**, because when tokens
are broadly elevated any random channel reproduces the overlap. IoU/AUROC and both nulls are
computed and plotted, but do not drive the verdict.

**Outputs** (under `output_dir/`): `summary.json` (verdict + reading + all medians +
logged deviations), `per_prompt.csv`, and `fig_variance_explained.png` (E1 ρ-vs-k curve),
`fig_overlap.png` (E2, deconfounded vs both nulls vs the confounded artifact),
`fig_norm_profile.png` (norm + bimodality across depth, Darcet Fig. 4a analogue),
`fig_norm_hist.png` (final-layer histogram with the derived cutoff),
`fig_spatial_panels.png` (rgb / `m` / `N_full` / `N_ex` side by side),
`fig_position_stability.png` (do speckles sit at fixed grid slots across prompts?).

**Note on the high-norm threshold.** Darcet's 150 is DINOv2-specific and stated to vary by
model, so it is derived per run by 2-means on log-norms. Whether a two-mode reading is even
warranted is checked independently with Sarle's bimodality coefficient (≈ 1/3 = unimodal,
> 5/9 = bimodal) — deliberately not a 2-means split-quality metric, which scores ~0.64 on a
plain Gaussian *and* on a heavy-tailed lognormal and so cannot tell a separated high-norm
mode from a mere long tail.

```bash
python -m src.experiments norms --config configs/highnorm_tokens.yaml
```

## Text and image streams

`src/experiments/text_stream_qualitative.py` (Colab Part 4). Everything above analyzes the
**image** stream; this points the *same channel lens* (rank channels by mean|abs|, per-token
L2 norm, post-hoc "norm minus the massive channels") at the **text** tokens. Text is a 1-D
sequence, so outputs are log-scale per-token-position plots (full norm and the norm after
excluding massive channels, with prompt/EOS/padding identities), not spatial heatmaps. It asks
which text positions have unusually large norms, whether the selected channels account for them,
and whether those channels are **shared with the image stream** (channel-overlap Jaccard, per
prompt/layer). Excluding channels is post-hoc component attribution, not a causal forward-pass
ablation.

**The text source differs by architecture, handled automatically (`--text-source auto`):**
- **FLUX (MMDiT)** — text is a live per-DiT-layer residual stream. Each block returns a
  `(text[512], image[4096])` tuple; we capture `out[0]` (`register_capture_hooks(capture_text=True)`,
  `_extract_text_stream`).
- **PixArt (cross-attn DiT)** — the DiT has **no** text stream (text is a frozen T5 encoding
  used via cross-attention). The real text stream is inside the **T5 encoder**, captured per T5
  layer (`register_text_encoder_hooks`). Slicing image-only DiT outputs would capture image positions, so this analysis uses T5.
  T5 encoder depth and FLUX DiT depth represent different computational stages.

Outputs are foldered by source: `<output_dir>/text_{dit|t5}/text_L<layer>_ch<base_k>.png`.
The numeric core is `src/common/highnorm.py`, reused unchanged; only the capture (text slice /
T5 hook) and the 1-D visualization are new.

```bash
python -m src.experiments text --config configs/highnorm_tokens.yaml --layers all
```

## Q7 generation-level function

Part 5 of the Colab provides a `Q7_ACTIVE_MODEL` dropdown for `flux-schnell`,
`flux1-dev`, and `pixart-sigma`. All model-specific Q7 values come from the notebook's shared `MODEL_PRESETS`:
checkpoint, resolution, schedule, guidance, memory threshold, dominant channel, and depth
zones. Schnell uses four steps without CFG; Dev uses 28 steps with guidance 3.5. The
denoising thirds are derived from the selected preset's step count automatically. PixArt uses
20 steps, guidance 4.5, channel 293, and its shorter block-13-to-20 register interval.

The implementation dispatches to architecture-specific adapters. FLUX suppresses natural sinks
in joint attention while retaining text-query outputs. PixArt suppresses sinks in the image-only
`attn1` self-attention and leaves its separate T5 cross-attention (`attn2`) unchanged. PixArt's
real-CFG batch is `[unconditional, conditional]`; residual and sink edits affect only the last,
conditional row, preserving the unconditional branch exactly. FLUX.2 remains unsupported because
its register zones and causal channel have not been calibrated for this design.

The Colab defaults to `Q7_RUN_MODE="screen"`. Its cross-shaped design tests all three denoising
phases at mid-register depth and all four depth zones during the middle phase (six unique cells),
using one prompt/seed: 32 generations and 31 temporary PNGs rather than the full grid's 558
generations and 549 PNGs. `smoke` runs seven generations solely as an API check; `full` retains
the complete 3-prompt × 3-seed × 3-phase × 4-zone design. Screen mode estimates temporal and
depth main effects but cannot rule out phase-by-depth interactions, so confirm important cells
or use `full` for the final factorial result.

Set `Q7_SCREEN_PROMPTS=3` to screen three different prompts and obtain three contact
sheets (approximately three times the default screen generation workload). Full mode
already includes three prompts; smoke mode still uses one.

Q7's full PNG grid is written to `/content/q7_work`, not Google Drive. Raw residual and attention
traces remain in memory and are never serialized. After evaluation, the notebook optionally
copies only CSV/JSON audit files, the small `vstar` direction, and rendered figures to
`<DRIVE_ROOT>/<model>/q7_compact/<mode>`; the `images/` directory is explicitly excluded. Local
images disappear when the Colab runtime resets, so finish evaluation in the same session. Prompt
conditioning is encoded once per unique prompt, calibration and intervention generation share one
loaded pipeline, clean evaluation features are cached, and a completed scenario is skipped on
resume when its local files still exist.

`vstar` calibration follows the paper's scope: unit-normalized high-norm tokens are pooled
across configured prompts, seeds, denoising steps, and only the preset's register-zone
layers (FLUX 18-39; PixArt 13-20), not unrelated outliers from the full network. A versioned run identity prevents
results produced by an older intervention implementation from being resumed as current runs.
The depth conditions are sustained zone interventions (FLUX: 1/4/12/5 layers; PixArt:
1/2/3/2), not dose-matched single-layer ablations; interpret the causal map as regime-level
suppression and use the recorded per-layer fire counts when comparing effect magnitude.

For GenEval-style counting, attribute, and spatial-relation evaluation, set the Colab's
`Q7_STRUCTURED_SCORES_PATH` to a CSV produced by the chosen structured evaluator. It must
contain the exact key columns `run_identity,prompt_id,seed,condition,phase,zone` and may
contain clean/edited pairs named `geneval_counting_{clean,edited}`,
`geneval_attribute_{clean,edited}`, `geneval_spatial_{clean,edited}`, and
`geneval_overall_{clean,edited}`. Scores are validated in `[0,1]`, merged only into their
exact run cells, converted to edited-minus-clean effects, bootstrapped in `summary.csv`, and
shown in the prompt-fidelity figure. This interface does not silently substitute CLIP for a
structured evaluator.

Evaluation writes `paired_metrics.csv`, `summary.csv`, and these figures under
`<Q7_OUTPUT_DIR>/figures/`:

- `q7_causal_map.png` — LPIPS heatmaps across phase and depth for every intervention.
- `q7_frequency_profile.png` — low-frequency/layout versus high-frequency/detail change.
- `q7_prompt_fidelity.png` — paired CLIP and, when available, ImageReward deltas with
  bootstrap intervals and visible sample counts.
- `q7_vstar_loadings.png` — the largest calibrated `v*` channel coefficients.
- `q7_representative_contact_sheet.png` — clean, edited, and amplified-difference images.

Contact sheets include up to three distinct prompts: the first keeps the filename above,
and additional sheets use `q7_representative_contact_sheet_prompt_<id>.png`. Each sheet
uses the first complete phase/depth/seed cell for its prompt in manifest order, with the
prompt and settings printed above the images; examples are not selected by effect size.
Plot labels use **Remove high-norm states** (zero selected token vectors) and
**Renormalize high-norm tokens** (scale their norms to the ordinary-token median).
The stored condition IDs `remove_top_registers` and `norm_only` remain unchanged.

Figures are generated automatically by `--evaluate`. They can be regenerated without a
model or GPU after changing plotting code:

```bash
python -m src.experiments.generation_function --config configs/q7_colab.json --plot
```

## Q9: text states and image-register coupling

Use [Q9_Colab.ipynb](Q9_Colab.ipynb) independently of the older notebook. Its model dropdown
uses shared presets in `text_image_coupling.py` for FLUX.1-dev (28 steps, guidance 3.5) and
Schnell (4 steps, guidance 0). The bidirectional adapter deliberately rejects PixArt:
its static T5 conditioning cannot read back from the image DiT. See [SPEC_Q9.md](SPEC_Q9.md).

- `smoke`: two calibration prompts, two held-out prompts, one seed/step, projected T5 input
  and block 17. Tests all applicable methods, replay agreement, actual edits and cleanup.
- `discovery`: 12 prompts x 2 seeds. Measures T5 and DiT token classes, norms, directions,
  channel rankings and exact, chunked attention reductions at selected steps/layers.
- `screen`: four held-out prompts x two seeds; individual transformer forwards branched
  from identical clean latent states at selected denoising steps. No edited image decoding.
- `confirm`: 24 additional prompts x three seeds; by default four text-state contrasts
  plus four direction-removal rescue comparisons, at block 17/step 0. This is 576 edited
  trajectories, so freeze a small contrast set after screening and review the printed budget.
  Calibration is separate from evaluation and may be recomputed when sites/steps change.

Every edit targets one site at one step. Site `-1` edits the output of the T5-to-DiT projection;
nonnegative sites edit post-block text states. Routing methods target attention inside the
selected block instead. Block-17 residual edits precede the hypothesized block-18 writer.
The defaults are hypotheses, not automatic discovery of the correct birth layer.
`readout_layer` defaults to block 39 for the causal map, while response curves retain all
observed downstream blocks. Change the readout after inspecting independent calibration;
an empty register population at a readout is missing evidence, not a zero causal effect.

Operations include direction removal, exact resulting-norm matching, candidate-channel
suppression, zeroing, same-class ordinary-token zeroing, norm/edit-energy-matched random
rotations, and equal-token-class-layout donor swaps. Donor forwards hold the recipient image
latent and pooled CLIP vector fixed. Native empty-prompt baselines are separately labelled;
they are not represented as semantic-length-matched unconditional controls.

Routing tests distinguish image queries reading candidate text, candidate text queries reading
image registers, and candidate text queries reading content text. Score blocking renormalizes
attention; value blocking removes only the selected weighted-value contribution. Reverse
image-register lesions read out later text. Rescue restores either the clean image-register
projection or its complete state, with ordinary-position and identity-patch comparisons.

```bash
pip install -e ".[q9]"
python -m src.experiments.text_image_coupling --config /content/q9_config.json
python -m src.experiments.text_image_coupling --config /content/q9_config.json --plot
python -m src.experiments.text_image_coupling --config /content/q9_config.json \
  --export-compact /content/drive/MyDrive/Research/MA/q9_compact
```

The Colab creates the config; its advanced cell exposes prompts, seeds, methods, sites,
steps, candidate selection, and rescue settings. Pinning Diffusers 0.37.0 and Transformers
4.57.6 is part of the tested Q9 adapter contract. CPU model-interface tests use miniature
random FLUX/T5 networks; pretrained GPU runs and scientific outcomes require Colab execution.

The default `Q9_OPTIMIZE_PROBES=True` reuses clean diagnostic rows only before the
intervention can affect them, avoids redundant image-state transfers between adjacent
observed blocks, and captures only required donor states. All post-intervention readouts
are still measured; the attention probe keeps every query rather than subsampling.
`Q9_SKIP_UNAVAILABLE=True` records missing candidates, directions, matched controls or
rescue targets without running an ineffective edited trajectory. These jobs remain
explicitly marked `skipped_unavailable`, never counted as successful interventions.
Both switches can be disabled for comparison; per-job `work` counters record the saved
diagnostics. These optimizations do not reduce the denoising steps or experimental grid,
and do not imply a measured A100 speedup.

Results are under `output_dir/runs/<identity>/`. Main artifacts:

- `paired_metrics.csv.gz`, `summary.csv`: internal paired effects and prompt-cluster CIs.
- `audits.csv`, `manifest.json`: actual edits, token counts, invalid controls, exact job keys.
- `birth.csv`: generic outlier onset and separately channel-dominated register onset
  (mean selected-token channel energy >=0.5); absent birth remains missing.
- `directions.csv`, `direction_stability.csv`, `t5_summary.csv`: calibration and text summaries.
- `image_metrics.csv`, `image_summary.csv`: confirm-mode frequency, LPIPS and paired CLIP
  scores; external structured/prompt-fidelity scores may also be merged.
- `figures/`: norm atlas, direction energy, routing, causal response maps/curves and confirm
  example image pairs. Blank cells have no eligible measurement, not a zero effect.

External scores use unique `identity,job_id` keys from `manifest.json`. Supported clean/edited
columns are `counting_{clean,edited}`, `attribute_{clean,edited}`, `spatial_{clean,edited}`,
`overall_{clean,edited}` (all in [0,1]), and `image_reward_{clean,edited}` (finite scores).
Set `structured_scores` to that CSV before evaluation. These are explicit integration points;
Q9 does not silently claim to run GenEval or ImageReward when no scores are supplied.

No-candidate, unavailable-direction and unmatched-control jobs remain in the audit but are
excluded from causal summaries. One-prompt results have no confidence interval. Direction
fits with <50% leading energy are flagged; single-axis lesions there do not establish absence
of a higher-dimensional mechanism. Curves are exploratory unless contrasts were predefined;
the code does not perform a family-wide multiple-testing correction or a semantic decoder.
An optional equivalence bound applies only to image channel-energy changes, and does not
by itself establish global independence of the streams.

Raw activations, dense attention tensors and replay inputs are never serialized. Full PNGs
and per-job records remain in `/content`. Drive export whitelists compact files and refuses
exports larger than 250 MiB; it excludes the full image grid. Colab runtime reset loses the
local work unless retained separately, so resume only skips exact completed local jobs.

## Stage 1 cache storage

The older Stage 1 pipeline can persist its potentially large activation shards by pointing both
directories at a Drive mount. This is separate from Q7; with limited Drive capacity, leave these
paths under `/content` as well. Q7 deliberately exports no raw activation cache or full image grid.

```python
from google.colab import drive; drive.mount('/content/drive')
```
```yaml
output_dir:           /content/drive/MyDrive/activation_studies/outputs
activation_cache_dir: /content/drive/MyDrive/activation_studies/cache
```

Reduced artifacts are batched `cache_batch_size` (~25–50) prompts per shard file to keep
the Drive FUSE mount happy. Swapping to `rclone` (S3/GCS/B2) or a private HF dataset repo
later is the same config swap — no code change.

## Tests

```bash
uv run pytest -q
```

Covers the CPU-testable core: config precedence/validation, the abs-then-mean ranking
invariant, seeded random draws, Stage 3 normalization + foreground selection, IoU,
nearest upsampling, resumable cache round-trip, and torch-free importability of every
stage. The model-touching paths (FLUX.2-klein capture, BiRefNet) are exercised on Colab.
