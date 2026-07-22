# Fig 3 — Massive Activations in Diffusion Transformers 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BrendanGho/massive-activations-fig3/blob/main/Figure3_Colab.ipynb)

Reproduction of **Figure 3 / Section 3.2** of *"Few Channels Draw The Whole
Picture: Revealing Massive Activations in Diffusion Transformers"*
(arXiv:2605.13974) for FLUX.2-klein. (Not the Section 3.1 disruption experiment.)

The claim under test: a tiny number of "massive-activation" channels in a diffusion
transformer already localize the image subject. Selecting the **top-k** such channels
per layer and clustering their per-token activations yields a foreground mask that
matches a segmentation pseudo-ground-truth (BiRefNet), while **bottom-k** channels are
diffuse and **random-k** sit in between.

Two outputs:
- **Qualitative** (a few example prompts): per-layer heatmap + binary mask for top-k
  vs bottom-k channels (Fig 3A–C).
- **Quantitative** (Fig 3D, all 1,600 GenAI-Bench prompts): layer-wise mIoU curve vs
  BiRefNet pseudo-GT — one line each for top-k / bottom-k / random-k.

## Layout

```
configs/default.yaml            # single source of config (see precedence below)
src/common/config.py            # config loading + strict validation
src/common/model_utils.py       # FLUX.2-klein + BiRefNet loading, capture hooks (lazy torch)
src/common/clustering.py        # Stage 3 numeric core (normalize -> KMeans(2) -> mask)
src/common/io.py                # reduced-cache shards, prompt loading, IoU, upsampling
src/stage1_generate_and_cache.py
src/stage2_channel_ranking.py
src/stage3_mask_construction.py
src/stage4_evaluate_figure3d.py
scripts/run_pipeline.sh         # resumable end-to-end wrapper
tests/test_fig3_*.py            # CPU tests for the numeric core
outputs/  cache/                # runtime only (gitignored)
```

## Install

```bash
uv sync                    # core + test deps (numpy, scikit-learn) — enough to run the tests
uv sync --extra fig3       # + torch/diffusers/transformers/matplotlib for the real run (GPU/Colab)
# or: pip install -e ".[fig3]"
```

The numeric core (`config`, `clustering`, `io`, ranking) is pure `numpy`/`scikit-learn`
and importable without a GPU. `torch`/`diffusers`/`transformers`/`matplotlib` are imported
**lazily** inside the model-touching stages, so the tests run anywhere.

## Configure

Fill the five **required** keys in `configs/default.yaml` (or override them — the loader
**fails loudly** if any is empty, because this runs unattended):

| key | example |
|---|---|
| `model_ckpt` | `black-forest-labs/FLUX.2-klein-4B` (ungated) or `-9B` (gated) — HF id or local dir |
| `prompt_source` | 1,600 GenAI-Bench prompts; defaults to the bundled `data/genai_prompts.jsonl` — or point it at another `.txt` / `.json` / `.jsonl` / `.parquet` file, or an HF dataset id |
| `birefnet_weights` | `ZhengPeng7/BiRefNet` |
| `output_dir` | where CSV / plots / qualitative / `run_metadata.json` land |
| `activation_cache_dir` | where reduced cache shards land |

**Override precedence (highest wins): CLI flag > `FIG3_*` env var > YAML.**

```bash
# env override
FIG3_TOP_K=12 FIG3_SEED=1 python -m src.stage1_generate_and_cache --config configs/default.yaml
# CLI override (repeatable)
python -m src.stage1_generate_and_cache --config configs/default.yaml --set seed=1 --set device=cuda
```

## Run

```bash
# fused (default, recommended for the full 1,600-prompt run)
python -m src.stage1_generate_and_cache --config configs/default.yaml --fused
python -m src.stage4_evaluate_figure3d  --config configs/default.yaml

# or the resumable wrapper (skips already-cached prompts)
scripts/run_pipeline.sh configs/default.yaml
```

`FLUX.2-klein-4B` is ~16 GB in half precision (Qwen3-4B text encoder + 4B transformer),
so it needs a ≥24 GB GPU to load fully; on a 16 GB T4 set `offload: true` (or `--set
offload=true`) to enable `enable_model_cpu_offload` (fits, slower).

**Fused mode** (default) runs Stages 2+3 in-process right after each capture and persists
only small reduced artifacts (scores, channel indices, binary masks, decoded RGB,
qualitative PNGs). The full `[N_I, D]` per-layer tensor is discarded — at FLUX hidden
width across ~all layers × 1,600 prompts it would be hundreds of GB.

**Debugging the stages standalone** (on a handful of cached prompts):

```bash
python -m src.stage1_generate_and_cache --config cfg.yaml --no-fused --limit 4  # persist full acts
python -m src.stage2_channel_ranking    --config cfg.yaml --prompt-id 0
python -m src.stage3_mask_construction  --config cfg.yaml --prompt-id 0 --qualitative
python -m src.stage4_evaluate_figure3d  --config cfg.yaml
```

## Invariants (do not "simplify" these)

1. **Only image-stream tokens, only the last denoising timestep.** The text/image split
   is derived at runtime from the model's packed sequence (`N_I` = image-latent token
   count), never a hard-coded offset. Last timestep is captured by hooks overwriting a
   per-layer buffer each forward.
2. **Channel score = `mean(abs(activations))` over tokens — abs *then* mean**, not
   `abs(mean(...))`. The ordering changes the ranking. (`test_fig3_ranking.py` pins this.)
3. **Ranking stats are per-sample / per-layer / per-stream** — activations/scores are
   never averaged across samples before ranking.
4. **Stage 3 order:** min-max normalize each channel across tokens **first**, then
   KMeans(2) on the **k-dim per-token vectors** (not a collapsed scalar); foreground is
   the cluster with the higher mean of `s[n] = normalized[n,:].sum()` (the Fig 3B heatmap).

## Outputs

- `outputs/run_metadata.json` — resolved config, package versions, prompt-source hash,
  and the **logged ambiguity choices** (random-k trial count, KMeans init/seed defaults,
  capture conventions).
- `outputs/figure3d_results.csv` — `layer, strategy, mean_miou, std_miou, n`.
- `outputs/figure3d_curve.png` — the three mIoU curves.
- `outputs/qualitative/prompt_XXXXX/` — heatmap+mask PNGs for the first
  `num_example_prompts` prompts.

**Sanity targets** (printed as warnings, not asserted): top-k dominates every layer,
bottom-k flat ≈ 0.2, random-k between, FLUX.2-klein top-k peak ≈ 0.5 near layer 10.
If the shape is off, the likely culprits are the normalization order or the abs-then-mean
ranking.

## Part 2 — per-generation channel stability (the main experiment)

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

**Scores.** Primary = `mean(abs(activations))` over tokens (the fig3 massive-activation score;
invariant #2 above). Secondary (`secondary_metric: p999`) = 99.9th percentile of
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

## Part 3 — are the sparse outlier tokens the same as ViT "high-norm" tokens?

`src/experiments/highnorm_tokens.py` + `configs/highnorm_tokens.yaml`, numeric core in
`src/common/highnorm.py`, spec in [`SPEC_highnorm.md`](SPEC_highnorm.md). Question:
isolating the top 1–2 massive channels at layer 18 renders a near-black image with sparse
bright speckles — are those speckle tokens the **high-norm / register tokens** of
Darcet et al. ([arXiv:2309.16588](https://arxiv.org/abs/2309.16588))? Where Part 2 is
channel-space only, this is a **token-space** question.

**Start here — the qualitative look.** `src/experiments/highnorm_qualitative.py` is the
simplest version: no statistics, one row per prompt — generated image | isolated top-1
channel (the speckles) | high-norm tokens (full norm) | high-norm tokens (norm minus the
top channel). The full-norm panel is a carbon copy of the speckles (that's the confound
below, made visible); whether the last panel still lights up at those spots is the whole
question. `python -m src.experiments.highnorm_qualitative --config configs/highnorm_tokens.yaml`

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

→ **H1** (selectivity ≫ 1, elevation ≈ 1): same phenomenon; massive activations *are* how
these tokens get their norm. **H2** (both ≫ 1): genuine register tokens, broadly elevated.
**H3** (selectivity ≈ 1): channel is uniformly large, no speckles to explain.

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
python -m src.experiments.highnorm_tokens --config configs/highnorm_tokens.yaml
```

## Colab storage

Point both dirs at a Drive mount so writes survive a session ending mid-run (no code change):

```python
from google.colab import drive; drive.mount('/content/drive')
```
```yaml
output_dir:           /content/drive/MyDrive/figure3_repro/outputs
activation_cache_dir: /content/drive/MyDrive/figure3_repro/cache
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
