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
| `model_ckpt` | `black-forest-labs/FLUX.2-klein` (HF id or local dir) |
| `prompt_source` | 1,600 GenAI-Bench prompts: `.txt` / `.json` / `.jsonl` / `.parquet`, or an HF dataset id |
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
2. **Channel score = `abs(mean_over_tokens)` — mean *then* abs**, not `mean(abs(...))`.
   The ordering changes the ranking. (`test_fig3_ranking.py` pins this.)
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
If the shape is off, the likely culprits are the normalization order or the mean-then-abs
ranking.

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

Covers the CPU-testable core: config precedence/validation, the mean-then-abs ranking
invariant, seeded random draws, Stage 3 normalization + foreground selection, IoU,
nearest upsampling, resumable cache round-trip, and torch-free importability of every
stage. The model-touching paths (FLUX.2-klein capture, BiRefNet) are exercised on Colab.
