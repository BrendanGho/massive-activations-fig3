# SPEC - Q7 generation-level function of the register/channel/sink circuit

## Question

What selective image-generation function is causally supported by the FLUX/PixArt register
state, its dominant channel (FLUX 154; PixArt 293), and the associated attention sink?

## Design

Run paired, same-seed generations.  A baseline trace fixes the natural register positions for
every denoising step and block.  Interventions use that trace rather than reselecting tokens after
the model has been perturbed.

Conditions:

- `remove_vstar`: subtract the projection onto a calibrated unit direction only at natural
  register tokens.
- `suppress_channel`: zero the model preset's dominant channel for every image token (FLUX 154;
  PixArt 293).
- `suppress_sink`: trace the strongest natural image key separately for every attention head
  using image-key-renormalized incoming attention, then block image queries from attending to
  that head's sink while leaving text-query routing and the residual token unchanged. For PixArt,
  this targets image self-attention (`attn1`) and leaves T5 cross-attention (`attn2`) untouched.
- `remove_top_registers`: zero the complete residual vectors of the highest-norm natural
  register tokens.
- `norm_only`: scale natural register vectors to the ordinary-token median norm while preserving
  their direction (mechanistically matched control from the paper).

Cross each condition with early/middle/late denoising thirds and writer/early-register/
mid-register/dissolution depth zones. Prompt, seed, initial generator state, scheduler, guidance,
resolution, and step count are identical within every pair.

The confirmatory configuration uses the full Cartesian product. The optional exploratory screen
uses a preregistered cross: all three phases at `mid_register`, plus all four zones at `middle`
(six unique cells). It estimates main temporal/depth patterns cheaply but is not evidence that
unmeasured phase-by-depth interactions are absent.

FLUX.1 uses its 57-block joint-attention layout and register zones 18-39. PixArt-Sigma uses its
28 image-only DiT blocks, block 13 as the writer, and the shorter 13-20 register interval. Because
PixArt uses real CFG, interventions edit only the conditional (last) row of `[uncond, cond]`.

## Outputs

- Clean/intervened PNGs and a resumable per-run manifest.
- `paired_metrics.csv`: LPIPS, CLIP, optional ImageReward, low-frequency and high-frequency
  paired distances, plus optional GenEval-style scores supplied by a structured evaluator.
- `summary.csv` with paired bootstrap confidence intervals by condition, time phase, and zone.
- Optional externally computed GenEval-style clean/edited scores are matched by the full run-cell
  identity, range-checked, converted to paired deltas, and included in the summary and figures.
- `figures/q7_causal_map.png` and `q7_frequency_profile.png` for temporal/depth and
  low-/high-frequency effects, plus paired prompt-fidelity and `vstar` loading plots.
- A representative contact sheet that exposes clean/intervened/amplified-difference images.
- Calibration file containing the fitted `vstar` and its audit metadata; natural register/sink
  traces are paired in memory with each scenario and are not serialized.
- In Colab, the full image grid is temporary under `/content`; compact metrics, audit metadata,
  `vstar`, and figures may be exported to Drive, but raw activation traces and the image grid are
  not copied.

## Acceptance criteria

- Pure intervention operators preserve untouched tokens/channels exactly and the norm-only
  operator preserves direction while matching the requested norm.
- Phase and depth targeting are inclusive, deterministic, and fire only in the requested cells.
- Register masks are selected from the clean trace with the paper's 3x-median, top-8 rule.
- `vstar` is fitted only from high-norm tokens in the configured register-zone layers,
  pooled across the calibration prompts, seeds, denoising steps, and those layers.
- Sink suppression edits attention routing only; it does not overwrite the residual-stream token.
  Edited image-query rows use the pinned Diffusers attention dispatcher, model dtype, and active
  backend with only the per-head natural-sink key masked; clean text-query outputs are retained.
  PixArt uses the native scaled-dot-product path of its pinned `AttnProcessor2_0`, changes only
  conditional image self-attention, and retains the unconditional CFG row and cross-attention.
- A smoke mode runs one baseline plus every condition in one phase/zone and asserts both total
  forward counts and actual targeted edit counts before a full run.
- Every generated pair records all generation parameters and hashes the experiment config.
- Prompt conditioning is cached per unique prompt without changing its tensors, and combined
  calibration/run mode reuses one loaded pipeline. Resume skips a scenario only when every
  requested run cell and its clean image exist for the exact run identity.
- CPU tests cover the operators, targeting, metrics, and paired aggregation without importing
  diffusers or downloading model weights.
