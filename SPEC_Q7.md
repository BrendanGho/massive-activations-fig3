# SPEC - Q7 generation-level function of the register/channel/sink circuit

## Question

What selective image-generation function is causally supported by the FLUX register state,
its dominant channel (154), and the associated attention sink?

## Design

Run paired, same-seed generations.  A baseline trace fixes the natural register positions for
every denoising step and block.  Interventions use that trace rather than reselecting tokens after
the model has been perturbed.

Conditions:

- `remove_vstar`: subtract the projection onto a calibrated unit direction only at natural
  register tokens.
- `suppress_channel_154`: zero channel 154 for every image token.
- `suppress_sink`: trace the strongest natural image key separately for every attention head
  using image-key-renormalized incoming attention, then block image queries from attending to
  that head's sink while leaving text-query routing and the residual token unchanged.
- `remove_top_registers`: zero the complete residual vectors of the highest-norm natural
  register tokens.
- `norm_only`: scale natural register vectors to the ordinary-token median norm while preserving
  their direction (mechanistically matched control from the paper).

Cross each condition with early/middle/late denoising thirds and writer/early-register/
mid-register/dissolution depth zones. Prompt, seed, initial generator state, scheduler, guidance,
resolution, and step count are identical within every pair.

## Outputs

- Clean/intervened PNGs and a resumable per-run manifest.
- `paired_metrics.csv`: LPIPS, CLIP, optional ImageReward, low-frequency and high-frequency
  paired distances, plus optional GenEval-style scores supplied by a structured evaluator.
- `summary.csv` with paired bootstrap confidence intervals by condition, time phase, and zone.
- `figures/q7_causal_map.png` and `q7_frequency_profile.png` for temporal/depth and
  low-/high-frequency effects, plus paired prompt-fidelity and `vstar` loading plots.
- A representative contact sheet that exposes clean/intervened/amplified-difference images.
- Calibration file containing the fitted `vstar` and natural register/sink traces.

## Acceptance criteria

- Pure intervention operators preserve untouched tokens/channels exactly and the norm-only
  operator preserves direction while matching the requested norm.
- Phase and depth targeting are inclusive, deterministic, and fire only in the requested cells.
- Register masks are selected from the clean trace with the paper's 3x-median, top-8 rule.
- Sink suppression edits attention routing only; it does not overwrite the residual-stream token.
- A smoke mode runs one baseline plus every condition in one phase/zone and asserts both total
  forward counts and actual targeted edit counts before a full run.
- Every generated pair records all generation parameters and hashes the experiment config.
- CPU tests cover the operators, targeting, metrics, and paired aggregation without importing
  diffusers or downloading model weights.
