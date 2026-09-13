# Q9 experimental plan: text states and causal coupling to image registers

Status: FLUX runner, reports, CPU integration tests and standalone Q9 Colab implemented.
Pretrained GPU experiments have not been executed locally. Primary model: FLUX.1-dev.

Implementation entry point: `python -m src.experiments.text_image_coupling --config ...`.
`Q9_Colab.ipynb` creates model-preset-based discovery/smoke/screen/confirm configurations.
The sections below retain the scientific design, including optional follow-ups. The current
release fits and tests a single leading direction, flags weak single-axis fits, and leaves
subspace interventions, a semantic decoder, and family-wide multiplicity correction as
explicit follow-ups. External structured/ImageReward scores are imported, not generated.
Single-site/step targeting is implemented; sustained equal-width windows can be studied in
a follow-up rather than implicitly changing intervention dose in the current comparisons.

## Question and hypotheses

Do candidate text states seed, maintain, or read back from the image-register circuit?
High norms and incoming attention establish candidates, not a functional register claim.

- Seed: perturbing text before image-register birth changes the subsequent image register.
- Maintain: perturbing text after birth disrupts an already established image register.
- Read back: perturbing image registers changes later text states through image-to-text
  information transfer; test whether those altered text states subsequently affect images.
- Independence: effective text perturbations change text computation or image semantics,
  but leave the specified image-register measurements within a predefined small-effect bound.

## Architecture and scope

Separate T5 encoder activations (computed during conditioning) from the evolving text
residuals inside the diffusion transformer. In FLUX, test projected T5 inputs and live DiT
text states separately. Hold pooled CLIP conditioning fixed during isolated T5-path edits;
a separate whole-prompt control may change both conditioning paths.

FLUX has both dual-stream blocks and later single-stream blocks with identifiable text/image
positions. Instrument both. A DiT text edit persists downstream within that forward, but the
next denoising forward starts again from the conditioning embeddings. Effects across denoising
steps can persist through the changed image latent. Record this distinction explicitly.

Use FLUX.1-dev as the primary causal model and Schnell as a separate replication, with its
own calibration. Its four denoising steps do not resolve temporal phases as finely as Dev.
PixArt can test T5-to-image conditioning and image-query/text-key cross-attention, but its
static text conditioning cannot receive image information through a live DiT text stream.
Report that pathway as architecturally unavailable, not an empirical negative result.

FLUX.1 EOS/padding tokens are different token types from the chat-template tokens studied
in recent LLM-conditioned diffusion work. Preserve exact tokenizer IDs and actual attention
masks; do not import the template-token conclusion into FLUX.1 by assumption.

## Stage A: clean discovery and calibration

Use 12 calibration prompts covering short/long descriptions, objects, attributes, counting,
and relations, with two seeds each. Keep these separate from evaluation prompts. Include
tokenizer-verified equal-length prompt pairs differing in object, attribute, or relation.

1. Capture T5 layer summaries and DiT text/image summaries indexed by model, prompt, seed,
   denoising step, layer, token class, and (for attention) head.
2. Classify real positions as content, EOS, padding, and any actual additional special tokens.
   Record token IDs, content length, sequence length, and effective masks. Never assume padding
   is inaccessible in attention. Stratify statistics by class so padding does not dominate.
3. Measure norms, norm/median ratios, massive-channel energy fractions, channel rankings,
   and norm after excluding dominant channels. Keep the full distribution and candidate counts.
4. Identify norm candidates and attention sinks independently; quantify their overlap.
   A candidate must meet a calibrated excess-attention criterion, not merely rank first.
   Report incoming mass both relative to all keys and conditional on text keys, plus
   per-token mass and head entropy. Account for token-class size and number of valid keys.
5. Fit unit-vector uncentered SVD directions separately per stage/layer or a demonstrated
   stable layer interval. Report leading energy fraction, held-out projection energy,
   absolute cosine agreement between prompt fits, and top-channel stability. Use a small
   subspace if no single direction explains the candidates well. Do not force a shared axis.
6. Probe prompt specificity with equal-length donor prompts, within-prompt/across-seed
   comparisons, and held-out stability. If adding a semantic decoder, train/test by prompt
   and compare to token-position/class baselines. A stable norm direction does not imply
   that its orthogonal residual contains no semantics.
7. Rediscover image-register onset and persistence per model and step. Use Q7 layer 18 as
   a reference hypothesis, not a required discovery result. Freeze evaluation targets after
   calibration; include before/after block probes near onset.

The existing text plots select a top fraction and retain final-step text activations. Q9
must retain step-indexed summaries and must not equate plotted top-fraction points with
threshold-defined Q7 registers. Use the Q7 3x-median image definition for comparability,
but audit uncapped counts and cap saturation. Calibrate text selection separately, with
token-class checks; do not automatically import the top-eight cap into text.

## Attention notation and interventions

Use query/key labels in every output to avoid reversing information flow:

| Query rows | Key/value columns | Information delivered to |
| --- | --- | --- |
| Text | Text | Text from text |
| Image | Text | Image from text |
| Text | Image | Text from image |
| Image | Image | Image from image |

For candidate text states, measure all applicable blocks. Within image columns/rows,
separate clean-detected registers from ordinary image tokens.

Estimate attention from the actual normalized, position-encoded Q/K and active masks.
Stream query chunks and save reductions. Validate summaries against exact small attention
matrices. If sampling queries for discovery, record sampling/error and verify selected heads
with all queries before confirmation. Attention mass alone is not a measure of transferred
semantic information; include value-path interventions and downstream residual readouts.

## Stage B: short causal probes before full image generations

At selected clean denoising states, branch matched transformer forwards from identical latent,
conditioning, timestep, and guidance. Read downstream blocks and the final noise/velocity
prediction without decoding an image. These measure within-forward effects without accumulating
earlier trajectory divergence. They do not replace full generation-level validation.

Begin with single-block, single-step interventions. For a writer near block 18, a text edit
after block 17 can influence block 18; an edit after block 18 cannot test formation inside
that block. Choose one pre-birth, one established-register, and one later site from calibration.
Use three representative denoising steps, fixed in advance after calibration.

Candidate state operations at clean-selected text positions R_T:

- Direction removal: x' = x - (x dot v_text) v_text.
- Magnitude-matched control: x' = x * ||x_direction_removed|| / ||x||. This preserves
  direction and matches the resulting norm of direction removal for each token.
- Candidate-channel suppression at R_T only; all-text channel suppression is a separate,
  broader intervention. Do not assume a text channel shares the image channel index.
- Complete zeroing at R_T as a broad lesion, with equal-count control token lesions.
- Same-position donor-state replacement between tokenizer-length-matched prompts;
  distinguish encoder-output swaps from later DiT text-state swaps.

Use same-class noncandidate tokens when available, matched token counts, and random-direction
controls calibrated to comparable edit energy. Record actual norm change and edited token count;
label controls lacking adequate matches. Q7 ordinary-median scaling can be an additional
comparison, but is not exactly magnitude-matched to direction removal. Scaling residuals does
not by itself guarantee preserved downstream sink identity; measure routing after the edit.

Pathway tests:

- Suppress image-query attention to candidate text keys: tests text influence on images.
- Suppress candidate-text-query attention to image-register keys: tests register read-back.
- Suppress candidate-text-query attention to content-text keys: distinguishes direct textual
  acquisition from image-mediated acquisition.
- Reverse lesion: remove image-register v_image or states and measure later candidate text
  states, compared with equal-count ordinary-image-token controls.

Score masking renormalizes remaining weights. For promising edges, also remove their weighted
value contribution while retaining original attention weights to separate routing competition
from value content. Apply output projections correctly and preserve untargeted query outputs.

## Primary and secondary outcomes

Primary outcomes are downstream image-register measurements, not image quality:

- Absolute channel-154 amplitude and channel-154 energy fraction for FLUX.
- Projection onto frozen clean v_image, squared alignment, and norms.
- Register count, birth layer, persistence, and spatial positions.
- Incoming attention to image registers, sink identity, and head-specific sink concentration.

Measure both fixed clean register positions and independently redetected edited positions.
The former tracks loss at original locations; the latter detects relocation or compensation.
Keep v_image fixed for causal comparisons, rather than refitting it to the edited run.
Define handling of empty candidate sets explicitly and do not treat no targets as evidence
that a successful intervention was inert.

Secondary outcomes: candidate text norms/directions/sinks, full prediction change, LPIPS,
paired prompt-fidelity deltas, structured object/attribute/relation scores, and spatial-frequency
differences. Broad/fine frequency differences are proxies, not semantic layout/detail labels.

## Stage C: full trajectories and rescue

Validate selected effects with same-seed full generations at the original model preset.
Start with isolated step/layer edits; use equal-width windows for sustained interventions.
Reuse Q7 early/middle/late timings for later confirmation, but report actual timesteps/sigmas
and intervention counts. Avoid interpreting unequal-width depth zones as equal-dose tests.

For the strongest text-to-register effect compare:

1. Clean baseline.
2. Text perturbation.
3. Text perturbation + restore only the clean image-register projection along v_image at
   a specified downstream site: x' = x + ((x_clean - x) dot v_image) v_image.
4. Text perturbation + full clean-register-state restoration as a broader rescue.
5. Matched restoration at ordinary image positions and a clean-to-clean sham patch.

Measure later internal recovery and final image recovery. Restoration at the patched site
itself is tautological; recovery must survive downstream. Rescue supports mediation but does
not prove the register is the only route. Whole-state rescue may restore ordinary image content.
If read-back is supported, transplant late text states into a recipient continuation to test
whether acquired information can subsequently influence the image.

## Prompt and unconditional controls

Separate three comparisons: native meaningful vs empty prompt; equal-token-length semantic
minimal pairs; and candidate-state swaps with content positions and masks unchanged.
Equal padded tensor length does not equal equal semantic length or equal EOS position.
Report the residual confound in native empty comparisons rather than inventing a supposedly
neutral padded sentence. Preserve native encoder/pipeline mask behavior.

For FLUX-dev, label empty-prompt generation as empty-prompt conditioning; do not equate its
guidance embedding with a conventional conditional/unconditional CFG pair. In PixArt, native
CFG branches are available: specify which branch is edited and retain the other branch.

## Budget, statistics, and execution auditing

- Discovery: 12 prompts x 2 seeds = 24 clean Dev trajectories; derive many summaries per run.
- Screen: four held-out prompts x two seeds, using short forward probes over selected sites,
  followed by full trajectories only for the most informative interventions.
- Confirmation: starting target of 24 additional held-out prompts x three seeds for a small
  preregistered contrast set. Determine whether to expand from pilot variance, not significance.
- Estimate runtime from measured clean tracing and forward-probe timings on the assigned GPU.
  Do not promise Q7-based wall times or exact prefix-replay savings without a benchmark.
- Bootstrap by prompt, retaining seeds/repeated sites within prompt; avoid treating tokens,
  heads, or layers as independent samples. Freeze primary contrasts and use appropriate
  multiple-comparison control for any confirmatory sweeps.
- For a negative result, verify perturbation strength, positive controls, target availability,
  and confidence intervals within a predefined smallest relevant effect. Otherwise call it
  inconclusive. General text conditioning can matter while this particular register is stable.
- Audit tokenizer alignment, masks, untouched streams, actual nonempty edits, affected token
  counts, no-op agreement, and clean replay. Report numerical tolerances, not assumed bit equality.
- Record checkpoint/revision, dtype/backend, scheduler config, guidance, steps, seed/noise,
  prompt/token IDs, calibration identity, intervention location/operator, and output identity.
- Keep raw activations, attention matrices, and temporary replay states in RAM/local Colab
  storage only. Export compact summaries, calibration vectors, manifests, and selected figures
  to Drive, with an estimated byte budget. Do not persist full activation traces to Drive.

## Planned figures and implementation sequence

1. Text token-class norm/sink atlas across layer and denoising step.
2. Held-out direction and massive-channel stability plots.
3. Four query/key routing blocks, with image registers separated from ordinary image tokens.
4. Text intervention -> image-register response maps, including birth/position changes.
5. Reverse image intervention -> text-state response plots.
6. Rescue curves and selected baseline/perturbation/rescue image comparisons.

Implement step-indexed summaries and mask audits first, short causal probes second, then
generation/rescue and Colab discovery/screen/confirm presets. Reuse Q7 model loading,
conditioning caching, paired manifests and metrics where valid. Do not build the full
token-class x method x layer x step x model Cartesian grid before discovery.

## References

- Text Template Tokens Are Implicit Semantic Registers in Diffusion Transformers:
  https://arxiv.org/abs/2607.19139
- Pinned FLUX implementation (Diffusers v0.37.0):
  https://github.com/huggingface/diffusers/blob/v0.37.0/src/diffusers/models/transformers/transformer_flux.py
- Local reference implementations: src/experiments/text_stream_qualitative.py,
  src/common/model_utils.py, and SPEC_Q7.md.
