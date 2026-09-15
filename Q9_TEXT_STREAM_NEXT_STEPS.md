# Q9 text-stream audit and next experiments

## What the current pilot tests

The pilot tests whether selected EOS/padding **DiT text states after block 17**, at
denoising step 0, affect the image stream downstream. It uses three evaluation prompts,
one seed, three separate calibration prompts, and four state interventions (12 edited
forwards in Screen). Attention is observed at blocks 17, 18, 19 and 39; residual summaries
are recorded across blocks at the selected step. Confirm completes edited trajectories
and measures final images, but its default internal observations still cover step 0 only.

T5 encoder block states/attention are observed when prompts are encoded. They are distinct
from the evolving DiT text states. Site -1 edits the **projected T5 conditioning at the
DiT entrance**, not a hidden layer inside T5. Every new denoising forward begins with
conditioning again; a text edit can influence later denoising through the changed image
latent. Pooled CLIP conditioning stays fixed for isolated text-state edits.

## Implementation audit

Checked against `q9_runtime.py`, `text_image_coupling.py`, `q9_report.py` and CPU tests
using actual miniature FLUX/T5 modules from the pinned libraries:

- Token labels are checked against the actual T5 input IDs. The actual encoder mask is
  recorded separately from the tokenizer mask. Padding is not assumed to be masked out.
- Native T5 attention is captured block by block; hooks are removed even on failure.
  Native-versus-instrumented embeddings are checked for exact agreement.
- A residual edit is applied after the selected block. An attention-path edit acts inside
  its selected block. The pilot's post-17 edit can influence block 18, not block 17 attention.
- Clean candidate positions and fitted per-layer directions are frozen for edited runs.
  Both fixed-position and redetected image-register measurements are retained.
- Q9 norm matching preserves the original direction while matching the resulting norm
  of direction removal. It differs from Q7's ordinary-token-median scaling.
- Image queries attending text keys carry text values into image outputs. Text queries
  attending image keys carry image values into text outputs. Score/value edits preserve
  untargeted query rows; value subtraction excludes output-projection bias.
- Screen starts from the clean saved latent and timestep. Confirm uses the same prompt,
  seed and generation parameters, checks initial-noise identity, and audits edit attempts.
- Probe reuse stops when an intervention could affect the measurement. Tests compare
  optimized/unoptimized predictions and every readout, including a multi-step trajectory.
- Reports exclude unavailable/no-effect edits from causal summaries and bootstrap by
  prompt, rather than counting tokens as independent observations.

### Corrected reporting error

The previous attention `overlap_count` intersected sinks with the selected candidate set.
With selection `union`, ordinary-norm sinks were therefore counted as norm/sink overlap.
Revision `q9-v3` intersects the independently thresholded norm mask with sinks instead,
within the configured token classes. It also reports norm, eligible-sink and final-candidate
counts/positions separately. This correction does not change which states are selected or
how they are edited. Old saved overlap counts should not be interpreted as norm/sink overlap;
obtaining corrected counts requires a new trace. The new identity prevents mixing revisions.

### Limits that remain

1. **Singleton EOS control:** there is usually only one EOS. If it is selected, the
   equal-count same-class ordinary-token control cannot be constructed. The entire
   `ordinary_zero` comparison is marked unavailable, including when padding candidates
   themselves could have been matched. This is honest missing evidence, not a successful
   negative control. `report_status.json` now summarizes statuses and explains this case.
2. **Matching strength:** available ordinary tokens are matched by class/count and nearest
   available norm. There is no maximum norm-distance requirement; zeroing comparisons may
   have different edit magnitudes. The norm-matched direction control is the cleaner test
   of a direction-specific effect.
3. **Selection by union:** the default includes sinks that are not high-norm. Call these
   candidate text states, not exclusively high-norm text registers. At unprobed attention
   layers only norm information is available; sink-only/intersection selections there
   cannot supply an attention-based candidate population.
   The shared direction also pools selected EOS and padding states; unit normalization
   does not give the two classes equal weight when their candidate counts differ.
4. **Limited claim:** three prompts, one seed, one site and one step cannot establish
   general independence, stability across seeds, maintenance, or read-back. Block 18 is
   a hypothesis carried from image-stream work; verify the clean birth measurements.
5. **Measurements:** attention mass alone is not semantic information. A channel-energy
   fraction can stay constant while absolute channel amplitude changes; inspect both.
   Weak leading-direction fits may require a subspace experiment.
6. CPU integration tests do not replace a run of pretrained FLUX on the intended A100.

## Prioritized follow-ups

Budgets below count edited single-forward trials for three prompts and one seed. Clean
calibration/baseline forwards are additional, and changing a configuration may require
new calibration and a new run identity. They are suggestions, not added default jobs.

| Priority | Experiment | Question and essential control | Small screening budget | Support |
| --- | --- | --- | --- | --- |
| 1 | Separate EOS and padding | Does EOS drive the effect, or do padding states contribute? Compare direction removal with norm matching within each class. Assess ordinary padding controls where available. | 2 classes × 2 edits × 3 prompts = 12 | Existing candidate-class configuration |
| 2 | Norm candidates versus attention sinks | Is the effect associated with large magnitude, sink routing, or their intersection? Compare `norm`, `sink`, `intersection`; retain norm matching. Record token count/edit magnitude because sets differ. | 3 selections × 2 edits × 3 prompts = 18 | Existing candidate-source configuration |
| 3 | Projected T5 input versus post-block 17 | Is the causal component already present in the conditioning, or does it develop inside the DiT? Fit each site's direction separately and compare removal/norm matching. Unequal downstream depth limits effect-size comparisons. | 2 sites × 2 edits × 3 prompts = 12 | Existing sites -1 and 17 |
| 4 | Reverse image-to-text test | Does disrupting an established image register change later text? Compare image-register zeroing with ordinary image-token zeroing at one calibrated post-birth site. Confirm ordinary controls exist and inspect edit magnitudes. | 2 edits × 3 prompts = 6 | Existing reverse methods |
| 5 | Read-back pathway test | Does candidate text read image-register values? Compare `text_reads_register_score` and `text_reads_register_value` at one post-birth site, then inspect later text and image states. A specificity claim also needs an ordinary-image-key pathway control. | 2 edits × 3 prompts = 6, before the additional control | Main lesions exist; ordinary-key pathway control needs implementation |
| 6 | Matched-prompt donor swap | Does swapping EOS/padding states transfer an attribute or relation? Use tokenizer-verified equal-layout prompts, e.g. the pilot's left/right pair if tokenization matches; hold recipient image latent and pooled CLIP fixed. Add a recipient-to-itself swap. | Depends on number of valid donor pairs | Donor swap exists; explicit self-swap control and paired semantic scoring need implementation |

Run the first one or two follow-ups before expanding the grid. EOS/padding separation
is especially useful given the singleton-control issue, and uses the existing mechanisms.
Both Colabs now expose `eos` and `pad` separately in the optional candidate-class dropdown.

## Further ideas after a reproducible effect

- **Before/after birth:** compare post-17 and post-19 edits, equal token rules and edit type.
  This separates a formation hypothesis from a maintenance hypothesis. It is a layer
  comparison within one forward, not a comparison of early/late denoising.
- **Orthogonal semantic content:** decompose a donor state into its shared-direction
  component and the orthogonal remainder, transplant each separately, and norm-match.
  This asks whether a stable massive component supports routing while other dimensions
  carry prompt-specific content. This needs a new operator.
- **Dose response:** remove 25%, 50%, 100% of the projection, with corresponding norm controls.
  A reproducible graded response is more informative than a single complete deletion.
- **Image-latent dependence:** hold conditioning/timestep fixed and change only the image
  latent, then compare later text states and sink routing. Repeat while blocking image-to-text
  values. This distinguishes image-dependent text states from inherited encoder content.
- **Prompt-length sensitivity:** use tokenization-verified equal-length prompt pairs first.
  A later padded-length manipulation must preserve meaningful embeddings and positions;
  otherwise changes in T5 context and attention normalization confound the result.
- **Small rescue:** once a text-to-image effect exists, restore the clean image-register
  projection downstream and compare with ordinary-position and sham controls. Measure
  recovery at later blocks and the final image, not merely at the patched site.

## Relation to existing work

[Text Template Tokens Are Implicit Semantic Registers in Diffusion Transformers](https://arxiv.org/html/2607.19139v2)
reports template-token semantics/read-back in models with LLM/VLM text conditioning.
FLUX.1's T5 EOS/padding tokens are a different setting. Our useful extension is to test
whether those text states couple specifically to the image-register circuit and its
shared direction/channel, rather than assuming the template-token mechanism transfers.

[Attention Sinks in Diffusion Transformers: A Causal Analysis](https://arxiv.org/abs/2605.09313)
motivates distinguishing score-path suppression from value-path removal and assessing
semantic alignment separately from trajectory changes. These are methodological starting
points, not evidence that the proposed FLUX.1 mechanism is already established.
