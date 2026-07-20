# SPEC — Are massive-activation outlier tokens the same tokens as ViT "high-norm" tokens?

Companion experiment to `SPEC.md` (Figure 3 reproduction). Separate file on purpose: the
Figure 3 spec is still live and this is an independent question.

## Question

Isolating the top 1–2 massive-activation channels at a FLUX double-stream block (layer 18)
produces a near-black image with sparse bright speckles. Are those speckle locations the same
tokens that DiT/ViT work calls **high-norm tokens** / **registers**
(Darcet et al., ICLR 2024, arXiv:2309.16588)?

Darcet's criterion, reproduced faithfully: the **L2 norm of the patch-token embedding at the
model output**, thresholded at a hand-picked cutoff (150 for DINOv2) read off a **clearly
bimodal** norm histogram; ~2.4% of tokens qualify; they emerge around layer 15 of 40 and
persist; they sit on patches redundant with their neighbours and hold little local information.

## The confound this spec exists to control

`‖x[n,:]‖² = Σ_d x[n,d]²`. A token with a massive value in channel *c* has its norm dominated
by that one channel, so "massive-activation tokens are high-norm tokens" is **true by
construction**. Any measurement correlating `|x[:,c]|` with `‖x[n,:]‖₂` will report overlap
≈ 1.0 and mean nothing.

Every overlap statistic in this spec is therefore computed against `N_ex`, the norm with the
massive channels **excised**, never against the full norm. Three outcomes are distinguishable:

The verdict rests on two **effect sizes**, deliberately not on set overlap:

- `selectivity` = median `m`[outlier] / median `m`[typical] — is the channel token-sparse,
  i.e. are there speckles at all?
- `elevation` = median `N_ex`[outlier] / median `N_ex`[typical] — do those tokens stay
  high-norm once the massive channels are excised? *This is the question, stated numerically.*

| Hypothesis | Signature | Reading |
|---|---|---|
| **H1** one mechanism | selectivity ≫ 1, **elevation ≈ 1** | Massive activations *are* how these tokens get their norm |
| **H2** co-located, distinct | selectivity ≫ 1, **elevation ≫ 1** | Register tokens broadly elevated; massive channels one facet |
| **H3** premise fails | **selectivity ≈ 1** | Channel is uniformly large, not token-sparse; no speckles to explain |

H1 is the prior. H1-vs-H2 is the finding.

**Why not overlap-versus-null.** Two measurements that look natural are both unusable, and
the tests pin this down so the rule does not get "simplified" back into them:

- **`ρ` cannot separate H1 from H3.** Both show `ρ` ≈ 1 (planted data: 1.000 vs 0.993) —
  a uniformly large channel dominates every token's norm just as a sparse one dominates its
  outliers'. Only `selectivity` distinguishes them (1130 vs 3.4).
- **"IoU beats the scale-matched null" cannot detect H2.** Under genuine H2 the outlier
  tokens are elevated across the whole channel dimension, so a random scale-matched channel
  reproduces almost the same overlap (planted data: null 0.82 vs observed 1.00). The test is
  dead on arrival for the one hypothesis it would need to catch.

IoU, AUROC and both nulls are still computed and reported — they characterise the geometry
and populate `fig_overlap.png` — but they do not drive the verdict.

## Definitions

Per prompt, at a fixed layer, with image stream `X` of shape `(N_I, D)` — the block-output
residual stream already captured by `model_utils.register_capture_hooks`:

- `C_k` — top-*k* channels by `mean(abs(X))` over tokens (stage 2's abs-then-mean invariant)
- `m[n] = Σ_{c∈C_k} |X[n,c]|` — massive-channel score; this is what makes the speckles
- `N_full[n] = ‖X[n,:]‖₂`, `N_ex[n] = ‖X[n, D∖C_k]‖₂` — norm with massive channels excised
- `ρ[n] = ‖X[n,C_k]‖² / ‖X[n,:]‖²` — fraction of squared norm owned by the massive channels
- outlier set `O` = top `outlier_frac` tokens by `m` at a fixed `base_k` (default 2)

## Acceptance criteria (testable)

Automated, CPU-only, no torch/diffusers/matplotlib (`uv run pytest`):

- **AC1 — channel selection.** `top_channels` ranks by `mean(abs(X))` (abs then mean, matching
  `stage2.channel_scores`) with stable tie-breaking by ascending channel index; `k > D` raises.
- **AC2 — norm decomposition.** `‖X[n,C]‖² + ‖X[n,D∖C]‖² == ‖X[n,:]‖²` to float tolerance for
  arbitrary `C`; `norm_fraction` ∈ [0,1]; zero-norm tokens yield 0.0, not NaN.
- **AC3 — variance-explained curve (E1).** On planted data where one channel carries almost all
  the energy of a known token subset, `rho_outlier` ≈ 1 at k=1 while `rho_typical` stays small;
  the curve is monotone non-decreasing in k.
- **AC3b — decision rule.** `summarize` returns H1 on planted single-channel data, H2 on
  broadly-elevated data, and H3 on a uniformly-large (non-sparse) channel. Two guard tests
  assert the negative results above: `ρ` fails to separate H1/H3, and the scale-matched null
  tracks the signal under H2.
- **AC4 — overlap statistics (E2).** `overlap_stats` returns IoU, observed/expected intersection
  and a hypergeometric P(X ≥ obs); the p-value matches a brute-force enumeration; two empty sets
  give IoU 1.0; disjoint sets give IoU 0.0 and p ≈ 1.
- **AC5 — rank statistics.** `auroc` equals the Mann-Whitney statistic (checked against a
  brute-force pair count, ties counted as 0.5); `spearman` matches Pearson-on-ranks with
  average-rank tie handling; both agree with `scipy`/`sklearn` on random data.
- **AC6 — scale-matched null.** `scale_matched_channels` never returns a channel in `C_k`,
  always returns `k` distinct channels inside the requested descending-rank window, and is
  reproducible from a seeded generator.
- **AC7 — bimodality.** `bimodality_split` recovers a planted two-mode threshold on log-norms
  within tolerance, and its Sarle bimodality coefficient exceeds 5/9 for a 97/3 two-mode
  mixture while staying near 1/3 for **both** a single Gaussian **and** a heavy-tailed
  lognormal — a long tail is not Darcet's separated high-norm mode. (A 2-means split-quality
  metric fails this: it scores ~0.64 on all three.)
- **AC8 — neighbour redundancy.** `neighbor_cosine_similarity` averages cosine similarity over
  the 4-neighbourhood, handles grid edges (2–3 neighbours) and matches a hand-computed value on
  a small grid.
- **AC9 — lazy imports.** `src.common.highnorm` and `src.experiments.highnorm_tokens` import
  with no torch/diffusers/matplotlib installed.

Manual / Colab-only (require GPU + FLUX weights):

- **AC10 — artifacts.** A run writes `per_prompt.csv`, `summary.json`, and the figures
  `fig_variance_explained.png`, `fig_overlap.png`, `fig_norm_profile.png`, `fig_norm_hist.png`,
  `fig_spatial_panels.png`, `fig_position_stability.png`.
- **AC11 — verdict.** `summary.json` records H1/H2/H3 by the decision rule above, plus the
  deconfounded IoU against both nulls, and the layer at which each signature emerges.

## Nulls (both required)

1. **Scale-matched random channels** — drawn from descending-rank window `[null_rank_lo,
   null_rank_hi]` (default 50–500), *not* uniform over `D`. Uniform draws mostly hit
   near-dead channels and make any effect look significant.
2. **Token permutation** — shuffle `m` across tokens to fix the spatial-sparsity baseline.

## Deviations from Darcet et al. (logged to `summary.json`, not silently picked)

- **Neighbour redundancy** uses the layer-0 block output as the "early representation" proxy;
  FLUX has no ViT patch-embedding layer with the same semantics.
- **Threshold** is derived per-run by a 2-means split on log-norms rather than the hard-coded
  150, which is DINOv2-specific and stated in the paper to vary across models.
- **Last denoising step only**, inherited from the Figure 3 capture convention.
